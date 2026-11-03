// D4 prototype: KV cache in FP8 (E4M3) versus the production FP16 layout.
//
// The bench keeps the geometry of Qwen3.8-Flash-Next full attention (24 query
// heads, 2 KV heads, head_dim 256) and runs the same scalar causal-attention
// kernel twice: once reading FP16 K/V, once reading E4M3 K/V with one scale per
// (token, KV head). Both arms share the arithmetic, so the timing delta is the
// price of the narrower storage plus the on-the-fly dequantization, and the
// output comparison bounds the quality cost.
//
// Build (standalone, without CMake):
//   tools/bench/build.sh kv_fp8_attention  # in the dev shell
// Standalone compile used for this measurement (store paths from the nix
// toolchain):
//   amdclang++ -x hip --offload-arch=gfx1151 -O3 kv_fp8_attention.hip \
//     --rocm-path=<clr> --rocm-device-lib-path=<device-libs>/amdgcn/bitcode \
//     -L<clr>/lib -L<rocm-runtime>/lib -lamdhip64
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kQHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kGqa = kQHeads / kKvHeads;
constexpr float kScale = 1.0F / 16.0F;  // 1/sqrt(kDim)

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

// OCP E4M3 (fn): 1 sign, 4 exponent (bias 7), 3 mantissa, S.1111.111 is NaN.
float E4M3Value(std::uint32_t code) {
  const bool negative = (code & 0x80U) != 0;
  const std::uint32_t exponent = (code >> 3) & 0xFU;
  const std::uint32_t mantissa = code & 0x7U;
  if (exponent == 0xFU && mantissa == 0x7U)
    return std::nanf("");
  const float magnitude =
      exponent == 0 ? static_cast<float>(mantissa) / 512.0F
                    : (1.0F + static_cast<float>(mantissa) / 8.0F) *
                          std::ldexp(1.0F, static_cast<int>(exponent) - 7);
  return negative ? -magnitude : magnitude;
}

std::uint8_t E4M3Encode(float value) {
  static const std::vector<float> kValues = [] {
    std::vector<float> values;
    for (std::uint32_t code = 0; code < 0x80U; ++code) {
      const float v = E4M3Value(code);
      if (!std::isnan(v))
        values.push_back(v);
    }
    return values;
  }();
  const bool negative = value < 0.0F;
  const float magnitude = std::min(std::fabs(value), 448.0F);
  // Nearest representative on the positive side; the table is ascending.
  const auto it = std::lower_bound(kValues.begin(), kValues.end(), magnitude);
  std::uint32_t code = 0;
  if (it == kValues.begin()) {
    code = 0;
  } else if (it == kValues.end()) {
    code = 0x7EU;
  } else {
    const auto below = std::prev(it);
    code = static_cast<std::uint32_t>(it - kValues.begin());
    if (magnitude - *below < *it - magnitude)
      code = static_cast<std::uint32_t>(below - kValues.begin());
  }
  // Recover the code from the value so sign is applied to the same magnitude.
  for (std::uint32_t candidate = 0; candidate < 0x80U; ++candidate) {
    if (E4M3Value(candidate) == kValues[code]) {
      code = candidate;
      break;
    }
  }
  return static_cast<std::uint8_t>(code | (negative ? 0x80U : 0));
}

__device__ __forceinline__ float E4M3ToFloat(std::uint32_t code) {
  const bool negative = (code & 0x80U) != 0;
  const std::uint32_t exponent = (code >> 3) & 0xFU;
  const std::uint32_t mantissa = code & 0x7U;
  float magnitude;
  if (exponent == 0) {
    magnitude = static_cast<float>(mantissa) * (1.0F / 512.0F);
  } else {
    magnitude = (1.0F + static_cast<float>(mantissa) * 0.125F) *
                exp2f(static_cast<float>(exponent) - 7.0F);
  }
  return negative ? -magnitude : magnitude;
}

__device__ __forceinline__ float WarpSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, value, offset);
  return value;
}

// One warp per (query row, query head); eight warps per block. Online softmax
// over the causal prefix. k/v are [token][kKvHeads][kDim].
__global__ void AttentionF16Kernel(const float* q, const __half* k,
                                   const __half* v, float* out,
                                   std::uint32_t rows, std::uint32_t depth) {
  const std::uint32_t item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= rows * kQHeads)
    return;
  const std::uint32_t lane = threadIdx.x % 32;
  const std::uint32_t row = item / kQHeads;
  const std::uint32_t head = item % kQHeads;
  const std::uint32_t kvh = head / kGqa;
  const std::uint32_t position = depth - rows + row;
  float qv[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i)
    qv[i] = q[(row * kQHeads + head) * kDim + lane + i * 32];
  float acc[8] = {};
  float max = -INFINITY;
  float sum = 0.0F;
  for (std::uint32_t t = 0; t <= position; ++t) {
    const __half* key = k + (t * kKvHeads + kvh) * kDim;
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i)
      score += qv[i] * __half2float(key[lane + i * 32]);
    score = WarpSum(score) * kScale;
    score = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, score, 0);
    const float next = fmaxf(max, score);
    const float correction = expf(max - next);
    const float weight = expf(score - next);
    max = next;
    sum = sum * correction + weight;
    const __half* value = v + (t * kKvHeads + kvh) * kDim;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i)
      acc[i] =
          acc[i] * correction + weight * __half2float(value[lane + i * 32]);
  }
  float* destination = out + (row * kQHeads + head) * kDim;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i)
    destination[lane + i * 32] = acc[i] / sum;
}

// Same kernel over E4M3 K/V with one scale per (token, KV head).
__global__ void AttentionFp8Kernel(const float* q, const std::uint8_t* k,
                                   const float* k_scale, const std::uint8_t* v,
                                   const float* v_scale, float* out,
                                   std::uint32_t rows, std::uint32_t depth) {
  const std::uint32_t item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= rows * kQHeads)
    return;
  const std::uint32_t lane = threadIdx.x % 32;
  const std::uint32_t row = item / kQHeads;
  const std::uint32_t head = item % kQHeads;
  const std::uint32_t kvh = head / kGqa;
  const std::uint32_t position = depth - rows + row;
  float qv[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i)
    qv[i] = q[(row * kQHeads + head) * kDim + lane + i * 32];
  float acc[8] = {};
  float max = -INFINITY;
  float sum = 0.0F;
  for (std::uint32_t t = 0; t <= position; ++t) {
    const std::uint8_t* key = k + (t * kKvHeads + kvh) * kDim;
    const float ks = k_scale[t * kKvHeads + kvh];
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i)
      score += qv[i] * E4M3ToFloat(key[lane + i * 32]) * ks;
    score = WarpSum(score) * kScale;
    score = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, score, 0);
    const float next = fmaxf(max, score);
    const float correction = expf(max - next);
    const float weight = expf(score - next);
    max = next;
    sum = sum * correction + weight;
    const std::uint8_t* value = v + (t * kKvHeads + kvh) * kDim;
    const float vs = v_scale[t * kKvHeads + kvh];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i)
      acc[i] =
          acc[i] * correction + weight * E4M3ToFloat(value[lane + i * 32]) * vs;
  }
  float* destination = out + (row * kQHeads + head) * kDim;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i)
    destination[lane + i * 32] = acc[i] / sum;
}

struct Result {
  double median_ms;
  std::vector<float> out;
};

Result Time(const std::vector<float>& q, const std::vector<__half>& k,
            const std::vector<__half>& v, const std::vector<std::uint8_t>& k8,
            const std::vector<float>& ks, const std::vector<std::uint8_t>& v8,
            const std::vector<float>& vs, std::uint32_t rows,
            std::uint32_t depth, bool fp8, int reps) {
  float* d_q = Upload(q);
  float* d_out = nullptr;
  CheckHip(hipMalloc(&d_out, q.size() / kQHeads * kDim * sizeof(float)), "out");
  __half* d_k = nullptr;
  __half* d_v = nullptr;
  std::uint8_t* d_k8 = nullptr;
  std::uint8_t* d_v8 = nullptr;
  float* d_ks = nullptr;
  float* d_vs = nullptr;
  if (fp8) {
    d_k8 = Upload(k8);
    d_v8 = Upload(v8);
    d_ks = Upload(ks);
    d_vs = Upload(vs);
  } else {
    d_k = Upload(k);
    d_v = Upload(v);
  }
  const std::uint32_t threads = 256;
  const std::uint32_t blocks =
      (rows * kQHeads + threads / 32 - 1) / (threads / 32);
  const auto launch = [&] {
    if (fp8)
      AttentionFp8Kernel<<<blocks, threads>>>(d_q, d_k8, d_ks, d_v8, d_vs,
                                              d_out, rows, depth);
    else
      AttentionF16Kernel<<<blocks, threads>>>(d_q, d_k, d_v, d_out, rows,
                                              depth);
  };
  launch();
  CheckHip(hipDeviceSynchronize(), "warmup");
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  std::vector<float> times(reps);
  for (int i = 0; i < reps; ++i) {
    CheckHip(hipEventRecord(start), "record");
    launch();
    CheckHip(hipEventRecord(stop), "record");
    CheckHip(hipEventSynchronize(stop), "sync");
    CheckHip(hipEventElapsedTime(&times[i], start, stop), "elapsed");
  }
  std::sort(times.begin(), times.end());
  Result result;
  result.median_ms = times[reps / 2];
  result.out.resize(q.size() / kQHeads * kDim);
  CheckHip(hipMemcpy(result.out.data(), d_out,
                     result.out.size() * sizeof(float), hipMemcpyDeviceToHost),
           "download");
  for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_out),
                  static_cast<void*>(d_k), static_cast<void*>(d_v),
                  static_cast<void*>(d_k8), static_cast<void*>(d_v8),
                  static_cast<void*>(d_ks), static_cast<void*>(d_vs)})
    if (p != nullptr)
      CheckHip(hipFree(p), "free");
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return result;
}

}  // namespace

int main(int argc, char** argv) try {
  std::uint32_t depth = argc > 1 ? std::atoi(argv[1]) : 2048;
  std::uint32_t rows = argc > 2 ? std::atoi(argv[2]) : 64;
  int reps = argc > 3 ? std::atoi(argv[3]) : 5;

  std::uint32_t seed = 0x9E3779B9U;
  const auto next = [&seed] {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  const auto value = [&next](float scale) {
    return scale * (static_cast<float>(next() % 2001) / 1000.0F - 1.0F);
  };
  std::vector<float> q(static_cast<std::size_t>(rows) * kQHeads * kDim);
  for (float& x : q)
    x = value(1.0F);
  const std::size_t kv = static_cast<std::size_t>(depth) * kKvHeads * kDim;
  std::vector<__half> k(kv), v(kv);
  for (std::size_t i = 0; i < kv; ++i) {
    k[i] = __float2half(value(1.0F));
    v[i] = __float2half(value(1.0F));
  }
  // E4M3 with one absmax scale per (token, KV head).
  std::vector<std::uint8_t> k8(kv), v8(kv);
  std::vector<float> k_scale(static_cast<std::size_t>(depth) * kKvHeads);
  std::vector<float> v_scale(static_cast<std::size_t>(depth) * kKvHeads);
  for (std::uint32_t t = 0; t < depth; ++t) {
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      const std::size_t base =
          (static_cast<std::size_t>(t) * kKvHeads + h) * kDim;
      float kmax = 0.0F;
      float vmax = 0.0F;
      for (std::uint32_t d = 0; d < kDim; ++d) {
        kmax = std::max(kmax, std::fabs(__half2float(k[base + d])));
        vmax = std::max(vmax, std::fabs(__half2float(v[base + d])));
      }
      const float kscale = kmax > 0.0F ? kmax / 448.0F : 1.0F;
      const float vscale = vmax > 0.0F ? vmax / 448.0F : 1.0F;
      k_scale[static_cast<std::size_t>(t) * kKvHeads + h] = kscale;
      v_scale[static_cast<std::size_t>(t) * kKvHeads + h] = vscale;
      for (std::uint32_t d = 0; d < kDim; ++d) {
        k8[base + d] = E4M3Encode(__half2float(k[base + d]) / kscale);
        v8[base + d] = E4M3Encode(__half2float(v[base + d]) / vscale);
      }
    }
  }

  const auto f16 =
      Time(q, k, v, k8, k_scale, v8, v_scale, rows, depth, false, reps);
  const auto fp8 =
      Time(q, k, v, k8, k_scale, v8, v_scale, rows, depth, true, reps);
  double worst = 0.0;
  double sq = 0.0;
  double ref_sq = 0.0;
  double dot = 0.0;
  double out_sq = 0.0;
  for (std::size_t i = 0; i < f16.out.size(); ++i) {
    const double a = f16.out[i];
    const double b = fp8.out[i];
    worst = std::max(worst, std::abs(a - b));
    sq += (a - b) * (a - b);
    ref_sq += a * a;
    dot += a * b;
    out_sq += b * b;
  }
  const double rms_error = std::sqrt(sq / static_cast<double>(f16.out.size()));
  const double ref_rms =
      std::sqrt(ref_sq / static_cast<double>(f16.out.size()));
  const double cosine = dot / std::sqrt(ref_sq * out_sq + 1e-30);
  const double fp16_bytes = 2.0 * kKvHeads * kDim * sizeof(__half);
  const double fp8_bytes = 2.0 * kKvHeads * kDim * sizeof(std::uint8_t) +
                           2.0 * kKvHeads * sizeof(float);
  std::printf(
      "{\n"
      "  \"geometry\": {\"depth\": %u, \"query_rows\": %u, \"q_heads\": %u, "
      "\"kv_heads\": %u, \"head_dim\": %u},\n"
      "  \"bytes_per_token\": {\"fp16\": %.0f, \"fp8_e4m3\": %.0f, "
      "\"ratio\": %.2f},\n"
      "  \"speed_ms_median\": {\"fp16\": %.3f, \"fp8_e4m3\": %.3f, "
      "\"slowdown_pct\": %.1f},\n"
      "  \"quality\": {\"rms_error\": %.5f, \"rms_over_reference\": %.5f, "
      "\"max_abs_error\": %.5f, \"cosine\": %.6f}\n"
      "}\n",
      depth, rows, kQHeads, kKvHeads, kDim, fp16_bytes, fp8_bytes,
      fp16_bytes / fp8_bytes, f16.median_ms, fp8.median_ms,
      100.0 * (fp8.median_ms - f16.median_ms) / f16.median_ms, rms_error,
      rms_error / (ref_rms + 1e-30), worst, cosine);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
