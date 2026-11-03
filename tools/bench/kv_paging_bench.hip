// Paged vs contiguous KV addressing for decode attention.
//
// The production change replaces `(layer * max_context + position)` with a
// block-table lookup. This benchmark measures what that indirection costs on
// gfx1151 before any production kernel is touched, and proves the two address
// forms produce identical output.
//
// Geometry follows Qwen3.8-Flash-Next: 2 KV heads, head_dim 256, one block per
// 128 positions, one block covering every layer. Each lane reads one 16-byte
// fragment per position for K and one for V, which is the access pattern the
// WMMA route uses after its 16-key tiles are staged.
//
//   nix develop -c tools/bench/build.sh kv_paging_bench
//   /tmp/kv_paging_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::uint32_t kBlockTokens = 128;
constexpr std::uint32_t kLayers = 12;
constexpr std::uint32_t kFragments = kHeadDim / 8;  // 16-byte fragments per row

void Check(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
    std::exit(1);
  }
}

template<typename T>
struct Buffer {
  T* data{nullptr};
  explicit Buffer(std::size_t count) {
    Check(hipMalloc(&data, count * sizeof(T)), "hipMalloc");
    Check(hipMemset(data, 0, count * sizeof(T)), "hipMemset");
  }
  ~Buffer() {
    if (data != nullptr)
      (void)hipFree(data);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

// One wave per query head, exactly the decode shape: lane `l` owns the 8
// dimensions [l*8, l*8+8), so one 16-byte fragment load per lane covers the
// whole 256-dim row.
//
// kMode 0 reads the contiguous cache, 1 resolves the block table per position
// (the naive form), 2 hoists the lookup to the start of each block, which is
// what a 16-key WMMA tile can do because a tile never crosses a block.
template<int kMode>
__global__ void DecodeAttention(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, float* __restrict__ out,
    const std::uint32_t* __restrict__ table, std::uint64_t block_stride,
    std::uint32_t layer, std::uint32_t position, std::uint32_t max_context,
    std::uint32_t kv_head) {
  const std::uint32_t lane = threadIdx.x;
  const float* query = q + (lane * 8);
  const std::uint32_t head_offset = kv_head * kHeadDim;

  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  float accumulator[8] = {};

  std::uint64_t block_base = 0;
  if constexpr (kMode != 0)
    block_base = static_cast<std::uint64_t>(table[0]) * block_stride +
                 static_cast<std::uint64_t>(layer) * kBlockTokens * kKvWidth;
  const std::uint64_t contiguous_base =
      static_cast<std::uint64_t>(layer) * max_context * kKvWidth;

  for (std::uint32_t pos = 0; pos <= position; ++pos) {
    std::uint64_t row = 0;
    if constexpr (kMode == 0) {
      row = contiguous_base + static_cast<std::uint64_t>(pos) * kKvWidth;
    } else if constexpr (kMode == 1) {
      row =
          static_cast<std::uint64_t>(table[pos / kBlockTokens]) * block_stride +
          static_cast<std::uint64_t>(layer) * kBlockTokens * kKvWidth +
          static_cast<std::uint64_t>(pos % kBlockTokens) * kKvWidth;
    } else {
      if ((pos & (kBlockTokens - 1)) == 0)
        block_base =
            static_cast<std::uint64_t>(table[pos / kBlockTokens]) *
                block_stride +
            static_cast<std::uint64_t>(layer) * kBlockTokens * kKvWidth;
      row = block_base +
            static_cast<std::uint64_t>(pos & (kBlockTokens - 1)) * kKvWidth;
    }
    row += head_offset;

    const uint4 key_fragment =
        *reinterpret_cast<const uint4*>(k_cache + row + lane * 8);
    const uint4 value_fragment =
        *reinterpret_cast<const uint4*>(v_cache + row + lane * 8);

    const auto* key_half = reinterpret_cast<const __half*>(&key_fragment);
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t d = 0; d < 8; ++d)
      score += query[d] * __half2float(key_half[d]);
#pragma unroll
    for (std::uint32_t offset = 16; offset > 0; offset >>= 1)
      score += __shfl_xor(score, offset);
    score *= scale;

    const float new_max = fmaxf(running_max, score);
    const float previous = expf(running_max - new_max);
    const float weight = expf(score - new_max);
    running_sum = running_sum * previous + weight;
    running_max = new_max;
    const auto* value_half = reinterpret_cast<const __half*>(&value_fragment);
#pragma unroll
    for (std::uint32_t d = 0; d < 8; ++d) {
      accumulator[d] =
          accumulator[d] * previous + weight * __half2float(value_half[d]);
    }
  }

  float* row_out = out + (lane * 8);
#pragma unroll
  for (std::uint32_t d = 0; d < 8; ++d) {
    row_out[d] = accumulator[d] / running_sum;
  }
}

struct Case {
  std::uint32_t context;
  std::uint32_t repetitions;
};

double Median(std::vector<double>& values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int Run(std::uint32_t context, std::uint32_t repetitions, std::uint32_t layer) {
  const std::uint32_t blocks = (context + kBlockTokens - 1) / kBlockTokens + 3;
  const std::uint64_t block_stride =
      std::uint64_t{kBlockTokens} * kLayers * kKvWidth;
  const std::size_t cache_elements =
      std::uint64_t{kLayers} * context * kKvWidth;

  Buffer<__half> contiguous_k(cache_elements);
  Buffer<__half> contiguous_v(cache_elements);
  Buffer<__half> paged_k(std::size_t{blocks} * block_stride);
  Buffer<__half> paged_v(std::size_t{blocks} * block_stride);
  Buffer<std::uint32_t> table(blocks);
  Buffer<float> q(kHeads * kHeadDim);
  Buffer<float> out_contiguous(kHeadDim);
  Buffer<float> out_paged(kHeadDim);

  std::vector<__half> host_k(cache_elements);
  std::vector<__half> host_v(cache_elements);
  for (std::size_t i = 0; i < cache_elements; ++i) {
    host_k[i] = __float2half_rn(std::sin(static_cast<double>(i) * 1e-3));
    host_v[i] = __float2half_rn(std::cos(static_cast<double>(i) * 7e-4));
  }
  Check(hipMemcpy(contiguous_k.data, host_k.data(),
                  cache_elements * sizeof(__half), hipMemcpyHostToDevice),
        "k upload");
  Check(hipMemcpy(contiguous_v.data, host_v.data(),
                  cache_elements * sizeof(__half), hipMemcpyHostToDevice),
        "v upload");

  // Deliberately permuted block order, so sequential table entries walk the
  // pool backwards. A cache-friendly mapping would understate the cost.
  std::vector<std::uint32_t> host_table(blocks);
  for (std::uint32_t i = 0; i < blocks; ++i)
    host_table[i] = blocks - 1 - i;
  Check(hipMemcpy(table.data, host_table.data(), blocks * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice),
        "table upload");

  // Copy every layer's rows into the permuted blocks.
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    for (std::uint32_t pos = 0; pos < context; ++pos) {
      const std::uint32_t index = pos / kBlockTokens;
      const std::uint32_t block = host_table[index];
      const std::uint64_t source =
          (std::uint64_t{l} * context + pos) * kKvWidth;
      const std::uint64_t target = std::uint64_t{block} * block_stride +
                                   std::uint64_t{l} * kBlockTokens * kKvWidth +
                                   std::uint64_t{pos % kBlockTokens} * kKvWidth;
      Check(hipMemcpy(paged_k.data + target, host_k.data() + source,
                      kKvWidth * sizeof(__half), hipMemcpyHostToDevice),
            "k block copy");
      Check(hipMemcpy(paged_v.data + target, host_v.data() + source,
                      kKvWidth * sizeof(__half), hipMemcpyHostToDevice),
            "v block copy");
    }
  }

  std::vector<float> host_q(kHeads * kHeadDim);
  for (std::size_t i = 0; i < host_q.size(); ++i)
    host_q[i] = std::sin(static_cast<double>(i) * 0.017);
  Check(hipMemcpy(q.data, host_q.data(), host_q.size() * sizeof(float),
                  hipMemcpyHostToDevice),
        "q upload");

  // One KV head, as in the real kernel: heads 0..11 share it.
  const std::uint32_t kv_head = 0;
  auto launch = [&](int mode, float* out) {
    switch (mode) {
      case 0:
        DecodeAttention<0><<<1, 32>>>(q.data, contiguous_k.data,
                                      contiguous_v.data, out, nullptr, 0, layer,
                                      context - 1, context, kv_head);
        break;
      case 1:
        DecodeAttention<1><<<1, 32>>>(q.data, paged_k.data, paged_v.data, out,
                                      table.data, block_stride, layer,
                                      context - 1, context, kv_head);
        break;
      default:
        DecodeAttention<2><<<1, 32>>>(q.data, paged_k.data, paged_v.data, out,
                                      table.data, block_stride, layer,
                                      context - 1, context, kv_head);
        break;
    }
  };

  std::vector<float> reference(kHeadDim);
  const std::array<std::uint32_t, 3> outputs = {static_cast<std::uint32_t>(0),
                                                static_cast<std::uint32_t>(1),
                                                static_cast<std::uint32_t>(2)};
  Buffer<float> out_a(kHeadDim);
  Buffer<float> out_b(kHeadDim);
  Buffer<float> out_c(kHeadDim);
  float* out_buffers[3] = {out_a.data, out_b.data, out_c.data};

  launch(0, out_a.data);
  Check(hipDeviceSynchronize(), "sync");
  Check(hipMemcpy(reference.data(), out_a.data, kHeadDim * sizeof(float),
                  hipMemcpyDeviceToHost),
        "reference");

  double worst[3] = {0.0, 0.0, 0.0};
  for (std::uint32_t mode = 1; mode < 3; ++mode) {
    std::vector<float> result(kHeadDim);
    launch(static_cast<int>(mode), out_buffers[mode]);
    Check(hipDeviceSynchronize(), "sync");
    Check(hipMemcpy(result.data(), out_buffers[mode], kHeadDim * sizeof(float),
                    hipMemcpyDeviceToHost),
          "result");
    for (std::uint32_t i = 0; i < kHeadDim; ++i)
      worst[mode] =
          std::max(worst[mode],
                   static_cast<double>(std::fabs(reference[i] - result[i])));
  }

  std::array<std::vector<double>, 3> samples;
  hipEvent_t start_event{};
  hipEvent_t stop_event{};
  Check(hipEventCreate(&start_event), "event");
  Check(hipEventCreate(&stop_event), "event");

  for (std::uint32_t r = 0; r < repetitions + 2; ++r) {
    for (std::uint32_t mode = 0; mode < 3; ++mode) {
      Check(hipEventRecord(start_event), "record");
      launch(static_cast<int>(mode), out_buffers[mode]);
      Check(hipEventRecord(stop_event), "record");
      Check(hipEventSynchronize(stop_event), "sync");
      float ms = 0.0F;
      Check(hipEventElapsedTime(&ms, start_event, stop_event), "elapsed");
      if (r >= 2)
        samples[mode].push_back(ms);
    }
  }
  Check(hipEventDestroy(start_event), "destroy");
  Check(hipEventDestroy(stop_event), "destroy");

  // One head reads `context * head_dim` halves for K and the same for V.
  const double bytes = 2.0 * context * kHeadDim * sizeof(__half);
  const double contiguous = Median(samples[0]);
  const double per_position = Median(samples[1]);
  const double hoisted = Median(samples[2]);
  std::printf(
      "context %6u | contiguous %7.3f ms | per-position %7.3f ms (%+5.1f%%) "
      "| hoisted %7.3f ms (%+5.1f%%) | %6.2f GB/s contiguous | max|diff| "
      "%.3g/%.3g\n",
      context, contiguous, per_position,
      100.0 * (per_position - contiguous) / contiguous, hoisted,
      100.0 * (hoisted - contiguous) / contiguous, bytes / (contiguous * 1e6),
      worst[1], worst[2]);

  if (worst[1] > 0.0 || worst[2] > 0.0) {
    std::fprintf(stderr, "paged and contiguous results differ\n");
    return 1;
  }
  (void)outputs;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t layer = 0;
  if (argc > 1)
    layer = static_cast<std::uint32_t>(std::atoi(argv[1]));

  std::printf(
      "gfx1151 paged-vs-contiguous decode attention, layer %u, "
      "block %u tokens, %u KiB per block\n",
      layer, kBlockTokens,
      static_cast<unsigned>(std::uint64_t{kBlockTokens} * kLayers * kKvWidth *
                            sizeof(__half) / 1024));

  int failures = 0;
  for (const Case& item :
       {Case{1024, 20}, Case{4096, 12}, Case{16384, 6}, Case{65536, 3}}) {
    failures += Run(item.context, item.repetitions, layer);
  }
  return failures == 0 ? 0 : 1;
}
