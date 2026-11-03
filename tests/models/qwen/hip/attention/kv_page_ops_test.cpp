// Block-managed KV reads in the WMMA attention route.
//
// The paged path must be a pure addressing change: the same bytes, placed in
// blocks at a deliberately permuted order, have to produce a byte-identical
// output array. This test packs the canonical FP16 cache with a contiguous
// launch, copies the rows into a permuted block pool, and then re-runs the
// kernel through the block table. Any difference is an addressing bug, not a
// numerical one.
//
// The pool holds several layers and the run reads layer 1, so a wrong layer
// offset inside a block fails here as well.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/kv_page.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/kv_blocks.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kNumHeads = 24;
constexpr std::uint32_t kNumKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvWidth = kNumKvHeads * kHeadDim;
constexpr std::uint32_t kMaxContext = 1024;
constexpr std::uint32_t kBatch = 64;
constexpr std::uint32_t kLayer = 1;
constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kBlockTokens = gufo::models::qwen::kKvBlockTokens;
constexpr std::uint32_t kBlockShift = 7;
constexpr std::uint32_t kPoolBlocks = (kMaxContext / kBlockTokens) + 2;
constexpr std::size_t kPagedElements =
    std::uint64_t{kPoolBlocks} * kBlockTokens * kLayers * kKvWidth;

static_assert((1u << kBlockShift) == kBlockTokens, "block shift matches size");

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }

private:
  std::uint32_t state_;
};

struct DeviceBuffer {
  void* data{nullptr};
  explicit DeviceBuffer(std::size_t bytes) {
    HIP_CHECK(hipMalloc(&data, bytes));
    HIP_CHECK(hipMemset(data, 0, bytes));
  }
  ~DeviceBuffer() {
    if (data != nullptr)
      (void)hipFree(data);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void Fail(const char* message) {
  std::cerr << "paged KV attention test failure: " << message << '\n';
  std::abort();
}

}  // namespace

int main() {
  if (gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "qwen_kv_page_ops_test") != 0) {
    return gufo::test::kCtestSkipReturnCode;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t query_elements =
      static_cast<std::size_t>(kBatch) * attention_width;
  const std::size_t kv_elements =
      static_cast<std::size_t>(kLayers) * kMaxContext * kKvWidth;

  DeviceBuffer d_q(query_elements * sizeof(float));
  DeviceBuffer d_k(static_cast<std::size_t>(kBatch) * kKvWidth * sizeof(float));
  DeviceBuffer d_v(static_cast<std::size_t>(kBatch) * kKvWidth * sizeof(float));
  DeviceBuffer d_gate(query_elements * sizeof(float));
  DeviceBuffer d_cache(static_cast<std::size_t>(kLayers) * kMaxContext *
                       kKvWidth * sizeof(__half));
  DeviceBuffer d_cache_v(static_cast<std::size_t>(kLayers) * kMaxContext *
                         kKvWidth * sizeof(__half));
  DeviceBuffer d_out(query_elements * sizeof(float));
  DeviceBuffer d_paged(kPagedElements * sizeof(__half));
  DeviceBuffer d_paged_v(kPagedElements * sizeof(__half));

  std::vector<float> host_q(query_elements);
  std::vector<float> host_k(static_cast<std::size_t>(kBatch) * kKvWidth);
  std::vector<float> host_v(host_k.size());
  std::vector<float> host_gate(query_elements);
  Rng rng(90210);
  for (auto& value : host_q)
    value = rng.Uniform(-1.0F, 1.0F);
  for (auto& value : host_gate)
    value = rng.Uniform(-4.0F, 4.0F);
  for (std::size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = rng.Uniform(-1.0F, 1.0F);
    host_v[i] = rng.Uniform(-1.0F, 1.0F);
  }

  HIP_CHECK(hipMemcpy(d_q.data, host_q.data(), host_q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate.data, host_gate.data(),
                      host_gate.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k.data, host_k.data(), host_k.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v.data, host_v.data(), host_v.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  // Contiguous launch writes the canonical FP16 cache and its output.
  if (!gufo::hip::LaunchQwenWmmaAttention(
          static_cast<const float*>(d_q.data),
          static_cast<const float*>(d_k.data),
          static_cast<const float*>(d_v.data),
          static_cast<const float*>(d_gate.data), nullptr, nullptr,
          d_cache.data, d_cache_v.data, static_cast<float*>(d_out.data), kLayer,
          0, kBatch, kMaxContext, kNumHeads, kNumKvHeads, kHeadDim)) {
    Fail("contiguous launch rejected the production shape");
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Read both packed planes back and rebuild them in permuted blocks.
  gufo::models::qwen::KvBlockPool pool(kPoolBlocks);
  gufo::models::qwen::KvBlockTable table(kBlockTokens, kMaxContext);
  std::vector<std::uint32_t> host_table(pool.capacity());
  for (std::uint32_t index = 0; index < pool.capacity(); ++index) {
    host_table[index] = pool.capacity() - 1 - index;  // reverse the pool
  }
  if (table.Append(pool, kMaxContext) != kMaxContext)
    Fail("block table could not cover the context");

  const std::uint64_t block_stride =
      std::uint64_t{kBlockTokens} * kLayers * kKvWidth;
  const std::size_t plane_elements =
      static_cast<std::size_t>(kLayers) * kMaxContext * kKvWidth;

  const auto repack = [&](void* device_plane, DeviceBuffer& destination) {
    std::vector<__half> source(plane_elements);
    HIP_CHECK(hipMemcpy(source.data(), device_plane,
                        source.size() * sizeof(__half), hipMemcpyDeviceToHost));
    std::vector<__half> target(kPagedElements);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
      for (std::uint32_t pos = 0; pos < kMaxContext; ++pos) {
        const std::uint32_t entry = pos / kBlockTokens;
        const std::uint64_t from =
            (std::uint64_t{layer} * kMaxContext + pos) * kKvWidth;
        const std::uint64_t to =
            std::uint64_t{host_table[entry]} * block_stride +
            std::uint64_t{layer} * kBlockTokens * kKvWidth +
            std::uint64_t{pos % kBlockTokens} * kKvWidth;
        std::memcpy(target.data() + to, source.data() + from,
                    kKvWidth * sizeof(__half));
      }
    }
    HIP_CHECK(hipMemcpy(destination.data, target.data(),
                        target.size() * sizeof(__half), hipMemcpyHostToDevice));
  };
  repack(d_cache.data, d_paged);
  repack(d_cache_v.data, d_paged_v);

  DeviceBuffer d_table(pool.capacity() * sizeof(std::uint32_t));
  HIP_CHECK(hipMemcpy(d_table.data, host_table.data(),
                      host_table.size() * sizeof(std::uint32_t),
                      hipMemcpyHostToDevice));

  const gufo::hip::KvPageView view{
      .table = static_cast<const std::uint32_t*>(d_table.data),
      .block_stride = block_stride,
      .layer_stride = std::uint64_t{kBlockTokens} * kKvWidth,
      .context_stride = std::uint64_t{kMaxContext} * kKvWidth,
      .block_shift = kBlockShift,
  };

  DeviceBuffer d_paged_out(query_elements * sizeof(float));
  if (!gufo::hip::LaunchQwenWmmaAttention(
          static_cast<const float*>(d_q.data), nullptr, nullptr,
          static_cast<const float*>(d_gate.data), nullptr, nullptr,
          d_paged.data, d_paged_v.data, static_cast<float*>(d_paged_out.data),
          kLayer, 0, kBatch, kMaxContext, kNumHeads, kNumKvHeads, kHeadDim,
          nullptr, nullptr, /*key_begin=*/0, /*skip_kv_write=*/true, {}, {},
          /*first_query=*/0, &view)) {
    Fail("paged launch rejected the production shape");
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> contiguous_out(query_elements);
  std::vector<float> paged_out(query_elements);
  HIP_CHECK(hipMemcpy(contiguous_out.data(), d_out.data,
                      contiguous_out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(paged_out.data(), d_paged_out.data,
                      paged_out.size() * sizeof(float), hipMemcpyDeviceToHost));

  if (std::memcmp(contiguous_out.data(), paged_out.data(),
                  contiguous_out.size() * sizeof(float)) != 0) {
    double worst = 0.0;
    for (std::size_t i = 0; i < contiguous_out.size(); ++i)
      worst = std::max(worst, static_cast<double>(
                                  std::fabs(contiguous_out[i] - paged_out[i])));
    std::cerr << "paged attention output differs, worst |diff| " << worst
              << '\n';
    return 1;
  }

  std::cout
      << "paged KV attention matches the contiguous cache byte for byte\n";
  return 0;
}
