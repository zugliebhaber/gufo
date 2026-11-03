// Block-managed KV reads in the Flash-Next attention routes.
//
// Both readers (the WMMA route for wide batches and the per-token route used
// by decode and verification) must be pure addressing changes: the same bytes,
// placed in blocks at a deliberately permuted order, have to produce a
// byte-identical output. Dense WMMA tiles and the sparse per-token selection
// both resolve the block table, so a wrong block, layer or in-block position
// fails here.
//
// The pool holds two layers and the run reads layer 1.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/kv_page.hpp"
#include "src/models/qwen/kv_blocks.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;

namespace {

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kMaxContext = 1024;
constexpr std::uint32_t kTokens = 64;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kLayer = 1;
constexpr std::uint32_t kBlockTokens = gufo::models::qwen::kKvBlockTokens;
constexpr std::uint32_t kBlockShift = 7;
constexpr std::uint32_t kPoolBlocks = (kMaxContext / kBlockTokens) + 2;
constexpr std::size_t kPagedElements =
    std::uint64_t{kPoolBlocks} * kBlockTokens * kLayers * kKvWidth;

static_assert((1u << kBlockShift) == kBlockTokens, "block shift matches size");

void Fail(const char* message) {
  std::cerr << "flash-next paged KV test failure: " << message << '\n';
  std::abort();
}

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

struct Case {
  const char* name;
  bool wmma;
};

}  // namespace

int main() {
  if (gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "qwen38_flash_next_kv_page_ops_test") != 0) {
    return gufo::test::kCtestSkipReturnCode;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t query_elements =
      static_cast<std::size_t>(kTokens) * attention_width;
  const std::size_t plane_elements =
      static_cast<std::size_t>(kLayers) * kMaxContext * kKvWidth;

  DeviceBuffer d_q(query_elements * sizeof(float));
  DeviceBuffer d_gate(query_elements * sizeof(float));
  DeviceBuffer d_cache(plane_elements * sizeof(__half));
  DeviceBuffer d_cache_v(plane_elements * sizeof(__half));
  DeviceBuffer d_paged(kPagedElements * sizeof(__half));
  DeviceBuffer d_paged_v(kPagedElements * sizeof(__half));
  DeviceBuffer d_start(2 * sizeof(std::uint32_t));
  DeviceBuffer d_table(kPoolBlocks * sizeof(std::uint32_t));

  // Distinct, deterministic rows: a wrong block or layer shows up as a
  // different value, not as rounding noise (the comparison is byte-exact).
  std::vector<__half> host_k(plane_elements);
  std::vector<__half> host_v(plane_elements);
  for (std::size_t i = 0; i < plane_elements; ++i) {
    host_k[i] = __float2half_rn(std::sin(static_cast<double>(i) * 1e-3));
    host_v[i] = __float2half_rn(std::cos(static_cast<double>(i) * 7e-4));
  }
  HIP_CHECK(hipMemcpy(d_cache.data, host_k.data(),
                      host_k.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_cache_v.data, host_v.data(),
                      host_v.size() * sizeof(__half), hipMemcpyHostToDevice));

  // Repack both planes so that block ids walk the pool backwards.
  gufo::models::qwen::KvBlockPool pool(kPoolBlocks);
  gufo::models::qwen::KvBlockTable table(kBlockTokens, kMaxContext);
  std::vector<std::uint32_t> host_table(kPoolBlocks);
  for (std::uint32_t index = 0; index < kPoolBlocks; ++index)
    host_table[index] = kPoolBlocks - 1 - index;
  if (table.Append(pool, kMaxContext) != kMaxContext)
    Fail("block table could not cover the context");
  HIP_CHECK(hipMemcpy(d_table.data, host_table.data(),
                      host_table.size() * sizeof(std::uint32_t),
                      hipMemcpyHostToDevice));

  const std::uint64_t block_stride =
      std::uint64_t{kBlockTokens} * kLayers * kKvWidth;
  const auto repack = [&](const std::vector<__half>& source,
                          DeviceBuffer& destination) {
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
  repack(host_k, d_paged);
  repack(host_v, d_paged_v);

  const gufo::hip::KvPageView view{
      .table = static_cast<const std::uint32_t*>(d_table.data),
      .block_stride = block_stride,
      .layer_stride = std::uint64_t{kBlockTokens} * kKvWidth,
      .context_stride = std::uint64_t{kMaxContext} * kKvWidth,
      .block_shift = kBlockShift,
  };

  std::vector<float> host_q(query_elements);
  std::vector<float> host_gate(query_elements);
  for (std::size_t i = 0; i < host_q.size(); ++i) {
    host_q[i] = std::sin(static_cast<double>(i) * 0.017);
    host_gate[i] = std::cos(static_cast<double>(i) * 0.011);
  }
  HIP_CHECK(hipMemcpy(d_q.data, host_q.data(), host_q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate.data, host_gate.data(),
                      host_gate.size() * sizeof(float), hipMemcpyHostToDevice));

  const __half* k_layer =
      static_cast<const __half*>(d_cache.data) +
      static_cast<std::size_t>(kLayer) * kMaxContext * kKvWidth;
  const __half* v_layer =
      static_cast<const __half*>(d_cache_v.data) +
      static_cast<std::size_t>(kLayer) * kMaxContext * kKvWidth;

  for (const Case& item :
       {Case{"wmma-dense", true}, Case{"per-token", false}}) {
    DeviceBuffer d_ref(query_elements * sizeof(float));
    DeviceBuffer d_new(query_elements * sizeof(float));

    if (item.wmma) {
      if (!q::WmmaCausalAttention(
              static_cast<const float*>(d_q.data),
              static_cast<const float*>(d_gate.data), k_layer, v_layer, nullptr,
              0, static_cast<float*>(d_ref.data), kTokens, 0, kHeads, kKvHeads,
              kHeadDim, kRatio, nullptr)) {
        Fail("contiguous WMMA launch rejected the model geometry");
      }
      if (!q::WmmaCausalAttention(
              static_cast<const float*>(d_q.data),
              static_cast<const float*>(d_gate.data),
              static_cast<const __half*>(d_paged.data),
              static_cast<const __half*>(d_paged_v.data), nullptr, 0,
              static_cast<float*>(d_new.data), kTokens, 0, kHeads, kKvHeads,
              kHeadDim, kRatio, nullptr, /*last_only=*/false, &view, kLayer)) {
        Fail("paged WMMA launch rejected the model geometry");
      }
    } else {
      q::Attention(static_cast<const float*>(d_q.data), k_layer, v_layer,
                   nullptr, 0, static_cast<float*>(d_ref.data), nullptr, 1,
                   kTokens, static_cast<const std::uint32_t*>(d_start.data),
                   kHeads, kKvHeads, kHeadDim, kRatio, nullptr);
      q::Attention(static_cast<const float*>(d_q.data),
                   static_cast<const __half*>(d_paged.data),
                   static_cast<const __half*>(d_paged_v.data), nullptr, 0,
                   static_cast<float*>(d_new.data), nullptr, 1, kTokens,
                   static_cast<const std::uint32_t*>(d_start.data), kHeads,
                   kKvHeads, kHeadDim, kRatio, nullptr, nullptr, nullptr, 0,
                   &view, kLayer);
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> reference(query_elements);
    std::vector<float> candidate(query_elements);
    HIP_CHECK(hipMemcpy(reference.data(), d_ref.data,
                        reference.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(candidate.data(), d_new.data,
                        candidate.size() * sizeof(float),
                        hipMemcpyDeviceToHost));

    if (std::memcmp(reference.data(), candidate.data(),
                    reference.size() * sizeof(float)) != 0) {
      double worst = 0.0;
      for (std::size_t i = 0; i < reference.size(); ++i)
        worst = std::max(
            worst, static_cast<double>(std::fabs(reference[i] - candidate[i])));
      std::cerr << item.name << " output differs, worst |diff| " << worst
                << '\n';
      return 1;
    }
    std::cout << item.name
              << ": paged KV matches the contiguous cache byte for byte\n";
  }
  return 0;
}
