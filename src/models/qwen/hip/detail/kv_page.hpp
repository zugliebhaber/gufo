#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_KV_PAGE_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_KV_PAGE_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#endif

namespace gufo::hip {

/// Device view of the attention KV store.
///
/// `table == nullptr` selects the contiguous per-session layout that predates
/// paging, so one kernel body serves both and the contiguous path keeps its
/// original arithmetic. With a table, a position maps to a physical block id
/// from `gufo::models::qwen::KvBlockTable`.
///
/// The executor allocates the pool as `[block][layer][block_tokens][kv_width]`,
/// which makes `block_stride` and `layer_stride` launch constants. Resolve the
/// block once per key tile: measured on gfx1151, a lookup per position costs
/// 4-6%, a lookup hoisted per 128-position block costs nothing
/// (`tools/bench/kv_paging_bench.hip`).
struct KvPageView {
  /// One block id per `1 << block_shift` positions, in device memory.
  const std::uint32_t* table{nullptr};
  /// Elements in one block across every layer.
  std::uint64_t block_stride{0};
  /// Elements in one layer inside a block.
  std::uint64_t layer_stride{0};
  /// Elements in one layer of the contiguous layout (`max_context * kv_width`).
  std::uint64_t context_stride{0};
  /// log2 of the block's position count; block sizes are powers of two.
  std::uint32_t block_shift{0};
};

#if defined(ENGINE_ENABLE_HIP)
__device__ __forceinline__ std::uint64_t KvBlockBase(const KvPageView& view,
                                                     std::uint64_t block,
                                                     std::uint32_t layer) {
  return block * view.block_stride +
         static_cast<std::uint64_t>(layer) * view.layer_stride;
}

/// Element offset of `position` in `layer`, before any KV-head offset.
__device__ __forceinline__ std::uint64_t KvRowOffset(const KvPageView& view,
                                                     std::uint32_t layer,
                                                     std::uint64_t position,
                                                     std::uint32_t kv_width) {
  if (view.table == nullptr) {
    return static_cast<std::uint64_t>(layer) * view.context_stride +
           position * kv_width;
  }
  const std::uint64_t mask = (std::uint64_t{1} << view.block_shift) - 1;
  return KvBlockBase(view, view.table[position >> view.block_shift], layer) +
         (position & mask) * kv_width;
}

/// Row pointer for a plane whose base pointer is `plane`.
///
/// In the contiguous layout `plane` names the layer, so the address is the
/// pre-paging one. With a table, `plane` names the whole pool and `layer`
/// selects the layer inside the block.
__device__ __forceinline__ __half* KvHalfRow(__half* plane,
                                             const KvPageView& view,
                                             std::uint32_t layer,
                                             std::uint32_t position,
                                             std::uint32_t kv_width) {
  if (view.table == nullptr)
    return plane + static_cast<std::uint64_t>(position) * kv_width;
  return plane + KvRowOffset(view, layer, position, kv_width);
}

__device__ __forceinline__ const __half* KvHalfRow(const __half* plane,
                                                   const KvPageView& view,
                                                   std::uint32_t layer,
                                                   std::uint32_t position,
                                                   std::uint32_t kv_width) {
  if (view.table == nullptr)
    return plane + static_cast<std::uint64_t>(position) * kv_width;
  return plane + KvRowOffset(view, layer, position, kv_width);
}
#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_KV_PAGE_HPP_
