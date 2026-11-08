#ifndef GUFO_MODELS_QWEN_KV_BLOCKS_HPP_
#define GUFO_MODELS_QWEN_KV_BLOCKS_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::qwen {

/// Tokens per attention KV block.
///
/// One block covers every full-attention layer and both planes, so a session
/// needs a single table rather than one per layer; every layer of Qwen3.8-27B
/// and Qwen3.8-Flash-Next uses the same KV head count and head dim. The value
/// is a compromise: a larger block means fewer table loads in a prefill tile
/// and fewer entries, but a coarser copy-on-write tail.
///
/// At 24 KiB per token (Flash-Next), 128 tokens are 3 MiB per block.
inline constexpr std::uint32_t kKvBlockTokens = 128;

/// log2(kKvBlockTokens). Device addressing uses shifts, and a static_assert
/// keeps the two definitions in step.
inline constexpr std::uint32_t kKvBlockShift = 7;

static_assert((std::uint32_t{1} << kKvBlockShift) == kKvBlockTokens,
              "block shift must match the block size");

/// Empty table slot. Block ids are dense in [0, capacity).
inline constexpr std::uint32_t kInvalidKvBlock = 0xFFFFFFFFu;

/// Point-in-time accounting for one KV pool.
///
/// `high_water_blocks` is the largest simultaneous live count since the pool
/// was created. Comparing it with `capacity_blocks` is what distinguishes a
/// reservation that the workload needs from one that only `sessions x context`
/// implied: the difference is device memory that no request ever touched.
struct KvPoolSnapshot {
  std::uint32_t capacity_blocks{};
  std::uint32_t live_blocks{};
  std::uint32_t free_blocks{};
  std::uint32_t high_water_blocks{};
  /// Live blocks referenced by more than one table (prefix sharing / COW).
  std::uint32_t shared_blocks{};
};

/// Reference-counted pool of attention KV blocks.
///
/// The pool owns ids, not bytes. The caller lays blocks out in device memory
/// (see `KvBlockTable::ByteOffset` usage in the model executor) so the pool
/// stays a host-side, allocation-free piece of bookkeeping that can be tested
/// without a GPU.
///
/// Ownership: the model executor's single owner thread drives every call, as
/// with the rest of the scheduler state. No operation takes a lock.
class KvBlockPool {
public:
  explicit KvBlockPool(std::uint32_t capacity_blocks);

  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  /// Blocks with at least one reference.
  [[nodiscard]] std::uint32_t live_blocks() const noexcept {
    return capacity_ - free_count_;
  }
  [[nodiscard]] std::uint32_t free_blocks() const noexcept {
    return free_count_;
  }
  /// Highest simultaneous live-block count so far.
  [[nodiscard]] std::uint32_t high_water_blocks() const noexcept {
    return high_water_;
  }
  /// Blocks a second table references. These are the prefix rows SGLang-style
  /// RadixAttention shares instead of copying.
  [[nodiscard]] std::uint32_t shared_blocks() const noexcept {
    std::uint32_t count = 0;
    for (const auto references : references_) {
      if (references > 1)
        ++count;
    }
    return count;
  }
  [[nodiscard]] KvPoolSnapshot Snapshot() const noexcept {
    return {capacity_, live_blocks(), free_blocks(), high_water_,
            shared_blocks()};
  }

  /// Returns a fresh block with reference count 1, or `kInvalidKvBlock` when
  /// the pool is exhausted.
  /// Hands out the block that has been free longest, so a just-released
  /// prefix keeps its rows long enough for a repeated prompt to adopt them
  /// before the next capture overwrites the block.
  [[nodiscard]] std::uint32_t Acquire() noexcept;
  /// Takes one more reference on a live block (checkpoint or fork sharing).
  void Retain(std::uint32_t block) noexcept;
  /// Drops one reference; the block rejoins the free list at zero.
  void Release(std::uint32_t block) noexcept;
  [[nodiscard]] std::uint32_t references(std::uint32_t block) const noexcept;

private:
  std::uint32_t capacity_;
  std::uint32_t high_water_{0};
  // Zero means free. A block that is live has at least one entry here.
  std::vector<std::uint16_t> references_;
  // Free ids in reuse order: the oldest release leaves first. A ring keeps
  // both Acquire and Release constant time without growing with releases.
  std::vector<std::uint32_t> free_list_;
  std::uint32_t free_head_{0};
  std::uint32_t free_count_{0};
};

/// A block that must be copied before the caller writes its tail.
///
/// `ForkFrom` and `Truncate` never move bytes: they report the device copy the
/// caller has to make, so the pool has no HIP dependency and a test can verify
/// the aliasing rule with plain memory.
struct KvTailDetach {
  bool required{false};
  /// Table entry that now points at the private copy.
  std::uint32_t entry{0};
  /// Block the tail rows currently live in.
  std::uint32_t source{kInvalidKvBlock};
  /// Fresh block the caller must copy the first `rows` rows into.
  std::uint32_t target{kInvalidKvBlock};
  /// Rows (positions) to copy, counted from the start of the block.
  std::uint32_t rows{0};
};

/// Per-session map from logical position to physical block.
///
/// Positions below `position()` are committed. The table never owns bytes and
/// never allocates device memory; it acquires and releases ids from a pool and
/// reports the copies a caller must make on the device.
class KvBlockTable {
public:
  /// `context_capacity` positions are addressable; entries are reserved for
  /// `ceil(context_capacity / block_tokens)` blocks up front.
  KvBlockTable(std::uint32_t block_tokens, std::uint32_t context_capacity);

  [[nodiscard]] std::uint32_t block_tokens() const noexcept {
    return block_tokens_;
  }
  [[nodiscard]] std::uint32_t position_capacity() const noexcept {
    return static_cast<std::uint32_t>(entries_.size()) * block_tokens_;
  }
  /// Largest position a table of this shape can reach.
  [[nodiscard]] std::uint32_t max_position() const noexcept;
  /// Committed positions.
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  /// Committed blocks, that is `ceil(position / block_tokens)`.
  [[nodiscard]] std::uint32_t block_count() const noexcept;
  /// Physical block for a table entry, or `kInvalidKvBlock`.
  [[nodiscard]] std::uint32_t entry(std::uint32_t index) const noexcept;
  /// Entry that holds a committed position; the caller must check
  /// `position() > pos`.
  [[nodiscard]] std::uint32_t block_for(std::uint32_t pos) const noexcept;

  /// Appends up to `tokens` positions.
  ///
  /// Returns the number appended, which is smaller than `tokens` when the
  /// table reaches its capacity or the pool runs dry. Partial appends keep the
  /// table consistent so a chunked prefill can report actual progress.
  ///
  /// A committed partial block is always private: `ForkFrom` copies the tail
  /// and `Truncate` detaches it. Should a table ever reach a shared partial
  /// block, this call stops at the block, reports the copy in `detach`, and
  /// appends nothing into it until the caller has copied the rows and calls
  /// again.
  [[nodiscard]] std::uint32_t Append(KvBlockPool& pool, std::uint32_t tokens,
                                     KvTailDetach* detach = nullptr);

  /// Truncates to `position`, releasing whole blocks past it.
  ///
  /// When the last surviving block is shared, the remaining tail is copied
  /// into a private block first; `detach` then tells the caller which bytes to
  /// move. Returns false without changing the table when the pool cannot
  /// supply that private block.
  [[nodiscard]] bool Truncate(KvBlockPool& pool, std::uint32_t position,
                              KvTailDetach* detach = nullptr);

  /// Shares `other`'s prefix and copies its partial tail.
  ///
  /// Full blocks are retained by reference; the tail block is private to this
  /// table and reported in `detach` for the caller to fill. `position` is
  /// clamped to `other.position()`.
  [[nodiscard]] bool ForkFrom(KvBlockPool& pool, const KvBlockTable& other,
                              std::uint32_t position,
                              KvTailDetach* detach = nullptr);

  /// Shares `block` at `index`: takes one more pool reference and drops the
  /// table's previous block there.
  ///
  /// Only full blocks are safe to adopt. A committed partial block receives
  /// further writes through its tail, so it must stay private; the caller
  /// guarantees the rows this entry covers are immutable for as long as the
  /// share lives. Returns false, without changing the table, when `index` is
  /// outside the table or `block` is not live in the pool.
  [[nodiscard]] bool Adopt(KvBlockPool& pool, std::uint32_t index,
                           std::uint32_t block) noexcept;

  /// Releases every block. The table is empty afterwards.
  void ReleaseAll(KvBlockPool& pool) noexcept;

  /// Byte offset of `position` in the layer `layer` of a device plane laid out
  /// as `[block][layer][block_tokens][row_bytes]`.
  ///
  /// `layer_count` is the number of layers stored in one block, which fixes
  /// the stride between blocks. The executor allocates the pool buffer with
  /// exactly this layout; keeping the arithmetic here lets the pool tests lock
  /// it down without a GPU.
  [[nodiscard]] std::uint64_t ByteOffset(std::uint32_t layer,
                                         std::uint32_t position,
                                         std::uint32_t layer_count,
                                         std::uint64_t row_bytes) const;

private:
  std::uint32_t block_tokens_;
  std::uint32_t position_{0};
  std::vector<std::uint32_t> entries_;
};

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_KV_BLOCKS_HPP_
