#include "src/models/qwen/kv_blocks.hpp"

#include <algorithm>
#include <stdexcept>

namespace gufo::models::qwen {
namespace {

std::uint32_t EntriesFor(std::uint32_t block_tokens,
                         std::uint32_t context_capacity) {
  return (context_capacity + block_tokens - 1) / block_tokens;
}

}  // namespace

KvBlockPool::KvBlockPool(std::uint32_t capacity_blocks)
    : capacity_(capacity_blocks),
      references_(capacity_blocks, 0),
      free_list_(capacity_blocks),
      free_count_(capacity_blocks) {
  // Hand out low ids first so a short run stays inside a small address span.
  // Filling in ascending order makes the FIFO ring return 0, 1, 2, ... while
  // the pool is still untouched.
  for (std::uint32_t block = 0; block < capacity_blocks; ++block)
    free_list_[block] = block;
}

std::uint32_t KvBlockPool::Acquire() noexcept {
  if (free_count_ == 0)
    return kInvalidKvBlock;
  const std::uint32_t block = free_list_[free_head_];
  free_head_ = (free_head_ + 1) % capacity_;
  --free_count_;
  references_[block] = 1;
  high_water_ = std::max(high_water_, live_blocks());
  return block;
}

void KvBlockPool::Retain(std::uint32_t block) noexcept {
  if (block >= capacity_ || references_[block] == 0)
    return;
  if (references_[block] != 0xFFFFu)
    ++references_[block];
}

void KvBlockPool::Release(std::uint32_t block) noexcept {
  if (block >= capacity_ || references_[block] == 0)
    return;
  if (--references_[block] == 0) {
    free_list_[(free_head_ + free_count_) % capacity_] = block;
    ++free_count_;
  }
}

std::uint32_t KvBlockPool::references(std::uint32_t block) const noexcept {
  return block < capacity_ ? references_[block] : 0;
}

KvBlockTable::KvBlockTable(std::uint32_t block_tokens,
                           std::uint32_t context_capacity)
    : block_tokens_(block_tokens),
      entries_(EntriesFor(block_tokens, context_capacity), kInvalidKvBlock) {
  if (block_tokens == 0)
    throw std::invalid_argument("KV block size must be positive");
}

std::uint32_t KvBlockTable::max_position() const noexcept {
  return static_cast<std::uint32_t>(entries_.size()) * block_tokens_;
}

std::uint32_t KvBlockTable::block_count() const noexcept {
  return (position_ + block_tokens_ - 1) / block_tokens_;
}

std::uint32_t KvBlockTable::entry(std::uint32_t index) const noexcept {
  return index < entries_.size() ? entries_[index] : kInvalidKvBlock;
}

std::uint32_t KvBlockTable::block_for(std::uint32_t pos) const noexcept {
  return pos < position_ ? entries_[pos / block_tokens_] : kInvalidKvBlock;
}

std::uint32_t KvBlockTable::Append(KvBlockPool& pool, std::uint32_t tokens,
                                   KvTailDetach* detach) {
  if (detach != nullptr)
    *detach = {};
  std::uint32_t appended = 0;
  const std::uint32_t limit = std::min(tokens, max_position() - position_);
  while (appended < limit) {
    const std::uint32_t index = position_ / block_tokens_;
    const std::uint32_t in_block = position_ % block_tokens_;
    if (entries_[index] == kInvalidKvBlock) {
      const std::uint32_t block = pool.Acquire();
      if (block == kInvalidKvBlock)
        break;
      entries_[index] = block;
    } else if (in_block != 0 && pool.references(entries_[index]) > 1) {
      // Never write into rows another table can read. Stop at the block and
      // let the caller make the device copy it already owns.
      const std::uint32_t source = entries_[index];
      const std::uint32_t fresh = pool.Acquire();
      if (fresh == kInvalidKvBlock)
        break;
      entries_[index] = fresh;
      pool.Release(source);
      if (detach != nullptr) {
        *detach = {.required = true,
                   .entry = index,
                   .source = source,
                   .target = fresh,
                   .rows = in_block};
      }
      break;
    }
    const std::uint32_t step =
        std::min(limit - appended, block_tokens_ - in_block);
    position_ += step;
    appended += step;
  }
  return appended;
}

bool KvBlockTable::Truncate(KvBlockPool& pool, std::uint32_t position,
                            KvTailDetach* detach) {
  if (detach != nullptr)
    *detach = {};
  if (position > position_)
    return false;
  if (position == position_)
    return true;

  const std::uint32_t kept_blocks =
      (position + block_tokens_ - 1) / block_tokens_;
  const std::uint32_t previous_blocks = block_count();
  const bool needs_tail = position % block_tokens_ != 0 && kept_blocks > 0;
  const std::uint32_t tail_index = kept_blocks > 0 ? kept_blocks - 1 : 0;

  // Acquire the private copy before dropping anything, so a full pool leaves
  // the table untouched instead of half truncated.
  std::uint32_t replacement = kInvalidKvBlock;
  std::uint32_t tail_block = kInvalidKvBlock;
  if (needs_tail) {
    tail_block = entries_[tail_index];
    if (pool.references(tail_block) > 1) {
      replacement = pool.Acquire();
      if (replacement == kInvalidKvBlock)
        return false;
    }
  }

  for (std::uint32_t index = kept_blocks; index < previous_blocks; ++index) {
    const std::uint32_t block = entries_[index];
    if (block == kInvalidKvBlock)
      continue;
    entries_[index] = kInvalidKvBlock;
    pool.Release(block);
  }

  if (replacement != kInvalidKvBlock) {
    entries_[tail_index] = replacement;
    pool.Release(tail_block);
    if (detach != nullptr) {
      *detach = {.required = true,
                 .entry = tail_index,
                 .source = tail_block,
                 .target = replacement,
                 .rows = position % block_tokens_};
    }
  }
  position_ = position;
  return true;
}

bool KvBlockTable::ForkFrom(KvBlockPool& pool, const KvBlockTable& other,
                            std::uint32_t position, KvTailDetach* detach) {
  if (detach != nullptr)
    *detach = {};
  if (position_ != 0)
    return false;
  const std::uint32_t target = std::min(position, other.position_);
  const std::uint32_t full_blocks = target / block_tokens_;
  if (full_blocks + (target % block_tokens_ != 0 ? 1u : 0u) > entries_.size())
    return false;

  const auto undo = [&](std::uint32_t keep) {
    for (std::uint32_t index = 0; index < keep; ++index) {
      if (entries_[index] == kInvalidKvBlock)
        continue;
      pool.Release(entries_[index]);
      entries_[index] = kInvalidKvBlock;
    }
  };

  for (std::uint32_t index = 0; index < full_blocks; ++index) {
    const std::uint32_t block = other.entries_[index];
    if (block == kInvalidKvBlock) {
      undo(index);
      return false;
    }
    pool.Retain(block);
    entries_[index] = block;
  }

  const std::uint32_t tail_rows = target % block_tokens_;
  if (tail_rows != 0) {
    const std::uint32_t source = other.entries_[full_blocks];
    if (source == kInvalidKvBlock) {
      undo(full_blocks);
      return false;
    }
    const std::uint32_t fresh = pool.Acquire();
    if (fresh == kInvalidKvBlock) {
      undo(full_blocks);
      return false;
    }
    entries_[full_blocks] = fresh;
    if (detach != nullptr) {
      *detach = {.required = true,
                 .entry = full_blocks,
                 .source = source,
                 .target = fresh,
                 .rows = tail_rows};
    }
  }

  position_ = target;
  return true;
}

bool KvBlockTable::Adopt(KvBlockPool& pool, std::uint32_t index,
                         std::uint32_t block) noexcept {
  if (index >= entries_.size() || block == kInvalidKvBlock ||
      pool.references(block) == 0)
    return false;
  if (entries_[index] == block)
    return true;
  // Take the new reference before dropping the old one: the two may name the
  // same block only when the entry was already adopted, handled above.
  pool.Retain(block);
  const std::uint32_t previous = entries_[index];
  entries_[index] = block;
  if (previous != kInvalidKvBlock)
    pool.Release(previous);
  return true;
}

void KvBlockTable::ReleaseAll(KvBlockPool& pool) noexcept {
  for (auto& block : entries_) {
    if (block == kInvalidKvBlock)
      continue;
    pool.Release(block);
    block = kInvalidKvBlock;
  }
  position_ = 0;
}

std::uint64_t KvBlockTable::ByteOffset(std::uint32_t layer,
                                       std::uint32_t position,
                                       std::uint32_t layer_count,
                                       std::uint64_t row_bytes) const {
  const std::uint32_t block = entries_[position / block_tokens_];
  const std::uint64_t rows_per_block =
      static_cast<std::uint64_t>(block_tokens_);
  const std::uint64_t block_bytes = rows_per_block * layer_count * row_bytes;
  return static_cast<std::uint64_t>(block) * block_bytes +
         static_cast<std::uint64_t>(layer) * rows_per_block * row_bytes +
         static_cast<std::uint64_t>(position % block_tokens_) * row_bytes;
}

}  // namespace gufo::models::qwen
