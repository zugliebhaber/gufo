#include "src/models/qwen/kv_blocks.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using gufo::models::qwen::kInvalidKvBlock;
using gufo::models::qwen::kKvBlockTokens;
using gufo::models::qwen::KvBlockPool;
using gufo::models::qwen::KvBlockTable;
using gufo::models::qwen::KvTailDetach;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Qwen KV block test failure: " << message << '\n';
    std::abort();
  }
}

void CheckEqual(std::uint64_t actual, std::uint64_t expected,
                const char* message) {
  if (actual != expected) {
    std::cerr << "Qwen KV block test failure: " << message << " (actual "
              << actual << ", expected " << expected << ")\n";
    std::abort();
  }
}

/// Device-memory stand-in. Every token row is one byte, so a test can see
/// exactly which block a write landed in.
struct FakePlane {
  static constexpr std::uint64_t kRowBytes = 8;
  static constexpr std::uint32_t kLayers = 2;

  std::vector<std::uint8_t> bytes;

  explicit FakePlane(std::uint32_t pool_blocks)
      : bytes(static_cast<std::size_t>(pool_blocks) * BlockBytes(), 0) {}

  static std::uint64_t BlockBytes() {
    return std::uint64_t{kKvBlockTokens} * kLayers * kRowBytes;
  }

  void Write(const KvBlockTable& table, std::uint32_t layer,
             std::uint32_t position, std::uint8_t value) {
    const auto offset = table.ByteOffset(layer, position, kLayers, kRowBytes);
    Check(offset + kRowBytes <= bytes.size(), "write stays inside the plane");
    bytes[static_cast<std::size_t>(offset)] = value;
  }

  [[nodiscard]] std::uint8_t Read(const KvBlockTable& table,
                                  std::uint32_t layer,
                                  std::uint32_t position) const {
    const auto offset = table.ByteOffset(layer, position, kLayers, kRowBytes);
    Check(offset + kRowBytes <= bytes.size(), "read stays inside the plane");
    return bytes[static_cast<std::size_t>(offset)];
  }

  void CopyRows(std::uint32_t source, std::uint32_t target,
                std::uint32_t rows) {
    const auto from = static_cast<std::size_t>(source) * BlockBytes();
    const auto to = static_cast<std::size_t>(target) * BlockBytes();
    const auto count = static_cast<std::size_t>(rows) * kLayers * kRowBytes;
    Check(to + count <= bytes.size(), "copy stays inside the plane");
    for (std::size_t byte = 0; byte < count; ++byte)
      bytes[to + byte] = bytes[from + byte];
  }
};

void PoolAcquireAndRelease() {
  KvBlockPool pool(3);
  CheckEqual(pool.capacity(), 3, "capacity");
  CheckEqual(pool.free_blocks(), 3, "starts free");

  const auto first = pool.Acquire();
  const auto second = pool.Acquire();
  Check(first != kInvalidKvBlock && second != kInvalidKvBlock, "acquire");
  Check(first != second, "ids are distinct");
  CheckEqual(pool.references(first), 1, "fresh reference");
  CheckEqual(pool.live_blocks(), 2, "live count");

  pool.Retain(first);
  CheckEqual(pool.references(first), 2, "retain adds a reference");
  pool.Release(first);
  CheckEqual(pool.references(first), 1, "release drops one reference");
  pool.Release(first);
  CheckEqual(pool.references(first), 0, "unreferenced block is free");
  CheckEqual(pool.free_blocks(), 2, "free list grows");
  CheckEqual(pool.high_water_blocks(), 2, "high water");

  pool.Release(99);
  CheckEqual(pool.free_blocks(), 2, "releasing an unknown id changes nothing");
}

void PoolReusesTheOldestFreeBlock() {
  // A hash-addressed prefix cache needs a released block to keep its rows a
  // little longer, so the id that has been free longest leaves first instead
  // of the block just returned. Reusing the newest release would overwrite a
  // prefix that a repeated prompt could still adopt.
  KvBlockPool pool(4);
  const auto a = pool.Acquire();
  const auto b = pool.Acquire();
  const auto c = pool.Acquire();
  const auto d = pool.Acquire();
  CheckEqual(pool.free_blocks(), 0, "all blocks live");
  pool.Release(c);
  pool.Release(a);
  pool.Release(d);
  CheckEqual(pool.Acquire(), c, "oldest release leaves first");
  CheckEqual(pool.Acquire(), a, "second oldest leaves next");
  CheckEqual(pool.Acquire(), d, "most recent release leaves last");
  CheckEqual(pool.free_blocks(), 0, "every block live again");
  // The ring wraps without losing the order.
  pool.Release(b);
  pool.Release(a);
  CheckEqual(pool.Acquire(), b, "wrapped ring keeps the oldest first");
  CheckEqual(pool.Acquire(), a, "wrapped ring keeps the newest last");
}

void AppendAllocatesPerBlock() {
  KvBlockPool pool(8);
  KvBlockTable table(kKvBlockTokens, 1024);
  CheckEqual(table.max_position(), 1024, "max position covers the capacity");

  CheckEqual(table.Append(pool, 100), 100, "partial block");
  CheckEqual(table.block_count(), 1, "one block for a partial fill");
  CheckEqual(pool.live_blocks(), 1, "one live block");

  CheckEqual(table.Append(pool, 28), 28, "fills the block");
  CheckEqual(table.block_count(), 1, "still one block");
  CheckEqual(pool.live_blocks(), 1, "no second block yet");

  CheckEqual(table.Append(pool, 1), 1, "crosses the boundary");
  CheckEqual(table.block_count(), 2, "second block");
  CheckEqual(pool.live_blocks(), 2, "second live block");
  Check(table.block_for(0) != table.block_for(kKvBlockTokens),
        "blocks differ across the boundary");
  CheckEqual(table.block_for(kKvBlockTokens), table.entry(1),
             "block_for matches the entry");
  CheckEqual(table.block_for(9999), kInvalidKvBlock,
             "uncommitted position has no block");
}

void AppendStopsAtPoolExhaustion() {
  KvBlockPool pool(2);
  KvBlockTable table(kKvBlockTokens, 4096);
  CheckEqual(table.Append(pool, 1000), 2 * std::uint64_t{kKvBlockTokens},
             "partial append when the pool runs dry");
  CheckEqual(table.position(), 2 * std::uint64_t{kKvBlockTokens},
             "position reflects the partial append");
  CheckEqual(pool.free_blocks(), 0, "pool exhausted");
  CheckEqual(table.Append(pool, 10), 0, "no progress without blocks");
}

void AppendStopsAtTableCapacity() {
  KvBlockPool pool(8);
  KvBlockTable table(kKvBlockTokens, 200);
  CheckEqual(table.position_capacity(), 256, "capacity rounds up to blocks");
  CheckEqual(table.Append(pool, 1000), 256, "append clamps to capacity");
  CheckEqual(table.Append(pool, 1), 0, "table full");
}

void AppendDetachesSharedTail() {
  KvBlockPool pool(4);
  KvBlockTable table(kKvBlockTokens, 1024);
  CheckEqual(table.Append(pool, 100), 100, "partial block");
  const std::uint32_t block = table.entry(0);
  pool.Retain(block);
  CheckEqual(pool.references(block), 2, "block is shared with a reader");

  KvTailDetach detach{};
  CheckEqual(table.Append(pool, 64, &detach), 0,
             "no write into a shared partial block");
  Check(detach.required, "the caller must copy the rows first");
  CheckEqual(detach.rows, 100, "rows to copy");
  CheckEqual(detach.source, block, "copy source");
  CheckEqual(pool.references(block), 1, "reader keeps its reference");
  CheckEqual(detach.target, table.entry(0), "entry rebound to the copy");
  CheckEqual(table.position(), 100, "position unchanged");

  CheckEqual(table.Append(pool, 64), 64, "append continues after the copy");
  CheckEqual(table.position(), 164, "position advanced");
}

void TruncateReleasesWholeBlocks() {
  KvBlockPool pool(8);
  KvBlockTable table(kKvBlockTokens, 1024);
  CheckEqual(table.Append(pool, 300), 300, "append");
  CheckEqual(pool.live_blocks(), 3, "three blocks committed");

  KvTailDetach detach{};
  Check(table.Truncate(pool, kKvBlockTokens, &detach),
        "truncate to a block boundary");
  Check(!detach.required, "boundary truncation needs no copy");
  CheckEqual(table.position(), kKvBlockTokens, "position after truncation");
  CheckEqual(pool.live_blocks(), 1, "later blocks were released");
  CheckEqual(pool.free_blocks(), 7, "released blocks are reusable");
}

void TruncateKeepsUnsharedTail() {
  KvBlockPool pool(8);
  KvBlockTable table(kKvBlockTokens, 1024);
  CheckEqual(table.Append(pool, 200), 200, "append");

  KvTailDetach detach{};
  Check(table.Truncate(pool, 100, &detach), "truncate inside the block");
  Check(!detach.required, "a private tail is written in place");
  CheckEqual(table.position(), 100, "position after truncation");
  CheckEqual(pool.live_blocks(), 1, "the tail block survives");
}

void TruncateDetachesSharedTail() {
  KvBlockPool pool(8);
  FakePlane plane(pool.capacity());
  KvBlockTable shared(kKvBlockTokens, 1024);
  CheckEqual(shared.Append(pool, kKvBlockTokens), kKvBlockTokens, "append");
  for (std::uint32_t pos = 0; pos < kKvBlockTokens; ++pos)
    plane.Write(shared, 0, pos, 7);

  KvBlockTable fork(kKvBlockTokens, 1024);
  Check(fork.ForkFrom(pool, shared, kKvBlockTokens, nullptr),
        "fork a full block");
  CheckEqual(pool.references(shared.entry(0)), 2, "block is shared");

  KvTailDetach detach{};
  Check(fork.Truncate(pool, 64, &detach), "truncate into the shared block");
  Check(detach.required, "a shared tail must be copied");
  CheckEqual(detach.rows, 64, "copied rows");
  CheckEqual(detach.source, shared.entry(0), "copy source");
  CheckEqual(pool.references(detach.source), 1,
             "source keeps only its own reference");
  CheckEqual(pool.references(detach.target), 1, "target is private");

  plane.CopyRows(detach.source, detach.target, detach.rows);
  plane.Write(fork, 0, 10, 0xCD);
  CheckEqual(plane.Read(shared, 0, 10), 7, "shared block is not mutated");
  CheckEqual(plane.Read(fork, 0, 10), 0xCD, "the copy took the write");
}

void TruncateRefusesWithoutABlock() {
  KvBlockPool pool(2);
  KvBlockTable shared(kKvBlockTokens, 1024);
  CheckEqual(shared.Append(pool, 2 * std::uint64_t{kKvBlockTokens}),
             2 * std::uint64_t{kKvBlockTokens}, "append");
  KvBlockTable fork(kKvBlockTokens, 1024);
  Check(fork.ForkFrom(pool, shared, 2 * kKvBlockTokens, nullptr),
        "fork shares both blocks");
  CheckEqual(pool.free_blocks(), 0, "pool is empty");

  KvTailDetach detach{};
  Check(!fork.Truncate(pool, 64, &detach),
        "no private block available for the shared tail");
  Check(!detach.required, "a refused truncation reports no copy");
  CheckEqual(fork.position(), 2 * std::uint64_t{kKvBlockTokens},
             "table is unchanged");
  CheckEqual(fork.block_count(), 2, "blocks are unchanged");
}

void ForkCopiesOnlyTheTail() {
  KvBlockPool pool(8);
  FakePlane plane(pool.capacity());
  KvBlockTable source(kKvBlockTokens, 1024);
  CheckEqual(source.Append(pool, 200), 200, "append");
  for (std::uint32_t pos = 0; pos < 200; ++pos)
    plane.Write(source, 0, pos, static_cast<std::uint8_t>(pos));

  KvBlockTable fork(kKvBlockTokens, 1024);
  KvTailDetach detach{};
  Check(fork.ForkFrom(pool, source, 200, &detach), "fork");
  Check(detach.required, "tail copy is required");
  CheckEqual(detach.rows, 200 - kKvBlockTokens, "tail rows");
  plane.CopyRows(detach.source, detach.target, detach.rows);

  for (std::uint32_t pos = 0; pos < 200; ++pos) {
    CheckEqual(plane.Read(fork, 0, pos), plane.Read(source, 0, pos),
               "fork reads the same bytes as the source");
  }
  CheckEqual(pool.live_blocks(), 3, "two shared rows and one private tail");

  CheckEqual(fork.Append(pool, 64), 64, "append");
  plane.Write(fork, 0, 200, 0xAB);
  CheckEqual(plane.Read(source, 0, 199), 199,
             "source tail is untouched by the fork");
  CheckEqual(plane.Read(fork, 0, 200), 0xAB, "fork write landed");
  CheckEqual(pool.references(source.entry(0)), 2,
             "the shared block is still shared");
}

void ByteOffsetLayout() {
  KvBlockPool pool(4);
  KvBlockTable table(kKvBlockTokens, 512);
  CheckEqual(table.Append(pool, 200), 200, "append");
  const auto first = std::uint64_t{table.entry(0)};
  const auto second = std::uint64_t{table.entry(1)};
  const std::uint64_t block_bytes = std::uint64_t{kKvBlockTokens} * 2 * 8;

  CheckEqual(table.ByteOffset(0, 0, 2, 8), first * block_bytes, "first row");
  CheckEqual(table.ByteOffset(1, 5, 2, 8),
             first * block_bytes + std::uint64_t{kKvBlockTokens} * 8 + 5 * 8,
             "layer and row offset");
  CheckEqual(table.ByteOffset(0, 140, 2, 8), second * block_bytes + 12 * 8,
             "positions past a block boundary");
}

void ReleaseAllReturnsBlocks() {
  KvBlockPool pool(4);
  KvBlockTable table(kKvBlockTokens, 512);
  CheckEqual(table.Append(pool, 400), 400, "append");
  CheckEqual(pool.live_blocks(), 4, "all blocks in use");
  table.ReleaseAll(pool);
  CheckEqual(pool.free_blocks(), 4, "every block returned");
  CheckEqual(table.position(), 0, "table reset");
  CheckEqual(table.block_count(), 0, "no committed blocks");
}

void AdoptSharesAFullBlock() {
  KvBlockPool pool(4);
  KvBlockTable source(kKvBlockTokens, 1024);
  CheckEqual(source.Append(pool, kKvBlockTokens), kKvBlockTokens,
             "source owns a full block");
  KvBlockTable destination(kKvBlockTokens, 1024);
  CheckEqual(destination.Append(pool, kKvBlockTokens), kKvBlockTokens,
             "destination owns a private block");
  const std::uint32_t source_block = source.entry(0);
  const std::uint32_t private_block = destination.entry(0);
  Check(source_block != private_block, "distinct physical blocks");

  Check(destination.Adopt(pool, 0, source_block), "adopt the shared block");
  CheckEqual(destination.entry(0), source_block, "entry points at the share");
  CheckEqual(pool.references(source_block), 2, "source and share both hold it");
  CheckEqual(pool.references(private_block), 0, "the private block was freed");
  CheckEqual(pool.live_blocks(), 1, "one physical block now backs both tables");

  // Adopting the same block again is idempotent; it must not leak a reference.
  Check(destination.Adopt(pool, 0, source_block), "re-adopt is a no-op");
  CheckEqual(pool.references(source_block), 2, "no reference was added");

  destination.ReleaseAll(pool);
  CheckEqual(pool.references(source_block), 1, "source keeps the block");
  source.ReleaseAll(pool);
  CheckEqual(pool.free_blocks(), 4, "the block returns to the pool");
}

void AdoptRefusesInvalidBlocks() {
  KvBlockPool pool(2);
  KvBlockTable source(kKvBlockTokens, 1024);
  CheckEqual(source.Append(pool, kKvBlockTokens), kKvBlockTokens, "append");
  KvBlockTable destination(kKvBlockTokens, 1024);
  CheckEqual(destination.Append(pool, kKvBlockTokens), kKvBlockTokens,
             "append");
  const std::uint32_t before = destination.entry(0);

  Check(!destination.Adopt(pool, 0, kInvalidKvBlock), "invalid id refused");
  Check(!destination.Adopt(pool, 0, pool.capacity()),
        "out-of-range id refused");
  Check(!destination.Adopt(pool, 99, source.entry(0)), "index refused");
  CheckEqual(destination.entry(0), before,
             "refusals leave the table unchanged");
  CheckEqual(pool.references(source.entry(0)), 1, "no stray reference");
}

}  // namespace

int main() {
  PoolAcquireAndRelease();
  PoolReusesTheOldestFreeBlock();
  AppendAllocatesPerBlock();
  AppendStopsAtPoolExhaustion();
  AppendStopsAtTableCapacity();
  TruncateReleasesWholeBlocks();
  AppendDetachesSharedTail();
  TruncateKeepsUnsharedTail();
  TruncateDetachesSharedTail();
  TruncateRefusesWithoutABlock();
  ForkCopiesOnlyTheTail();
  AdoptSharesAFullBlock();
  AdoptRefusesInvalidBlocks();
  ByteOffsetLayout();
  ReleaseAllReturnsBlocks();
  std::cout << "Qwen KV block tests passed\n";
  return 0;
}
