#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_ADAPTER_HPP_

#include <mutex>

#include "src/cache/adapter.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/models/qwen38_flash_next/continuation_layout.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace gufo::models::qwen {
class KvBlockTable;
}  // namespace gufo::models::qwen

namespace gufo::models::qwen38_flash_next {

// The caller supplies the complete artifact/tokenizer/template identity. The
// adapter adds its state ABI, context and execution policy. Slots own public
// model sessions, so the runner uses the same Sync/Decode API as legacy
// serving.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model>, core::SessionMode,
                      std::uint32_t context, cache::Identity);
  ~ContinuationAdapter() override;
  [[nodiscard]] Session& GetSession(cache::Slot&);
  // Capture synchronously while the coherent boundary is exposed; settle
  // all transfers before the callback returns and the shared arena is reused.
  void PrefillThrough(cache::Slot&, std::span<const std::int32_t>,
                      std::uint32_t boundary, const std::function<void()>&);
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const {
    return layout_.BackingPlan(ram_bytes, model_->DecodeConcurrency());
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  [[nodiscard]] std::vector<cache::ComponentPosition>
  PrefillCheckpointPositions(const cache::Slot&, cache::Rows) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows,
      cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;
  [[nodiscard]] std::optional<cache::KvPoolGauges> PoolGauges() const override;
  [[nodiscard]] bool RetainDeviceRows() const override;

private:
  class State;
  struct Region;
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  [[nodiscard]] Region PrivateRegion(State&, const ContinuationComponent&,
                                     bool loading) const;
  [[nodiscard]] void* RowData(State&, const ContinuationComponent&) const;
  /// Contiguous byte runs covering rows [first, end) of a component.
  /// A paged KV plane yields one run per block; every other component is one
  /// run. The concatenation is always `(end - first) * row_bytes`.
  struct RowSpan {
    std::byte* data;
    cache::Rows rows;
  };
  [[nodiscard]] std::vector<RowSpan> RowSpans(State&,
                                              const ContinuationComponent&,
                                              cache::Rows first,
                                              cache::Rows end) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(gufo::hip::TransferStream&)>&);
  /// A device row range a slot lent to the cache while it was still live.
  ///
  /// A restore that receives exactly these bytes may adopt the source's
  /// physical blocks instead of copying rows: full blocks are append-only, so
  /// sharing them cannot change their contents. Ranges are one block wide
  /// because a block is the largest run that is contiguous in the pool.
  struct ShareRecord {
    const models::qwen::KvBlockTable* table{nullptr};
    const std::byte* data{nullptr};
    cache::ComponentId id{};
    cache::Rows first{0}, end{0};
  };
  void RecordShare(const models::qwen::KvBlockTable*, cache::ComponentId,
                   cache::Rows first, cache::Rows end,
                   const std::byte* data) const;
  /// Whether a retained checkpoint may borrow device KV rows. False when the
  /// shared pool cannot cover one full context per resident session: lending
  /// would then pin blocks that waiting requests need.
  [[nodiscard]] bool LendDeviceRows() const;
  /// Forgets every share of a table whose rows are about to change. Sharing
  /// must stop before the source overwrites the block, or a restore would
  /// adopt rows that no longer match its checkpoint.
  void DropShares(const models::qwen::KvBlockTable*) const;
  /// Drops a table's shares when rows of one component are about to change.
  void DropShares(const models::qwen::KvBlockTable*, cache::ComponentId,
                  cache::Rows first, cache::Rows end) const;
  /// Releases the session's block table and maps `tokens` positions again, so
  /// no adopted share survives into a new request or restore. A restore passes
  /// its target frontier; a fresh request passes zero and grows the table as
  /// execution writes into it.
  static void RemapTable(rocm::Session&, std::uint32_t tokens);
  /// Adopts the source blocks covered by a live borrow. Returns the number of
  /// rows adopted, or zero when the bytes are not a live share (the caller
  /// then copies rows as usual).
  cache::Rows AdoptSharedRows(State&, const ContinuationComponent&,
                              cache::Rows first, cache::Rows end,
                              const std::byte* data);
  mutable std::mutex shares_mutex_;
  mutable std::vector<ShareRecord> shares_;
  /// Bounds the registry; the guard drops a table's shares when it mutates, so
  /// this only protects against a long-lived source with many captures.
  static constexpr std::size_t kMaxShareRecords = 1 << 16;
  std::shared_ptr<Model> model_;
  core::SessionMode mode_;
  std::uint32_t context_;
  ContinuationLayout layout_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};
}  // namespace gufo::models::qwen38_flash_next
#endif
