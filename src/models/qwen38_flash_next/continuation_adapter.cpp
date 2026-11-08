#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/core/hip/snapshot_copy.hpp"
#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
bool Draft(ContinuationPart part) {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftPooledKeys ||
         part == ContinuationPart::kDraftRawKeys ||
         part == ContinuationPart::kDraftResidual;
}
bool Pool(ContinuationPart part) {
  return part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftPooledKeys;
}
struct Metadata {
  std::uint32_t version{2};
  std::uint32_t target{}, draft{}, blocks{}, draft_blocks{}, hidden_base{};
  std::uint32_t residual_valid{}, image_bytes{};
  NgramHistory ngram;
  MtpLengthState policy;
  speculative::PromptLookup<>::State lookup;
  std::array<std::uint8_t, 32> image{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
/// A range a shared block already covers: nothing is transferred, so the
/// restore sees a settled completion without opening a device job.
class SettledSignal final : public cache::CompletionSignal {
public:
  bool Ready() const noexcept override { return true; }
  TransferResult Wait() noexcept override { return TransferResult::kSucceeded; }
};
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->continuation_hooks_ = this;
    this->session->session_->continuation_hooks_ = this;
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    BeforeRelease();
    session->continuation_hooks_ = nullptr;
    session->session_->continuation_hooks_ = nullptr;
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed && (guarding || session->IsValid());
  }
  void BeforeExecution() override {
    if (!IsValid() || pending || checkpoint)
      throw std::logic_error("Flash-Next slot is not ready for execution");
  }
  void BeforeWrite(bool draft, std::uint32_t first, std::uint32_t end,
                   std::uint32_t first_block,
                   std::uint32_t end_block) override {
    // Model execution temporarily disables its public validity flag. The
    // adapter's restore/failure status and outstanding transfers still gate
    // every native write, including graph replay and speculative rollback.
    if (restoring || failed || pending || checkpoint)
      throw std::logic_error("Flash-Next slot is not ready for mutation");
    CaptureDuringGuard readable(*this);
    auto* table = session->session_ != nullptr
                      ? session->session_->kv_blocks_.get()
                      : nullptr;
    for (const auto& component : adapter.layout_.Components()) {
      if (component.descriptor.kind != cache::ComponentKind::kAppendRows ||
          Draft(component.part) != draft)
        continue;
      const auto a = Pool(component.part) ? first_block : first;
      const auto b = Pool(component.part) ? end_block : end;
      if (a < b)
        guard.BeforeOverwrite(component.descriptor.id, a, b);
      // The rows are about to change: a restore must no longer adopt them.
      if (table != nullptr && a < b &&
          (component.part == ContinuationPart::kKey ||
           component.part == ContinuationPart::kValue))
        adapter.DropShares(table, component.descriptor.id, a, b);
    }
  }
  void BeforeReset() override {
    if (invalidating)
      return;
    if (restoring || pending || checkpoint)
      throw std::logic_error("cannot reset a busy Flash-Next slot");
    GuardAll();
  }
  void AfterReset() override {
    // Native Reset queues zero fills. Settle them before exposing an empty
    // slot to captures on independent streams.
    Check(hipStreamSynchronize(adapter.model_->executor_->stream()));
  }
  void BeforeRestore() override {
    BeforeExecution();
    GuardAll();
  }
  void BeforeRelease() noexcept override {
    CaptureDuringGuard readable(*this);
    adapter.DropShares(session->session_ != nullptr
                           ? session->session_->kv_blocks_.get()
                           : nullptr);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(
            component.descriptor.id, 0,
            Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  void GuardAll() {
    CaptureDuringGuard readable(*this);
    auto* table = session->session_ != nullptr
                      ? session->session_->kv_blocks_.get()
                      : nullptr;
    adapter.DropShares(table);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(
            component.descriptor.id, 0,
            Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  const rocm::PrefillCheckpoint* checkpoint{};
  rocm::Session& CaptureState() const {
    return checkpoint ? *checkpoint->state : *session->session_;
  }
  struct CaptureDuringGuard {
    explicit CaptureDuringGuard(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~CaptureDuringGuard() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  bool restoring{false}, guarding{false}, invalidating{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  /// Block indices this restore adopted from another slot.
  std::vector<std::uint32_t> adopted;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         core::SessionMode mode,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      mode_(mode),
      context_(context),
      layout_(model_ ? model_->config() : Config{},
              mode == core::SessionMode::kSpeculative, context,
              model_ ? model_->executor_->max_speculative() : 0),
      identity_(std::move(identity)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasMtp()))
    throw std::invalid_argument("invalid Flash-Next adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Private components attest the target checkpoint boundary.
  put(Session::kSnapshotPayloadVersion);
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  put(model_->executor_->max_speculative());
  put(model_->PrefillCapacity());
  put(model_->DecodeConcurrency());
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Flash-Next adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.BeforeExecution();
  return *state.session;
}
void ContinuationAdapter::PrefillThrough(cache::Slot& slot,
                                         std::span<const std::int32_t> prompt,
                                         std::uint32_t boundary,
                                         const std::function<void()>& capture) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (!capture)
    throw std::invalid_argument("missing component checkpoint consumer");
  std::string error;
  const auto consume = [&](const rocm::PrefillCheckpoint& checkpoint) {
    struct BoundaryScope {
      State& state;
      bool guarding;
      BoundaryScope(State& state, const rocm::PrefillCheckpoint& checkpoint)
          : state(state), guarding(std::exchange(state.guarding, true)) {
        state.checkpoint = &checkpoint;
      }
      ~BoundaryScope() {
        // The checkpoint arena may be reused by the next native pass. Even a
        // rejected consumer must finish reading it before leaving this scope.
        {
          const std::lock_guard lock(state.jobs_mutex);
          for (const auto& job : state.jobs)
            (void)job->Wait();
        }
        state.checkpoint = nullptr;
        state.guarding = guarding;
      }
    } scope(state, checkpoint);
    capture();
    if (state.pending)
      throw std::logic_error("in-pass capture returned with pending transfers");
  };
  if (!state.session->SyncThroughComponents(prompt, boundary, consume, &error))
    throw std::runtime_error("component in-pass prefill failed: " + error);
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(mode_, context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Flash-Next slot");
  const auto& native = state.CaptureState();
  return layout_.Positions(native.position_, native.mtp_.position,
                           native.blocks_, native.mtp_.blocks);
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Flash-Next prefill plan");
  std::vector<Rows> result;
  while (first < end) {
    const auto remaining = end - first;
    const auto capacity = remaining <= model_->PrefillThroughCapacity()
                              ? model_->PrefillThroughCapacity()
                              : model_->PrefillCapacity();
    first += std::min<Rows>(remaining, capacity);
    result.push_back(first);
  }
  return result;
}
std::vector<cache::ComponentPosition>
ContinuationAdapter::PrefillCheckpointPositions(const cache::Slot& slot,
                                                Rows boundary) const {
  const auto& state = As(slot);
  if (!state.IsValid() || state.pending || state.checkpoint)
    throw std::logic_error(
        "Flash-Next slot is not ready for checkpoint planning");
  if (boundary <= state.session->Position() || boundary > context_ ||
      !state.session->PrefillCheckpointBytes(
          static_cast<std::uint32_t>(boundary)))
    throw std::invalid_argument("unsupported Flash-Next in-pass boundary");
  const auto& native = *state.session->session_;
  const auto& config = model_->config();
  const auto draft =
      mode_ == core::SessionMode::kSpeculative ? boundary - 1 : 0;
  return layout_.Positions(
      boundary, draft,
      boundary > config.indexer_top_k ? boundary / config.compress_ratio
                                      : native.blocks_,
      draft > config.indexer_top_k ? draft / config.compress_ratio
                                   : native.mtp_.blocks);
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Flash-Next component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  auto& native = state.CaptureState();
  switch (component.part) {
    case ContinuationPart::kTokens:
      return state.session->tokens_.data();
    case ContinuationPart::kKey:
      return native.attention_[component.layer].k_cache;
    case ContinuationPart::kValue:
      return native.attention_[component.layer].v_cache;
    case ContinuationPart::kPooledKeys:
      return native.attention_[component.layer].block_k;
    case ContinuationPart::kDraftKey:
      return native.mtp_.k_cache;
    case ContinuationPart::kDraftValue:
      return native.mtp_.v_cache;
    case ContinuationPart::kDraftPooledKeys:
      return native.mtp_.block_k;
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

std::vector<ContinuationAdapter::RowSpan> ContinuationAdapter::RowSpans(
    State& state, const ContinuationComponent& component, cache::Rows first,
    cache::Rows end) const {
  auto* base = static_cast<std::byte*>(RowData(state, component));
  const auto row = component.descriptor.row_bytes;
  const auto whole = [&]() {
    return std::vector<RowSpan>{{base + first * row, end - first}};
  };
  if (component.part != ContinuationPart::kKey &&
      component.part != ContinuationPart::kValue) {
    return whole();
  }
  const auto& native = state.CaptureState();
  const auto& attention = native.attention_[component.layer];
  if (attention.paged.table == nullptr)
    return whole();
  const std::uint64_t block_bytes =
      attention.paged.block_stride * sizeof(__half);
  const std::uint64_t layer_bytes =
      attention.paged.layer_stride * sizeof(__half);
  const cache::Rows block_rows = models::qwen::kKvBlockTokens;
  std::vector<RowSpan> spans;
  for (cache::Rows pos = first; pos < end;) {
    const cache::Rows in_block = pos % block_rows;
    const cache::Rows rows = std::min(block_rows - in_block, end - pos);
    // The cache stores rows in logical order; the bytes live in the physical
    // block the table assigned. Using the logical index would read or write
    // another session's block as soon as the pool is not laid out in order.
    // A prefill checkpoint is only a view of the session's device state, so
    // its own table is null: resolve ids through the owning session, whose
    // eager mapping never moves a block.
    cache::Rows physical = pos / block_rows;
    const auto* table = state.session->session_ != nullptr
                            ? state.session->session_->kv_blocks_.get()
                            : native.kv_blocks_.get();
    if (table != nullptr) {
      const auto block = static_cast<std::uint32_t>(pos / block_rows);
      const auto mapped = table->entry(block);
      if (mapped != models::qwen::kInvalidKvBlock)
        physical = mapped;
    }
    spans.push_back({base + std::uint64_t{physical} * block_bytes +
                         std::uint64_t{attention.layer_index} * layer_bytes +
                         in_block * row,
                     rows});
    pos += rows;
  }
  return spans;
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Flash-Next restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Flash-Next frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto frontier = [&](ContinuationPart part) -> std::uint32_t {
    for (std::size_t i = 0; i < ordered.size(); ++i)
      if (layout_.Components()[i].part == part) {
        if (ordered[i].valid_rows > context_)
          throw std::invalid_argument("Flash-Next frontier exceeds context");
        return static_cast<std::uint32_t>(ordered[i].valid_rows);
      }
    return 0;
  };
  const auto target = frontier(ContinuationPart::kTokens);
  const auto draft = frontier(ContinuationPart::kDraftKey);
  const auto expected =
      layout_.Positions(target, draft, frontier(ContinuationPart::kPooledKeys),
                        frontier(ContinuationPart::kDraftPooledKeys));
  if (ordered != expected)
    throw std::invalid_argument("inconsistent Flash-Next component frontiers");
  // Allocate host capacity and coverage first; any refusal leaves execution
  // valid. Guard all ranges while the original state is still readable.
  state.session->tokens_.reserve(target);
  state.session->logits_.reserve(model_->VocabSize());
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  Check(hipStreamSynchronize(model_->executor_->stream()));
  // A restore writes a different frontier: never reuse a block this slot
  // adopted from another session. Remapping also re-uploads the device table.
  RemapTable(*state.session->session_, target);
  state.session->session_->PreserveSnapshots(0, 0);
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.adopted.clear();
  state.metadata.fill(std::byte{});
  state.restoring = true;
  state.session->valid_ = false;
  // The restore rewrites the tokens in place, not through RestoreSnapshot.
  state.session->lookup_.Clear();
  state.session->tokens_.resize(target);
  state.session->ForgetDraftVocab();
  state.session->logits_.resize(model_->VocabSize());
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component, bool loading) const {
  auto& session = *state.session;
  auto& native = state.CaptureState();
  Region result;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false,
                       std::size_t offset = 0) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host});
  };
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(state.checkpoint ? const_cast<float*>(state.checkpoint->logits.data())
                           : session.logits_.data(),
          bytes, true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), bytes, true);
      break;
    case ContinuationPart::kConvolution:
      add(native.linear_[component.layer].conv_state, bytes);
      break;
    case ContinuationPart::kRecurrent:
      add(native.linear_[component.layer].state, bytes);
      break;
    case ContinuationPart::kPleHistory:
      add(native.ple_history_, bytes);
      break;
    case ContinuationPart::kDraftResidual:
      if (loading || native.mtp_.residual_valid)
        add(native.mtp_.h, bytes);
      break;
    case ContinuationPart::kKeptHidden:
      add(native.mtp_.target_hidden,
          loading ? bytes
                  : std::size_t{state.checkpoint ? state.checkpoint->hidden_rows
                                                 : session.KeptHiddenRows()} *
                        model_->config().HcDim() * sizeof(float));
      break;
    case ContinuationPart::kRawKeys:
    case ContinuationPart::kDraftRawKeys: {
      const auto positions = loading ? state.expected : Positions(state);
      const bool draft = Draft(component.part);
      std::uint32_t position = 0, blocks = 0;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto part = layout_.Components()[i].part;
        if (part ==
            (draft ? ContinuationPart::kDraftKey : ContinuationPart::kTokens))
          position = static_cast<std::uint32_t>(positions[i].valid_rows);
        if (part == (draft ? ContinuationPart::kDraftPooledKeys
                           : ContinuationPart::kPooledKeys))
          blocks = static_cast<std::uint32_t>(positions[i].valid_rows);
      }
      const auto& c = model_->config();
      const auto begin = blocks * c.compress_ratio;
      const auto rows = position - begin;
      const auto physical = begin & (native.index_capacity_ - 1);
      const auto tail = std::min(rows, native.index_capacity_ - physical);
      const auto width = std::size_t{c.indexer_head_dim} * sizeof(float);
      auto* pointer = reinterpret_cast<std::byte*>(
          draft ? native.mtp_.index_k
                : native.attention_[component.layer].index_k);
      add(pointer + physical * width, tail * width);
      add(pointer, (rows - tail) * width, false, tail * width);
      break;
    }
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Flash-Next private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.session->session_->spec_tokens_)
    throw std::logic_error(
        "cannot capture an invalid or speculative Flash-Next frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flash-Next private piece");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  if (component.part == ContinuationPart::kMetadata) {
    auto& session = *state.session;
    auto& native = state.CaptureState();
    Metadata metadata{};
    metadata.target = native.position_;
    metadata.draft = native.mtp_.position;
    metadata.blocks = native.blocks_;
    metadata.draft_blocks = native.mtp_.blocks;
    metadata.hidden_base =
        state.checkpoint ? native.position_ - state.checkpoint->hidden_rows
                         : session.hidden_base_;
    metadata.residual_valid = native.mtp_.residual_valid;
    metadata.ngram = native.ngram_;
    metadata.policy = session.draft_length_.State();
    metadata.lookup = session.lookup_.Save();
    const auto identity = session.ImageIdentity(native.position_);
    metadata.image_bytes = static_cast<std::uint32_t>(identity.size());
    if (!identity.empty()) {
      if (identity.size() != metadata.image.size())
        throw std::logic_error("invalid Flash-Next image identity");
      std::ranges::copy(identity, metadata.image.begin());
    }
    std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
  }
  auto region = PrivateRegion(state, component, false);
  if (component.part == ContinuationPart::kMetadata)
    region.pieces = {{host_metadata.data(), 0, host_metadata.size(), true}};
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes, which
        // serialize as zeros: clearing the whole recycled copy on the host
        // dominated capture time.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Flash-Next private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Flash-Next private piece");
    const auto region = PrivateRegion(state, component, true);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Flash-Next slot");
  const auto& native = state.CaptureState();
  const auto rows =
      component.part == ContinuationPart::kTokens ? native.position_
      : Pool(component.part)
          ? (Draft(component.part) ? native.mtp_.blocks : native.blocks_)
          : (Draft(component.part) ? native.mtp_.position : native.position_);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > rows || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Flash-Next row capture");
  const auto spans = bytes.empty() ? std::vector<RowSpan>{}
                                   : RowSpans(state, component, first, end);
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        std::size_t offset = 0;
        for (const auto& span : spans) {
          const std::size_t count = span.rows * row;
          if (component.part == ContinuationPart::kTokens)
            std::memcpy(bytes.data() + offset, span.data, count);
          else
            Check(hip::CopySnapshotToHost(bytes.data() + offset, span.data,
                                          count, transfer.Native()));
          offset += count;
        }
      });
}
std::span<const std::byte> ContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& component = Component(id);
  // The host token vector can move when execution appends. Device append
  // allocations stay fixed for the session and are guarded before mutation.
  if (component.part == ContinuationPart::kTokens)
    return {};
  if (!state.IsValid() ||
      component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::logic_error("cannot borrow invalid Flash-Next rows");
  const auto positions = Positions(slot);
  const auto position =
      std::ranges::find(positions, id, &cache::ComponentPosition::id);
  const auto row = component.descriptor.row_bytes;
  if (position == positions.end() || first >= end || end > position->valid_rows)
    throw std::invalid_argument("invalid Flash-Next borrowed range");
  const auto& native = As(slot).CaptureState();
  if ((component.part == ContinuationPart::kKey ||
       component.part == ContinuationPart::kValue) &&
      native.attention_[component.layer].paged.table != nullptr) {
    // A retained checkpoint that borrows device rows pins the shared pool until
    // it is evicted. When the pool budget cannot cover one full context per
    // session, that pinning starves requests still waiting to prefill: their
    // blocks never return while completed requests keep the cached snapshot.
    // Such a pool keeps the rows on the host instead (swap-out), so ending a
    // request releases its blocks immediately and the cache stays usable.
    if (!LendDeviceRows())
      return {};
    // A paged plane is contiguous only inside one block, and cache chunks are
    // exactly one block wide. A contiguous run can therefore be lent as is: the
    // cache keeps the source's device bytes and a restore adopts the physical
    // block instead of copying its rows. A run that crosses a block boundary
    // cannot be lent, so the cache copies it.
    const auto spans =
        RowSpans(const_cast<State&>(state), component, first, end);
    if (spans.size() != 1 || spans.front().rows != end - first)
      return {};
    const auto* table = state.session->session_ != nullptr
                            ? state.session->session_->kv_blocks_.get()
                            : native.kv_blocks_.get();
    if (table != nullptr && first % qwen::kKvBlockTokens == 0 &&
        end % qwen::kKvBlockTokens == 0)
      RecordShare(table, id, first, end, spans.front().data);
    return {spans.front().data, static_cast<std::size_t>((end - first) * row)};
  }
  return {static_cast<const std::byte*>(
              RowData(const_cast<State&>(state), component)) +
              first * row,
          (end - first) * row};
}
bool ContinuationAdapter::LendDeviceRows() const {
  if (model_ && !model_->DeviceRowLending())
    return false;
  const auto gauges = PoolGauges();
  if (!gauges)
    return true;
  const std::uint32_t per_session =
      (context_ + models::qwen::kKvBlockTokens - 1) /
      models::qwen::kKvBlockTokens;
  // A live borrow shares blocks the source already owns; one full context is
  // enough for that. Retaining a completed frontier needs the larger headroom
  // checked by RetainDeviceRows below.
  return gauges->capacity_blocks >= per_session;
}
void ContinuationAdapter::RecordShare(const models::qwen::KvBlockTable* table,
                                      cache::ComponentId id, Rows first,
                                      Rows end, const std::byte* data) const {
  if (table == nullptr || data == nullptr)
    return;
  const std::lock_guard lock(shares_mutex_);
  // Bounded registry: the guard drops a table's shares when its rows change,
  // and this cap only protects a pathological long-lived source.
  if (shares_.size() >= kMaxShareRecords)
    shares_.erase(
        shares_.begin(),
        shares_.begin() + static_cast<std::ptrdiff_t>(shares_.size() / 4));
  shares_.push_back({table, data, id, first, end});
}
void ContinuationAdapter::DropShares(
    const models::qwen::KvBlockTable* table) const {
  if (table == nullptr)
    return;
  const std::lock_guard lock(shares_mutex_);
  std::erase_if(shares_, [table](const ShareRecord& share) {
    return share.table == table;
  });
}
void ContinuationAdapter::DropShares(const models::qwen::KvBlockTable* table,
                                     cache::ComponentId id, Rows first,
                                     Rows end) const {
  if (table == nullptr)
    return;
  const std::lock_guard lock(shares_mutex_);
  const bool overlap = std::any_of(
      shares_.begin(), shares_.end(), [&](const ShareRecord& share) {
        return share.table == table && share.id == id && share.first < end &&
               first < share.end;
      });
  if (overlap)
    std::erase_if(shares_, [table](const ShareRecord& share) {
      return share.table == table;
    });
}
void ContinuationAdapter::RemapTable(rocm::Session& native,
                                     std::uint32_t tokens) {
  auto* table = native.kv_blocks_.get();
  auto* pool = native.owner_ != nullptr ? native.owner_->kv_pool() : nullptr;
  if (table == nullptr || pool == nullptr)
    return;
  table->ReleaseAll(*pool);
  // Map only what the caller needs: a restore maps its target frontier, a fresh
  // request maps nothing and grows the table as execution writes into it. The
  // copy path resolves destination blocks through this table, so a restore must
  // map its frontier here rather than relying on the lazy growth path.
  if (tokens != 0 && table->Append(*pool, tokens) < tokens)
    throw std::runtime_error("attention KV pool exhausted");
  native.d_kv_table_rows_ = 0;
}
cache::Rows ContinuationAdapter::AdoptSharedRows(
    State& state, const ContinuationComponent& component, Rows first, Rows end,
    const std::byte* data) {
  if (component.part != ContinuationPart::kKey &&
      component.part != ContinuationPart::kValue)
    return 0;
  const auto block_rows = Rows{qwen::kKvBlockTokens};
  if (data == nullptr || first >= end || first % block_rows != 0 ||
      end % block_rows != 0)
    return 0;
  auto& native = *state.session->session_;
  auto* destination = native.kv_blocks_.get();
  auto* pool = native.owner_ != nullptr ? native.owner_->kv_pool() : nullptr;
  if (destination == nullptr || pool == nullptr)
    return 0;
  std::optional<ShareRecord> match;
  {
    const std::lock_guard lock(shares_mutex_);
    for (const auto& share : shares_)
      if (share.id == component.descriptor.id && share.first == first &&
          share.end == end && share.data == data) {
        match = share;
        break;
      }
  }
  if (!match || match->table == nullptr || match->table == destination)
    return 0;
  const auto begin = static_cast<std::uint32_t>(first / block_rows);
  const auto finish = static_cast<std::uint32_t>(end / block_rows);
  if (finish * block_rows > destination->max_position())
    return 0;
  // Validate the whole range first: a refusal must leave the table untouched,
  // because the caller then copies the rows into the destination's own blocks.
  for (auto index = begin; index < finish; ++index) {
    const auto block = match->table->entry(index);
    if (block == qwen::kInvalidKvBlock || pool->references(block) == 0)
      return 0;
  }
  for (auto index = begin; index < finish; ++index) {
    const auto block = match->table->entry(index);
    if (!destination->Adopt(*pool, index, block))
      throw std::logic_error("Flash-Next block share failed after validation");
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.adopted.push_back(index);
    }
  }
  // Adopted entries replace rows the device mirror already uploaded; force a
  // complete table upload before the next execution reads a block.
  native.d_kv_table_rows_ = 0;
  return end - first;
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Flash-Next row restore");
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    if (component.part == ContinuationPart::kKey ||
        component.part == ContinuationPart::kValue) {
      const auto block_rows = Rows{qwen::kKvBlockTokens};
      const bool aligned = first % block_rows == 0 && end % block_rows == 0;
      if (aligned && first < end) {
        // Every layer shares one block index. Once a component adopted a
        // block, the remaining layers already find their rows in place; a
        // copy would write into the shared block.
        bool adopted = true;
        {
          const std::lock_guard lock(state.jobs_mutex);
          for (auto index = first / block_rows; index < end / block_rows;
               ++index)
            if (std::ranges::find(state.adopted,
                                  static_cast<std::uint32_t>(index)) ==
                state.adopted.end()) {
              adopted = false;
              break;
            }
        }
        if (adopted)
          return cache::Completion(std::make_unique<SettledSignal>());
        // Writing a block another table reads would corrupt it. Sharing is
        // only ever entered with all layers' borrows live together, so this is
        // a safety net; refuse rather than alias silently.
        auto& native = *state.session->session_;
        auto* table = native.kv_blocks_.get();
        auto* pool =
            native.owner_ != nullptr ? native.owner_->kv_pool() : nullptr;
        if (table != nullptr && pool != nullptr)
          for (auto index = first / block_rows; index < end / block_rows;
               ++index)
            if (pool->references(
                    table->entry(static_cast<std::uint32_t>(index))) > 1)
              throw std::logic_error(
                  "Flash-Next restore would write a shared KV block");
      }
    }
    // A full block of another resident session is append-only: adopting it
    // costs one pool reference instead of copying its rows. The cache hands
    // back the source's own device bytes for a live borrow, so the pointer
    // both identifies the share and proves it is still unmodified.
    if (const auto adopted =
            AdoptSharedRows(state, component, first, end, bytes.data());
        adopted != 0)
      return cache::Completion(std::make_unique<SettledSignal>());
    const auto spans = bytes.empty() ? std::vector<RowSpan>{}
                                     : RowSpans(state, component, first, end);
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          std::size_t offset = 0;
          for (const auto& span : spans) {
            const std::size_t count = span.rows * row;
            if (component.part == ContinuationPart::kTokens)
              std::memcpy(span.data, bytes.data() + offset, count);
            else
              Check(hip::CopySnapshotToDevice(span.data, bytes.data() + offset,
                                              count, transfer.Native()));
            offset += count;
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Flash-Next validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  auto& session = *state.session;
  auto& native = *session.session_;
  auto policy = session.draft_length_;
  if (metadata.version != 2 || metadata.hidden_base > metadata.target ||
      metadata.target - metadata.hidden_base >
          model_->executor_->max_speculative() ||
      metadata.residual_valid > 1 ||
      (metadata.residual_valid && !metadata.draft) ||
      (metadata.image_bytes != 0 && metadata.image_bytes != 32) ||
      !policy.Restore(metadata.policy) ||
      !speculative::PromptLookup<>::Valid(metadata.lookup, metadata.target))
    return refuse();
  try {
    if (layout_.Positions(metadata.target, metadata.draft, metadata.blocks,
                          metadata.draft_blocks) != state.expected)
      return refuse();
    const auto identity = session.ImageIdentity(metadata.target);
    if (metadata.image_bytes && (identity.size() != 32 ||
                                 !std::ranges::equal(identity, metadata.image)))
      return refuse();
    if (metadata.image_bytes || (session.image_prompt_ && identity.empty())) {
      // Preserve the attached request's future images as well as its matching
      // computed prefix, including checkpoints before the first image.
      native.RestoreVisionLayout(session.image_prompt_->rope,
                                 model_->executor_->stream());
    } else {
      native.ConfigureVision(nullptr, nullptr, model_->executor_->stream());
      session.image_prompt_.reset();
    }
    Check(hipStreamSynchronize(model_->executor_->stream()));
  } catch (...) {
    return refuse();
  }
  native.position_ = metadata.target;
  native.blocks_ = metadata.blocks;
  native.mtp_.position = metadata.draft;
  native.mtp_.blocks = metadata.draft_blocks;
  native.mtp_.residual_valid = metadata.residual_valid;
  native.ngram_ = metadata.ngram;
  native.spec_base_ = metadata.target;
  native.spec_tokens_ = 0;
  native.snapshot_lineage_.reset();
  session.hidden_base_ = metadata.hidden_base;
  session.draft_length_ = policy;
  session.lookup_.Restore(metadata.lookup);
  session.draft_token_ = 0;
  session.stats_ = {};
  session.valid_ = true;
  if (!state.adopted.empty()) {
    std::ranges::sort(state.adopted);
    state.adopted.erase(std::unique(state.adopted.begin(), state.adopted.end()),
                        state.adopted.end());
    if (server::Logger::Enabled(server::LogLevel::kInfo)) {
      std::string detail =
          "schema=component-cache-v1 event=kv_blocks_shared blocks=" +
          std::to_string(state.adopted.size());
      const auto& native = *state.session->session_;
      if (native.owner_ != nullptr && native.owner_->kv_pool() != nullptr) {
        const auto* pool = native.owner_->kv_pool();
        detail +=
            " pool_live=" + std::to_string(pool->live_blocks()) +
            " pool_free=" + std::to_string(pool->free_blocks()) +
            " pool_capacity=" + std::to_string(pool->capacity()) +
            " pool_high_water=" + std::to_string(pool->high_water_blocks());
      }
      server::Logger::Info("cache", detail);
    }
  }
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  state.adopted.clear();
  return true;
}
bool ContinuationAdapter::RetainDeviceRows() const {
  if (!LendDeviceRows())
    return false;
  const auto gauges = PoolGauges();
  if (!gauges)
    return true;
  const std::uint32_t per_session =
      (context_ + models::qwen::kKvBlockTokens - 1) /
      models::qwen::kKvBlockTokens;
  const std::uint64_t coverage =
      std::uint64_t{std::max<std::uint32_t>(1, model_->DecodeConcurrency())} *
      per_session;
  // Keeping a completed live frontier pins its device rows until the slot is
  // reused. Only do that when the pool can cover one full context per resident
  // session; otherwise the host snapshot is the durable copy and the slot is
  // released immediately.
  return gauges->capacity_blocks >= coverage;
}
std::optional<cache::KvPoolGauges> ContinuationAdapter::PoolGauges() const {
  if (!model_ || !model_->executor_)
    return std::nullopt;
  const auto* pool = model_->executor_->kv_pool();
  if (pool == nullptr)
    return std::nullopt;
  const auto& c = model_->config();
  if (c.full_attention_interval == 0)
    return std::nullopt;
  const std::uint64_t layers = c.num_layers / c.full_attention_interval;
  const std::uint64_t block_bytes = 2ULL * models::qwen::kKvBlockTokens *
                                    layers * c.AttentionKvDim() *
                                    sizeof(std::uint16_t);
  const auto snapshot = pool->Snapshot();
  return cache::KvPoolGauges{
      snapshot.capacity_blocks,   snapshot.live_blocks,   snapshot.free_blocks,
      snapshot.high_water_blocks, snapshot.shared_blocks, block_bytes};
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending || state.checkpoint)
      return false;
    resetting = &state;
    state.BeforeRelease();
    struct ResetScope {
      bool& active;
      ~ResetScope() { active = false; }
    } reset_scope{state.invalidating};
    state.invalidating = true;
    // Reassign a healthy resident slot without rebuilding its MTP graphs.
    // Failed state and explicit Session::Reset still release all scratch.
    state.session->ResetImpl(!state.failed && state.session->IsValid());
    // The next request may write any row: release every adopted share and map
    // a private block table again before it prefills. Zero maps nothing; the
    // first prefill appends the blocks it writes into.
    RemapTable(*state.session->session_, 0);
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.adopted.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}
}  // namespace gufo::models::qwen38_flash_next
