#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/generation_metrics.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/cli/serve/trace.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::server {
namespace {

constexpr std::size_t kDecodeProgressInterval = 50;
constexpr std::size_t kNoSession = std::numeric_limits<std::size_t>::max();

struct OutputBudget {
  explicit OutputBudget(std::size_t byte_limit) : limit(byte_limit) {}

  [[nodiscard]] bool TryReserve(std::size_t bytes) noexcept {
    std::size_t current = buffered_bytes.load(std::memory_order_relaxed);
    while (current <= limit && bytes <= limit - current) {
      const std::size_t updated = current + bytes;
      if (buffered_bytes.compare_exchange_weak(current, updated,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
        std::size_t high_water =
            max_buffered_bytes.load(std::memory_order_relaxed);
        while (high_water < updated &&
               !max_buffered_bytes.compare_exchange_weak(
                   high_water, updated, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
        return true;
      }
    }
    return false;
  }

  void Release(std::size_t bytes) noexcept {
    if (bytes > 0) {
      buffered_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
    }
  }

  const std::size_t limit;
  std::atomic<std::size_t> buffered_bytes{0};
  std::atomic<std::size_t> max_buffered_bytes{0};
};

struct ScheduledRequest;

/// Requests holding each execution session, for `/slots`. No other mutex is
/// acquired while this table's mutex is held.
struct SessionTable {
  explicit SessionTable(std::size_t count) : requests(count) {}

  std::mutex mutex;
  std::vector<std::shared_ptr<const ScheduledRequest>> requests;
};

struct OutputPiece {
  std::string text;
  std::optional<DecodedLogprobs> logprobs;
};

struct ScheduledRequest {
  std::uint64_t id{0};
  std::string client_id{"anonymous"};
  std::vector<TextRunnerToken> prompt;
  std::shared_ptr<const TextPromptContext> prompt_context;
  bool cache_prompt{true};
  bool cache_persist{true};
  std::size_t cache_prefix_tokens{0};
  bool stop_at_eos{true};
  std::size_t token_limit{1};
  /// True when the client supplied a positive max_tokens. Unlimited requests
  /// reserve a conservative short-chat budget for admission instead of the
  /// whole remaining context.
  bool output_limit_explicit{false};
  sampling::SamplingConfig sampling;
  TextGenerationScheduler::CancellationCheck external_cancellation;
  bool publish_token_pieces{false};
  bool publish_prompt_progress{false};
  TextGenerationScheduler::Clock::time_point request_start;
  TextGenerationScheduler::Clock::time_point stream_start_deadline;
  std::optional<TextGenerationScheduler::Clock::time_point> deadline;
  std::size_t max_output_bytes{0};
  std::size_t max_buffered_output_bytes{0};
  std::shared_ptr<OutputBudget> output_budget;
  std::shared_ptr<SessionTable> sessions;

  std::atomic<bool> cancellation_requested{false};
  std::atomic<TextRequestPhase> phase{TextRequestPhase::kQueued};

  TextRunnerPool::Request runner_request;
  // Opt-in shared-prefix/request coalescing. A follower has no runner request
  // and no session; it receives the leader's greedy token stream.
  std::vector<TextRunnerToken> coalesce_prompt;
  bool coalesce_eligible{false};
  bool coalesced_follower{false};
  std::weak_ptr<ScheduledRequest> coalesce_leader;
  std::vector<std::shared_ptr<ScheduledRequest>> coalesce_followers;
  TextGenerationScheduler::Result result;
  std::optional<TextGenerationScheduler::Clock::time_point> previous_token;
  std::chrono::duration<double, std::milli> inter_token_total{0};
  std::size_t inter_token_samples{0};
  std::size_t last_decode_progress_tokens{0};
  double last_decode_progress_ms{0.0};
  std::optional<TextGenerationScheduler::Clock::time_point> prefill_start;
  bool decode_due{false};
  std::optional<TextRunnerToken> preview_token;
  bool advance_pending{false};
  bool preview_stops{false};
  /// RECOMPUTE preemption releases the model state and keeps only this
  /// request's tokens; these accumulate how often that happened.
  std::size_t preemptions{0};
  /// Times the device pool refused blocks and the scheduler degraded this
  /// request to its host snapshot instead of failing it.
  std::size_t kv_deferrals{0};
  /// First degradation of the current episode. A request keeps waiting while
  /// residents make progress, so the deadline bounds the wait, not the count.
  std::optional<TextGenerationScheduler::Clock::time_point> kv_deferred_since;
  /// Earliest re-admission after such a degradation; the residents need time
  /// to free memory before the retained snapshot can be restored.
  std::optional<TextGenerationScheduler::Clock::time_point> kv_retry_at;
  std::optional<sampling::SamplerState> resume_sampler;
  std::size_t generated_output_bytes{0};
  StopSequenceFilter stop_filter;
  LogprobDecoder logprob_decoder;

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::deque<OutputPiece> output_pieces;
  std::optional<TextGenerationBackend::PromptProgress> pending_progress;
  // Set at admission for streams; the consumer reports its start at most once.
  bool admitted{false};
  std::size_t buffered_output_bytes{0};
  // A stalled consumer trips backpressure on every token, so the debug line is
  // written once per request. Only the scheduler thread touches this.
  bool backpressure_logged{false};
  // Only the scheduler worker changes this, from admission through cleanup.
  bool counted_processing{false};
  /// First admission only: a replay keeps its original queue time.
  bool admitted_once{false};
  // Admission waits at most once for a resident request to publish the
  // prompt prefix both share. Only the scheduler thread touches these.
  std::shared_ptr<ScheduledRequest> prefix_leader;
  std::size_t prefix_position{0};
  bool prefix_considered{false};
  std::optional<TextGenerationScheduler::Clock::time_point> prefix_wait_start;
  // Admission also waits, once and bounded, for a disk job writing a much
  // deeper prefix of the prompt (a conversation larger than the RAM budget).
  std::optional<TextGenerationScheduler::Clock::time_point>
      publication_wait_start;
  bool publication_waiting{false};
  TextPendingPublication publication;
  detail::PublicationWaitEstimate::Milliseconds publication_wait_limit{};
  // The held `/slots` session while counted_processing is set. Only the
  // scheduler worker changes it.
  std::size_t session{kNoSession};
  // Copies of result counters for `/slots`, which reads them from another
  // thread while the scheduler worker updates `result`.
  std::atomic<std::size_t> live_cached_prompt_tokens{0};
  std::atomic<std::size_t> live_prefill_tokens{0};
  std::atomic<std::size_t> live_generated_tokens{0};
  std::exception_ptr failure;
  bool terminal{false};

  ~ScheduledRequest() {
    // A completed stream may be discarded without ever consuming its output.
    // Return any remaining charge when the last request owner releases it.
    if (output_budget != nullptr)
      output_budget->Release(buffered_output_bytes);
  }
};

/// Token accounting for the shared paged-KV pool, when the runner has one.
struct SharedKvPoolTokens {
  std::size_t capacity{0};
  std::size_t free{0};
  std::size_t block{1};
};

struct PendingClient {
  std::string client_id;
  std::deque<std::shared_ptr<ScheduledRequest>> requests;
};

void LogPublicationWait(const std::shared_ptr<ScheduledRequest>& request,
                        std::string_view stage, std::string_view release = {}) {
  if (!Logger::Enabled(LogLevel::kInfo))
    return;
  std::ostringstream line;
  line << "event=publication_wait request=" << request->id << " stage=" << stage
       << " checkpoint_id=" << request->publication.checkpoint_id
       << " pending_tokens=" << request->publication.pending_tokens
       << " available_tokens=" << request->publication.available_tokens
       << " limit_ms=" << request->publication_wait_limit.count();
  if (request->publication_wait_start)
    line << " wait_ms="
         << std::chrono::duration<double, std::milli>(
                TextGenerationScheduler::Clock::now() -
                *request->publication_wait_start)
                .count();
  if (!release.empty())
    line << " release=" << release;
  Logger::Info("cache", line.str());
}

void EndPublicationWait(const std::shared_ptr<ScheduledRequest>& request,
                        std::string_view reason) noexcept {
  if (!std::exchange(request->publication_waiting, false))
    return;
  try {
    LogPublicationWait(request, "release", reason);
  } catch (...) {
    // Observability cannot prevent cancellation or failure completion.
  }
}

bool IsTerminal(const std::shared_ptr<ScheduledRequest>& request) noexcept {
  return request->phase.load(std::memory_order_acquire) ==
         TextRequestPhase::kTerminal;
}

/// True while a request degraded by the device pool waits in the deferred
/// queue for residents to free memory. It owns no runner lease and must not be
/// re-queued among residents.
bool AwaitingKvRetry(
    const std::shared_ptr<ScheduledRequest>& request) noexcept {
  return request->kv_retry_at.has_value();
}

void ReleaseBufferedOutputLocked(
    const std::shared_ptr<ScheduledRequest>& request) noexcept {
  request->output_pieces.clear();
  request->output_budget->Release(request->buffered_output_bytes);
  request->buffered_output_bytes = 0;
}

void PublishTerminal(const std::shared_ptr<ScheduledRequest>& request,
                     std::exception_ptr failure = {},
                     bool discard_pending_output = false) noexcept {
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (discard_pending_output) {
      ReleaseBufferedOutputLocked(request);
    }
    request->failure = std::move(failure);
    request->terminal = true;
    if (std::exchange(request->counted_processing, false)) {
      if (request->session != kNoSession) {
        const std::lock_guard<std::mutex> sessions_lock(
            request->sessions->mutex);
        request->sessions->requests[std::exchange(request->session, kNoSession)]
            .reset();
      }
      detail::RequestsProcessing().fetch_sub(1, std::memory_order_relaxed);
    }
    request->phase.store(TextRequestPhase::kTerminal,
                         std::memory_order_release);
  }
  request->output_condition.notify_all();
}

/// Queue cost of one streamed piece. A token whose decoded text is empty still
/// occupies a queue slot, so it is charged one byte: a zero cost would let an
/// unbounded number of them accumulate behind a slow consumer.
[[nodiscard]] std::size_t QueuedPieceCost(const OutputPiece& piece) noexcept {
  return std::max<std::size_t>(piece.text.size(), 1) +
         (piece.logprobs ? piece.logprobs->StorageBytes() : 0);
}

// Debug-only backpressure line, extracted from the publish path: the fast
// path needs the `reason.empty()` branch, not fifteen lines of formatting.
// The once-per-request flag lives on the request; the budgets read here are
// only ever loaded, never taken, so this runs unlocked after PublishPiece
// releases output_mutex.
void LogBackpressure(const std::shared_ptr<ScheduledRequest>& request,
                     std::string_view reason, std::size_t buffered_bytes,
                     std::size_t piece_bytes) {
  if (!Logger::Enabled(LogLevel::kDebug) || request->backpressure_logged) {
    return;
  }
  request->backpressure_logged = true;
  Logger::Debug("scheduler",
                "event=backpressure request=" + std::to_string(request->id) +
                    " reason=" + std::string(reason) +
                    " buffered_bytes=" + std::to_string(buffered_bytes) +
                    " piece_bytes=" + std::to_string(piece_bytes) +
                    " request_limit_bytes=" +
                    std::to_string(request->max_buffered_output_bytes) +
                    " total_bytes=" +
                    std::to_string(request->output_budget->buffered_bytes.load(
                        std::memory_order_relaxed)) +
                    " total_limit_bytes=" +
                    std::to_string(request->output_budget->limit));
}

[[nodiscard]] bool PublishPiece(
    const std::shared_ptr<ScheduledRequest>& request, std::string text,
    const DecodedLogprobs* logprobs = nullptr) {
  if (!request->publish_token_pieces) {
    return true;
  }
  OutputPiece piece{std::move(text),
                    logprobs ? std::optional(*logprobs) : std::nullopt};
  const std::size_t piece_bytes = QueuedPieceCost(piece);
  std::string_view reason;
  std::size_t buffered = 0;
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (request->buffered_output_bytes > request->max_buffered_output_bytes ||
        piece_bytes > request->max_buffered_output_bytes -
                          request->buffered_output_bytes) {
      reason = "request_buffer";
      buffered = request->buffered_output_bytes;
    } else if (!request->output_budget->TryReserve(piece_bytes)) {
      reason = "total_buffer";
      buffered = request->buffered_output_bytes;
    } else {
      try {
        request->buffered_output_bytes += piece_bytes;
        request->result.max_buffered_output_bytes =
            std::max(request->result.max_buffered_output_bytes,
                     request->buffered_output_bytes);
        request->output_pieces.push_back(std::move(piece));
      } catch (...) {
        request->output_budget->Release(piece_bytes);
        request->buffered_output_bytes -= piece_bytes;
        throw;
      }
    }
  }
  if (reason.empty()) {
    request->output_condition.notify_one();
    return true;
  }
  // Reported after releasing output_mutex: the line names which budget refused
  // the piece, which is the question an operator asks when a stream stalls.
  LogBackpressure(request, reason, buffered, piece_bytes);
  return false;
}

/// Lets a streaming consumer commit its response before prompt processing.
void PublishAdmission(const std::shared_ptr<ScheduledRequest>& request) {
  if (!request->publish_token_pieces) {
    return;
  }
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    request->admitted = true;
  }
  request->output_condition.notify_one();
}

/// Replaces unread progress, so a slow consumer holds at most one update.
void PublishPromptProgress(const std::shared_ptr<ScheduledRequest>& request) {
  if (!request->publish_prompt_progress) {
    return;
  }
  const auto now = TextGenerationScheduler::Clock::now();
  if (!request->prefill_start.has_value()) {
    request->prefill_start = now;
  }
  const std::size_t total = request->result.prompt_tokens;
  const std::size_t cache =
      std::min(request->result.cached_prompt_tokens, total);
  const TextGenerationBackend::PromptProgress progress{
      .total = total,
      .cache = cache,
      .processed = std::min(cache + request->result.prefill_tokens, total),
      .time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     now - *request->prefill_start)
                     .count(),
  };
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    request->pending_progress = progress;
  }
  request->output_condition.notify_one();
}

bool CancellationRequested(const std::shared_ptr<ScheduledRequest>& request) {
  if (request->cancellation_requested.load(std::memory_order_acquire)) {
    return true;
  }
  return request->external_cancellation && request->external_cancellation();
}

bool DeadlineExceeded(const std::shared_ptr<ScheduledRequest>& request,
                      bool auto_sessions) {
  // A KV-deferred request in auto mode is waiting for memory, not stalling.
  // The configured request timeout must not turn that wait into an abort.
  if (auto_sessions && request->kv_deferred_since)
    return false;
  return request->deadline.has_value() &&
         TextGenerationScheduler::Clock::now() >= *request->deadline;
}

/// The `--trace` generation record as known at submission: the effective
/// limits and sampling, and the prompt as the model reads it. Captured on the
/// submitting thread before the scheduler takes the prompt over.
struct GenerationTrace {
  json::Value record;
  std::string prompt;
};

// The shortest decimal that round-trips the float, so a configured 0.7 reads
// 0.7 rather than its widened 0.699999988.
double TraceFloat(float value) {
  std::array<char, 32> buffer{};
  const auto [end, error] =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error != std::errc{}) {
    return value;
  }
  return std::strtod(std::string(buffer.data(), end).c_str(), nullptr);
}

GenerationTrace StartGenerationTrace(const ScheduledRequest& request,
                                     const TextModelRunner& runner) {
  GenerationTrace trace{
      .record = Trace::Record("generation", Trace::CurrentRequest()),
      // Special tokens decode to their spelling, so this is the rendered
      // template for every model without a per-model hook.
      .prompt = Trace::Text(runner.Decode(request.prompt)),
  };
  trace.record["generation"] = request.id;
  trace.record["max_tokens"] = request.token_limit;
  json::Value sampling = json::Value::object();
  sampling["temperature"] = TraceFloat(request.sampling.temperature);
  sampling["top_k"] = request.sampling.top_k;
  sampling["top_p"] = TraceFloat(request.sampling.top_p);
  sampling["min_p"] = TraceFloat(request.sampling.min_p);
  sampling["seed"] = request.sampling.seed;
  sampling["repeat_penalty"] = TraceFloat(request.sampling.repeat_penalty);
  sampling["repeat_last_n"] = request.sampling.repeat_last_n;
  sampling["frequency_penalty"] =
      TraceFloat(request.sampling.frequency_penalty);
  sampling["presence_penalty"] = TraceFloat(request.sampling.presence_penalty);
  sampling["constrained"] = request.sampling.constraint != nullptr;
  trace.record["sampling"] = std::move(sampling);
  return trace;
}

const char* FinishName(const TextGenerationScheduler::Result& result) {
  if (result.cancelled) {
    return "cancelled";
  }
  switch (result.finish_reason) {
    case TextGenerationBackend::FinishReason::kStop:
      return "stop";
    case TextGenerationBackend::FinishReason::kStopSequence:
      return "stop_sequence";
    case TextGenerationBackend::FinishReason::kLength:
      return "length";
    case TextGenerationBackend::FinishReason::kCancelled:
      return "cancelled";
  }
  return "stop";
}

void WriteGenerationTrace(GenerationTrace trace,
                          const TextGenerationScheduler::Result& result,
                          const std::exception_ptr& failure) {
  auto& record = trace.record;
  if (result.component_cache)
    record["component_cache"] = ComponentCacheJson(*result.component_cache);
  record["prompt_tokens"] = result.prompt_tokens;
  record["cache"] = result.cache_disk_hit ? "disk"
                    : result.cache_hit    ? "memory"
                                          : "miss";
  record["cached_tokens"] = result.cached_prompt_tokens;
  if (!result.cache_hit && !result.cache_miss_reason.empty()) {
    record["cache_miss_reason"] = result.cache_miss_reason;
    record["common_prefix_tokens"] = result.cache_common_prefix_tokens;
    record["nearest_checkpoint_tokens"] = result.cache_checkpoint_tokens;
  }
  record["generated_tokens"] = result.tokens.size();
  record["finish"] = FinishName(result);
  if (!result.stop_sequence.empty()) {
    record["stop_sequence"] = Trace::Text(result.stop_sequence);
  }
  if (failure != nullptr) {
    try {
      std::rethrow_exception(failure);
    } catch (const std::exception& error) {
      record["error"] = Trace::Text(error.what());
    } catch (...) {
      record["error"] = "unknown";
    }
  }
  record["prompt"] = std::move(trace.prompt);
  record["output"] = Trace::Text(result.text);
  Trace::Write(record);
}

std::size_t ResidentKiB() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.starts_with("VmRSS:")) {
      std::istringstream value(line.substr(6));
      std::size_t kib = 0;
      if (value >> kib)
        return kib;
    }
  }
  return 0;
}

// The first constrained request would otherwise build the vocabulary inside
// its queue time (about 30 ms for Flash-Next's 248,320 tokens).
void PrepareStructuredOutput(const TextModelRunner& runner) {
  const auto resident = ResidentKiB();
  const auto started = std::chrono::steady_clock::now();
  const auto tokens = runner.PrepareConstraintVocabulary();
  if (tokens == 0)
    return;
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - started;
  std::ostringstream line;
  line << "event=structured_output_prepared vocabulary_tokens=" << tokens
       << " prepare_ms=" << std::fixed << std::setprecision(1)
       << elapsed.count() << " rss_delta_kib="
       << static_cast<long long>(ResidentKiB()) -
              static_cast<long long>(resident)
       << ' ' << Logger::MemoryStatus();
  Logger::Info("loader", line.str());
}

}  // namespace

struct TextGenerationScheduler::Request::Impl {
  explicit Impl(std::shared_ptr<ScheduledRequest> scheduled_request)
      : request(std::move(scheduled_request)) {}

  std::shared_ptr<ScheduledRequest> request;
  bool waited{false};
  /// Present only while `--trace` is armed; touched only by the consumer.
  std::optional<GenerationTrace> trace;
};

struct CompletedStats {
  std::uint64_t id{0};
  std::size_t prompt_tokens{0};
  std::size_t generated_tokens{0};
  double ttft_ms{0.0};
  bool failed{false};
  bool cancelled{false};
};

/// Cumulative counters captured at the start of each statistics interval. The
/// periodic table reports the difference between two snapshots, so every row
/// describes the last ten seconds rather than the lifetime of the server.
struct StatsSnapshot {
  std::chrono::steady_clock::time_point time{};
  std::uint64_t total_prompt_tokens{0};
  std::uint64_t total_generated_tokens{0};
  std::uint64_t admitted{0};
  std::uint64_t completed{0};
  std::uint64_t cancelled{0};
  std::uint64_t failed{0};
  std::uint64_t prompt_tokens{0};
  std::uint64_t generated_tokens{0};
  double prompt_seconds{0.0};
  double gen_seconds{0.0};
  double ttft_ms{0.0};
  std::uint64_t cache_hits{0};
};

struct TextGenerationScheduler::Impl {
  Impl(std::shared_ptr<TextRunnerPool> model_runner_pool,
       TextPrefillPolicy model_prefill_policy,
       TextSchedulerPolicy model_scheduler_policy)
      : runner_pool(std::move(model_runner_pool)),
        prefill_policy(model_prefill_policy),
        scheduler_policy(model_scheduler_policy),
        batch_limit(runner_pool != nullptr
                        ? (model_scheduler_policy.auto_sessions
                               ? std::size_t{1}
                               : runner_pool->capacity())
                        : 0),
        output_budget(std::make_shared<OutputBudget>(
            model_scheduler_policy.max_buffered_output_bytes_total)) {
    if (runner_pool == nullptr) {
      throw std::invalid_argument(
          "text generation scheduler runner pool must not be null");
    }
    sessions = std::make_shared<SessionTable>(runner_pool->capacity());
    if (prefill_policy.decode_active_tokens == 0 ||
        prefill_policy.min_batched_tokens >
            prefill_policy.decode_active_tokens) {
      throw std::invalid_argument(
          "batched prefill budget and its floor must be positive and ordered");
    }
    // A burst as wide as the admitted batch must fit in the pending queue, so
    // raising max_num_seqs never turns into admission refusals.
    max_pending_requests =
        scheduler_policy.max_pending_requests != 0
            ? scheduler_policy.max_pending_requests
            : std::max(kDefaultPendingRequests, runner_pool->capacity());
    max_pending_requests_per_client =
        scheduler_policy.max_pending_requests_per_client != 0
            ? scheduler_policy.max_pending_requests_per_client
            : max_pending_requests;
    if (max_pending_requests_per_client > max_pending_requests ||
        scheduler_policy.max_output_bytes_per_request == 0 ||
        scheduler_policy.max_buffered_output_bytes_per_request == 0 ||
        scheduler_policy.max_buffered_output_bytes_total == 0 ||
        scheduler_policy.request_timeout.count() < 0 ||
        scheduler_policy.stall_timeout.count() < 0) {
      throw std::invalid_argument("invalid text scheduler limits");
    }
    // How many requests may share one prefill pass. The 32-request cap is the
    // measured optimum at 1024 concurrent requests; a pass takes only requests
    // already waiting to prefill, so the actual cohort is
    // min(cap, prefilling queue depth) and shrinks automatically at low load.
    // GUFO_PREFILL_COHORT overrides the cap for experiments.
    if (const char* cohort = std::getenv("GUFO_PREFILL_COHORT")) {
      const auto requested = std::strtoul(cohort, nullptr, 10);
      prefill_cohort_limit =
          std::clamp<std::size_t>(static_cast<std::size_t>(requested), 1, 64);
    }
    incremental_prefill_supported =
        runner_pool->runner().Descriptor().capabilities.incremental_prefill;
    final_token_advance_required =
        runner_pool->runner()
            .Descriptor()
            .capabilities.final_token_advance_required;
    incremental_text_is_exact = runner_pool->runner()
                                    .Descriptor()
                                    .capabilities.incremental_text_is_exact;
    multi_token_decode =
        runner_pool->runner().Descriptor().capabilities.multi_token_decode;
    batched_multi_token_decode = runner_pool->runner()
                                     .Descriptor()
                                     .capabilities.batched_multi_token_decode;
    batched_multi_token_decode_max_width =
        runner_pool->runner()
            .Descriptor()
            .capabilities.batched_multi_token_decode_max_width;
    device_lost_requests.reserve(runner_pool->capacity());
    if (const char* text = std::getenv("GUFO_ASSEMBLE_DECODE_ROWS")) {
      char* end = nullptr;
      const auto parsed = std::strtoul(text, &end, 10);
      if (end != text && *end == '\0' && parsed > 0) {
        scheduler_policy.assemble_decode_rows = std::min<std::size_t>(
            static_cast<std::size_t>(parsed), runner_pool->capacity());
      }
    } else if (scheduler_policy.auto_sessions) {
      // Hybrid plan scaling: fill the decode cohort up to the resolved session
      // capacity, which is 256 at small contexts and the memory-bound count at
      // long contexts.
      scheduler_policy.assemble_decode_rows =
          std::min<std::size_t>(models::qwen38_flash_next::kMaxBatchSessions,
                                runner_pool->capacity());
    }
    PrepareStructuredOutput(runner_pool->runner());
    stats_last.time = std::chrono::steady_clock::now();
    stats_last.total_prompt_tokens = detail::TotalPromptTokens().load();
    stats_last.total_generated_tokens = detail::TotalGenTokens().load();
    stats_last.cache_hits = detail::PrefixCacheHits().load();
    if (scheduler_policy.log_stats) {
      stats_monitor = std::jthread(
          [this](const std::stop_token& stop_token) { RunStats(stop_token); });
    }
    worker = std::jthread(
        [this](const std::stop_token& stop_token) { Run(stop_token); });
    if (scheduler_policy.stall_timeout.count() > 0) {
      stall_monitor = std::jthread([this](const std::stop_token& stop_token) {
        MonitorStalls(stop_token);
      });
    }
  }

  ~Impl() {
    stats_monitor.request_stop();
    if (stats_monitor.joinable()) {
      stats_monitor.join();
    }
    stall_monitor.request_stop();
    if (stall_monitor.joinable()) {
      stall_monitor.join();
    }
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    worker.request_stop();
    queue_condition.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  static double TokensPerSecond(std::size_t tokens, double elapsed_ms) {
    return elapsed_ms > 0.0 ? static_cast<double>(tokens) * 1000.0 / elapsed_ms
                            : 0.0;
  }

  void LogPrefillProgress(const std::shared_ptr<ScheduledRequest>& request,
                          std::size_t chunk_tokens,
                          double chunk_ms) const noexcept {
    if (!scheduler_policy.log_progress || !Logger::Enabled(LogLevel::kInfo)) {
      return;
    }
    try {
      const std::size_t cached = std::min(request->result.cached_prompt_tokens,
                                          request->result.prompt_tokens);
      const std::size_t total = request->result.prompt_tokens - cached;
      const std::size_t processed =
          std::min(request->result.prefill_tokens, total);
      const double percentage =
          total > 0 ? static_cast<double>(processed) * 100.0 / total : 100.0;
      std::ostringstream message;
      message << "request=" << request->id
              << " phase=prefill tokens=" << processed << '/' << total
              << " percentage=" << std::fixed << std::setprecision(1)
              << percentage
              << " chunk_tps=" << TokensPerSecond(chunk_tokens, chunk_ms)
              << " avg_tps="
              << TokensPerSecond(request->result.prefill_tokens,
                                 request->result.prefill_ms);
      Logger::Info("progress", message.str());
    } catch (...) {
    }
  }

  void LogDecodeProgress(const std::shared_ptr<ScheduledRequest>& request,
                         bool final = false) const noexcept {
    if (!scheduler_policy.log_progress || !Logger::Enabled(LogLevel::kInfo)) {
      return;
    }
    const std::size_t generated = request->result.tokens.size();
    if (generated == 0 || generated == request->last_decode_progress_tokens ||
        (!final &&
         generated / kDecodeProgressInterval ==
             request->last_decode_progress_tokens / kDecodeProgressInterval)) {
      return;
    }
    try {
      const std::size_t chunk_tokens =
          generated - request->last_decode_progress_tokens;
      const double chunk_ms =
          request->result.decode_ms - request->last_decode_progress_ms;
      const double percentage =
          request->token_limit > 0
              ? static_cast<double>(generated) * 100.0 / request->token_limit
              : 100.0;
      std::ostringstream message;
      message << "request=" << request->id
              << " phase=decode tokens=" << generated << '/'
              << request->token_limit << " percentage=" << std::fixed
              << std::setprecision(1) << percentage
              << " chunk_tps=" << TokensPerSecond(chunk_tokens, chunk_ms)
              << " avg_tps="
              << TokensPerSecond(generated, request->result.decode_ms);
      if (request->result.draft_tokens > 0) {
        message << " draft_accepted=" << request->result.draft_accepted_tokens
                << " draft_proposed=" << request->result.draft_tokens
                << " acceptance_percentage="
                << static_cast<double>(request->result.draft_accepted_tokens) *
                       100.0 / request->result.draft_tokens;
      }
      Logger::Info("progress", message.str());
      request->last_decode_progress_tokens = generated;
      request->last_decode_progress_ms = request->result.decode_ms;
    } catch (...) {
    }
  }

  /// Named for the `--sessions` pool: the batch bound is a runtime value.
  [[nodiscard]] std::size_t EffectiveCapacity() const noexcept {
    return batch_limit.load(std::memory_order_acquire);
  }

  /// Conservative output budget for an unlimited request. A request that
  /// names max_tokens reserves exactly that amount instead.
  static constexpr std::size_t kAutoOutputReserveTokens = 512;
  static constexpr auto kKvAdmissionRetry = std::chrono::milliseconds{20};

  [[nodiscard]] static std::size_t OutputReserveTokens(
      const ScheduledRequest& request) noexcept {
    if (request.output_limit_explicit)
      return request.token_limit;
    return std::min<std::size_t>(request.token_limit, kAutoOutputReserveTokens);
  }

  [[nodiscard]] static std::size_t AdmissionEstimateTokens(
      const ScheduledRequest& request) noexcept {
    return request.prompt.size() + OutputReserveTokens(request);
  }

  [[nodiscard]] std::optional<SharedKvPoolTokens> SharedKvPool()
      const noexcept {
    try {
      const auto claim = runner_pool->runner().ResourceClaim();
      if (!claim.shared_kv_capacity_tokens.has_value() ||
          !claim.shared_kv_free_tokens.has_value())
        return std::nullopt;
      return SharedKvPoolTokens{
          .capacity = *claim.shared_kv_capacity_tokens,
          .free = *claim.shared_kv_free_tokens,
          .block = claim.shared_kv_block_tokens.value_or(1),
      };
    } catch (...) {
      return std::nullopt;
    }
  }

  /// KV tokens the resident batch has already claimed but not mapped yet.
  /// Admission subtracts this from the pool's free count so a burst cannot
  /// reserve the same free blocks several times before the first prefill maps
  /// them.
  [[nodiscard]] std::size_t ResidentKvReservation(
      const std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      const std::deque<std::shared_ptr<ScheduledRequest>>& capturing,
      std::size_t block_tokens) const noexcept {
    const std::array<const std::deque<std::shared_ptr<ScheduledRequest>>*, 3>
        lists{&prefilling, &decoding, &capturing};
    std::size_t total = 0;
    for (const auto* list : lists) {
      for (const auto& request : *list) {
        if (!request->runner_request)
          continue;
        const std::size_t mapped = request->runner_request.prefill_position() +
                                   request->result.tokens.size();
        const std::size_t needed = AdmissionEstimateTokens(*request);
        if (needed <= mapped)
          continue;
        std::size_t pending = needed - mapped;
        if (block_tokens > 1) {
          const auto remainder = pending % block_tokens;
          if (remainder != 0)
            pending += block_tokens - remainder;
        }
        if (pending > std::numeric_limits<std::size_t>::max() - total)
          return std::numeric_limits<std::size_t>::max();
        total += pending;
      }
    }
    return total;
  }

  [[nodiscard]] bool CanAdmitKv(
      const ScheduledRequest& request,
      const std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      const std::deque<std::shared_ptr<ScheduledRequest>>& capturing) const {
    const auto pool = SharedKvPool();
    if (!pool)
      return true;
    // The oldest request must always be able to start when nothing else is
    // resident: otherwise a prompt whose budget slightly exceeds the free
    // blocks would wait forever. Its prefill adapts to the free capacity and
    // decode defers when the pool fills, so it still makes progress.
    if (prefilling.empty() && decoding.empty() && capturing.empty())
      return true;
    const auto estimate = AdmissionEstimateTokens(request);
    const auto block = std::max<std::size_t>(1, pool->block);
    const auto pending =
        ResidentKvReservation(prefilling, decoding, capturing, block);
    std::size_t needed = estimate;
    if (const auto remainder = needed % block; remainder != 0)
      needed += block - remainder;
    return pool->free >= pending && pool->free - pending >= needed;
  }

  /// Leaves the request queued and wakes admission after a short retry rather
  /// than failing it. The completion path also notifies the queue condition,
  /// so released blocks usually resume it sooner.
  void DeferAdmission(
      const std::shared_ptr<ScheduledRequest>& request,
      const std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      const std::deque<std::shared_ptr<ScheduledRequest>>& capturing) {
    const auto now = Clock::now();
    const auto retry = now + kKvAdmissionRetry;
    const bool first = !kv_admission_retry_at;
    kv_admission_retry_at = retry;
    if (!first || !Logger::Enabled(LogLevel::kInfo))
      return;
    std::size_t free = 0;
    std::size_t block = 1;
    if (const auto pool = SharedKvPool()) {
      free = pool->free;
      block = pool->block;
    } else {
      const std::array<const std::deque<std::shared_ptr<ScheduledRequest>>*, 3>
          residents{&prefilling, &decoding, &capturing};
      if (const auto tokens =
              FreeKvTokens(ResidentTokens(residents), decoding.size())) {
        free = *tokens;
      } else {
        block = 0;
      }
    }
    Logger::Info("scheduler",
                 "event=kv_admission_deferred request=" +
                     std::to_string(request->id) + " estimate_tokens=" +
                     std::to_string(AdmissionEstimateTokens(*request)) +
                     " free_tokens=" + std::to_string(free) +
                     " block_tokens=" + std::to_string(block) +
                     " retry_ms=" + std::to_string(kKvAdmissionRetry.count()));
  }

  /// Auto mode: one resident initially, then one more per scheduler turn while
  /// the queue has work and the KV estimate has room. With no resident or
  /// waiting request it returns to one without preempting live work.
  void AutoScaleBatch(
      const std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      const std::deque<std::shared_ptr<ScheduledRequest>>& capturing,
      const std::deque<std::shared_ptr<ScheduledRequest>>& waiting) {
    if (!scheduler_policy.auto_sessions)
      return;
    const std::size_t capacity = runner_pool->capacity();
    const std::size_t current = EffectiveCapacity();
    const std::size_t resident =
        prefilling.size() + decoding.size() + capturing.size();
    std::size_t queued = 0;
    std::size_t estimate = 0;
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      queued = queued_count;
      if (!queued_clients.empty() && !queued_clients.front().requests.empty())
        estimate =
            AdmissionEstimateTokens(*queued_clients.front().requests.front());
    }
    if (queued == 0 && waiting.empty() && kv_deferred_requests.empty()) {
      if (resident == 0 && current != 1) {
        SetBatchLimit(1);
        if (Logger::Enabled(LogLevel::kDebug)) {
          Logger::Debug("scheduler",
                        "event=auto_batch_limit value=1 reason=idle");
        }
      }
      return;
    }
    if (current >= capacity || LowMemory())
      return;
    const std::array<const std::deque<std::shared_ptr<ScheduledRequest>>*, 3>
        resident_lists{&prefilling, &decoding, &capturing};
    if (estimate != 0) {
      if (const auto pool = SharedKvPool()) {
        const auto block = std::max<std::size_t>(1, pool->block);
        auto needed = estimate;
        if (const auto remainder = needed % block; remainder != 0)
          needed += block - remainder;
        const auto pending =
            ResidentKvReservation(prefilling, decoding, capturing, block);
        if (pool->free < pending || pool->free - pending < needed)
          return;
      } else if (const auto free = FreeKvTokens(ResidentTokens(resident_lists),
                                                decoding.size());
                 free.has_value() && *free < estimate) {
        return;
      }
    }
    // Grow toward the visible demand in one step. The per-request KV
    // reservation still stops admission long before the pool is oversubscribed,
    // so this only removes the old one-session-per-turn ramp on a large burst.
    const std::size_t demand =
        resident + queued + waiting.size() + kv_deferred_requests.size();
    const std::size_t next = std::min(capacity, std::max(current + 1, demand));
    SetBatchLimit(next);
    if (Logger::Enabled(LogLevel::kDebug)) {
      Logger::Debug("scheduler",
                    "event=auto_batch_limit value=" + std::to_string(next) +
                        " queued=" + std::to_string(queued) +
                        " estimate_tokens=" + std::to_string(estimate));
    }
  }

  /// KV the runner claims for the whole session pool, in tokens.
  ///
  /// Derived from the runner's own byte claim instead of a constant, so a model
  /// that shares or quantises KV reports the capacity its attention kernel
  /// really has. Nullopt when the claim cannot be measured.
  [[nodiscard]] std::optional<std::size_t> KvCapacityTokens() const noexcept {
    if (const auto pool = SharedKvPool())
      return pool->capacity;
    try {
      const auto claim = runner_pool->runner().ResourceClaim();
      const auto context = runner_pool->runner().Descriptor().max_context;
      if (!claim.state_capacity_bytes.has_value() ||
          !claim.per_request_state_bytes.has_value() || context == 0 ||
          *claim.per_request_state_bytes == 0)
        return std::nullopt;
      const std::size_t bytes = *claim.state_capacity_bytes;
      // Tokens the claim pays for: bytes * (tokens per session byte). Rounding
      // down keeps the bound conservative, and the overflow guard keeps the
      // product inside size_t for very large unified-memory claims.
      if (bytes > std::numeric_limits<std::size_t>::max() / context)
        return bytes / *claim.per_request_state_bytes * context;
      return (bytes * context) / *claim.per_request_state_bytes;
    } catch (...) {
      return std::nullopt;
    }
  }

  /// Unified memory still free for KV growth, after the configured reserve.
  [[nodiscard]] std::optional<std::size_t> MemoryHeadroomBytes()
      const noexcept {
    if (!scheduler_policy.available_memory)
      return std::nullopt;
    std::optional<std::size_t> available;
    try {
      available = scheduler_policy.available_memory();
    } catch (...) {
      return std::nullopt;
    }
    if (!available.has_value())
      return std::nullopt;
    return *available > scheduler_policy.memory_reserve_bytes
               ? *available - scheduler_policy.memory_reserve_bytes
               : 0;
  }

  /// KV tokens the free unified memory still pays for.
  ///
  /// Unified memory means a KV byte and a weight byte compete, so the pool's
  /// own claim is only half of the available KV: the host reading bounds it
  /// too. Nullopt when either side cannot be measured.
  [[nodiscard]] std::optional<std::size_t> MemoryTokenHeadroom()
      const noexcept {
    // A paged pool already owns every KV byte it can map. Its free-block count
    // is the bound; deriving a second token budget from the session's base
    // state would be meaningless.
    if (SharedKvPool().has_value())
      return std::nullopt;
    const auto headroom = MemoryHeadroomBytes();
    if (!headroom.has_value())
      return std::nullopt;
    try {
      const auto claim = runner_pool->runner().ResourceClaim();
      const auto context = runner_pool->runner().Descriptor().max_context;
      if (!claim.per_request_state_bytes.has_value() || context == 0)
        return std::nullopt;
      const std::size_t bytes_per_token =
          std::max<std::size_t>(1, *claim.per_request_state_bytes / context);
      return *headroom / bytes_per_token;
    } catch (...) {
      return std::nullopt;
    }
  }

  /// KV pressure from free unified memory: below the reserve, the batch cannot
  /// grow without competing with the rest of the system.
  [[nodiscard]] bool LowMemory() const noexcept {
    const auto headroom = MemoryHeadroomBytes();
    return headroom.has_value() && *headroom == 0;
  }

  /// Largest prompt chunk one iteration may prefetch.
  ///
  /// `max_num_batched_tokens` caps the chunk, and free KV capacity caps it
  /// again: the tokens this request would add land in the same unified pool as
  /// every resident session's decode state. Pending decoders reserve one token
  /// each, so a nearly full pool shrinks the chunk instead of starving the
  /// batch; the floor keeps the adaptation from collapsing to noise.
  /// KV tokens the resident batch leaves free for one more prefill chunk.
  /// Nullopt when the runner's capacity cannot be derived.
  [[nodiscard]] std::optional<std::size_t> FreeKvTokens(
      std::size_t resident_tokens, std::size_t pending_decoders) const {
    if (const auto pool = SharedKvPool())
      return pool->free;
    std::optional<std::size_t> free;
    if (const auto capacity = KvCapacityTokens()) {
      const std::size_t reserved = resident_tokens + pending_decoders;
      free = reserved >= *capacity ? 0 : *capacity - reserved;
    }
    if (const auto memory = MemoryTokenHeadroom()) {
      free = free.has_value() ? std::min(*free, *memory) : *memory;
    }
    return free;
  }

  [[nodiscard]] std::size_t PrefillTokenBudget(
      std::size_t resident_tokens, std::size_t pending_decoders) const {
    const std::size_t configured = prefill_policy.decode_active_tokens;
    if (!prefill_policy.adapt_to_kv_capacity)
      return configured;
    const auto free = FreeKvTokens(resident_tokens, pending_decoders);
    if (!free.has_value())
      return configured;
    const std::size_t floor =
        prefill_policy.min_batched_tokens != 0
            ? std::min(configured, prefill_policy.min_batched_tokens)
            : std::min(configured, kMinimumAdaptedPrefillTokens);
    return std::clamp(std::min(configured, *free), floor, configured);
  }

  /// Prompt tokens every resident session already holds in model state.
  [[nodiscard]] static std::size_t ResidentTokens(
      std::span<const std::deque<std::shared_ptr<ScheduledRequest>>* const>
          lists) {
    std::size_t total = 0;
    for (const auto* list : lists) {
      for (const auto& request : *list) {
        if (!request->runner_request)
          continue;
        total += request->runner_request.prefill_position() +
                 request->result.tokens.size();
      }
    }
    return total;
  }

  /// Minimum chunk the adaptation may still grant for this prompt.
  [[nodiscard]] std::size_t PrefillFloor(std::size_t remaining_prompt) const {
    const std::size_t configured = prefill_policy.decode_active_tokens;
    const std::size_t floor =
        prefill_policy.min_batched_tokens != 0
            ? std::min(configured, prefill_policy.min_batched_tokens)
            : std::min(configured, kMinimumAdaptedPrefillTokens);
    return std::min(floor, std::max<std::size_t>(1, remaining_prompt));
  }

  /// Reclaims KV from the newest decoder when the prefilling request cannot
  /// even get its floor.
  ///
  /// This is the early memory-pressure signal: free KV capacity, not the
  /// configured batched-token budget, is what limits the batch. The victim is
  /// preempted with RECOMPUTE — its state is released and replayed later, never
  /// swapped to host memory. Returns true when the caller should re-decide.
  [[nodiscard]] bool RelieveKvPressure(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      std::deque<std::shared_ptr<ScheduledRequest>>& preempted,
      std::size_t resident_tokens) {
    if (!prefill_policy.adapt_to_kv_capacity || prefilling.empty() ||
        decoding.empty())
      return false;
    const auto& pending = prefilling.front();
    if (!pending->runner_request || pending->id == kv_relieved_request)
      return false;
    const std::size_t remaining =
        pending->runner_request.prompt_tokens() -
        std::min(pending->runner_request.prefill_position(),
                 pending->runner_request.prompt_tokens());
    const std::size_t floor = PrefillFloor(remaining);
    const auto free = FreeKvTokens(resident_tokens, decoding.size());
    if (!free.has_value() || *free >= floor)
      return false;
    // One episode may reclaim at most all but one decoder: the batch keeps a
    // decode slot, and a request that still cannot get its floor after that
    // proceeds with the floor chunk instead of preempting in a cycle.
    if (kv_reliefs_in_episode + 1 >= decoding.size() + 1)
      return false;
    auto victim = std::move(decoding.back());
    decoding.pop_back();
    const std::size_t released =
        victim->runner_request ? victim->runner_request.prefill_position() +
                                     victim->result.tokens.size()
                               : 0;
    kv_reliefs_in_episode += 1;
    // Remember the request that caused this relief, so it is not preempted
    // again until it has consumed the KV it asked for. When one victim is not
    // enough, the next iteration reclaims another decoder rather than
    // thrashing the same one.
    kv_relieved_request = pending->id;
    if (Logger::Enabled(LogLevel::kInfo)) {
      const auto headroom = MemoryHeadroomBytes();
      Logger::Info("scheduler",
                   "event=kv_pressure reason=" +
                       std::string(headroom.has_value() && *headroom == 0
                                       ? "host_memory"
                                       : "kv_capacity") +
                       " request=" + std::to_string(pending->id) +
                       " free_tokens=" + std::to_string(*free) +
                       " floor_tokens=" + std::to_string(floor) +
                       " released_tokens=" + std::to_string(released) +
                       (headroom.has_value() ? " memory_headroom_bytes=" +
                                                   std::to_string(*headroom)
                                             : std::string{}) +
                       " strategy=recompute");
    }
    try {
      if (CompleteIfStopped(victim)) {
        return true;
      }
      if (PreemptForRecompute(victim)) {
        preempted.push_back(std::move(victim));
      } else if (!IsTerminal(victim)) {
        CompleteFailure(victim, std::make_exception_ptr(std::runtime_error(
                                    "cannot preempt a request without state")));
      }
    } catch (...) {
      CompleteFailure(victim, std::current_exception());
    }
    return true;
  }

  void SetBatchLimit(std::size_t value) {
    if (value == 0 || value > runner_pool->capacity()) {
      throw std::invalid_argument(
          "max_num_seqs must be between one and the preallocated sessions");
    }
    const std::size_t previous = batch_limit.exchange(value);
    if (previous == value)
      return;
    if (Logger::Enabled(LogLevel::kDebug)) {
      Logger::Debug("scheduler",
                    "event=batch_limit value=" + std::to_string(value) +
                        " previous=" + std::to_string(previous) +
                        " strategy=recompute");
    }
    queue_condition.notify_all();
  }

  /// Drops one resident request's processing slot and `/slots` entry so its
  /// state can be reclaimed or rebuilt by a later admission.
  void ReleaseProcessingSlot(const std::shared_ptr<ScheduledRequest>& request) {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (std::exchange(request->counted_processing, false)) {
      if (request->session != kNoSession) {
        const std::lock_guard<std::mutex> sessions_lock(
            request->sessions->mutex);
        request->sessions->requests[std::exchange(request->session, kNoSession)]
            .reset();
      }
      detail::RequestsProcessing().fetch_sub(1, std::memory_order_relaxed);
    }
    request->phase.store(TextRequestPhase::kQueued, std::memory_order_release);
  }

  /// Releases a resident request's model state and schedules its replay.
  ///
  /// The preemption policy is RECOMPUTE: no suspended state is kept, so the
  /// victim's prompt grows by the tokens it already generated and re-admission
  /// prefills them again. Its sampler travels with the tokens, so the replayed
  /// trajectory resumes the same draw instead of sampling afresh. Returns false
  /// when the request holds nothing that could be replayed.
  [[nodiscard]] bool PreemptForRecompute(
      const std::shared_ptr<ScheduledRequest>& request) {
    if (!request->runner_request || request->runner_request.SnapshotPending())
      return false;
    const auto prompt = request->runner_request.prompt();
    std::vector<TextRunnerToken> replay;
    replay.reserve(prompt.size() + request->result.tokens.size());
    replay.assign(prompt.begin(), prompt.end());
    replay.insert(replay.end(), request->result.tokens.begin(),
                  request->result.tokens.end());
    request->resume_sampler = request->runner_request.Sampler();
    const std::size_t replayed = replay.size();

    request->runner_request.Invalidate();
    request->runner_request = TextRunnerPool::Request{};
    request->prompt = std::move(replay);
    request->preview_token.reset();
    request->preview_stops = false;
    request->advance_pending = false;
    request->decode_due = true;
    request->prefix_leader.reset();
    // A replay owns no lease to share, so it must not park behind a peer.
    request->prefix_considered = true;
    request->preemptions += 1;
    request->result.preemptions = request->preemptions;
    request->result.preempted_tokens += replayed;
    ReleaseProcessingSlot(request);
    detail::TotalPreemptions().fetch_add(1, std::memory_order_relaxed);
    detail::TotalRecomputeTokens().fetch_add(replayed,
                                             std::memory_order_relaxed);
    if (Logger::Enabled(LogLevel::kDebug)) {
      Logger::Debug(
          "scheduler",
          "event=preemption request=" + std::to_string(request->id) +
              " strategy=recompute replay_tokens=" + std::to_string(replayed) +
              " generated=" + std::to_string(request->result.tokens.size()) +
              " preemptions=" + std::to_string(request->preemptions));
    }
    return true;
  }

  /// How long a degraded request waits for residents to free memory. A burst
  /// larger than the pool converges within a few resident turns; the bound only
  /// stops a request whose peers never release anything. The same bound covers
  /// admission refusals, whose first attempt precedes any retained snapshot.
  static constexpr auto kKvDegradationPatience = std::chrono::seconds(30);
  static constexpr std::size_t kMaxKvDeferrals = 4096;

  /// Pool occupancy as log fields. A degradation is only evidence that the pool
  /// really ran out when the numbers show it: a full pool at the refusal names
  /// capacity, live and free blocks, so "the request degraded" and "the pool
  /// was too small" stay distinguishable in the server log.
  static std::string KvPoolFields(const TextRunnerPool::Request& request) {
    const auto metrics = request.component_cache_metrics();
    if (!metrics || !metrics->kv_pool)
      return " pool=unavailable";
    const auto& pool = *metrics->kv_pool;
    return " pool_capacity=" + std::to_string(pool.capacity_blocks) +
           " pool_live=" + std::to_string(pool.live_blocks) +
           " pool_free=" + std::to_string(pool.free_blocks) +
           " pool_block_bytes=" + std::to_string(pool.block_bytes);
  }

  /// Degrades a request whose step failed because the device KV pool refused
  /// blocks: its retained host snapshot keeps the completed work, the device
  /// state is released, and the request waits for residents to free memory.
  /// Returns false when degrading cannot help — no other resident holds
  /// blocks, a capture is still running, or this request already degraded too
  /// often — so the caller fails it normally instead.
  bool DegradeKvExhausted(const std::shared_ptr<ScheduledRequest>& request) {
    const auto refuse = [&](std::string_view reason) {
      if (Logger::Enabled(LogLevel::kInfo)) {
        Logger::Info("scheduler",
                     "event=kv_degradation_refused request=" +
                         std::to_string(request->id) +
                         " reason=" + std::string(reason) +
                         " deferrals=" + std::to_string(request->kv_deferrals) +
                         " residents=" + std::to_string(resident_count_hint) +
                         KvPoolFields(request->runner_request));
      }
      return false;
    };
    if (!request->runner_request)
      return refuse("no-request");
    if (!scheduler_policy.auto_sessions &&
        request->kv_deferrals >= kMaxKvDeferrals)
      return refuse("deferral-limit");
    // Auto mode must never turn a capture in flight into a request failure:
    // Cancel() joins the existing capture and releases the device state.
    if (!scheduler_policy.auto_sessions &&
        request->runner_request.SnapshotPending())
      return refuse("snapshot-pending");
    const auto now = Clock::now();
    if (!scheduler_policy.auto_sessions && request->kv_deferred_since &&
        now - *request->kv_deferred_since > kKvDegradationPatience)
      return refuse("patience");
    const std::size_t position = request->runner_request.prefill_position();
    const std::size_t generated = request->result.tokens.size();
    const auto pool_fields = KvPoolFields(request->runner_request);
    // Admission moved the prompt into the lease; rebuild it from the lease so
    // re-admission presents the same identity. A decode-phase failure carries
    // generated tokens: replay them with the prompt so the retained snapshot
    // covers the whole trajectory. A prefill failure resumes at the captured
    // frontier instead.
    const auto prompt = request->runner_request.prompt();
    if (prompt.empty())
      return refuse("empty-prompt");
    std::vector<TextRunnerToken> replay;
    replay.reserve(prompt.size() + generated);
    replay.assign(prompt.begin(), prompt.end());
    if (generated != 0) {
      replay.insert(replay.end(), request->result.tokens.begin(),
                    request->result.tokens.end());
      request->resume_sampler = request->runner_request.Sampler();
    }
    request->prompt = std::move(replay);
    const auto retained = request->runner_request.Cancel();
    request->runner_request = TextRunnerPool::Request{};
    request->kv_deferrals += 1;
    if (!request->kv_deferred_since)
      request->kv_deferred_since = now;
    request->result.kv_deferrals = request->kv_deferrals;
    request->advance_pending = false;
    request->decode_due = true;
    request->preview_token.reset();
    request->preview_stops = false;
    request->prefix_leader.reset();
    request->prefix_considered = true;
    ReleaseProcessingSlot(request);
    request->kv_retry_at = Clock::now() + std::chrono::milliseconds(20);
    kv_deferred_requests.push_back(request);
    detail::TotalKvDeferrals().fetch_add(1, std::memory_order_relaxed);
    if (Logger::Enabled(LogLevel::kInfo)) {
      Logger::Info("scheduler",
                   "event=kv_exhaustion_degraded request=" +
                       std::to_string(request->id) +
                       " prefill_position=" + std::to_string(position) +
                       " generated=" + std::to_string(generated) +
                       " retained_snapshot_bytes=" +
                       std::to_string(retained.snapshot_bytes) + " deferrals=" +
                       std::to_string(request->kv_deferrals) + pool_fields);
    }
    return true;
  }

  /// Applies a lowered `max_num_seqs`. Newest residents release their state
  /// first: they hold the least invested work, and their replay is shortest.
  /// A request whose snapshot copy is still running keeps its lease until the
  /// copy completes, so it moves to `capturing` and is handled next iteration.
  void EnforceBatchLimit(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      std::deque<std::shared_ptr<ScheduledRequest>>& capturing,
      std::deque<std::shared_ptr<ScheduledRequest>>& preempted) {
    const std::size_t limit = EffectiveCapacity();
    while (prefilling.size() + decoding.size() + capturing.size() > limit) {
      std::shared_ptr<ScheduledRequest> victim;
      if (!decoding.empty()) {
        victim = std::move(decoding.back());
        decoding.pop_back();
      } else if (!prefilling.empty()) {
        victim = std::move(prefilling.back());
        prefilling.pop_back();
      } else {
        // Only in-flight snapshot copies remain. They cannot release a state
        // they are reading; the next iteration re-checks the bound.
        return;
      }
      try {
        if (CompleteIfStopped(victim))
          continue;
        if (!PreemptForRecompute(victim)) {
          CompleteFailure(victim,
                          std::make_exception_ptr(std::runtime_error(
                              "cannot preempt a request without state")));
          continue;
        }
        preempted.push_back(std::move(victim));
      } catch (...) {
        CompleteFailure(victim, std::current_exception());
      }
    }
  }

  /// Re-admits a preempted request before any queued arrival, reserving its
  /// `/slots` session exactly as `PopQueued` does.
  [[nodiscard]] std::shared_ptr<ScheduledRequest> PopPreempted(
      std::deque<std::shared_ptr<ScheduledRequest>>& preempted) {
    if (preempted.empty())
      return {};
    auto request = std::move(preempted.front());
    preempted.pop_front();
    {
      const std::lock_guard<std::mutex> sessions_lock(sessions->mutex);
      const auto free = std::find(sessions->requests.begin(),
                                  sessions->requests.end(), nullptr);
      if (free != sessions->requests.end()) {
        request->session =
            static_cast<std::size_t>(free - sessions->requests.begin());
        *free = request;
      }
    }
    request->counted_processing = true;
    detail::RequestsProcessing().fetch_add(1, std::memory_order_relaxed);
    return request;
  }

  /// Keeps the process-wide deferred gauge in step. Requires queue_mutex.
  void SetQueuedCountLocked(std::size_t count) {
    detail::RequestsDeferred().fetch_add(
        static_cast<std::int64_t>(count) -
            static_cast<std::int64_t>(queued_count),
        std::memory_order_relaxed);
    queued_count = count;
  }

  [[nodiscard]] std::shared_ptr<ScheduledRequest> PeekQueued() const {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    if (queued_clients.empty() || queued_clients.front().requests.empty())
      return {};
    return queued_clients.front().requests.front();
  }

  [[nodiscard]] std::shared_ptr<ScheduledRequest> PopQueued() {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    if (queued_clients.empty()) {
      return {};
    }
    auto client = std::move(queued_clients.front());
    queued_clients.pop_front();
    auto request = std::move(client.requests.front());
    client.requests.pop_front();
    // Admission reserves a session before potentially slow cache preparation.
    // Publish the transfer before removing it from the deferred count. Admit
    // holds fewer than capacity requests here, so a session is always free.
    {
      const std::lock_guard<std::mutex> sessions_lock(sessions->mutex);
      const auto free = std::find(sessions->requests.begin(),
                                  sessions->requests.end(), nullptr);
      if (free != sessions->requests.end()) {
        request->session =
            static_cast<std::size_t>(free - sessions->requests.begin());
        *free = request;
      }
    }
    request->counted_processing = true;
    detail::RequestsProcessing().fetch_add(1, std::memory_order_relaxed);
    SetQueuedCountLocked(queued_count - 1);
    if (!client.requests.empty()) {
      queued_clients.push_back(std::move(client));
    }
    return request;
  }

  [[nodiscard]] bool RemoveQueued(
      const std::shared_ptr<ScheduledRequest>& request) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto client = queued_clients.begin(); client != queued_clients.end();
         ++client) {
      const auto queued =
          std::find(client->requests.begin(), client->requests.end(), request);
      if (queued == client->requests.end()) {
        continue;
      }
      client->requests.erase(queued);
      SetQueuedCountLocked(queued_count - 1);
      if (client->requests.empty()) {
        queued_clients.erase(client);
      }
      return true;
    }
    return false;
  }

  [[nodiscard]] std::vector<TextGenerationBackend::SessionState>
  SessionSnapshot() const {
    std::vector<TextGenerationBackend::SessionState> states(
        runner_pool->capacity());
    const std::lock_guard<std::mutex> lock(sessions->mutex);
    for (std::size_t index = 0; index < states.size(); ++index) {
      auto& state = states[index];
      state.speculative = multi_token_decode;
      const auto& request = sessions->requests[index];
      if (request == nullptr) {
        continue;
      }
      // Identity, prompt size and limit are fixed before admission.
      const std::size_t generated =
          request->live_generated_tokens.load(std::memory_order_relaxed);
      state.processing = true;
      state.request_id = request->id;
      state.prompt_tokens = request->result.prompt_tokens;
      state.cached_prompt_tokens =
          request->live_cached_prompt_tokens.load(std::memory_order_relaxed);
      state.processed_prompt_tokens =
          request->live_prefill_tokens.load(std::memory_order_relaxed);
      state.generated_tokens = generated;
      state.remaining_tokens =
          request->token_limit - std::min(generated, request->token_limit);
    }
    return states;
  }

  [[nodiscard]] std::vector<std::shared_ptr<ScheduledRequest>> QueuedSnapshot()
      const {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    std::vector<std::shared_ptr<ScheduledRequest>> snapshot;
    snapshot.reserve(queued_count);
    for (const auto& client : queued_clients) {
      snapshot.insert(snapshot.end(), client.requests.begin(),
                      client.requests.end());
    }
    return snapshot;
  }

  /// Cancels and reaps coalesced followers. A follower owns no model session,
  /// so this is the only place its own deadline/cancellation is observed while
  /// the leader is still prefilling or between decode steps.
  void ProcessCoalescedFollowers(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      std::deque<std::shared_ptr<ScheduledRequest>>& capturing) {
    for (auto* list : {&prefilling, &decoding, &capturing}) {
      for (auto& leader : *list) {
        auto& followers = leader->coalesce_followers;
        for (auto it = followers.begin(); it != followers.end();) {
          if (!*it || IsTerminal(*it)) {
            it = followers.erase(it);
            continue;
          }
          if (CompleteIfStopped(*it)) {
            it = followers.erase(it);
            continue;
          }
          ++it;
        }
        if (IsTerminal(leader) && !followers.empty()) {
          for (auto& follower : followers)
            if (follower && !IsTerminal(follower))
              CompleteCancelled(follower);
          followers.clear();
        }
      }
    }
  }

  void FinalizeResult(const std::shared_ptr<ScheduledRequest>& request,
                      TextGenerationBackend::FinishReason finish_reason) {
    request->result.completion_tokens = request->result.tokens.size();
    request->result.finish_reason = finish_reason;
    if (request->inter_token_samples > 0) {
      request->result.mean_inter_token_ms =
          request->inter_token_total.count() /
          static_cast<double>(request->inter_token_samples);
    }
    if (!incremental_text_is_exact) {
      request->result.text =
          runner_pool->runner().Decode(request->result.tokens);
    }
  }

  void CompleteCancelled(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    EndPublicationWait(request, "cancelled");
    if (!request->coalesce_followers.empty()) {
      auto followers = std::move(request->coalesce_followers);
      request->coalesce_followers.clear();
      for (auto& follower : followers)
        if (follower && !IsTerminal(follower))
          CompleteCancelled(follower);
    }
    if (device_lost.load(std::memory_order_acquire)) {
      CompleteFailure(request, device_lost_failure);
      return;
    }
    try {
      if (request->runner_request) {
        const auto retained = request->runner_request.Cancel();
        request->result.component_cache = retained.component_cache;
        request->result.cache_shared_bytes =
            retained.component_cache ? retained.component_cache->shared_bytes
                                     : 0;
        request->result.cache_snapshot_bytes = retained.snapshot_bytes;
        request->result.cache_snapshot_ms = retained.snapshot_ms;
        request->result.cache_disk_queued_bytes = retained.disk_queued_bytes;
        request->result.cache_disk_enqueue_ms = retained.disk_enqueue_ms;
      }
      request->result.cancelled = true;
      FinalizeResult(request, TextGenerationBackend::FinishReason::kCancelled);
      LogDecodeProgress(request, true);
      RecordRequestMetrics(request->result);
      RecordCompletedStats(*request, false, true);
      PublishTerminal(request, {}, true);
    } catch (...) {
      RecordFailedRequest();
      PublishTerminal(request, std::current_exception(), true);
    }
  }

  void CompleteFailure(const std::shared_ptr<ScheduledRequest>& request,
                       std::exception_ptr failure,
                       bool classified = false) noexcept {
    EndPublicationWait(request, "failed");
    if (IsTerminal(request))
      return;
    if (!classified)
      failure = ClassifyFailure(std::move(failure));
    if (!request->coalesce_followers.empty()) {
      auto followers = std::move(request->coalesce_followers);
      request->coalesce_followers.clear();
      for (auto& follower : followers)
        if (follower && !IsTerminal(follower))
          CompleteFailure(follower, failure, true);
    }
    if (request->runner_request) {
      if (device_lost.load(std::memory_order_acquire)) {
        // Retain the lease, snapshots and their workers until process exit.
        // Resetting state, freeing snapshots or joining transfers on the dead
        // context can block or crash before the supervisor can restart us.
        device_lost_requests.push_back(request);
      } else {
        request->runner_request.Invalidate();
      }
    }
    request->result.completion_tokens = request->result.tokens.size();
    LogDecodeProgress(request, true);
    RecordFailedRequest();
    RecordCompletedStats(*request, true, false);
    PublishTerminal(request, std::move(failure), true);
  }

  void MarkDeviceLost(std::string_view reason) noexcept {
    if (device_lost.exchange(true, std::memory_order_acq_rel))
      return;
    detail::DeviceLostTotal().fetch_add(1, std::memory_order_relaxed);
    try {
      Logger::Error("scheduler", "event=device_lost remedy=restart reason=" +
                                     std::string(reason));
    } catch (...) {
      // Loss remains observable even if logging fails.
    }
  }

  // A device call that never returns blocks its thread, so nothing on that
  // thread can report it. This thread only reads atomics; it never touches
  // the device or the scheduler's queues.
  void MonitorStalls(const std::stop_token& stop_token) noexcept {
    const auto timeout = scheduler_policy.stall_timeout;
    const auto interval = std::clamp<std::chrono::milliseconds>(
        timeout / 4, std::chrono::milliseconds(1), std::chrono::minutes(1));
    std::mutex mutex;
    std::condition_variable_any condition;
    std::unique_lock<std::mutex> lock(mutex);
    std::uint64_t seen = progress.load(std::memory_order_relaxed);
    // Stalled time is counted in wake-ups, not read from the clock, so a
    // process paused as a whole (SIGSTOP, a cgroup freeze) is credited one
    // interval for the pause however long it lasted.
    std::chrono::milliseconds stalled{0};
    while (!stop_token.stop_requested()) {
      condition.wait_for(lock, stop_token, interval, [] { return false; });
      if (stop_token.stop_requested() ||
          device_lost.load(std::memory_order_acquire))
        return;
      const std::uint64_t now = progress.load(std::memory_order_relaxed);
      if (now != seen || !loop_working.load(std::memory_order_relaxed)) {
        seen = now;
        stalled = std::chrono::milliseconds(0);
        continue;
      }
      stalled += interval;
      if (stalled < timeout)
        continue;
      try {
        MarkDeviceLost("scheduler made no progress for " +
                       std::to_string(timeout.count()) + " ms");
      } catch (...) {
        MarkDeviceLost("scheduler made no progress");
      }
      return;
    }
  }

  // A GPU reset leaves this process's device context permanently unusable,
  // and every later work unit then fails with a raw driver message. Probe the
  // device only after a model failure, on this thread between work units, so
  // the probe never overlaps this scheduler's own GPU work.
  [[nodiscard]] std::exception_ptr ClassifyFailure(
      std::exception_ptr failure) noexcept {
    if (failure == nullptr)
      return failure;
    if (device_lost.load(std::memory_order_acquire))
      return device_lost_failure;
    try {
      std::string reason = "unknown failure";
      try {
        std::rethrow_exception(failure);
      } catch (const TextGenerationError&) {
        return failure;
      } catch (const std::exception& error) {
        reason = error.what();
      } catch (...) {
        // A non-standard exception keeps the generic reason.
      }
      if (!device_lost.load(std::memory_order_acquire)) {
        if (runner_pool->runner().DeviceUsable())
          return failure;
        MarkDeviceLost(reason);
      }
      return device_lost_failure;
    } catch (...) {
      return failure;
    }
  }

  void CompleteDeadline(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    EndPublicationWait(request, "deadline");
    CompleteFailure(request, std::make_exception_ptr(TextGenerationError(
                                 TextGenerationErrorCode::kDeadlineExceeded,
                                 "text generation request deadline exceeded")));
  }

  [[nodiscard]] bool CompleteIfStopped(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    if (IsTerminal(request))
      return true;
    if (device_lost.load(std::memory_order_acquire)) {
      CompleteFailure(request, device_lost_failure);
      return true;
    }
    try {
      if (DeadlineExceeded(request, scheduler_policy.auto_sessions)) {
        CompleteDeadline(request);
        return true;
      }
      if (CancellationRequested(request)) {
        CompleteCancelled(request);
        return true;
      }
      return false;
    } catch (...) {
      CompleteFailure(request, std::current_exception());
      return true;
    }
  }

  void CompleteSuccess(const std::shared_ptr<ScheduledRequest>& request,
                       TextGenerationBackend::FinishReason finish_reason) {
    if (!request->coalesce_followers.empty()) {
      auto followers = std::move(request->coalesce_followers);
      request->coalesce_followers.clear();
      for (auto& follower : followers)
        if (follower && !IsTerminal(follower))
          CompleteSuccess(follower, finish_reason);
    }
    if (request->stop_filter.enabled()) {
      auto tail = request->stop_filter.Finish();
      request->result.text += tail;
      if (!tail.empty() && !PublishPiece(request, std::move(tail)))
        throw TextGenerationError(
            TextGenerationErrorCode::kOutputBackpressure,
            "text generation buffered output limit exceeded");
    }
    if (request->runner_request) {
      // A stop may land on a preview, an unadvanced AR token, or inside an
      // accepted speculative block. Retain only correctly labelled executed
      // state, using the same cache reconciliation as an interrupted request.
      // Read before the commit returns the leased state.
      const auto ngram = request->runner_request.ngram_rows();
      request->result.ngram_rows = ngram.rows;
      request->result.ngram_dedup_rows = ngram.dedup_rows;
      request->result.ngram_cache_hits = ngram.cache_hits;
      request->result.ngram_disk_reads = ngram.disk_reads;
      const auto cache_commit =
          finish_reason == TextGenerationBackend::FinishReason::kStopSequence
              ? request->runner_request.Cancel()
              : request->runner_request.Commit();
      request->result.component_cache = cache_commit.component_cache;
      request->result.cache_shared_bytes =
          cache_commit.component_cache
              ? cache_commit.component_cache->shared_bytes
              : 0;
      request->result.cache_snapshot_bytes = cache_commit.snapshot_bytes;
      request->result.cache_snapshot_ms = cache_commit.snapshot_ms;
      request->result.cache_disk_queued_bytes = cache_commit.disk_queued_bytes;
      request->result.cache_disk_enqueue_ms = cache_commit.disk_enqueue_ms;
      request->result.cache_shared_prefix_snapshots =
          cache_commit.shared_prefix_snapshots;
      request->result.cache_shared_prefix_bytes =
          cache_commit.shared_prefix_bytes;
      request->result.cache_shared_prefix_ms = cache_commit.shared_prefix_ms;
    }
    FinalizeResult(request, finish_reason);
    LogDecodeProgress(request, true);
    // Recorded here rather than by the HTTP adapter, so a client that leaves
    // before reading the result still counts.
    RecordRequestMetrics(request->result);
    RecordCompletedStats(*request, false, false);
    PublishTerminal(request);
  }

  /// Counts one request accepted by Submit. This is the arrival count shown as
  /// `adm`; actual session admission can happen later behind the queue.
  void RecordAdmissionStats() noexcept {
    if (!scheduler_policy.log_stats)
      return;
    try {
      std::lock_guard<std::mutex> lock(stats_mutex);
      ++stats_admitted;
    } catch (...) {
      // Periodic statistics must never fail a request.
    }
  }

  /// Records one finished request for the periodic vLLM-style summary. The
  /// scheduler calls this after its normal completion path, including
  /// cancellations and failures; statistics are best effort.
  void RecordCompletedStats(const ScheduledRequest& request, bool failed,
                            bool cancelled) noexcept {
    if (!scheduler_policy.log_stats)
      return;
    try {
      const auto& result = request.result;
      const std::size_t prompt = result.prefill_tokens;
      const std::size_t generated = result.completion_tokens;
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats_prompt_tokens += prompt;
      stats_generated_tokens += generated;
      stats_prompt_seconds += result.prefill_ms / 1000.0;
      stats_gen_seconds += result.decode_ms / 1000.0;
      stats_ttft_ms += result.ttft_ms;
      if (failed)
        ++stats_failed;
      else if (cancelled)
        ++stats_cancelled;
      else
        ++stats_completed;
      stats_window.push_back(
          {request.id, prompt, generated, result.ttft_ms, failed, cancelled});
    } catch (...) {
      // Periodic statistics must never fail a request.
    }
  }

  /// Once-per-minute line naming the requests that finished in the window.
  /// The window is swapped before the log-level check so a quiet level does
  /// not accumulate an unbounded queue.
  void LogCompletedWindow() {
    std::deque<CompletedStats> window;
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      window.swap(stats_window);
    }
    if (!Logger::Enabled(LogLevel::kInfo))
      return;
    std::size_t failed = 0;
    std::size_t cancelled = 0;
    std::size_t prompt = 0;
    std::size_t generated = 0;
    for (const auto& entry : window) {
      failed += entry.failed;
      cancelled += entry.cancelled;
      prompt += entry.prompt_tokens;
      generated += entry.generated_tokens;
    }
    std::ostringstream out;
    out << "event=requests window_s=60 completed=" << window.size()
        << " failed=" << failed << " cancelled=" << cancelled
        << " prompt_tokens=" << prompt << " generated_tokens=" << generated
        << " ids=";
    constexpr std::size_t kIdLimit = 24;
    const auto append = [&out](const CompletedStats& entry) {
      out << 'r' << entry.id;
      if (entry.failed)
        out << ":fail";
      else if (entry.cancelled)
        out << ":cancel";
      else
        out << ":ok";
    };
    if (window.empty()) {
      out << "none";
    } else {
      const auto head =
          window.size() <= kIdLimit ? window.size() : kIdLimit / 2;
      for (std::size_t i = 0; i < head; ++i) {
        if (i != 0)
          out << ',';
        append(window[i]);
      }
      if (window.size() > kIdLimit) {
        out << ",+" << window.size() - kIdLimit << " more";
        for (std::size_t i = window.size() - kIdLimit / 2; i < window.size();
             ++i) {
          out << ',';
          append(window[i]);
        }
      }
    }
    Logger::Info("stats", out.str());
  }

  /// Ten-second interval over the whole text server. Every numeric column is
  /// the difference between the previous snapshot and this one, so an idle
  /// server reports zero instead of the lifetime average. Process-wide token
  /// counters include in-flight work; per-request rates and TTFT come from
  /// requests that completed in the interval.
  void LogStats(bool header) {
    StatsSnapshot current;
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      current.time = std::chrono::steady_clock::now();
      current.total_prompt_tokens = detail::TotalPromptTokens().load();
      current.total_generated_tokens = detail::TotalGenTokens().load();
      current.admitted = stats_admitted;
      current.completed = stats_completed;
      current.cancelled = stats_cancelled;
      current.failed = stats_failed;
      current.prompt_tokens = stats_prompt_tokens;
      current.generated_tokens = stats_generated_tokens;
      current.prompt_seconds = stats_prompt_seconds;
      current.gen_seconds = stats_gen_seconds;
      current.ttft_ms = stats_ttft_ms;
      current.cache_hits = detail::PrefixCacheHits().load();
    }
    const auto previous = stats_last;
    stats_last = current;
    if (!Logger::Enabled(LogLevel::kInfo))
      return;

    const auto delta = [](std::uint64_t now, std::uint64_t before) {
      return now >= before ? now - before : 0;
    };
    const double interval =
        std::chrono::duration<double>(current.time - previous.time).count();
    const auto prompt_tokens =
        delta(current.prompt_tokens, previous.prompt_tokens);
    const auto generated_tokens =
        delta(current.generated_tokens, previous.generated_tokens);
    const auto admitted = delta(current.admitted, previous.admitted);
    const auto completed = delta(current.completed, previous.completed);
    const auto cancelled = delta(current.cancelled, previous.cancelled);
    const auto failed = delta(current.failed, previous.failed);
    const auto cache_hits = delta(current.cache_hits, previous.cache_hits);
    const double prompt_seconds =
        std::max(0.0, current.prompt_seconds - previous.prompt_seconds);
    const double gen_seconds =
        std::max(0.0, current.gen_seconds - previous.gen_seconds);
    const double ttft_ms = std::max(0.0, current.ttft_ms - previous.ttft_ms);
    const auto finished = completed + cancelled + failed;
    const double pp_request =
        prompt_seconds > 0.0
            ? static_cast<double>(prompt_tokens) / prompt_seconds
            : 0.0;
    const double tg_request =
        gen_seconds > 0.0 ? static_cast<double>(generated_tokens) / gen_seconds
                          : 0.0;
    const double ttft =
        finished > 0 ? ttft_ms / static_cast<double>(finished) : 0.0;
    const double total_pp =
        interval > 0.0
            ? static_cast<double>(delta(current.total_prompt_tokens,
                                        previous.total_prompt_tokens)) /
                  interval
            : 0.0;
    const double total_tg =
        interval > 0.0
            ? static_cast<double>(delta(current.total_generated_tokens,
                                        previous.total_generated_tokens)) /
                  interval
            : 0.0;
    const auto concurrency = std::max<std::int64_t>(
        0, detail::RequestsProcessing().load(std::memory_order_relaxed));
    if (header) {
      Logger::Info(
          "stats",
          "conc |  pp/s req  tg/s req  TTFT ms |  win pp/s  win tg/s    adm   "
          "done  win s  fail  cache");
      Logger::Info("stats",
                   "-----------------------------------------------------------"
                   "-----------------------------------");
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << std::setw(4) << concurrency
        << " | " << std::setw(9) << pp_request << std::setw(9) << tg_request
        << std::setw(9) << std::setprecision(0) << ttft << " | " << std::setw(9)
        << std::setprecision(1) << total_pp << std::setw(9) << total_tg
        << std::setw(7) << admitted << std::setw(7) << finished << std::setw(7)
        << std::setprecision(2) << interval << std::setw(5) << failed
        << std::setw(6) << cache_hits;
    Logger::Info("stats", out.str());
  }

  /// One-second tick: 10 s interval statistics and a 60 s completed-request
  /// window. The window log runs first and marks the header due, so the table
  /// header appears only after the once-per-minute request line.
  void RunStats(const std::stop_token& stop_token) noexcept {
    using clock = std::chrono::steady_clock;
    auto last_stats = clock::now();
    auto last_window = last_stats;
    while (!stop_token.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (stop_token.stop_requested())
        break;
      const auto now = clock::now();
      if (now - last_window >= std::chrono::seconds(60)) {
        LogCompletedWindow();
        stats_header_due = true;
        last_window = now;
      }
      if (now - last_stats >= std::chrono::seconds(10)) {
        LogStats(stats_header_due);
        stats_header_due = false;
        last_stats = now;
      }
    }
  }

  void ProcessQueuedCancellations() {
    for (const auto& request : QueuedSnapshot()) {
      try {
        const bool deadline_exceeded =
            DeadlineExceeded(request, scheduler_policy.auto_sessions);
        const bool cancelled =
            !deadline_exceeded && CancellationRequested(request);
        if (!(deadline_exceeded || cancelled) || !RemoveQueued(request)) {
          continue;
        }
        if (deadline_exceeded) {
          CompleteDeadline(request);
        } else {
          CompleteCancelled(request);
        }
      } catch (...) {
        if (RemoveQueued(request)) {
          CompleteFailure(request, std::current_exception());
        }
      }
    }
  }

  /// Parks a cold request behind a resident one that is still prefilling the
  /// longest prefix both prompts share. That request publishes a checkpoint
  /// there, so this one restores it instead of prefilling the same tokens in
  /// lockstep. Prefill runs one request at a time, so waiting costs nothing.
  [[nodiscard]] bool WaitForSharedPrefix(
      const std::shared_ptr<ScheduledRequest>& request,
      std::span<const std::deque<std::shared_ptr<ScheduledRequest>>* const>
          residents,
      std::size_t waiting) {
    if (std::exchange(request->prefix_considered, true) ||
        !request->cache_prompt ||
        // Parked requests hold no runner lease; bound their reservations like
        // admitted ones.
        waiting + 1 >= EffectiveCapacity())
      return false;
    std::shared_ptr<ScheduledRequest> leader;
    std::size_t common = 0;
    const std::span<const TextRunnerToken> prompt(request->prompt);
    for (const auto* list : residents) {
      for (const auto& resident : *list) {
        if (IsTerminal(resident) || !resident->runner_request ||
            resident->runner_request.prefill_complete())
          continue;
        const auto other = resident->runner_request.prompt();
        const auto shared = static_cast<std::size_t>(
            std::ranges::mismatch(prompt, other).in1 - prompt.begin());
        if (shared <= common ||
            shared <= resident->runner_request.prefill_position())
          continue;
        const auto identity =
            request->prompt_context
                ? request->prompt_context->CacheIdentity(shared)
                : std::span<const std::uint8_t>{};
        if (!std::ranges::equal(
                identity, resident->runner_request.input_identity(shared)))
          continue;
        leader = resident;
        common = shared;
      }
    }
    if (leader == nullptr)
      return false;
    // A longer retained prefix, such as this conversation's previous turn,
    // already beats waiting for a peer.
    const auto cached =
        runner_pool->CachedPrefixTokens(prompt, request->prompt_context.get());
    // A leader may have leased a deeper native frontier than a sibling can
    // restore. Share its next planned checkpoint even for a short suffix:
    // replaying generated tokens through prefill can change recurrent state.
    // Ordinary cold-prefix sharing retains its minimum-work threshold.
    const auto minimum_gain =
        leader->runner_request.cached_prompt_tokens() > cached
            ? std::size_t{1}
            : TextRunnerPool::Request::kSharedPrefixMinTokens;
    if (common < cached + minimum_gain)
      return false;
    const auto position = leader->runner_request.ShareCheckpoint(common);
    if (position < cached + minimum_gain)
      return false;
    request->prefix_leader = std::move(leader);
    request->prefix_position = position;
    request->prefix_wait_start = Clock::now();
    if (Logger::Enabled(LogLevel::kDebug)) {
      std::ostringstream line;
      line << "event=shared_prefix_wait id=" << request->id
           << " leader=" << request->prefix_leader->id << " tokens=" << position
           << " cached=" << cached << " prompt_tokens=" << prompt.size();
      Logger::Debug("cache", line.str());
    }
    return true;
  }

  /// A parked request resumes once its leader's checkpoint is retained, or
  /// once the leader can no longer publish it.
  [[nodiscard]] bool SharedPrefixSettled(
      const std::shared_ptr<ScheduledRequest>& request) const {
    const auto& leader = request->prefix_leader;
    if (IsTerminal(leader) || !leader->runner_request ||
        leader->runner_request.prefill_complete() ||
        leader->runner_request.prefill_position() > request->prefix_position)
      return true;
    return leader->runner_request.prefill_position() ==
               request->prefix_position &&
           runner_pool->CachedPrefixTokens(request->prompt,
                                           request->prompt_context.get()) >=
               request->prefix_position;
  }

  /// Drops stopped parked requests and releases settled ones in arrival
  /// order. Released requests are admitted before newer queued ones.
  void UpdateWaiting(std::deque<std::shared_ptr<ScheduledRequest>>& waiting) {
    for (auto it = waiting.begin(); it != waiting.end();) {
      const auto& request = *it;
      if (CompleteIfStopped(request)) {
        it = waiting.erase(it);
        continue;
      }
      if (request->prefix_leader && SharedPrefixSettled(request)) {
        const double wait_ms = std::chrono::duration<double, std::milli>(
                                   Clock::now() - *request->prefix_wait_start)
                                   .count();
        request->result.shared_prefix_wait_ms = wait_ms;
        if (Logger::Enabled(LogLevel::kDebug)) {
          std::ostringstream line;
          line << "event=shared_prefix_ready id=" << request->id
               << " leader=" << request->prefix_leader->id
               << " tokens=" << request->prefix_position
               << " wait_ms=" << wait_ms;
          Logger::Debug("cache", line.str());
        }
        request->prefix_leader.reset();
      }
      if (request->publication_waiting) {
        const auto pending = runner_pool->PendingPublication(
            request->prompt, request->prompt_context.get(),
            request->publication.checkpoint_id);
        if (!pending) {
          // Published, failed or superseded: the matching job is no longer
          // pending. Its completion log carries this same checkpoint ID.
          EndPublicationWait(request, "not_pending");
        } else {
          request->publication = *pending;
          if (pending->Gain() <= TextPendingPublication::kMinimumGainTokens) {
            EndPublicationWait(request, "available");
          } else if (const auto limit =
                         publication_prefill.Limit(pending->Gain())) {
            // Better usable cache or faster recent prefill may shorten the
            // original deadline. Neither slower samples nor a new disk job
            // may extend it or restart this request's one publication wait.
            request->publication_wait_limit =
                std::min(request->publication_wait_limit, *limit);
            if (Clock::now() - *request->publication_wait_start >=
                request->publication_wait_limit)
              EndPublicationWait(
                  request, request->publication_wait_limit ==
                                   detail::PublicationWaitEstimate::kMaximumWait
                               ? "cap"
                               : "cost_bound");
          } else {
            EndPublicationWait(request, "no_estimate");
          }
        }
      }
      ++it;
    }
  }

  [[nodiscard]] static std::shared_ptr<ScheduledRequest> PopReleased(
      std::deque<std::shared_ptr<ScheduledRequest>>& waiting) {
    const auto released =
        std::ranges::find_if(waiting, [](const auto& request) {
          return request->prefix_leader == nullptr &&
                 !request->publication_waiting;
        });
    if (released == waiting.end())
      return {};
    auto request = std::move(*released);
    waiting.erase(released);
    return request;
  }

  void Admit(std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
             std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
             std::deque<std::shared_ptr<ScheduledRequest>>& capturing,
             std::deque<std::shared_ptr<ScheduledRequest>>& waiting,
             std::deque<std::shared_ptr<ScheduledRequest>>& preempted,
             const std::stop_token& stop_token) {
    UpdateWaiting(waiting);
    const std::array<const std::deque<std::shared_ptr<ScheduledRequest>>*, 2>
        residents{&prefilling, &capturing};
    // Captures retain their runner lease until decoding resumes.
    while (!stop_token.stop_requested() &&
           prefilling.size() + decoding.size() + capturing.size() <
               EffectiveCapacity()) {
      // A preempted request owns the next free session: its replay is the
      // work an older arrival already paid for.
      auto request = PopPreempted(preempted);
      if (request == nullptr) {
        request = PopReleased(waiting);
      }
      if (request == nullptr) {
        // Parked followers already reserve a visible admission session. They
        // may resume in place, but new arrivals must respect those
        // reservations.
        if (prefilling.size() + decoding.size() + capturing.size() +
                waiting.size() >=
            EffectiveCapacity())
          return;
        const auto next = PeekQueued();
        if (next && !CanAdmitKv(*next, prefilling, decoding, capturing)) {
          DeferAdmission(next, prefilling, decoding, capturing);
          return;
        }
        request = PopQueued();
        if (request)
          kv_admission_retry_at.reset();
      }
      if (request == nullptr) {
        return;
      }
      try {
        if (CompleteIfStopped(request)) {
          continue;
        }
        // A second byte-identical greedy request joins the resident leader
        // instead of acquiring a model session of its own. The leader must be
        // at least as long-lived, so a request with a larger token limit stays
        // independent and becomes its own leader.
        if (request->coalesce_eligible) {
          std::shared_ptr<ScheduledRequest> coalesce_leader;
          for (auto* list : {&prefilling, &decoding, &capturing}) {
            for (auto& candidate : *list) {
              if (candidate && candidate->coalesce_eligible &&
                  !candidate->coalesced_follower && !IsTerminal(candidate) &&
                  candidate->result.tokens.empty() &&
                  request->token_limit <= candidate->token_limit &&
                  request->stop_at_eos == candidate->stop_at_eos &&
                  request->cache_prompt == candidate->cache_prompt &&
                  request->coalesce_prompt == candidate->coalesce_prompt) {
                coalesce_leader = candidate;
                break;
              }
            }
            if (coalesce_leader)
              break;
          }
          if (coalesce_leader) {
            request->coalesced_follower = true;
            request->coalesce_leader = coalesce_leader;
            request->phase.store(TextRequestPhase::kPrefilling,
                                 std::memory_order_release);
            request->result.execution_plan = "coalesced";
            request->result.physical_execution_width = 1;
            request->admitted_once = true;
            coalesce_leader->coalesce_followers.push_back(request);
            continue;
          }
        }
        if (WaitForSharedPrefix(request, residents, waiting.size())) {
          waiting.push_back(std::move(request));
          continue;
        }
        if (request->cache_prompt && !request->publication_wait_start) {
          if (const auto pending = runner_pool->PendingPublication(
                  request->prompt, request->prompt_context.get())) {
            request->publication = *pending;
            if (const auto limit = publication_prefill.Limit(pending->Gain())) {
              request->publication_wait_limit = *limit;
              request->publication_wait_start = Clock::now();
              request->publication_waiting = true;
              LogPublicationWait(request, "start");
              waiting.push_back(std::move(request));
              continue;
            }
            LogPublicationWait(request, "skip", "no_estimate");
          }
        }

        const std::weak_ptr<ScheduledRequest> weak_request = request;
        // The context outlives this admission: a preempted request is
        // re-admitted and must present the same prompt identity again.
        request->runner_request = runner_pool->Acquire(
            std::move(request->prompt), request->sampling,
            [weak_request, stop_token,
             auto_sessions = scheduler_policy.auto_sessions] {
              const auto request = weak_request.lock();
              return stop_token.stop_requested() || request == nullptr ||
                     CancellationRequested(request) ||
                     DeadlineExceeded(request, auto_sessions);
            },
            request->prompt_context, request->cache_prompt,
            request->cache_prefix_tokens, request->stop_at_eos,
            request->cache_persist, std::move(request->resume_sampler));
        // Each admission may restore or capture a checkpoint of its own.
        progress.fetch_add(1, std::memory_order_relaxed);
        if (!request->runner_request) {
          CompleteCancelled(request);
          continue;
        }

        request->result.component_cache =
            request->runner_request.component_cache_metrics();
        request->result.cache_hit = request->runner_request.cache_hit();
        const auto lookup = request->runner_request.cache_lookup();
        request->result.cache_miss_reason = lookup.miss_reason;
        request->result.cache_common_prefix_tokens =
            lookup.common_prefix_tokens;
        request->result.cache_checkpoint_tokens = lookup.checkpoint_tokens;
        request->result.cached_prompt_tokens =
            request->runner_request.cached_prompt_tokens();
        request->live_cached_prompt_tokens.store(
            request->result.cached_prompt_tokens, std::memory_order_relaxed);
        request->result.cache_restore_bytes =
            request->runner_request.cache_restore_bytes();
        request->result.cache_restore_ms =
            request->runner_request.cache_restore_ms();
        request->result.cache_disk_hit =
            request->runner_request.cache_disk_hit();
        request->result.incremental_prefill_supported =
            incremental_prefill_supported;
        // A replayed request keeps the queue time of its original arrival.
        if (!request->admitted_once) {
          request->admitted_once = true;
          request->result.queue_ms = std::chrono::duration<double, std::milli>(
                                         Clock::now() - request->request_start)
                                         .count();
          request->result.resident_requests_at_admission =
              prefilling.size() + decoding.size() + capturing.size() + 1;
        }
        PublishAdmission(request);
        PublishPromptProgress(request);
        request->phase.store(TextRequestPhase::kAdmitted,
                             std::memory_order_release);
        request->phase.store(request->runner_request.prefill_complete()
                                 ? TextRequestPhase::kDecodeReady
                                 : TextRequestPhase::kPrefilling,
                             std::memory_order_release);
        if (request->runner_request.prefill_complete()) {
          request->decode_due = true;
          decoding.push_back(std::move(request));
        } else {
          // The last-served prefill just moved to the back of its round.
          // Give an arrival its first turn before repeating that work, while
          // retaining the order of peers that have already been waiting.
          auto next = prefilling.end();
          if (!prefilling.empty() &&
              prefilling.back()->id == last_prefill_request_id) {
            --next;
          }
          prefilling.insert(next, std::move(request));
        }
      } catch (...) {
        const auto failure = std::current_exception();
        const auto now = Clock::now();
        const bool patient =
            scheduler_policy.auto_sessions ||
            (request->kv_deferrals < kMaxKvDeferrals &&
             (!request->kv_deferred_since ||
              now - *request->kv_deferred_since <= kKvDegradationPatience));
        if (IsKvExhaustion(failure) && patient &&
            (scheduler_policy.auto_sessions ||
             (prefilling.size() + decoding.size() + capturing.size()) != 0)) {
          request->kv_deferrals += 1;
          if (!request->kv_deferred_since)
            request->kv_deferred_since = now;
          request->result.kv_deferrals = request->kv_deferrals;
          request->prefix_leader.reset();
          request->prefix_considered = true;
          ReleaseProcessingSlot(request);
          request->kv_retry_at = now + std::chrono::milliseconds(20);
          if (Logger::Enabled(LogLevel::kInfo)) {
            Logger::Info("scheduler",
                         "event=kv_exhaustion_deferred request=" +
                             std::to_string(request->id) +
                             " stage=admission deferrals=" +
                             std::to_string(request->kv_deferrals));
          }
          kv_deferred_requests.push_back(std::move(request));
          continue;
        }
        CompleteFailure(request, failure);
      }
    }
  }

  /// Pops eligible peers for a batched prefill pass. `prefill_cohort_limit`
  /// is the upper bound; the pass takes only requests already waiting, so the
  /// actual cohort follows queue depth and never waits for peers. Eligibility
  /// is uniform: plain prompt arithmetic with no planned checkpoint, so the
  /// shared pass runs no per-request capture.
  std::vector<std::shared_ptr<ScheduledRequest>> TakePrefillPeers(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling) {
    std::vector<std::shared_ptr<ScheduledRequest>> peers;
    while (peers.size() + 1 < prefill_cohort_limit && !prefilling.empty() &&
           prefilling.front()->runner_request.CanBatchPrefill()) {
      peers.push_back(std::move(prefilling.front()));
      prefilling.pop_front();
    }
    return peers;
  }

  /// Routes a request that just ran a prefill chunk back into its queue.
  void SettlePrefilled(
      const std::shared_ptr<ScheduledRequest>& request, bool mark_decode_due,
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      std::deque<std::shared_ptr<ScheduledRequest>>& capturing) {
    if (!request || IsTerminal(request) || AwaitingKvRetry(request))
      return;
    if (request->runner_request.SnapshotPending()) {
      capturing.push_back(request);
      return;
    }
    if (request->runner_request.prefill_complete()) {
      if (mark_decode_due)
        request->decode_due = true;
      decoding.push_back(request);
      return;
    }
    prefilling.push_back(request);
  }

  void StepPrefill(
      const std::shared_ptr<ScheduledRequest>& request, bool decoder_runnable,
      std::size_t resident_tokens, std::size_t pending_decoders,
      bool peer_waiting = false,
      std::span<const std::shared_ptr<ScheduledRequest>> peers = {}) {
    try {
      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(TextRequestPhase::kPrefilling,
                           std::memory_order_release);
      last_prefill_request_id = request->id;
      // Bound work for short waiting prefills and captures as well as decoders.
      // Long prefills alone retain the model's efficient chunk size.
      const bool bounded_prefill =
          incremental_prefill_supported && (decoder_runnable || peer_waiting);
      // The configured max_num_batched_tokens, capped by the KV the resident
      // batch leaves free, so a long chunk never overruns the unified pool. An
      // idle prefill keeps its bulk work unit, but still never claims more KV
      // than the pool reports.
      std::size_t budget =
          bounded_prefill
              ? PrefillTokenBudget(resident_tokens, pending_decoders)
              : request->runner_request.prompt_tokens();
      if (prefill_policy.adapt_to_kv_capacity) {
        if (const auto free = FreeKvTokens(resident_tokens, pending_decoders)) {
          if (!bounded_prefill)
            budget = std::min(budget, std::max<std::size_t>(1, *free));
          // Early memory-pressure signal: report the iteration in which free
          // KV, not the configured budget, decided how much could run.
          const std::size_t needed = std::max<std::size_t>(
              1, request->runner_request.prompt_tokens() -
                     std::min(request->runner_request.prefill_position(),
                              request->runner_request.prompt_tokens()));
          const bool kv_limited =
              *free < std::min(needed, prefill_policy.decode_active_tokens);
          if (kv_limited) {
            request->result.prefill_kv_limited = true;
            detail::TotalKvLimited().fetch_add(1, std::memory_order_relaxed);
            if (Logger::Enabled(LogLevel::kDebug)) {
              Logger::Debug("scheduler",
                            "event=kv_pressure reason=chunk_limit request=" +
                                std::to_string(request->id) +
                                " free_tokens=" + std::to_string(*free) +
                                " needed_tokens=" + std::to_string(needed) +
                                " granted=" + std::to_string(budget));
            }
          }
        }
      }
      request->result.prefill_budget_tokens = budget;
      for (const auto& peer : peers) {
        peer->phase.store(TextRequestPhase::kPrefilling,
                          std::memory_order_release);
        peer->result.prefill_budget_tokens = budget;
      }
      if ((decoder_runnable || runner_pool->capacity() > 1) &&
          !incremental_prefill_supported) {
        request->result.prefill_fallback_reason =
            "incremental_prefill_unavailable";
        for (const auto& peer : peers)
          peer->result.prefill_fallback_reason =
              "incremental_prefill_unavailable";
      }
      const auto start = Clock::now();
      TextPrefillStep step;
      std::vector<TextPrefillStep> peer_steps;
      if (!peers.empty()) {
        std::vector<TextRunnerPool::Request*> batch;
        batch.reserve(peers.size());
        for (const auto& peer : peers)
          batch.push_back(&peer->runner_request);
        auto steps = request->runner_request.PrefillBatch(batch, budget);
        step = steps.front();
        peer_steps.assign(steps.begin() + 1, steps.end());
      } else {
        step = request->runner_request.Prefill(budget);
      }
      const double step_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      const auto record = [&](const std::shared_ptr<ScheduledRequest>& target,
                              const TextPrefillStep& value) {
        target->result.prefill_ms += step_ms;
        target->result.prefill_tokens += value.consumed_tokens;
        target->live_prefill_tokens.store(target->result.prefill_tokens,
                                          std::memory_order_relaxed);
        ++target->result.prefill_chunks;
        target->result.max_prefill_chunk_tokens = std::max(
            target->result.max_prefill_chunk_tokens, value.consumed_tokens);
        LogPrefillProgress(target, value.consumed_tokens, step_ms);
        PublishPromptProgress(target);
      };
      std::size_t step_tokens = step.consumed_tokens;
      for (const auto& peer_step : peer_steps)
        step_tokens += peer_step.consumed_tokens;
      if (!decoder_runnable && !peer_waiting)
        publication_prefill.Observe(step_tokens, step_ms);
      detail::TotalPromptTokens().fetch_add(step_tokens,
                                            std::memory_order_relaxed);
      record(request, step);
      for (std::size_t i = 0; i < peer_steps.size(); ++i)
        record(peers[i], peer_steps[i]);
      if (!peers.empty() && Logger::Enabled(LogLevel::kDebug)) {
        std::string tokens = std::to_string(step.consumed_tokens);
        for (const auto& peer_step : peer_steps)
          tokens += "+" + std::to_string(peer_step.consumed_tokens);
        Logger::Debug(
            "scheduler",
            "event=batched_prefill request=" + std::to_string(request->id) +
                " cohort=" + std::to_string(peers.size() + 1) +
                " tokens=" + tokens + " ms=" + std::to_string(step_ms));
      }

      if (decoder_runnable) {
        ++request->result.active_decode_prefill_chunks;
        // A peer that ran its chunk in this pass counts it against the same
        // active-decode episode.
        for (const auto& peer : peers)
          ++peer->result.active_decode_prefill_chunks;
        ++consecutive_active_prefill_chunks;
        request->result.max_consecutive_active_prefill_chunks =
            std::max(request->result.max_consecutive_active_prefill_chunks,
                     consecutive_active_prefill_chunks);
        for (const auto& peer : peers)
          peer->result.max_consecutive_active_prefill_chunks =
              std::max(peer->result.max_consecutive_active_prefill_chunks,
                       consecutive_active_prefill_chunks);
      } else {
        consecutive_active_prefill_chunks = 0;
      }

      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(step.decode_ready ? TextRequestPhase::kDecodeReady
                                             : TextRequestPhase::kPrefilling,
                           std::memory_order_release);
      for (std::size_t i = 0; i < peers.size(); ++i) {
        peers[i]->phase.store(peer_steps[i].decode_ready
                                  ? TextRequestPhase::kDecodeReady
                                  : TextRequestPhase::kPrefilling,
                              std::memory_order_release);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto& peer : peers) {
        if (IsTerminal(peer))
          continue;
        if (IsKvExhaustion(failure)) {
          (void)DegradeKvExhausted(peer);
        } else if (!CompleteIfStopped(peer)) {
          CompleteFailure(peer, failure);
        }
      }
      if (IsKvExhaustion(failure) && DegradeKvExhausted(request))
        return;
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  [[nodiscard]] std::optional<Clock::time_point> PrepareDecode(
      const std::shared_ptr<ScheduledRequest>& request) {
    if (CompleteIfStopped(request)) {
      return std::nullopt;
    }

    if (request->advance_pending) {
      if (!request->runner_request.PreparePromptSnapshot())
        return std::nullopt;
      request->advance_pending = false;
      if (!final_token_advance_required &&
          request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
        return std::nullopt;
      }
      // Snapshot capture and work for other requests are outside this step.
      return Clock::now();
    }
    request->phase.store(TextRequestPhase::kDecoding,
                         std::memory_order_release);
    const auto decode_start = Clock::now();
    const auto selection = request->runner_request.SelectNext();
    if (selection.stop) {
      PublishTerminalLogprobs(request, selection.logprobs);
      request->result.decode_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
              .count();
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
      return std::nullopt;
    }

    if (!PublishSelection(request, selection)) {
      return std::nullopt;
    }
    request->result.decode_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
            .count();
    if (request->stop_filter.stopped()) {
      CompleteSuccess(request,
                      TextGenerationBackend::FinishReason::kStopSequence);
      return std::nullopt;
    }
    request->advance_pending = true;
    return PrepareDecode(request);
  }

  const DecodedLogprobs* RetainLogprobs(
      const std::shared_ptr<ScheduledRequest>& request,
      const std::optional<sampling::TokenLogprobs>& values) {
    if (!values)
      return nullptr;
    auto decoded =
        request->logprob_decoder.DecodeToken(*values, [&](auto token) {
          return runner_pool->runner().Decode(std::span(&token, 1));
        });
    // Metadata is bounded by the token budget, not the text byte limit: at
    // top_logprobs 20 that limit would end a default request near 600 tokens.
    request->result.logprobs.push_back(std::move(decoded));
    return &request->result.logprobs.back();
  }

  void PublishTerminalLogprobs(
      const std::shared_ptr<ScheduledRequest>& request,
      const std::optional<sampling::TokenLogprobs>& values) {
    if (const auto* decoded = RetainLogprobs(request, values))
      if (!PublishPiece(request, {}, decoded))
        throw TextGenerationError(TextGenerationErrorCode::kOutputBackpressure,
                                  "logprobs exceed buffered output limit");
  }

  [[nodiscard]] bool PublishSelection(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextDecodeSelection& selection) {
    const auto now = Clock::now();
    if (!request->previous_token.has_value()) {
      request->result.ttft_ms = std::chrono::duration<double, std::milli>(
                                    now - request->request_start)
                                    .count();
    } else {
      const auto inter_token = now - *request->previous_token;
      request->inter_token_total += inter_token;
      request->result.max_inter_token_ms = std::max(
          request->result.max_inter_token_ms,
          std::chrono::duration<double, std::milli>(inter_token).count());
      ++request->inter_token_samples;
    }
    request->previous_token = now;
    if (selection.piece.size() >
        request->max_output_bytes - request->generated_output_bytes) {
      CompleteFailure(request,
                      std::make_exception_ptr(TextGenerationError(
                          TextGenerationErrorCode::kOutputLimit,
                          "text generation output byte limit exceeded")));
      return false;
    }
    request->generated_output_bytes += selection.piece.size();
    const auto* logprobs = RetainLogprobs(request, selection.logprobs);
    request->result.tokens.push_back(selection.token);
    request->live_generated_tokens.store(request->result.tokens.size(),
                                         std::memory_order_relaxed);
    detail::TotalGenTokens().fetch_add(1, std::memory_order_relaxed);
    const auto piece = request->stop_filter.enabled()
                           ? request->stop_filter.Push(selection.piece)
                           : selection.piece;
    if ((!piece.empty() || !request->stop_filter.enabled() || logprobs) &&
        !PublishPiece(request, piece, logprobs)) {
      CompleteFailure(request,
                      std::make_exception_ptr(TextGenerationError(
                          TextGenerationErrorCode::kOutputBackpressure,
                          "text generation buffered output limit exceeded")));
      return false;
    }
    if (incremental_text_is_exact) {
      request->result.text += piece;
    }
    if (request->stop_filter.stopped())
      request->result.stop_sequence = request->stop_filter.matched_sequence();
    // Coalesced replicas receive the leader's exact token stream. Their own
    // token limit may be smaller; a follower that reaches it is finalized
    // while the leader keeps running for the remaining replicas.
    if (!request->coalesce_followers.empty()) {
      const auto followers = request->coalesce_followers;
      for (const auto& follower : followers) {
        if (!follower || IsTerminal(follower))
          continue;
        if (!PublishSelection(follower, selection))
          continue;
        follower->phase.store(TextRequestPhase::kDecoding,
                              std::memory_order_release);
        if (follower->stop_filter.stopped()) {
          CompleteSuccess(follower,
                          TextGenerationBackend::FinishReason::kStopSequence);
          continue;
        }
        if (follower->result.tokens.size() >= follower->token_limit)
          CompleteSuccess(follower,
                          TextGenerationBackend::FinishReason::kLength);
      }
    }
    if (CompleteIfStopped(request)) {
      return false;
    }
    return true;
  }

  void FinishAdvanced(const std::shared_ptr<ScheduledRequest>& request,
                      Clock::time_point decode_start) {
    request->result.decode_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
            .count();
    LogDecodeProgress(request);
    if (CompleteIfStopped(request)) {
      return;
    }
    if (request->result.tokens.size() >= request->token_limit) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
    }
  }

  void StepDecode(const std::shared_ptr<ScheduledRequest>& request) {
    consecutive_active_prefill_chunks = 0;
    if (multi_token_decode) {
      StepMultiTokenDecode(request);
      return;
    }
    try {
      const auto decode_start = PrepareDecode(request);
      if (!decode_start.has_value()) {
        return;
      }
      request->runner_request.Advance();
      FinishAdvanced(request, *decode_start);
    } catch (...) {
      const auto failure = std::current_exception();
      if (IsKvExhaustion(failure) && DegradeKvExhausted(request))
        return;
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  void StepMultiTokenDecode(const std::shared_ptr<ScheduledRequest>& request) {
    consecutive_active_prefill_chunks = 0;
    try {
      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      if (!PrepareFirstSnapshot(request))
        return;
      const auto decode_start = Clock::now();
      const std::size_t remaining =
          request->token_limit - request->result.tokens.size() +
          (request->preview_token.has_value() ? 1 : 0);
      const auto step = request->runner_request.DecodeStep(remaining);
      request->result.draft_rounds += step.draft_rounds;
      request->result.draft_tokens += step.draft_tokens;
      request->result.draft_accepted_tokens += step.draft_accepted_tokens;
      request->result.lookup_tokens += step.lookup_tokens;
      request->result.lookup_accepted_tokens += step.lookup_accepted_tokens;

      CheckPreviewResult(request, step);
      for (const auto& selection : step.selections) {
        if (!PublishDecodedSelection(request, selection)) {
          return;
        }
        if (request->stop_filter.stopped())
          break;
      }

      request->result.decode_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
              .count();
      LogDecodeProgress(request);
      if (request->stop_filter.stopped()) {
        CompleteSuccess(request,
                        TextGenerationBackend::FinishReason::kStopSequence);
        return;
      }
      if (step.stop) {
        PublishTerminalLogprobs(request, step.terminal_logprobs);
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
        return;
      }
      if (request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      if (IsKvExhaustion(failure) && DegradeKvExhausted(request))
        return;
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  bool PrepareFirstSnapshot(const std::shared_ptr<ScheduledRequest>& request) {
    if (request->result.tokens.empty() && !request->preview_stops) {
      const auto preview_start = Clock::now();
      const auto preview = request->runner_request.PreviewFirstToken();
      if (preview && !preview->stop) {
        if (!PublishSelection(request, *preview))
          return false;
        request->preview_token = preview->token;
      }
      if (preview && preview->stop)
        PublishTerminalLogprobs(request, preview->logprobs);
      request->preview_stops = preview && preview->stop;
      request->result.decode_ms += std::chrono::duration<double, std::milli>(
                                       Clock::now() - preview_start)
                                       .count();
      LogDecodeProgress(request);
    }
    if (request->stop_filter.stopped()) {
      CompleteSuccess(request,
                      TextGenerationBackend::FinishReason::kStopSequence);
      return false;
    }
    if (!request->runner_request.PreparePromptSnapshot())
      return false;
    if (request->preview_stops) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
      return false;
    }
    return !CompleteIfStopped(request);
  }

  static void CheckPreviewResult(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextDecodeStep& step) {
    if (request->preview_token &&
        (step.selections.empty() ||
         step.selections.front().token != *request->preview_token)) {
      throw std::runtime_error("first-token preview disagrees with decoding");
    }
  }

  bool PublishDecodedSelection(const std::shared_ptr<ScheduledRequest>& request,
                               const TextDecodeSelection& selection) {
    if (request->preview_token) {
      request->preview_token.reset();
      return true;
    }
    return PublishSelection(request, selection);
  }

  void StepDecodeBatch(
      const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    const auto selected_plan = runner_pool->SelectDecodePlan(requests.size());
    if (batched_multi_token_decode &&
        (batched_multi_token_decode_max_width == 0 ||
         selected_plan.physical_width <=
             batched_multi_token_decode_max_width)) {
      StepMultiTokenDecodeBatch(requests);
      return;
    }
    if (requests.size() == 1) {
      StepDecode(requests.front());
      return;
    }

    consecutive_active_prefill_chunks = 0;
    struct PreparedRequest {
      std::shared_ptr<ScheduledRequest> request;
      Clock::time_point decode_start;
    };
    std::vector<PreparedRequest> prepared;
    prepared.reserve(requests.size());
    for (const auto& request : requests) {
      try {
        const auto decode_start = PrepareDecode(request);
        if (decode_start.has_value()) {
          prepared.push_back({
              .request = request,
              .decode_start = *decode_start,
          });
        }
      } catch (...) {
        CompleteFailure(request, std::current_exception());
      }
    }
    if (prepared.empty()) {
      return;
    }
    if (device_lost.load(std::memory_order_acquire)) {
      for (const auto& item : prepared)
        CompleteFailure(item.request, device_lost_failure);
      return;
    }
    if (prepared.size() == 1) {
      try {
        prepared.front().request->runner_request.Advance();
        FinishAdvanced(prepared.front().request, prepared.front().decode_start);
      } catch (...) {
        CompleteFailure(prepared.front().request, std::current_exception());
      }
      return;
    }

    const auto plan = runner_pool->SelectDecodePlan(prepared.size());
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      for (const auto& item : prepared) {
        if (CompleteIfStopped(item.request))
          continue;
        try {
          item.request->runner_request.Advance();
          FinishAdvanced(item.request, item.decode_start);
        } catch (...) {
          CompleteFailure(item.request, std::current_exception());
        }
      }
      return;
    }

    std::vector<TextRunnerPool::Request*> runner_requests;
    runner_requests.reserve(prepared.size());
    for (const auto& item : prepared) {
      runner_requests.push_back(&item.request->runner_request);
    }
    std::vector<std::exception_ptr> failures;
    try {
      failures = runner_pool->AdvanceBatch(runner_requests, plan);
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto& item : prepared) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failure);
      }
      return;
    }

    // Classify every failed lane before committing a successful peer: its
    // snapshot/commit may also touch the context that just failed.
    for (auto& failure : failures) {
      if (failure)
        failure = ClassifyFailure(std::move(failure));
      if (device_lost.load(std::memory_order_acquire)) {
        for (const auto& item : prepared)
          CompleteFailure(item.request, device_lost_failure);
        return;
      }
    }
    const std::string execution_plan =
        "batched-w" + std::to_string(plan.physical_width);
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const auto& item = prepared[i];
      if (failures[i]) {
        if (IsKvExhaustion(failures[i]) && DegradeKvExhausted(item.request))
          continue;
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failures[i], true);
        continue;
      }
      item.request->result.physical_execution_width = std::max(
          item.request->result.physical_execution_width, plan.physical_width);
      item.request->result.execution_plan = execution_plan;
      FinishAdvanced(item.request, item.decode_start);
    }
  }

  void StepMultiTokenDecodeBatch(
      const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    if (requests.size() == 1) {
      StepMultiTokenDecode(requests.front());
      return;
    }

    consecutive_active_prefill_chunks = 0;
    struct PreparedRequest {
      std::shared_ptr<ScheduledRequest> request;
      Clock::time_point decode_start;
      std::size_t remaining;
    };
    std::vector<PreparedRequest> prepared;
    prepared.reserve(requests.size());
    for (const auto& request : requests) {
      if (CompleteIfStopped(request)) {
        continue;
      }
      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      try {
        if (!PrepareFirstSnapshot(request))
          continue;
      } catch (...) {
        CompleteFailure(request, std::current_exception());
        continue;
      }
      prepared.push_back({
          .request = request,
          .decode_start = Clock::now(),
          .remaining = request->token_limit - request->result.tokens.size() +
                       (request->preview_token.has_value() ? 1 : 0),
      });
    }
    if (prepared.empty()) {
      return;
    }
    if (device_lost.load(std::memory_order_acquire)) {
      for (const auto& item : prepared)
        CompleteFailure(item.request, device_lost_failure);
      return;
    }
    if (prepared.size() == 1) {
      StepMultiTokenDecode(prepared.front().request);
      return;
    }

    const auto plan = runner_pool->SelectDecodePlan(prepared.size());
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      for (const auto& item : prepared) {
        StepMultiTokenDecode(item.request);
      }
      return;
    }

    std::vector<TextRunnerPool::Request*> runner_requests;
    std::vector<std::size_t> max_tokens;
    runner_requests.reserve(prepared.size());
    max_tokens.reserve(prepared.size());
    for (const auto& item : prepared) {
      runner_requests.push_back(&item.request->runner_request);
      max_tokens.push_back(item.remaining);
    }

    std::vector<TextDecodeStep> steps;
    try {
      steps = runner_pool->DecodeBatch(runner_requests, max_tokens, plan);
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto& item : prepared) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failure);
      }
      return;
    }

    for (auto& step : steps) {
      if (step.failure)
        step.failure = ClassifyFailure(std::move(step.failure));
      if (device_lost.load(std::memory_order_acquire)) {
        for (const auto& item : prepared)
          CompleteFailure(item.request, device_lost_failure);
        return;
      }
    }
    for (std::size_t index = 0; index < prepared.size(); ++index) {
      const auto& item = prepared[index];
      if (CompleteIfStopped(item.request))
        continue;
      const auto& step = steps[index];
      if (step.failure) {
        if (IsKvExhaustion(step.failure) && DegradeKvExhausted(item.request))
          continue;
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, step.failure, true);
        continue;
      }
      if (step.execution_plan.physical_width >=
          item.request->result.physical_execution_width) {
        item.request->result.physical_execution_width =
            step.execution_plan.physical_width;
        item.request->result.execution_plan =
            step.execution_plan.kind == TextExecutionPlanKind::kBatched
                ? "batched-w" +
                      std::to_string(step.execution_plan.physical_width)
                : (runner_pool->capacity() == 1 ? "serial-c1"
                                                : "serial-fallback");
      }
      item.request->result.draft_rounds += step.draft_rounds;
      item.request->result.draft_tokens += step.draft_tokens;
      item.request->result.draft_accepted_tokens += step.draft_accepted_tokens;
      item.request->result.lookup_tokens += step.lookup_tokens;
      item.request->result.lookup_accepted_tokens +=
          step.lookup_accepted_tokens;

      bool published = true;
      try {
        CheckPreviewResult(item.request, step);
      } catch (...) {
        CompleteFailure(item.request, std::current_exception());
        continue;
      }
      for (const auto& selection : step.selections) {
        if (!PublishDecodedSelection(item.request, selection)) {
          published = false;
          break;
        }
        if (item.request->stop_filter.stopped())
          break;
      }
      item.request->result.decode_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() -
                                                    item.decode_start)
              .count();
      LogDecodeProgress(item.request);
      if (!published || IsTerminal(item.request)) {
        continue;
      }
      try {
        if (item.request->stop_filter.stopped()) {
          CompleteSuccess(item.request,
                          TextGenerationBackend::FinishReason::kStopSequence);
        } else if (step.stop) {
          PublishTerminalLogprobs(item.request, step.terminal_logprobs);
          CompleteSuccess(item.request,
                          TextGenerationBackend::FinishReason::kStop);
        } else if (item.request->result.tokens.size() >=
                   item.request->token_limit) {
          CompleteSuccess(item.request,
                          TextGenerationBackend::FinishReason::kLength);
        }
      } catch (...) {
        CompleteFailure(item.request, std::current_exception());
      }
    }
  }

  [[nodiscard]] static bool HasDueDecoder(
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding) {
    return std::any_of(decoding.begin(), decoding.end(),
                       [](const auto& request) { return request->decode_due; });
  }

  static void MarkAllDecodersDue(
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding) {
    for (const auto& request : decoding) {
      request->decode_due = true;
    }
  }

  [[nodiscard]] static std::shared_ptr<ScheduledRequest> PopDecoder(
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      bool require_due) {
    const std::size_t candidates = decoding.size();
    for (std::size_t index = 0; index < candidates; ++index) {
      auto request = std::move(decoding.front());
      decoding.pop_front();
      if (!require_due || request->decode_due) {
        return request;
      }
      decoding.push_back(std::move(request));
    }
    auto request = std::move(decoding.front());
    decoding.pop_front();
    return request;
  }

  void CancelRemaining(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      std::deque<std::shared_ptr<ScheduledRequest>>& waiting,
      std::deque<std::shared_ptr<ScheduledRequest>>& preempted) noexcept {
    std::vector<std::shared_ptr<ScheduledRequest>> remaining_queued;
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      remaining_queued.reserve(queued_count);
      for (auto& client : queued_clients) {
        std::move(client.requests.begin(), client.requests.end(),
                  std::back_inserter(remaining_queued));
      }
      queued_clients.clear();
      SetQueuedCountLocked(0);
    }
    for (const auto& request : remaining_queued) {
      CompleteCancelled(request);
    }
    for (const auto& request : prefilling) {
      CompleteCancelled(request);
    }
    for (const auto& request : decoding) {
      CompleteCancelled(request);
    }
    for (const auto& request : waiting) {
      CompleteCancelled(request);
    }
    for (const auto& request : preempted) {
      CompleteCancelled(request);
    }
    for (const auto& request : kv_deferred_requests) {
      CompleteCancelled(request);
    }
    prefilling.clear();
    decoding.clear();
    waiting.clear();
    preempted.clear();
    kv_deferred_requests.clear();
  }

  void Run(const std::stop_token& stop_token) noexcept {
    std::deque<std::shared_ptr<ScheduledRequest>> prefilling;
    std::deque<std::shared_ptr<ScheduledRequest>> decoding;
    std::deque<std::shared_ptr<ScheduledRequest>> capturing;
    // Cold requests waiting for a resident prefill of their shared prefix.
    std::deque<std::shared_ptr<ScheduledRequest>> waiting;
    // Requests whose state was released for RECOMPUTE; admitted first.
    std::deque<std::shared_ptr<ScheduledRequest>> preempted;
    while (!stop_token.stop_requested()) {
      if (device_lost.load(std::memory_order_acquire))
        break;
      loop_working.store(true, std::memory_order_relaxed);
      ProcessQueuedCancellations();
      ProcessCoalescedFollowers(prefilling, decoding, capturing);

      for (std::size_t count = capturing.size(); count != 0; --count) {
        auto request = std::move(capturing.front());
        capturing.pop_front();
        if (request->runner_request.SnapshotPending()) {
          capturing.push_back(std::move(request));
          continue;
        }
        progress.fetch_add(1, std::memory_order_relaxed);
        if (!CompleteIfStopped(request)) {
          if (request->runner_request.prefill_complete()) {
            // Its capture already waited through peer prefill. Resume it
            // before another bounded chunk, as for any due decoder.
            request->decode_due = true;
            decoding.push_back(std::move(request));
          } else {
            prefilling.push_back(std::move(request));
          }
        }
      }
      AutoScaleBatch(prefilling, decoding, capturing, waiting);
      // A lowered max_num_seqs releases state before another admission can
      // take the session, so the shrink is visible to the very next work unit.
      EnforceBatchLimit(prefilling, decoding, capturing, preempted);
      // Degraded requests re-enter the waiting queue once residents have had a
      // chance to free memory; restoring their host snapshot is cheap and the
      // alternative is failing a request the pool refused blocks for.
      if (!kv_deferred_requests.empty()) {
        const auto now = Clock::now();
        for (auto deferred = kv_deferred_requests.begin();
             deferred != kv_deferred_requests.end();) {
          if (!(*deferred)->kv_retry_at || now >= *(*deferred)->kv_retry_at) {
            auto request = std::move(*deferred);
            deferred = kv_deferred_requests.erase(deferred);
            request->kv_retry_at.reset();
            waiting.push_back(std::move(request));
          } else {
            ++deferred;
          }
        }
      }
      if (!LowMemory()) {
        kv_reliefs_in_episode = 0;
      } else {
        const std::array<const std::deque<std::shared_ptr<ScheduledRequest>>*,
                         3>
            resident_lists{&prefilling, &decoding, &capturing};
        if (RelieveKvPressure(prefilling, decoding, preempted,
                              ResidentTokens(resident_lists))) {
          progress.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
      }
      Admit(prefilling, decoding, capturing, waiting, preempted, stop_token);
      if (device_lost.load(std::memory_order_acquire))
        break;
      if (prefilling.empty() && decoding.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        const auto wake = [&] {
          return stop_token.stop_requested() || stopping ||
                 !preempted.empty() ||
                 (queued_count != 0 &&
                  capturing.size() + waiting.size() < EffectiveCapacity() &&
                  (!kv_admission_retry_at ||
                   Clock::now() >= *kv_admission_retry_at)) ||
                 (!kv_deferred_requests.empty() &&
                  (!kv_deferred_requests.front()->kv_retry_at ||
                   Clock::now() >= *kv_deferred_requests.front()->kv_retry_at));
        };
        if (capturing.empty() && waiting.empty() &&
            kv_deferred_requests.empty()) {
          // Only the idle wait has a timer. Arrival interrupts it immediately;
          // no clock checks or device polling are added to active work units.
          while (!wake()) {
            loop_working.store(false, std::memory_order_relaxed);
            auto wait = scheduler_policy.device_probe_interval;
            if (kv_admission_retry_at) {
              const auto now = Clock::now();
              if (*kv_admission_retry_at > now) {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        *kv_admission_retry_at - now);
                wait = std::min(
                    wait, std::max(std::chrono::milliseconds{1}, remaining));
              } else {
                wait = std::chrono::milliseconds{1};
              }
            }
            const bool woken = queue_condition.wait_for(lock, wait, wake);
            progress.fetch_add(1, std::memory_order_relaxed);
            loop_working.store(true, std::memory_order_relaxed);
            if (woken)
              break;
            lock.unlock();
            try {
              if (runner_pool->runner().PollDevice() ==
                  TextModelRunner::DeviceProbeStatus::kLost)
                MarkDeviceLost("idle device probe failed");
            } catch (...) {
              // An inconclusive probe must not kill a usable model.
            }
            lock.lock();
            if (device_lost.load(std::memory_order_acquire))
              break;
          }
        } else
          queue_condition.wait_for(lock, std::chrono::milliseconds(1), wake);
        continue;
      }

      progress.fetch_add(1, std::memory_order_relaxed);
      const bool due_decoder = HasDueDecoder(decoding);
      const std::size_t resident_count = prefilling.size() + decoding.size();
      resident_count_hint = resident_count;
      // KV already committed by the resident batch, and the tokens this
      // iteration's due decoders need; both shrink a prefill chunk.
      const std::array<const std::deque<std::shared_ptr<ScheduledRequest>>*, 3>
          resident_lists{&prefilling, &decoding, &capturing};
      const std::size_t resident_tokens = ResidentTokens(resident_lists);
      const std::size_t pending_decoders = decoding.size();
      const bool preparing_multi_token_batch =
          multi_token_decode && resident_count > 1 && !prefilling.empty() &&
          runner_pool->SelectDecodePlan(resident_count).kind ==
              TextExecutionPlanKind::kBatched;
      if (preparing_multi_token_batch) {
        for (const auto& pending : prefilling) {
          pending->runner_request.PrepareBatchExecution();
        }
        for (const auto& ready : decoding) {
          ready->runner_request.PrepareBatchExecution();
        }
      }
      // Give simultaneous new requests one bounded chunk to form their first
      // batch. Once decoding starts, every due decoder runs before more
      // prefill.
      const bool assemble_initial_batch =
          preparing_multi_token_batch &&
          consecutive_active_prefill_chunks == 0 &&
          prefilling.front()->runner_request.prompt_tokens() -
                  prefilling.front()->runner_request.prefill_position() <=
              prefill_policy.decode_active_tokens &&
          std::all_of(decoding.begin(), decoding.end(),
                      [](const auto& request) {
                        return request->result.tokens.empty();
                      });
      const bool assemble_decode_cohort =
          scheduler_policy.assemble_decode_rows != 0 &&
          decoding.size() < scheduler_policy.assemble_decode_rows;
      if (!prefilling.empty() && !decoding.empty() &&
          (!due_decoder || assemble_initial_batch || assemble_decode_cohort)) {
        auto request = std::move(prefilling.front());
        prefilling.pop_front();
        // Cold prompts share one model pass when all of them are plain prompt
        // arithmetic: the expert weight read is paid once for the cohort.
        std::vector<std::shared_ptr<ScheduledRequest>> peers;
        if (request->runner_request.CanBatchPrefill())
          peers = TakePrefillPeers(prefilling);
        StepPrefill(request, true, resident_tokens, pending_decoders, false,
                    peers);
        SettlePrefilled(request, false, prefilling, decoding, capturing);
        for (const auto& peer : peers)
          SettlePrefilled(peer, false, prefilling, decoding, capturing);
        MarkAllDecodersDue(decoding);
        continue;
      }

      if (!decoding.empty()) {
        const std::size_t candidate_count =
            due_decoder
                ? static_cast<std::size_t>(std::count_if(
                      decoding.begin(), decoding.end(),
                      [](const auto& request) { return request->decode_due; }))
                : decoding.size();
        // The kernel tiles top out at 8/16/32/48 rows. An 8k+1 ready set
        // would open a tile for a single row that still reads the full
        // weights; leave that row due for the next step and run 8/16/24/32
        // full tiles. The deferred request advances one step later.
        const std::size_t planned =
            candidate_count > 8 && candidate_count % 8 == 1
                ? candidate_count - 1
                : candidate_count;
        const auto plan = runner_pool->SelectDecodePlan(planned);
        const std::size_t batch_size =
            plan.kind == TextExecutionPlanKind::kBatched
                ? std::min(planned, plan.physical_width)
                : 1;
        std::vector<std::shared_ptr<ScheduledRequest>> batch;
        batch.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index) {
          auto request = PopDecoder(decoding, due_decoder);
          request->decode_due = false;
          batch.push_back(std::move(request));
        }
        StepDecodeBatch(batch);
        for (auto& request : batch) {
          if (!IsTerminal(request) && !AwaitingKvRetry(request)) {
            if (request->runner_request.SnapshotPending())
              capturing.push_back(std::move(request));
            else
              decoding.push_back(std::move(request));
          }
        }
        continue;
      }

      auto request = std::move(prefilling.front());
      prefilling.pop_front();
      // Any other queued work makes this a chunked prefill: a 32K prompt must
      // not hold the scheduler for 20-30 s while peers wait for a turn. A lone
      // request keeps the bulk work unit, which is the efficient kernel path.
      const bool peer_waiting =
          !capturing.empty() || !prefilling.empty() || !waiting.empty();
      std::vector<std::shared_ptr<ScheduledRequest>> peers;
      if (request->runner_request.CanBatchPrefill())
        peers = TakePrefillPeers(prefilling);
      StepPrefill(request, false, resident_tokens, pending_decoders,
                  peer_waiting, peers);
      SettlePrefilled(request, true, prefilling, decoding, capturing);
      for (const auto& peer : peers)
        SettlePrefilled(peer, true, prefilling, decoding, capturing);
    }
    for (auto& request : capturing)
      decoding.push_back(std::move(request));
    CancelRemaining(prefilling, decoding, waiting, preempted);
  }

  std::shared_ptr<TextRunnerPool> runner_pool;
  TextPrefillPolicy prefill_policy;
  TextSchedulerPolicy scheduler_policy;
  /// Resident-request bound (`max_num_seqs`), at most the pool capacity.
  std::atomic<std::size_t> batch_limit;
  /// Effective pending-queue bounds from `scheduler_policy`, with zero fields
  /// derived from the batch size at construction.
  std::size_t max_pending_requests{0};
  std::size_t max_pending_requests_per_client{0};
  detail::PublicationWaitEstimate publication_prefill;
  std::shared_ptr<OutputBudget> output_budget;
  std::shared_ptr<SessionTable> sessions;
  bool incremental_prefill_supported{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  std::size_t batched_multi_token_decode_max_width{0};
  mutable std::mutex queue_mutex;
  std::condition_variable queue_condition;
  std::deque<PendingClient> queued_clients;
  std::size_t queued_count{0};
  bool stopping{false};
  std::atomic<bool> device_lost{false};
  const std::exception_ptr device_lost_failure =
      std::make_exception_ptr(TextGenerationError(
          TextGenerationErrorCode::kDeviceLost, kDeviceLostMessage));
  std::vector<std::shared_ptr<ScheduledRequest>> device_lost_requests;
  std::size_t consecutive_active_prefill_chunks{0};
  /// Requests that may share one prefill pass, including the lead request.
  std::size_t prefill_cohort_limit{32};
  /// Prefilling request that last triggered a KV relief; scheduler-thread only.
  std::uint64_t kv_relieved_request{0};
  /// Decoders reclaimed during the current pressure episode. Reset once the
  /// pressure clears, so a later episode may reclaim again.
  std::size_t kv_reliefs_in_episode{0};
  /// Requests released because the device pool refused blocks. They wait for
  /// residents to free memory and are re-admitted from their host snapshots.
  std::deque<std::shared_ptr<ScheduledRequest>> kv_deferred_requests;
  /// New arrivals blocked by the shared KV pool keep their queue position and
  /// retry after this point. The completion path also notifies the condition.
  std::optional<TextGenerationScheduler::Clock::time_point>
      kv_admission_retry_at;
  /// Resident count of the current iteration, for degradation decisions taken
  /// inside the step handlers. Scheduler-thread only.
  std::size_t resident_count_hint{0};
  std::uint64_t last_prefill_request_id{0};
  std::atomic<std::uint64_t> next_request_id{1};
  // Written by the scheduler thread and read by the stall monitor. `progress`
  // advances when a request is admitted, a work unit starts, a pending
  // capture completes or an idle wait returns; `loop_working` is false only
  // inside that idle wait. A loop that only polls a capture which never
  // completes makes no progress.
  std::atomic<std::uint64_t> progress{0};
  std::atomic<bool> loop_working{false};
  // Periodic vLLM-style summary. The scheduler thread records finished
  // requests; stats_monitor snapshots once per 10 s and once per minute.
  mutable std::mutex stats_mutex;
  std::deque<CompletedStats> stats_window;
  std::uint64_t stats_admitted{0};
  std::uint64_t stats_prompt_tokens{0};
  std::uint64_t stats_generated_tokens{0};
  std::uint64_t stats_completed{0};
  std::uint64_t stats_cancelled{0};
  std::uint64_t stats_failed{0};
  double stats_prompt_seconds{0.0};
  double stats_gen_seconds{0.0};
  double stats_ttft_ms{0.0};
  /// Snapshot of the previous 10 s row. Only the statistics thread touches it.
  StatsSnapshot stats_last{};
  bool stats_header_due{false};
  std::jthread stats_monitor;
  std::jthread worker;
  std::jthread stall_monitor;
};

TextGenerationScheduler::Request::Request() = default;

TextGenerationScheduler::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TextGenerationScheduler::Request::~Request() {
  Cancel();
}

TextGenerationScheduler::Request::Request(Request&&) noexcept = default;

TextGenerationScheduler::Request& TextGenerationScheduler::Request::operator=(
    Request&& other) noexcept {
  if (this != &other) {
    Cancel();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

TextGenerationScheduler::Request::operator bool() const noexcept {
  return impl_ != nullptr && impl_->request != nullptr;
}

std::uint64_t TextGenerationScheduler::Request::id() const noexcept {
  return *this ? impl_->request->id : 0;
}

TextRequestPhase TextGenerationScheduler::Request::phase() const noexcept {
  return *this ? impl_->request->phase.load(std::memory_order_acquire)
               : TextRequestPhase::kTerminal;
}

TextGenerationScheduler::Result TextGenerationScheduler::Request::Wait(
    const TokenCallback& on_token, const ProgressCallback& on_progress,
    const StartCallback& on_start) {
  if (!*this) {
    throw std::logic_error("text scheduler request is empty");
  }
  if (impl_->waited) {
    throw std::logic_error("text scheduler request was already consumed");
  }
  impl_->waited = true;

  bool deliver_pieces = true;
  // Only streams with a start consumer use the queue deadline.
  bool started = !on_start || !impl_->request->publish_token_pieces;
  bool consumer_cancelled = false;
  std::exception_ptr callback_failure;
  Result result;
  std::exception_ptr scheduler_failure;

  while (true) {
    OutputPiece piece;
    std::optional<TextGenerationBackend::PromptProgress> progress;
    bool start = false;
    bool has_piece = false;
    bool terminal = false;
    {
      std::unique_lock<std::mutex> lock(impl_->request->output_mutex);
      const auto ready = [&] {
        return impl_->request->terminal ||
               (!started && impl_->request->admitted) ||
               impl_->request->pending_progress.has_value() ||
               !impl_->request->output_pieces.empty();
      };
      if (started) {
        impl_->request->output_condition.wait(lock, ready);
      } else {
        // A request still queued at the deadline also starts, so transport
        // keepalives can run while it waits.
        const bool woke = impl_->request->output_condition.wait_until(
            lock, impl_->request->stream_start_deadline, ready);
        // Admission precedes this request's progress and output, even when
        // it also failed before the consumer woke.
        start = !woke || impl_->request->admitted;
      }
      if (start) {
        // Delivered below, before any progress or output.
      } else if (impl_->request->pending_progress.has_value()) {
        progress = std::exchange(impl_->request->pending_progress, {});
      } else if (!impl_->request->output_pieces.empty()) {
        has_piece = true;
        const std::size_t piece_bytes =
            QueuedPieceCost(impl_->request->output_pieces.front());
        piece = std::move(impl_->request->output_pieces.front());
        impl_->request->output_pieces.pop_front();
        impl_->request->buffered_output_bytes -= piece_bytes;
        impl_->request->output_budget->Release(piece_bytes);
      } else if (impl_->request->terminal) {
        result = impl_->request->result;
        scheduler_failure = impl_->request->failure;
        terminal = true;
      }
    }

    const auto deliver = [&](const auto& callback, const auto& value) {
      try {
        if (!callback(value)) {
          consumer_cancelled = true;
          deliver_pieces = false;
          Cancel();
        }
      } catch (...) {
        callback_failure = std::current_exception();
        deliver_pieces = false;
        Cancel();
      }
    };
    if (start) {
      started = true;
      if (deliver_pieces)
        deliver([&](bool) { return on_start(); }, true);
      continue;
    }
    if (progress.has_value() && deliver_pieces && on_progress) {
      deliver(on_progress, *progress);
    }
    if (has_piece && deliver_pieces && on_token) {
      deliver(on_token, TokenPiece(piece.text, piece.logprobs ? &*piece.logprobs
                                                              : nullptr));
    }
    if (terminal) {
      break;
    }
  }

  if (consumer_cancelled) {
    result.cancelled = true;
    result.finish_reason = TextGenerationBackend::FinishReason::kCancelled;
  }
  if (impl_->trace.has_value()) {
    WriteGenerationTrace(
        *std::exchange(impl_->trace, std::nullopt), result,
        callback_failure != nullptr ? callback_failure : scheduler_failure);
  }
  if (callback_failure != nullptr) {
    std::rethrow_exception(callback_failure);
  }
  if (scheduler_failure != nullptr) {
    std::rethrow_exception(scheduler_failure);
  }
  return result;
}

void TextGenerationScheduler::Request::Cancel() noexcept {
  if (*this && !IsTerminal(impl_->request)) {
    impl_->request->cancellation_requested.store(true,
                                                 std::memory_order_release);
  }
}

TextGenerationScheduler::TextGenerationScheduler(
    std::shared_ptr<TextRunnerPool> runner_pool,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy)
    : impl_(std::make_unique<Impl>(std::move(runner_pool), prefill_policy,
                                   scheduler_policy)) {}

TextGenerationScheduler::~TextGenerationScheduler() = default;

const TextModelRunner& TextGenerationScheduler::runner() const noexcept {
  return impl_->runner_pool->runner();
}

std::size_t TextGenerationScheduler::capacity() const noexcept {
  return impl_->runner_pool->capacity();
}

std::size_t TextGenerationScheduler::max_num_seqs() const noexcept {
  return impl_->EffectiveCapacity();
}

bool TextGenerationScheduler::memory_pressure() const noexcept {
  return impl_->LowMemory();
}

void TextGenerationScheduler::SetMaxNumSeqs(std::size_t max_num_seqs) {
  impl_->SetBatchLimit(max_num_seqs);
}

bool TextGenerationScheduler::device_lost() const noexcept {
  return impl_->device_lost.load(std::memory_order_acquire);
}

std::vector<TextGenerationScheduler::SessionState>
TextGenerationScheduler::SessionStates() const {
  return impl_->SessionSnapshot();
}

std::size_t TextGenerationScheduler::buffered_output_bytes() const noexcept {
  return impl_->output_budget->buffered_bytes.load(std::memory_order_relaxed);
}

std::size_t TextGenerationScheduler::max_buffered_output_bytes()
    const noexcept {
  return impl_->output_budget->max_buffered_bytes.load(
      std::memory_order_relaxed);
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool publish_token_pieces) {
  return Submit(std::move(prompt), max_tokens, sampling, is_cancelled,
                publish_token_pieces, RequestMetadata{});
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool publish_token_pieces,
    RequestMetadata metadata) {
  if (prompt.empty()) {
    throw std::invalid_argument("text scheduler prompt must not be empty");
  }
  const auto context = impl_->runner_pool->runner().Descriptor().max_context;
  if (prompt.size() >= context) {
    throw std::length_error("prompt has " + std::to_string(prompt.size()) +
                            " tokens but the context is " +
                            std::to_string(context) +
                            "; increase --context or shorten the conversation");
  }
  const std::size_t available = context - prompt.size();
  sampling.Validate();
  ValidateStopSequences(metadata.stop_sequences);
  if (!metadata.stop_sequences.empty() && !impl_->incremental_text_is_exact)
    throw std::invalid_argument(
        "stop sequences require exact incremental token decoding");

  auto request = std::make_shared<ScheduledRequest>();
  request->stop_filter = StopSequenceFilter(std::move(metadata.stop_sequences));
  request->id = impl_->next_request_id.fetch_add(1, std::memory_order_relaxed);
  request->client_id =
      metadata.client_id.empty() ? "anonymous" : std::move(metadata.client_id);
  request->result.prompt_tokens = prompt.size();
  request->result.token_metrics_recorded = true;
  request->result.client_id = request->client_id;
  request->result.configured_active_prefill_tokens =
      impl_->prefill_policy.decode_active_tokens;
  request->result.requested_logical_concurrency = impl_->EffectiveCapacity();
  request->result.execution_plan =
      impl_->EffectiveCapacity() == 1 ? "serial-c1" : "serial-fallback";
  request->prompt = std::move(prompt);
  request->prompt_context = std::move(metadata.prompt_context);
  request->cache_prompt = metadata.cache_prompt;
  request->cache_persist = metadata.cache_persist;
  request->cache_prefix_tokens = metadata.cache_prefix_tokens;
  request->stop_at_eos = metadata.stop_at_eos;
  request->token_limit =
      max_tokens > 0 ? std::min(max_tokens, available) : available;
  request->output_limit_explicit = max_tokens > 0;
  request->sampling = sampling;
  request->external_cancellation = is_cancelled;
  request->publish_token_pieces = publish_token_pieces;
  request->publish_prompt_progress =
      publish_token_pieces && metadata.return_progress;
  // Coalescing is only valid for deterministic, unfiltered greedy decoding:
  // every replica must receive exactly the leader's token stream. Different
  // stop sequences, logprobs, constraints, penalties or prompt progress keep
  // the ordinary independent path.
  request->coalesce_prompt = request->prompt;
  request->coalesce_eligible =
      std::getenv("GUFO_REQUEST_COALESCE") != nullptr &&
      request->sampling.temperature == 0.0F && request->sampling.top_k == 0 &&
      request->sampling.top_p == 1.0F && request->sampling.min_p == 0.0F &&
      request->sampling.min_keep == 0 && !request->sampling.logprobs &&
      !request->sampling.constraint && !request->sampling.reasoning_budget &&
      request->sampling.repeat_penalty == 1.0F &&
      request->sampling.frequency_penalty == 0.0F &&
      request->sampling.presence_penalty == 0.0F &&
      !request->stop_filter.enabled() && !request->publish_prompt_progress;
  request->request_start = metadata.request_start;
  request->stream_start_deadline =
      request->request_start + impl_->scheduler_policy.stream_start_delay;
  request->deadline = metadata.deadline;
  if (!request->deadline.has_value() &&
      impl_->scheduler_policy.request_timeout.count() > 0) {
    request->deadline =
        request->request_start + impl_->scheduler_policy.request_timeout;
  }
  request->max_output_bytes =
      impl_->scheduler_policy.max_output_bytes_per_request;
  request->max_buffered_output_bytes =
      impl_->scheduler_policy.max_buffered_output_bytes_per_request;
  request->output_budget = impl_->output_budget;
  request->sessions = impl_->sessions;

  // Decoded before the enqueue below hands the prompt to the scheduler thread.
  std::optional<GenerationTrace> trace;
  if (Trace::Enabled()) {
    trace = StartGenerationTrace(*request, impl_->runner_pool->runner());
  }

  // A refusal is reported and thrown after the lock below is released.
  // Admission-line inputs are snapshotted under the queue mutex and the
  // messages are assembled after the unlock: the timestamp and stderr write
  // behind Logger::Debug must not serialize every other Submit() behind this
  // one. The counters are the only mutable state read; `client_id`, the
  // limits and the submit depths written below are never rewritten by the
  // worker thread, and the refused request was never enqueued, so nothing
  // else can observe them in between.
  bool refused = false;
  TextGenerationErrorCode refusal_code = TextGenerationErrorCode::kQueueFull;
  std::string refusal_message;
  std::string_view refusal_reason;  // empty when the request was admitted
  std::size_t decision_queued = 0;
  std::size_t decision_client_queued = 0;
  // Snapshot for the admission log: once the block below unlocks, the
  // scheduler thread owns the request and moves `prompt` out of it, so any
  // later read of the vector would race even though the shared_ptr survives.
  std::size_t admitted_prompt_tokens = 0;
  {
    const std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    if (impl_->stopping) {
      throw TextGenerationError(TextGenerationErrorCode::kSchedulerStopping,
                                "text generation scheduler is stopping");
    }
    if (impl_->device_lost.load(std::memory_order_acquire)) {
      throw TextGenerationError(TextGenerationErrorCode::kDeviceLost,
                                kDeviceLostMessage);
    }
    if (impl_->queued_count >= impl_->max_pending_requests) {
      refused = true;
      refusal_code = TextGenerationErrorCode::kQueueFull;
      refusal_message = "text generation pending queue is full";
      refusal_reason = "queue_full";
      decision_queued = impl_->queued_count;
    } else {
      auto client = std::find_if(
          impl_->queued_clients.begin(), impl_->queued_clients.end(),
          [&](const PendingClient& pending) {
            return pending.client_id == request->client_id;
          });
      if (client != impl_->queued_clients.end() &&
          client->requests.size() >= impl_->max_pending_requests_per_client) {
        refused = true;
        refusal_code = TextGenerationErrorCode::kClientQueueFull;
        refusal_message = "text generation client pending queue is full";
        refusal_reason = "client_quota";
        decision_queued = impl_->queued_count;
        decision_client_queued = client->requests.size();
      } else {
        if (client == impl_->queued_clients.end()) {
          impl_->queued_clients.push_back({
              .client_id = request->client_id,
              .requests = {},
          });
          client = std::prev(impl_->queued_clients.end());
        }
        decision_queued = impl_->queued_count + 1;
        decision_client_queued = client->requests.size() + 1;
        request->result.queue_depth_at_submit = decision_queued;
        request->result.client_queue_depth_at_submit = decision_client_queued;
        admitted_prompt_tokens = request->prompt.size();
        client->requests.push_back(request);
        impl_->SetQueuedCountLocked(impl_->queued_count + 1);
      }
    }
  }
  if (refused) {
    if (Logger::Enabled(LogLevel::kDebug)) {
      const bool quota = refusal_reason == "client_quota";
      std::string detail =
          "event=admission_refused reason=" + std::string(refusal_reason) +
          " queued=" + std::to_string(decision_queued);
      if (quota) {
        detail += " client_queued=" + std::to_string(decision_client_queued);
      }
      detail += " limit=" +
                std::to_string(quota ? impl_->max_pending_requests_per_client
                                     : impl_->max_pending_requests) +
                " client_id=" + request->client_id;
      Logger::Debug("scheduler", detail);
    }
    throw TextGenerationError(refusal_code, refusal_message);
  }
  impl_->RecordAdmissionStats();
  impl_->queue_condition.notify_one();
  // Emitted after the notification, not between the enqueue and it: these
  // lines allocate, and a failure there would throw out of Submit for a
  // request the scheduler is already running. The counters are the locals
  // snapshotted above; every other value read here is immutable after Submit
  // set it up, so the worker thread cannot interleave with this line's reads.
  if (Logger::Enabled(LogLevel::kDebug)) {
    Logger::Debug(
        "scheduler",
        "event=admitted request=" + std::to_string(request->id) +
            " client_id=" + request->client_id +
            " queued=" + std::to_string(decision_queued) +
            " client_queued=" + std::to_string(decision_client_queued) +
            " prompt_tokens=" + std::to_string(admitted_prompt_tokens) +
            " max_tokens=" + std::to_string(request->token_limit));
  }
  auto handle = std::make_unique<Request::Impl>(std::move(request));
  handle->trace = std::move(trace);
  return Request(std::move(handle));
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  return Submit(std::move(prompt), max_tokens, config, is_cancelled,
                publish_token_pieces);
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces, RequestMetadata metadata) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  return Submit(std::move(prompt), max_tokens, config, is_cancelled,
                publish_token_pieces, std::move(metadata));
}

}  // namespace gufo::server
