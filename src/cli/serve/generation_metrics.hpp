#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

// Process-wide `/metrics` values. Token counters advance per prefill chunk and
// per generated token, so scrapes see work while requests are still running.
namespace detail {
inline std::atomic<std::uint64_t>& DeviceLostTotal() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalGenTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Request totals below are added once per request when it finishes.
inline std::atomic<std::uint64_t>& TotalCachedPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
/// Completed requests whose prompt reused at least one cached token.
inline std::atomic<std::uint64_t>& PrefixCacheHits() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
/// Completed requests that prefilled their whole prompt.
inline std::atomic<std::uint64_t>& PrefixCacheMisses() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& TotalPromptSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<double>& TotalGenSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<std::uint64_t>& MaxSequenceTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftRounds() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftAcceptedTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalLookupTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalLookupAcceptedTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& LastPromptSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
inline std::atomic<double>& LastGenSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
// Schedulers add their own deltas, so a draining scheduler and its
// replacement can both contribute during a model reload.
inline std::atomic<std::int64_t>& RequestsProcessing() {
  static std::atomic<std::int64_t> count{0};
  return count;
}
inline std::atomic<std::int64_t>& RequestsDeferred() {
  static std::atomic<std::int64_t> count{0};
  return count;
}
// RECOMPUTE preemption counters: model states released and tokens replayed,
// never swapped out and back.
inline std::atomic<std::uint64_t>& TotalPreemptions() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalRecomputeTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Iterations whose prefill chunk was bounded by free KV capacity rather than by
// the configured batched-token budget.
inline std::atomic<std::uint64_t>& TotalKvLimited() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Requests the device pool refused blocks for and the scheduler degraded to
// their retained host snapshot instead of failing.
inline std::atomic<std::uint64_t>& TotalKvDeferrals() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Finished-request outcomes. A request lands in exactly one bucket: completed
// and cancelled are recorded by RecordRequestMetrics, failures by
// RecordFailedRequest. Failed requests also keep the token counters of the
// work they did, like cancellations.
inline std::atomic<std::uint64_t>& TotalRequestsCompleted() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalRequestsCancelled() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalRequestsFailed() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// One fixed bucket layout for TTFT and end-to-end request duration, so a
// scrape is self-contained and no request samples stay in memory. The upper
// end covers long-context prefill; +Inf is implied by the count.
inline constexpr std::array<double, 26> kLatencyBuckets{
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08,  0.1,  0.25,
    0.5,   0.75,  1.0,  1.5,  2.0,  2.5,  5.0,   7.5,  10.0,
    15.0,  20.0,  30.0, 40.0, 50.0, 60.0, 120.0, 300.0};
template<std::size_t N>
struct Histogram {
  std::array<std::atomic<std::uint64_t>, N> cumulative{};
  std::atomic<std::uint64_t> count{0};
  std::atomic<double> sum{0.0};
};
inline Histogram<kLatencyBuckets.size()>& TtftHistogram() {
  static Histogram<kLatencyBuckets.size()> histogram;
  return histogram;
}
inline Histogram<kLatencyBuckets.size()>& RequestDurationHistogram() {
  static Histogram<kLatencyBuckets.size()> histogram;
  return histogram;
}
template<std::size_t N>
inline void ObserveHistogram(Histogram<N>& histogram,
                             const std::array<double, N>& bounds,
                             double seconds) noexcept {
  if (!(seconds >= 0.0))
    seconds = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    if (seconds <= bounds[i])
      histogram.cumulative[i].fetch_add(1, std::memory_order_relaxed);
  }
  histogram.count.fetch_add(1, std::memory_order_relaxed);
  double current = histogram.sum.load(std::memory_order_relaxed);
  while (!histogram.sum.compare_exchange_weak(current, current + seconds,
                                              std::memory_order_relaxed)) {
  }
}
}  // namespace detail

/// Records one request that ended in an error or a lost device. The scheduler
/// calls this from CompleteFailure, so a failed request is never counted as
/// completed even when it had already produced tokens.
inline void RecordFailedRequest() noexcept {
  detail::TotalRequestsFailed().fetch_add(1, std::memory_order_relaxed);
}

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
}

/// Share of this request's prompt that came from the prefix cache. One means
/// every prompt token was reused, so the whole prefill budget stayed available
/// for decode slots in the same iteration.
inline double PrefixCacheHitRate(const TextGenerationBackend::Result& result) {
  const auto cached =
      std::min(result.cached_prompt_tokens, result.prompt_tokens);
  return result.prompt_tokens == 0
             ? 0.0
             : static_cast<double>(cached) /
                   static_cast<double>(result.prompt_tokens);
}

/// Adds one finished request to the request totals and speed gauges. The
/// scheduler calls this when a request ends, whether or not its client reads
/// the result; failed requests are not added.
inline void RecordRequestMetrics(const TextGenerationBackend::Result& result) {
  const auto add_seconds = [](std::atomic<double>& total, double ms) {
    double current = total.load(std::memory_order_relaxed);
    while (!total.compare_exchange_weak(current, current + ms / 1000.0,
                                        std::memory_order_relaxed)) {
    }
  };
  if (result.cancelled) {
    detail::TotalRequestsCancelled().fetch_add(1, std::memory_order_relaxed);
  } else {
    detail::TotalRequestsCompleted().fetch_add(1, std::memory_order_relaxed);
    const double ttft_seconds = std::max(0.0, result.ttft_ms) / 1000.0;
    const double duration_seconds =
        (std::max(0.0, result.ttft_ms) + std::max(0.0, result.decode_ms)) /
        1000.0;
    detail::ObserveHistogram(detail::TtftHistogram(), detail::kLatencyBuckets,
                             ttft_seconds);
    detail::ObserveHistogram(detail::RequestDurationHistogram(),
                             detail::kLatencyBuckets, duration_seconds);
  }
  detail::TotalCachedPromptTokens().fetch_add(
      std::min(result.cached_prompt_tokens, result.prompt_tokens),
      std::memory_order_relaxed);
  if (std::min(result.cached_prompt_tokens, result.prompt_tokens) > 0)
    detail::PrefixCacheHits().fetch_add(1, std::memory_order_relaxed);
  else
    detail::PrefixCacheMisses().fetch_add(1, std::memory_order_relaxed);
  add_seconds(detail::TotalPromptSeconds(), result.prefill_ms);
  add_seconds(detail::TotalGenSeconds(), result.decode_ms);
  const std::uint64_t sequence_tokens =
      result.prompt_tokens + result.completion_tokens;
  std::uint64_t max_tokens =
      detail::MaxSequenceTokens().load(std::memory_order_relaxed);
  while (max_tokens < sequence_tokens &&
         !detail::MaxSequenceTokens().compare_exchange_weak(
             max_tokens, sequence_tokens, std::memory_order_relaxed)) {
  }
  detail::TotalDraftRounds().fetch_add(result.draft_rounds,
                                       std::memory_order_relaxed);
  detail::TotalDraftTokens().fetch_add(result.draft_tokens,
                                       std::memory_order_relaxed);
  detail::TotalDraftAcceptedTokens().fetch_add(result.draft_accepted_tokens,
                                               std::memory_order_relaxed);
  detail::TotalLookupTokens().fetch_add(result.lookup_tokens,
                                        std::memory_order_relaxed);
  detail::TotalLookupAcceptedTokens().fetch_add(result.lookup_accepted_tokens,
                                                std::memory_order_relaxed);
  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double tok_per_sec =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  if (prompt_per_second > 0.0) {
    detail::LastPromptSpeed().store(prompt_per_second,
                                    std::memory_order_relaxed);
  }
  if (tok_per_sec > 0.0) {
    detail::LastGenSpeed().store(tok_per_sec, std::memory_order_relaxed);
  }
}

inline std::string GenerationLogDetails(
    const TextGenerationBackend::Result& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << "prompt_tokens=" << result.prompt_tokens
      << " prefill_tokens=" << result.prefill_tokens
      << " generated_tokens=" << result.completion_tokens << " finish="
      << (result.cancelled ? "cancelled"
          : result.finish_reason == TextGenerationBackend::FinishReason::kLength
              ? "length"
          : result.finish_reason ==
                  TextGenerationBackend::FinishReason::kStopSequence
              ? "stop_sequence"
              : "stop")
      << " cache="
      << (result.cache_disk_hit ? "disk"
          : result.cache_hit    ? "memory"
                                : "miss")
      << " cached_tokens=" << result.cached_prompt_tokens
      << " cache_hit_rate=" << std::setprecision(4)
      << PrefixCacheHitRate(result) << std::setprecision(1)
      << " cache_restore_ms=" << result.cache_restore_ms
      << " queue_depth=" << result.queue_depth_at_submit
      << " resident_at_admission=" << result.resident_requests_at_admission
      << " queue_ms=" << result.queue_ms
      << " shared_prefix_wait_ms=" << result.shared_prefix_wait_ms
      << " ttft_ms=" << result.ttft_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result) << " decode_tps="
      << (result.decode_ms > 0
              ? 1000.0 * result.completion_tokens / result.decode_ms
              : 0.0)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens
      << " draft_rounds=" << result.draft_rounds;
  if (result.component_cache)
    out << " cache_schema=component-cache-v1 cache_reuse="
        << result.component_cache->reuse
        << " cache_captures=" << result.component_cache->captures
        << " cache_captured_capacity_bytes="
        << result.component_cache->captured_bytes
        << " cache_global_ledger_bytes="
        << result.component_cache->ledger.total_bytes;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens;
  if (result.lookup_tokens > 0)
    out << " lookup_accepted=" << result.lookup_accepted_tokens
        << " lookup_proposed=" << result.lookup_tokens;
  if (result.ngram_rows > 0)
    out << " ngram_rows=" << result.ngram_rows
        << " ngram_dedup_rows=" << result.ngram_dedup_rows
        << " ngram_cache_hits=" << result.ngram_cache_hits
        << " ngram_disk_reads=" << result.ngram_disk_reads;
  if (result.cache_snapshot_bytes > 0)
    out << " cache_snapshot_bytes=" << result.cache_snapshot_bytes;
  if (result.cache_disk_queued_bytes > 0)
    out << " cache_disk_queued_bytes=" << result.cache_disk_queued_bytes;
  if (!result.cache_hit && !result.cache_miss_reason.empty())
    out << " cache_miss_reason=" << result.cache_miss_reason
        << " common_prefix_tokens=" << result.cache_common_prefix_tokens
        << " nearest_checkpoint_tokens=" << result.cache_checkpoint_tokens;
  return out.str();
}

/// Timed prefill counts work actually executed; usage counts the full prompt.
inline json::Value GenerationTimings(
    const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  timings["prompt_n"] = result.prefill_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] =
      result.prefill_tokens > 0
          ? result.prefill_ms / static_cast<double>(result.prefill_tokens)
          : 0.0;
  timings["prompt_per_second"] = PrefillTokensPerSecond(result);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] =
      result.completion_tokens > 0
          ? result.decode_ms / static_cast<double>(result.completion_tokens)
          : 0.0;
  timings["predicted_per_second"] =
      result.decode_ms > 0.0 ? static_cast<double>(result.completion_tokens) *
                                   1000.0 / result.decode_ms
                             : 0.0;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  timings["draft_rounds"] = result.draft_rounds;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  if (result.component_cache)
    timings["component_cache"] = ComponentCacheJson(*result.component_cache);
  if (result.lookup_tokens > 0) {
    timings["lookup_n"] = result.lookup_tokens;
    timings["lookup_n_accepted"] = result.lookup_accepted_tokens;
  }
  return timings;
}

/// llama-server `prompt_progress` object for streamed `return_progress`.
inline json::Value PromptProgressJson(
    const TextGenerationBackend::PromptProgress& progress) {
  json::Value value = json::Value::object();
  value["total"] = progress.total;
  value["cache"] = progress.cache;
  value["processed"] = progress.processed;
  value["time_ms"] = progress.time_ms;
  return value;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_
