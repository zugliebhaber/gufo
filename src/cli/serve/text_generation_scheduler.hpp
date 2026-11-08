#ifndef GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
#define GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class TextRequestPhase : std::uint8_t {
  kQueued,
  kAdmitted,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kTerminal,
};

inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 16384;
/// Floor for a budget adapted to the KV the pool can still hold. A shorter
/// chunk would trade the whole iteration for a handful of tokens.
inline constexpr std::size_t kMinimumAdaptedPrefillTokens = 512;
/// Automatic pending-queue bound: a burst as wide as the admitted batch is
/// never refused for lack of a queue slot.
inline constexpr std::size_t kDefaultPendingRequests = 16;
/// Free unified memory that must remain before the scheduler stops treating a
/// prefill chunk as free to grow: KV and activations draw from the same pool as
/// the model weights on Strix Halo.
inline constexpr std::size_t kDefaultMemoryReserveBytes =
    static_cast<std::size_t>(2) << 30;
inline constexpr std::size_t kDefaultMaxOutputBytes =
    static_cast<std::size_t>(1024) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytes =
    static_cast<std::size_t>(64) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytesTotal =
    static_cast<std::size_t>(256) * 1024;

struct TextPrefillPolicy {
  /// `max_num_batched_tokens`: the largest prompt chunk one iteration may
  /// prefetch between active decode rounds. The default is deliberately well
  /// above 8192: long-prompt prefill measured 5-11% faster at 4096-token chunks
  /// than at 512, and a larger budget also keeps more decode slots fed when the
  /// chunk has to yield.
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
  /// Caps `decode_active_tokens` by the KV the runner can still hold: on
  /// unified memory a prefill chunk occupies the same pool as the resident
  /// decode state, so a nearly full pool shortens the chunk instead of
  /// overrunning it. False keeps the configured budget unconditionally.
  bool adapt_to_kv_capacity{true};
  /// Floor for an adapted budget, and the smallest chunk the adaptation may
  /// grant. Zero selects the smaller of kMinimumAdaptedPrefillTokens and
  /// `decode_active_tokens`, so a deliberately tiny budget keeps its own scale.
  std::size_t min_batched_tokens{0};
};

/// What happens to a resident request whose session must be reclaimed so the
/// admitted batch stays within `max_num_seqs`.
///
/// Only recompute exists: on unified memory a swap moves the same bytes twice
/// and lengthens the critical path, while replaying the victim's tokens from
/// its prompt keeps throughput on the remaining batch. There is deliberately
/// no swap policy to select.
enum class TextPreemptionPolicy : std::uint8_t {
  kRecompute,
};

struct TextSchedulerPolicy {
  /// Internal cadence; idle monitoring performs no work during inference.
  std::chrono::milliseconds device_probe_interval{std::chrono::seconds(5)};
  /// Internal bound after which a still-queued stream reports its start, so
  /// transports can keep the connection alive while it waits.
  std::chrono::milliseconds stream_start_delay{std::chrono::seconds(5)};
  /// Zero selects `max(kDefaultPendingRequests, admitted batch size)`, so a
  /// burst as wide as `max_num_seqs` is queued rather than refused.
  std::size_t max_pending_requests{0};
  /// Zero selects `max_pending_requests`.
  std::size_t max_pending_requests_per_client{0};
  TextPreemptionPolicy preemption{TextPreemptionPolicy::kRecompute};
  std::size_t max_output_bytes_per_request{kDefaultMaxOutputBytes};
  std::size_t max_buffered_output_bytes_per_request{
      kDefaultMaxBufferedOutputBytes};
  std::size_t max_buffered_output_bytes_total{
      kDefaultMaxBufferedOutputBytesTotal};
  /// Free unified memory in bytes. KV and prefill allocations draw from it, so
  /// a low reading trims prefill chunks and lets a waiting prefill reclaim KV
  /// from the newest decoder. Empty disables the signal (headless hosts).
  std::function<std::optional<std::size_t>()> available_memory;
  /// Free memory that must remain for the rest of the system.
  std::size_t memory_reserve_bytes{kDefaultMemoryReserveBytes};
  std::chrono::milliseconds request_timeout{0};
  /// A scheduler that has work but starts no work unit and completes no
  /// pending capture for this long is reported as a lost device. Zero
  /// disables it.
  std::chrono::milliseconds stall_timeout{0};
  bool log_progress{false};
  /// Emit the vLLM-style periodic summary: a 10 s interval table and a
  /// once-per-minute completed-request line. Every table row is a delta from
  /// the previous row. The CLI enables it for serving.
  bool log_stats{false};
  /// Starts the resident bound at one and grows it with queued demand while
  /// memory headroom allows. Memory-blocked requests are deferred
  /// indefinitely instead of being failed.
  bool auto_sessions{false};
  /// Fill the decode cohort to this many residents before running a decode
  /// wave. Auto sessions default this to the resolved pool capacity so the
  /// hybrid plan can use the widest cohort; zero keeps the qualified
  /// interleaving for fixed-session servers unless the environment overrides
  /// it.
  std::size_t assemble_decode_rows{0};
};

/// Single-owner scheduler for opaque text-model runner states.
///
/// Submitters never execute model code. One scheduler thread owns admission,
/// runner leases, prefill/decode work units, cancellation, and reclamation.
class TextGenerationScheduler {
public:
  using Clock = std::chrono::steady_clock;
  using Result = TextGenerationBackend::Result;
  using CancellationCheck = TextGenerationBackend::CancellationCheck;
  using TokenCallback = TextGenerationBackend::TokenCallback;
  using ProgressCallback = TextGenerationBackend::ProgressCallback;
  using StartCallback = TextGenerationBackend::StartCallback;
  using SessionState = TextGenerationBackend::SessionState;

  struct RequestMetadata {
    std::string client_id{"anonymous"};
    std::optional<Clock::time_point> deadline;
    Clock::time_point request_start{Clock::now()};
    std::shared_ptr<const TextPromptContext> prompt_context;
    bool cache_prompt{true};
    /// False keeps this request's checkpoints out of the disk cache.
    bool cache_persist{true};
    std::size_t cache_prefix_tokens{0};
    bool stop_at_eos{true};
    std::vector<std::string> stop_sequences;
    bool return_progress{false};
  };

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] TextRequestPhase phase() const noexcept;

    /// Consumes queued output pieces on the calling thread and waits for the
    /// scheduler-owned request to become terminal. Streaming requests report
    /// only the latest prompt progress, always before their first piece.
    /// Their start is reported once at admission, or after
    /// `stream_start_delay` in the queue, before progress and pieces.
    Result Wait(const TokenCallback& on_token = {},
                const ProgressCallback& on_progress = {},
                const StartCallback& on_start = {});
    void Cancel() noexcept;

  private:
    friend class TextGenerationScheduler;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  explicit TextGenerationScheduler(std::shared_ptr<TextRunnerPool> runner_pool,
                                   TextPrefillPolicy prefill_policy = {},
                                   TextSchedulerPolicy scheduler_policy = {});
  ~TextGenerationScheduler();

  TextGenerationScheduler(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler& operator=(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler(TextGenerationScheduler&&) = delete;
  TextGenerationScheduler& operator=(TextGenerationScheduler&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  /// Maximum resident requests (`max_num_seqs`). Starts at `capacity()` and may
  /// be lowered at runtime.
  [[nodiscard]] std::size_t max_num_seqs() const noexcept;
  /// True while the scheduler trims prefill work because KV capacity or free
  /// unified memory limits the batch. Clears when the pressure does.
  [[nodiscard]] bool memory_pressure() const noexcept;
  /// Lowers or restores the resident-request bound without reloading a model.
  ///
  /// Shrinking below the resident count reclaims the newest requests: their
  /// model state is released and their tokens replayed later, because the
  /// configured preemption policy is RECOMPUTE. Throws std::invalid_argument
  /// for zero or for more than the preallocated `capacity()`.
  void SetMaxNumSeqs(std::size_t max_num_seqs);
  /// Set once a failed work unit finds the runner's device unusable; later
  /// submissions and failures report TextGenerationErrorCode::kDeviceLost.
  [[nodiscard]] bool device_lost() const noexcept;
  /// One entry per session; admitted requests hold one until terminal.
  [[nodiscard]] std::vector<SessionState> SessionStates() const;
  [[nodiscard]] std::size_t buffered_output_bytes() const noexcept;
  [[nodiscard]] std::size_t max_buffered_output_bytes() const noexcept;

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using TextRequestMetadata = TextGenerationScheduler::RequestMetadata;

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
