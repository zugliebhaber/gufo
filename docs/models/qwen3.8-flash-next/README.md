# Qwen3.8 Flash-Next

> Part of the [model index](../README.md); see the [architecture](../../ARCHITECTURE.md), [CLI](../../CLI.md) and [README quickstart](../../../README.md#quickstart).

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [N1-N9 levers](N-LEVERS.md) · [ROCm 10.1 status](ROCM.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

### Serving settings at a glance

| Setting | Flag | Notes |
| --- | --- | --- |
| context per session | `--context N` | native 262144; above needs `--enable-yarn` |
| resident sessions | `--sessions N\|auto` | 1 = lowest memory/latency; 8-256 = wider decode plans; auto scales the plan with the resolved pool |
| prefill work unit | `--prefill-chunk N` | default 16384; capped by free KV |
| RAM prefix cache | `--cache-ram-bytes N\|off` | default auto; `off` for cold benchmarks |
| disk continuation | `--cache-disk DIR` | one process per directory |
| KV budget | `--kv-pool-bytes N\|off` | default free device memory, at most sessions x context |
| speculative decode | `--speculative mtp --mtp-model PATH` | single-user latency; keep off for exact AR A/B |
| draft width | `--draft-tokens N` | adaptive default; 1-7 proposals |
| reasoning | `--think on\|off\|auto`, `--reasoning-effort`, `--reasoning-budget` | thinking tokens count toward the response |
| request coalescing | `GUFO_REQUEST_COALESCE=1` | byte-identical greedy requests, opt-in |

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Eligible
requests draft over the first 98,304 token IDs plus a 1,024-row table containing
special tokens and other tokens already used by the request. The full head
remains the fallback when the anchor or two of the last 32 committed tokens
fall outside this vocabulary.

Single-session sampling, including the default thinking-on and thinking-off
presets, can use this smaller head with the existing acceptance/length
controller and AR backoff. Greedy requests, including penalties and
constraints, can use it in C1, when decoding alone and in jointly planned
batches; drafts ignore both, and verification applies them over the full
vocabulary. Greedy requests in unplanned multi-request batches retain the full
head.

The existing length controller and AR backoff apply to every request. The
target always verifies with its full vocabulary, including
exact sampled acceptance and residual correction. Sampled text can differ
across builds when proposals change. All-greedy C>1 batches retain their shared
planner and may use measured cycle costs. Each request keeps private caches,
rollback and RNG; snapshots preserve the learned acceptance history.
See [MTP qualification](QUALITY.md).

MTP decoding also proposes copies from the context, with one to eight sessions.
The context (committed tokens plus the chain so far) is matched against the
session's own earlier tokens by its last three tokens; when the best of the 16
most recent occurrences matches at least 12 tokens back for greedy requests
or 24 for sampled requests, the tokens that followed it fill the rest of the
chain and drafting stops. Greedy requests can
bypass the neural head when a match supplies at least three copies. Sampled
requests first make a neural proposal, then look for a copy extending it;
a repeated prefix alone can poorly predict new code or field values.
Their copy sources end at the request's starting context. Earlier parts of
the same new answer are not indexed for sampled copying; they become eligible
context on the next request. Greedy requests can also copy their current answer.
The request's first speculative step skips lookup to avoid delaying its first
token and records the start of generated text. Later matches cannot extend
back into the prompt's chat wrapper; a common tool header alone must not
qualify as a long matching answer. Copies
are verified like any draft: greedy text is byte-identical, and sampled
requests treat each copied token as a point-mass proposal under the same
rejection sampling, so outputs keep the target distribution (a seeded sampled
stream differs from one without copies, because random draws are consumed at
different points). Shorter matches
propose nothing; at temperature 1 an ungated 3-token rule displaced accepted
MTP drafts and was slower on every workload measured. A recently accepted
passage can resume after an edit with a five-token match; fewer than half of
the copied tokens being accepted disables that shorter gate, including after
a full match. Copies stop at EOS when the request enables it.
The index is private to the session and rebuilt after reset or snapshot
restore. Snapshots retain the request boundary and copy-resume cursor.
Lookup cycles and their following catch-up are excluded from the
batch planner's neural cost measurements. The planner measures short neural
chains before probing longer ones after copying. In all-greedy batches mixing
copies with new text, copied suffixes are bounded to half the verifier or the
planner's chosen width, whichever is larger, to limit pauses for other requests.
All-copy batches retain full width; sampled proposal streams are unchanged.
Timings report
`lookup_n` / `lookup_n_accepted`.

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144. A larger
context needs static YaRN and `--enable-yarn`: `gufo serve llm` refuses to
start with `--context` above 262144 without it, and `gufo bench` refuses a
workload whose depth plus prompt or generation tokens exceeds 262144. With
the flag, the factor is context / 262144 (`--context 409600` gives 1.5625).
The flag is an error at or below 262144, where it would have no effect.
Static YaRN applies to every request on that server, including short ones,
and changes the output distribution more as the factor grows; in the first
measurements, multiple-choice accuracy moved within noise at factors 1.5625
to 3.875. See the
[measurements](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323981940),
the
[opt-in proposal](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323982386)
and the
[accuracy check](https://github.com/gufo-org/gufo/pull/350#issuecomment-6234093998).
No GGUF key enables YaRN.
Prefill chunks whose visible context exceeds 262144 tokens use the per-token
attention kernel rather than the fused WMMA kernel. Memory grows with used
context and selected rollback depth (about 27.5 KiB per context token with
MTP, measured on gfx1151); admission reserves the configured capacity before
creating sessions.

## Batching, widths and limits

Decode runs in waves of the model's batch width. This model advertises batched
plans from two to **256 rows** (`kMaxBatchSessions`). Auto sessions fill the
decode cohort up to the resolved count, hard-capped at 64; fixed pools can use
wider plans up to 256 rows. Wide plans (>32 rows) use the
compacted expert buckets for the routed Q4_K projection; narrow tails keep the
grouped vector path, and the dense, attention and shared-expert work uses the
full plan width. MTP drafting keeps its eight-row gate, which means a wider
cohort decodes autoregressively.

Width is a throughput/latency trade. The retained 8/16/24/32 cold table shows
24 rows at about 94 % of the 32-row aggregate rate while keeping noticeably
more per-request decode rate and lower TTFT. For wide decode loads, 256 rows
measured **84.8 tok/s** total TG versus **76.5** at 32 rows (+10.8 %) on
256 requests × 128 output, and **61.0** versus **56.8 tok/s** (+7.4 %) at 1024
requests × 32 output. See [wide decode
plans](BENCHMARKS.md#wide-decode-plans-2026-11-06).

The 2026-11-07 wide MoE dispatch, GDN state writeback and wide Q8 Dense route
supersede those numbers on the same 4k/256 setup: 1024x16 is **144.4 s /
7.09 requests/s / 113.5 TG / 673.2 PP** with 256/256 deterministic output. See
[the dispatch and state
section](BENCHMARKS.md#wide-moe-dispatch-and-gdn-state-writeback-2026-11-07),
[the wide Q8 route](BENCHMARKS.md#wide-q8-projections-on-the-w8a8-route-2026-11-07)
and [BATCH-THROUGHPUT](../../BATCH-THROUGHPUT.md).

Prefill runs one pass at a time. A pass pays a fixed full expert-weight read
plus a per-token part, so short prompts spend most of the pass on that fixed
cost. Cold prompts are batched automatically: the cohort cap is 32 and the
actual size is `min(32, prefilling queue depth)`, so a lone prompt still runs
the serial pass. The batched path reproduces the serial stable-prefix chunking:
256/256 deterministic prompts match serial for cohort 8, 16 and 32, and at
1024×16 the automatic cohort improves wall time from 416.3 s to 214.4 s (TG
39.4 → 76.4 tok/s, PP 233.5 → 453.3 tok/s). Prefix-cache checkpoints from a
batched prefill are not yet restored by the component cache; the measurements
use cache off. See
[batch throughput levers](../../BATCH-THROUGHPUT.md#baseline-and-where-the-time-goes).

With base-state auto sizing, the uncapped fit is 150 sessions at 128k and 96
at 250k on this host; the product hard cap makes the default auto pool 64
sessions, so the hybrid decode plan reaches at most 64 rows. This deployment
lowers the cap to 40 and therefore uses plans of at most 40 rows.

**Current operator profile (2026-11-08).** This deployment lowers the cap with
`GUFO_AUTO_SESSIONS_MAX=40`, runs a 17 GiB KV pool (742,656 tokens) and a 2 GiB
RAM cache, and leaves about 8 GB host memory. The 40 states cost
`40 x 303.75 MB ~= 12.2 GB`. Short requests fill 40 sessions; 43 short requests
run 40 active and 3 waiting. A 26k prompt plus the 4k output horizon costs
~30k tokens, so the 17 GiB pool admits about 24 long requests; 43 long requests
run 24 active and 19 waiting.
Admission charges each request's complete private prompt plus a bounded output
horizon (4096 tokens for an explicit `max_tokens`, 512 for unlimited) against
the shared pool capacity, not its lagging free counter. A live shared prefix is
charged once: followers reserve only their private tail and output. Full prefix
blocks are adopted by refcount; `gufo_kv_pool_shared_blocks` exposes the live
sharing. The retained 128k sweep saturates aggregate decode near 76 tok/s from
16 clients;
MTP raises single-stream decode from 28.8 to 47.9 tok/s without moving the
aggregate. See [auto sessions at
250k](BENCHMARKS.md#auto-sessions-at-250k-2026-11-06) for the current run and
[the earlier 128k
sweep](BENCHMARKS.md#auto-sessions-at-128k-2026-11-05).

### Cache, coalescing and the canonical benchmark

The canonical benchmark gives every request a **unique, front-loaded nonce**
and runs with `--cache-ram-bytes off`: it measures cold prefill and unique
decode rows. Two separate workloads exercise reuse:

- **Shared-prefix cache:** leave `--cache-ram-bytes` at auto (or use
  `--cache-disk` for restart-safe continuation). A shared-prefix c32 run
  measured 94.67 -> 119.94 output tok/s (+26.7%) and 13.87 -> 0.223 s TTFT
  when the cache was enabled; the physical-token sum falls because cached
  prefill is logged as zero. Lookup uses the component cache's radix-style
  compressed token-prefix tree; live followers also adopt the leader's full
  prefix blocks by refcount, and only the partial tail is copied.
  `gufo_kv_pool_shared_blocks` reports the shared blocks. Report output and
  TTFT for this workload.
- **Request coalescing:** `GUFO_REQUEST_COALESCE=1` shares one greedy stream
  between byte-identical requests. A 32-identical batch measured 44.410 ->
  4.891 s (9.08x; an earlier 264-token prompt gave 9.53x), 4096 identical
  output tokens. The canonical unique-nonce benchmark does not trigger it and
  its on/off difference is noise. See
  [N-LEVERS](N-LEVERS.md#request-coalescing-opt-in) for scope and limits
  (greedy only; no stops/logprobs/penalties; followers have no KV; leader
  cancellation ends the group).

Use the [batch table](BENCHMARKS.md#2026-11-05-re-measurement-with-the-coalescing-build)
for the fresh 8/16/24/32 numbers and [N-LEVERS](N-LEVERS.md) for the lever
verdicts.

The scheduler's stall watchdog is on by default: generation that makes no
progress for `--stall-timeout-ms` (default 180000) reports `device_lost` and
exits, instead of holding sessions forever.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
