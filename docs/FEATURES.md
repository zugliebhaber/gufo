# Features added over upstream Gufo

This tree starts from upstream Gufo 0.28.0 (`version.txt`, `CHANGELOG.md`).
Upstream already provides continuous batching, chunked prefill, the component
cache and prefix reuse, MTP/speculative decoding, the OpenAI-compatible API and
the model coverage shown in the top-level README. Those are the baseline and
are not repeated here.

This page lists what this tree **adds or changes on top of that baseline**,
grouped by area, with the purpose, how to use it and where the evidence lives.
The inventory was checked against the upstream `v0.28.0` tree (new files,
`src/cache`, `kv_blocks`, the scheduler and the kernel sources), the
`16a3a024..working-tree` diff, and [PROGRESS.md](../PROGRESS.md). Where a
feature is a qualification or measurement of upstream behavior rather than a
code addition, the entry says so.

## At a glance

| Area | Feature | Purpose | Evidence |
| --- | --- | --- | --- |
| Serving | Recompute preemption | Reclaim KV from a preempted request without a swap buffer and keep the batch full | [PROGRESS](../PROGRESS.md) `TP-P1` |
| Serving | Dynamic prefill budget | Shrink one prefill pass to the free KV pool and defer instead of failing when it does not fit | [PROGRESS](../PROGRESS.md) `TP-P2`/`TP-P3`/`CP-P1` |
| Serving | Dynamic cohort prefill (default on) | Amortise the expert sweep over several cold prompts; the stable-prefix boundary keeps the deterministic gate exact | [BATCH-THROUGHPUT](BATCH-THROUGHPUT.md), [EXPERIMENTS](models/qwen3.8-flash-next/EXPERIMENTS.md) |
| Serving | Decode tile ladder and 8k+1 hold-back | Avoid the 9/17/25/33-row weight-read dips and reuse full 8/16/32/48 tiles | [BATCH-THROUGHPUT](BATCH-THROUGHPUT.md) |
| Serving | Wide decode plans up to 256 rows | Amortise dense projections and expert reads over a full resident cohort | [Flash-Next README](models/qwen3.8-flash-next/README.md) |
| Serving | `--sessions auto` | Resolve the session pool from measured state capacity, grow with demand, defer instead of failing | [SERVER](SERVER.md#auto-sessions) |
| Serving | Request coalescing (opt-in) | Let byte-identical greedy requests share one prefill and decode stream | [N-LEVERS](models/qwen3.8-flash-next/N-LEVERS.md#request-coalescing-opt-in) |
| Serving | Watchdog enabled by default | Turn a wedged device into a restartable `device_lost` instead of a silent hang | [SERVER](SERVER.md) |
| Serving | HTTP burst scaling | Accept and drain 256+ simultaneous connections without listener resets | [PROGRESS](../PROGRESS.md) |
| Cache | Paged attention and a shared KV pool | Bound KV memory with request block tables instead of a full private plane per session | [KV-CACHE-VISUAL](KV-CACHE-VISUAL.md) |
| Cache | Block sharing, adoption and copy-on-write tails | Serve shared prefixes without duplicating full blocks | [PROGRESS](../PROGRESS.md), [KV-CACHE](KV-CACHE.md) |
| Cache | Prefix-cache hit-rate metrics | Measure reuse per request and globally instead of inferring it from timing | [PROGRESS](../PROGRESS.md) |
| Cache | Lazy KV mapping and `--kv-pool-bytes` | Charge only mapped blocks and cap the pool against real demand | [KV-CACHE-VISUAL](KV-CACHE-VISUAL.md) |
| Cache | KV pool gauges | Expose capacity, live, free and high-water blocks in usage and logs | [PROGRESS](../PROGRESS.md) |
| Cache | KV exhaustion degrades to a host snapshot | Finish a round under a tight pool instead of failing requests | [PROGRESS](../PROGRESS.md) |
| Cache | Retention gating (`--cache-retain-cold`) | Do not retain unique cold prompts that cannot be reused | [PROGRESS](../PROGRESS.md) |
| Cache | Shared host-snapshot counters | Make cross-request snapshot dedup visible in usage metrics | [PROGRESS](../PROGRESS.md) |
| State | BF16 recurrent SSM state | Halve the per-session recurrent state while keeping F32 recurrence math | [PROGRESS](../PROGRESS.md) |
| Decode | Compacted MoE matrix dispatch for wide plans | Read each hot expert's weights once per bucket instead of once per eight-token group | [BATCH-THROUGHPUT](BATCH-THROUGHPUT.md) |
| Decode | Wide expert tile layout (64/128 rows) | Reuse expert weights over wide decode buckets | [BATCH-THROUGHPUT](BATCH-THROUGHPUT.md) |
| Decode | Vectorised BF16 GDN state writeback | Replace 32 scalar two-byte stores with four 16-byte stores per thread | [BATCH-THROUGHPUT](BATCH-THROUGHPUT.md) |
| Decode | Wide Q8 projections on the W8A8/F16 Dense route | Reuse each Q8 weight block across wider activation tiles instead of eight 32-row rereads | [BATCH-THROUGHPUT](BATCH-THROUGHPUT.md) |
| API | Model alias fallback | Keep OpenAI clients working when they send an alias or no model name | [PROGRESS](../PROGRESS.md) |
| API | Stop-sequence and text-span hardening | Avoid undefined behavior and lost text spans on edge inputs | [PROGRESS](../PROGRESS.md) |
| Diagnostics | `gufo diagnose` power-profile check | Make the clock state part of the measurement record | [ARCHITECTURE](ARCHITECTURE.md) |
| Tooling | `tools/bench/gather_bw.hip` | Separate sparse-gather from kernel inefficiency when calibrating DRAM bandwidth | [gather_bw.hip](../tools/bench/gather_bw.hip) |
| Tooling | Paging and FP8 KV benches | Measure block-table overhead and the FP8 memory/speed trade | [kv_paging_bench.hip](../tools/bench/kv_paging_bench.hip), [kv_fp8_attention.hip](../tools/bench/kv_fp8_attention.hip) |
| Tooling | Serving sweep tools | Sweep session count, chunk size and concurrency for a model | [serving_sweep.py](../tools/gufo/serving_sweep.py) |
| Tooling | Extended documentation validator | Check links, anchors, JSON blocks and Mermaid fences in CI | [README](README.md) |

## 1. Serving and scheduling extensions

Upstream batches resident decoders and exposes a pending queue and
`--prefill-chunk`. The additions below change how the batch is shaped and how
the server behaves under load.

### Recompute preemption

This tree makes recompute the only preemption policy: generated tokens are
appended to the preempted prompt, the sampler state (RNG and history) is
carried over, and the request is re-admitted ahead of new arrivals. There is
no swap path. The scheduler keeps the existing admission flow and reports the
reclaimed work in `preemptions` and `recompute_tokens`.

Purpose: keep the batch full and reclaim KV without reserving host swap space
for a paused request.

Evidence: `PROGRESS.md` `TP-P1`, `TextPreemptionPolicy::kRecompute`,
`text_generation_scheduler_test.cpp`.

### Dynamic prefill budget

The configured `--prefill-chunk` is an upper bound. The scheduler shrinks the
actual pass to what the free KV pool can hold and starts the next chunk only
when it fits. Admission charges each request's prompt plus requested output
against the pool; a request that does not fit stays queued and retries instead
of taking the whole pass or failing.

Purpose: a long prompt must not overrun the pool or starve the decoders, and a
short prompt must not wait behind a full-context prefill.

Evidence: `PROGRESS.md` `TP-P2`/`TP-P3`/`CP-P1`, `docs/SERVER.md`
("chunked prefill", "Auto sessions").

### Dynamic cohort prefill

Several cold prompts can share one prefill pass. The cohort cap defaults to
`min(32, prefilling queue depth)`; `GUFO_PREFILL_COHORT` overrides it for
experiments, and `GUFO_BATCHED_PREFILL=0` is a diagnostics kill switch. The
stable-prefix chunk boundary keeps the chunk arithmetic identical to the serial
frontier, so the deterministic 256-prompt comparison passes at cohort 8, 16 and
32.

Purpose: amortise the fixed expert-weight sweep over more prompt tokens and
raise prefill throughput without changing output.

Evidence: `docs/BATCH-THROUGHPUT.md` ("Dynamic prefill cohort"),
`docs/models/qwen3.8-flash-next/EXPERIMENTS.md`, `src/cli/serve/text_model_runner.cpp`.

### Decode tile ladder and 8k+1 hold-back

Decode projections use the MMVQ dispatcher's 1-32 row waves and the
16/32/48-token matrix tile instead of pre-splitting at eight rows. When the
ready-row count is `8k+1`, the scheduler holds one row back so the pass runs a
full tile and the next wave absorbs the tail.

Purpose: remove the measured 9/17/25/33-row throughput dips and reuse each
weight block across a full tile.

Evidence: `docs/BATCH-THROUGHPUT.md` ("Decode tile ladder"),
`docs/models/qwen3.8-flash-next/BENCHMARKS.md`, `batch.cpp` (`DenseBatch`).

### Wide decode plans up to 256 rows

Qwen3.8 Flash-Next advertises batched plans from two to 256 rows
(`kMaxBatchSessions`; the checked-in baseline caps it at 32). The hybrid plan
fills the widest cohort the resident sessions can supply: 96 rows on a 250k
auto server, 150 at 128k, 256 at short context. Dense, attention and shared-expert work use the full
plan; the routed Q4_K expert projection keeps its qualified chunking.
`GUFO_DECODE_MAX_ROWS` narrows the ceiling for A/B runs, and
`GUFO_ASSEMBLE_DECODE_ROWS` controls how many ready decoders a wave waits for.

Purpose: amortise dense weights and expert reads over the resident cohort; the
trade is lower per-stream decode rate for higher aggregate throughput.

Evidence: `src/models/qwen38_flash_next/config.hpp`,
`docs/models/qwen3.8-flash-next/README.md#batching-widths-and-limits`,
`docs/models/qwen3.8-flash-next/BENCHMARKS.md`.

### Auto sessions

`--sessions auto` sizes the LLM session pool after the model is resident. It
divides measured free state capacity by the per-session base state (block
table, index and recurrent state, not the shared KV rows) and keeps a 25 %
reserve. The scheduler starts with one resident session, grows toward visible
demand, returns to one when idle, and defers KV-blocked requests instead of
failing them. Fixed-session servers keep the old behavior; image, audio,
embedding and video servers do not accept `auto`.

Purpose: do not force the operator to choose between latency and aggregate
throughput before the workload is known.

Evidence: `docs/SERVER.md#auto-sessions`, `ARCHITECTURE.md` ADR-8,
`load_completed ... sessions=auto(N)`.

### Request coalescing (opt-in)

`GUFO_REQUEST_COALESCE=1` lets byte-identical greedy requests share one model
session: the leader prefills and decodes, followers receive the same token
stream. Eligibility requires identical prompt, sampling parameters and greedy
decoding.

Purpose: exact duplicate batch traffic (agent retries, fan-out, repeated
probes) can be served at a fraction of the work. Measured 9.08x wall/output on
32 identical requests; the canonical unique-nonce benchmark does not trigger
it.

Evidence:
`docs/models/qwen3.8-flash-next/N-LEVERS.md#request-coalescing-opt-in`,
`ARCHITECTURE.md` ADR-5, `text_generation_scheduler.cpp`.

### Watchdog enabled by default

Upstream ships `--stall-timeout-ms` but defaults it to 0 (off), so a wedged
device can leave a run silent. This tree defaults it to 180000 ms. A device
that stops heartbeating while generation is stalled is reported as
`device_lost` and the request ends; the server does not wait forever.

Purpose: make a wedged GPU a restartable failure instead of an
operator-visible hang.

Evidence: `docs/SERVER.md` ("stall watchdog"),
`docs/models/qwen3.8-flash-next/README.md`, `serve.cpp`.

### HTTP burst scaling

The listener uses a 1024-entry backlog floor (capped at `SOMAXCONN`), drains
the accept queue in a loop every 10 ms poll, and the loading responder uses the
same poll interval. The pending-request default in auto mode is sized from
`--max-connections`, not from the number of resident sessions.

Purpose: accept a full 256-request burst without kernel RSTs and without
turning demand into `429 queue_full`.

Evidence: `src/cli/serve/http_server.cpp`, `src/cli/serve/serve.cpp`; a
513/513 c256 run is retained in `PROGRESS.md`.

## 2. Paged KV cache and state

Upstream keeps the component cache and prefix reuse but has no paged KV block
pool. The paged path and its accounting are new.

### Paged attention and the shared KV pool

KV rows live in one pool of 128-token blocks (`kKvBlockTokens = 128`). Each
session owns a logical-to-physical block table; the executor addresses K and V
through that table instead of a contiguous private plane. The pool replaces the
old per-session planes at the same byte count.

Purpose: make block sharing, prefix adoption and a bounded pool possible
without copying full contexts.

Evidence: `docs/KV-CACHE-VISUAL.md`, `docs/KV-CACHE.md`,
`src/models/qwen/kv_blocks.hpp`, `PROGRESS.md` `PA-P1`/`PA-P3`.

### Block sharing, adoption and copy-on-write

A full block can be adopted by reference (`KvBlockTable::Adopt`) instead of
copied. The partial tail below 128 tokens is copied into a private block and
written copy-on-write. `BeforeWrite` invalidates share records before any
mutation, and a restore re-maps the table so an adopted block cannot leak into
a later request. The free list is a FIFO/LRU ring, so the oldest free block is
reused instead of the most recently released one.

Purpose: shared prefixes cost one physical copy, not one per request, and a
just-released block is not overwritten by the next capture before it can be
adopted.

Evidence: `PROGRESS.md` `PA-P4`/`KV-P1`/`KV-P2`/`B3`,
`docs/KV-CACHE-VISUAL.md`, `tests/models/qwen/kv_blocks_test.cpp`.

### Prefix-cache hit-rate metrics

Per request, `cache_hit_rate` and `cached_tokens` report reuse; globally
`llamacpp:prefix_cache_hits_total`, `llamacpp:prefix_cache_misses_total` and
`llamacpp:prefix_cache_hit_rate` expose it from `/metrics`. Counters advance
once per completed request, including cancellations, with no double count.

Purpose: measure cache value directly instead of inferring it from wall time.

Evidence: `PROGRESS.md` `KV-P3`, `docs/SERVER.md`.

### Paged attention row-address hoisting

The prefill attention kernel resolves the block base once per key block and
reuses it for the K and V passes instead of resolving per key and per plane.

Purpose: remove the paged-address overhead that grew with context depth;
measured dense paging overhead fell from 22.6 % to 2-4 %.

Evidence: `PROGRESS.md` ("Paged Attention: Zeilenadressierung pro Block
gehoisted").

### Lazy KV mapping and `--kv-pool-bytes`

A session maps blocks on write (`EnsureKvCapacity`), not the whole context at
start. The pool size comes from `--kv-pool-bytes N|off`: `0` (default) derives
it from free device memory with a reserve, capped at `sessions x context`;
`off` keeps one full-context session as the floor. The block size is 3 MiB per
128-token block on Flash-Next.

Purpose: a 2k-token conversation should not hold a full 4k/8k private plane;
the operator gets a hard device-memory bound.

Evidence: `docs/KV-CACHE-VISUAL.md`, `docs/KV-CACHE.md`,
`PROGRESS.md` ("Lazy-Mapping ..."), `serve.cpp` (`--kv-pool-bytes`).

### KV pool gauges

`KvPoolSnapshot` and the component-cache gauges expose capacity, live, free,
high-water blocks and block bytes under usage
(`usage.gufo.component_cache.global_kv_pool`) and in `kv_blocks_shared` log
lines (`pool_capacity`, `pool_high_water`).

Purpose: make the reservation verifiable against the actual peak and support
working-set sizing.

Evidence: `PROGRESS.md` ("KV-Pool-Buchhaltung sichtbar gemacht"),
`src/models/qwen/kv_blocks.hpp`.

### KV exhaustion degrades to a host snapshot

When a request cannot get device KV blocks, the scheduler keeps the completed
prefill as a host snapshot, releases device state, and retries admission after
residents finish. Log events `kv_exhaustion_degraded` and
`kv_exhaustion_deferred` record the path. A 700 MB pool under a 16-session load
completed 32/32 requests with 37 degradations and no device loss.

Purpose: a tight pool should cost latency, not correctness.

Evidence: `PROGRESS.md` `D6`, `docs/SERVER.md`.

### Retention gating for cold/unique traffic

`--cache-retain-cold` opts into retaining cold snapshots. Without it, the
`ColdRetentionGate` refuses retention unless there is evidence of reuse:
a hit, a waiting peer sharing the prefix, a repeated/extended prompt, or the
explicit flag. Unique cold traffic therefore retains no payload; the measured
ledger growth fell from 935 MB to 0.35 MB (2706x).

Purpose: stop unique single-turn prompts from filling the RAM ledger.

Evidence: `PROGRESS.md` `B4`, `tests/cache/retention_test.cpp`,
`docs/SERVER.md`.

### Shared host-snapshot counters

`cache_shared_bytes` and `retained_payload_bytes` report bytes restored from a
retained host checkpoint and the process-wide retained payload. A 16-client
identical-prefix run reports `unique / (clients x payload) = 0.109`.

Purpose: prove that concurrent same-prefix sessions share host snapshots
instead of duplicating them.

Evidence: `PROGRESS.md` `D3`, `docs/SERVER.md`.

### BF16 recurrent SSM state

`SsmState` is `hip_bfloat16`; upstream keeps it in F32. The GDN kernels load
the state into F32 registers, run the delta-rule recurrence in F32 and store the
result as BF16; rollback snapshots stay F32 because they are read once per
restore. Device state per session falls from 113,246,208 B to 56,623,104 B
(-56.6 MB, -50 %).

Purpose: the recurrent state is per-session and dominates the session budget
next to KV; halving it allows more sessions or longer contexts. The measured
quality delta stays inside the declared tolerance (KL mean, top-1 and NLL).

Evidence: `PROGRESS.md` `A1`,
`docs/models/qwen3.8-flash-next/QUALITY.md`,
`src/models/qwen38_flash_next/kernels/rocm/kernels.hpp`.

### FP8 KV prototype (not adopted)

A standalone E4M3 prototype halves KV bytes (2048 -> 1040 B/token) with cosine
0.9996 but costs 1.6-3.1x attention time on gfx1151 because E4M3 has no native
MMA path. It stays outside the serving path.

Purpose: quantify the memory/speed trade before committing to a KV-format
change.

Evidence: `PROGRESS.md` `D4`, `tools/bench/kv_fp8_attention.hip`.

## 3. Wide decode and expert execution

Upstream's decode expert path uses the grouped vector kernel. This tree adds a
wide-plan route and two memory-layout fixes on top.

### Compacted MoE dispatch for wide decode

Wide plans (>32 rows) delegate the routed experts to the compacted
expert-matrix route instead of the eight-token GEMV groups. The deferred WMMA
epilogue is settled before the batch combine. The routing in the 256-request
benchmark collapses onto the same ten experts, so the old path read each hot
expert's weights once per eight-token group; the compacted route reads them
once per 64/128-row bucket.

Measured (2x2 A/B, 1024x16): 210.2 -> 177.0 s, 78.0 -> 92.6 TG, 462 -> 549 PP.
At 256x128: 103.5 -> 81.0 s, 114.3 -> 145.8 TG. The deterministic 256-prompt
check is 256/256.

Purpose: decode on this model is routed-expert bandwidth bound; grouping the
hot experts is the largest single throughput lever found in this work.

Evidence: `docs/BATCH-THROUGHPUT.md` measure #28,
`src/models/qwen38_flash_next/kernels/rocm/batch.cpp`.

### Wide expert tile layout

Wide decode uses the 64-row down-projection tiles and 128-row paired gate/up
tiles instead of the 16-row decode tiles. Measured (2x2 A/B, 1024x16):
176.28 -> 174.59 s, 92.95 -> 93.85 TG; at 256x128: 145.0 -> 149.0 TG; 256/256
exact.

Purpose: read each hot expert's weights once or twice per layer rather than
once per 16 rows.

Evidence: `docs/BATCH-THROUGHPUT.md` measure #25,
`src/models/qwen38_flash_next/kernels/rocm/executor.cpp` (`RouteHints`).

### Vectorised BF16 GDN state writeback

The GDN kernel wrote the recurrent state back as 32 scalar two-byte stores per
thread at a 64-byte lane stride. It now packs eight BF16 values with the same
round-to-nearest-even encoding into one `uint4` (16 B), so the writeback emits
four `global_store_b128` instead of 32 `global_store_d16`.

Measured (2x2 A/B, 1024x16): 174.78 -> 156.48 s, 93.75 -> 104.7 TG,
555.6 -> 621.1 PP. At 256x16: 43.71 -> 39.52 s, 93.7 -> 103.7 TG. 256/256
exact; the change affects store width only.

Purpose: the recurrent state is the dominant per-step traffic after the expert
fix; the scalar stores were leaving most of the DRAM bandwidth unused.

Evidence: `docs/BATCH-THROUGHPUT.md` measure #51,
`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` (`StoreSsmState8`).

### Wide Q8 projections on the W8A8/F16 Dense route

At 256 rows the Q8_0 projections used 32-row MMVQ tiles, so each weight block
was re-read eight times per step; the 151k x 2560 vocabulary head was the
largest case. `DenseBatch` now sends Q8_0 rows above 48 through the W8A8/F16
`Dense` route, which tiles activations over wider blocks and reuses each weight
block across more token rows. Rows up to 48 keep the MMVQ tiles.

Measured (2x2 A/B, 4k/256, 256/256 exact): 1024x16 155.4 -> 145.3 s,
6.59 -> 7.05 requests/s, 105.4 -> 112.8 TG, 625 -> 668 PP; 256x128
64.1 -> 57.1 s, 184.3 -> 208.0 TG; 256x16 38.9 -> 37.0 s. The shipped binary
confirms 144.4 s / 7.09 requests/s / 113.5 TG / 673.2 PP. At 64 rows x 128
tokens the route is neutral per token.

Purpose: the decode step is weight-bandwidth bound; reading each Q8 block once
per wider activation tile instead of once per 32 rows is a direct throughput
gain with no output change on the deterministic gate.

Evidence: `docs/BATCH-THROUGHPUT.md` measure #52,
`src/models/qwen38_flash_next/kernels/rocm/batch.cpp` (`DenseBatch`),
`docs/models/qwen3.8-flash-next/BENCHMARKS.md`.

## 4. API and robustness fixes

These are correctness/compatibility changes on top of the upstream API, not new
endpoints.

- **Model alias fallback.** Upstream rejects an unknown or missing `model` name
  with 404. This process serves one loaded text model, so a known alias is
  echoed and a missing, empty or unknown name resolves to the loaded id.
  Purpose: OpenAI clients that send a different alias keep working. Evidence:
  `resolve_model` in `text_generation_backend.hpp`, `http_server_test`.
- **Stop-sequence and text-span hardening.** An empty stop-filter match throws
  instead of dereferencing `end()`; rewritten text spans swap into a fresh
  vector before appending so a moved-from vector is never reused. Purpose:
  remove edge-case undefined behavior and lost spans. Evidence:
  `stop_sequences.hpp`, `openai_chat.cpp`, `openai_chat_test`.
- **Artifact validator hardening.** Diagnostic artifact parsing uses bounded
  integers, `string_view` keys and explicit type errors. Purpose: never abort a
  diagnostics run on a malformed artifact. Evidence:
  `src/core/diagnostics/artifact_validator.cpp`.
- **Tool-parameter interpretation.** The upstream schema check is retained; the
  parameter bridge now validates string/number/boolean alternatives against the
  resolved schema before emitting an argument. Purpose: fewer schema-rejected
  tool calls. Evidence: `openai_chat.cpp`, `openai_chat_test`.

## 5. Diagnostics, benchmarking and documentation

- **`gufo diagnose` power-profile check.** The report reads the DRM
  `power_dpm_force_performance_level` and marks `high` as a reproducible
  baseline and `auto` as advisory. Purpose: make the clock state part of every
  measurement record. Evidence: `src/cli/diagnose/diagnose.cpp`,
  `src/core/diagnostics/linux_sysfs.cpp`, `ARCHITECTURE.md` ADR-7.
- **Decode-gather bandwidth probe.** `tools/bench/gather_bw.hip` reads sparse
  expert blocks and the same byte count contiguously with the same grid shape.
  It measured 231-232 GB/s for both, which rules out the gather pattern as the
  decode limit. Purpose: separate kernel inefficiency from the memory ceiling.
  Evidence: `tools/bench/gather_bw.hip`.
- **Paging and FP8 benches.** `tools/bench/kv_paging_bench.hip` measures the
  block-table overhead; `tools/bench/kv_fp8_attention.hip` is the FP8 KV
  prototype. Purpose: keep cache-layout experiments reproducible.
- **Serving sweep tools.** `tools/gufo/serving_sweep.py` and
  `tools/serving/gufo-serving-sweep.py` sweep session count, chunk size and
  concurrency for a model. Purpose: find the batch knee without a bespoke
  harness. Evidence: the two scripts and `PROGRESS.md` `TP-P1`.
- **Batch-throughput catalog.** `docs/BATCH-THROUGHPUT.md` ranks 52 measures
  with retained/rejected status and measured numbers. Purpose: keep optimization
  decisions evidence-based and prevent re-testing rejected ideas.
- **Documentation validator.** `tools/ci/check-docs.py` now checks relative
  links, anchors, JSON code blocks and Mermaid fences/kinds. Purpose: fail CI
  on stale docs. Evidence: `docs/README.md#documentation-checks`.
- **New documentation.** `ARCHITECTURE.md`, `KV-CACHE-VISUAL.md`,
  `N-LEVERS.md`, `BATCH-THROUGHPUT.md`, `paged-kv.md`, this page and
  `PROGRESS.md` document the design, the page tables, the measured levers and
  the negative results.

## 6. Upstream foundation (not repeated here)

The following are present in upstream Gufo 0.28.0 and are not claimed as new:
batched decode with the pending-queue/admission machinery, the
`--prefill-chunk` option, the component cache (`src/cache`), prefix
checkpoint/reuse, MTP/DFlash2/DSpark speculation, the OpenAI-compatible API
(chat, responses, completions, embeddings, audio, images, video), the image
input bounds (20 MiB, 32 MP), JSON-schema validation for tool calls, and the
model coverage and kernels listed in the top-level README. This tree measures,
qualifies or extends them; the model `BENCHMARKS.md` and `QUALITY.md` files
retain the numbers.

## Measured results (retained)

The numbers below use the same 4k/256 session setup, unique prompts and cache
off; each is a 2x2 alternating A/B on the same binary family.

| Workload | Before | After | Delta |
| --- | ---: | ---: | ---: |
| 1024x16, #28 wide MoE dispatch | 210.2 s / 78.0 TG / 462 PP | 177.0 s / 92.6 TG / 549 PP | -15.8 % wall, +18.7 % TG |
| 1024x16, #25 wide expert tiles | 176.28 s / 92.95 TG | 174.59 s / 93.85 TG | -1.0 % wall, +1.0 % TG |
| 1024x16, #51 GDN state writeback | 174.78 s / 93.75 TG / 555.6 PP | 156.48 s / 104.7 TG / 621.1 PP | -10.5 % wall, +11.7 % TG |
| 256x16, #51 on top of #28/#25 | 43.71 s / 93.7 TG | 39.52 s / 103.7 TG | -9.6 % wall, +10.7 % TG |
| 256x128, #28 | 103.5 s / 114.3 TG | 81.0 s / 145.8 TG | -21.7 % wall, +27.5 % TG |
| 1024x16, #52 wide Q8 route | 155.4 s / 6.59 req/s / 105.4 TG | 145.3 s / 7.05 req/s / 112.8 TG | -6.5 % wall, +7.0 % req/s |

The deterministic 256-prompt comparison is 256/256 for every adopted change.
The latest 1024x16 run with the shipped binary is **144.4 s / 7.09 req/s /
113.5 TG / 673.2 PP**; 256x128 improves from 64.1 to 57.1 s (184.3 -> 208.0 TG)
and 256x16 from 38.9 to 37.0 s.

## Tested and not adopted

Retained so the same idea is not re-tested without new evidence:

- **KV block size 256** (measure #34): 128 and 256 blocks are identical within
  noise; block stays 128.
- **32-token KV blocks** (measure D5): paged overhead is 1.3-2.1x higher than
  the 128-token block; block stays 128.
- **Q8_1 activation reuse** (measure #31): sharing one quantization across
  Q/K/V measured inside run noise; reverted.
- **Wider plans beyond 256 rows** (measure #22): 512 sessions exceeds the
  state capacity; 320 rows loads at 4k but has no total-TG gain over 256.
- **Fused decode HC-mixer route** (measure #27): the existing fused F16 route
  diverges on the 256-prompt gate (0/256 exact); the batch path keeps the
  separate combine and mixer.
- **FP8 KV cache** (measure D4): 1.97x smaller but 1.6-3.1x slower without a
  native FP8 MMA path.
- **Larger prefill cohorts / smaller decode assembly** (2026-11-07): cohort
  caps 64/48/24 measured 160.4/171.1/158.7 s against 144.4 s at 32; assembly
  192/128 measured 159.0/161.0 s against 156.4 s at 256 on the pre-#52 binary.
  The prefill/decode split stays at
  32 cohorts and 256 assembly. Merging the stable-prefix frontier into one pass
  is blocked by the frontier arithmetic (earlier 10.01 logit mismatch);
  prefill/decode overlap needs partitioned scratch and a second stream.

## Where to go next

- [PROGRESS.md](../PROGRESS.md): chronological engineering log with the
  original measurements and rejected alternatives.
- [BATCH-THROUGHPUT.md](BATCH-THROUGHPUT.md): the 52-measure catalog and current
  throughput baseline.
- [ARCHITECTURE.md](ARCHITECTURE.md): request flow, prefill/decode, MoE, cache
  tiers, coalescing and ADRs.
- [KV-CACHE.md](KV-CACHE.md) and [KV-CACHE-VISUAL.md](KV-CACHE-VISUAL.md):
  cache semantics, block pool, page tables and copy-on-write.
- [SERVER.md](SERVER.md) and [CLI.md](CLI.md): the flags and contracts for every
  feature above.
- Per-model [README](models/README.md), `BENCHMARKS.md`, `QUALITY.md` and
  `EXPERIMENTS.md`: retained model numbers and quality gates.
