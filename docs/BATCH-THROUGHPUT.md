# Batch throughput levers

This is a ranked catalogue of **50 measures** for **1024 concurrent requests**,
not a claim that every item is implemented. Measures 1-21 are the measured
baseline and adopted/reverted experiments; measures 22-50 are the ranked next
candidates. Measured results, estimates and opt-in experiments are labelled
separately.

## Baseline and where the time goes

Reference run: Qwen3.8 Flash-Next UD-Q4_K_XL, 8k context, 256 sessions, 4 GiB
KV pool, cache off, unique ~95-token prompts, 16 output tokens, host
networking, 1024-socket client pool, 256-row hybrid decode plan.

| Prefill path | Wall | agg TG | agg PP | HTTP | Output quality |
| --- | ---: | ---: | ---: | --- | --- |
| Serial (batching disabled, reference) | 416.3 s | 39.4 tok/s | 233.5 tok/s | 1024/1024 | reference |
| Batched, automatic (default, cohort 32) | **209.9 s** | **78.1 tok/s** | **462.9 tok/s** | 1024/1024 | 256/256 identical to serial |

The serial run is prefill-bound: one ~95-token prefill pass costs about
0.3-0.35 s (fixed expert-weight read plus `t/1450`), so 1024 prefills take
roughly 320-360 s of the 416 s wall. Decode is the smaller part at 16 output
tokens; at 32 output the 256-row plan measured 61.0 versus 56.8 tok/s for 32
rows (+7.4 %).

The decode work adopted after this table (#28 wide MoE dispatch, #25 wide
expert tiles, #51 GDN state writeback, #52 wide Q8 projections on the
W8A8/F16 Dense route) brings the same 1024x16 run to **144.4 s / 7.09 req/s /
113.5 TG / 673.2 PP**, with 256/256 deterministic output. That is the current
record; the rows above are the prefill baseline that established the cohort
path.

Batched prefill is the default. The scheduler picks the cohort automatically:
the cap is 32 and a pass takes only requests already waiting to prefill, so the
actual cohort is `min(32, prefilling queue depth)`. It never waits for peers,
and a lone prompt falls back to the serial pass.

Serial chat prefill stops at the rendered prompt's stable-prefix boundary (23
tokens for these requests) and feeds the 7-token assistant framing in a second
pass. The batched path applies the same per-item boundary, so each cohort runs
two row-batched passes instead of one. A deterministic 256-prompt check
(temperature 0, reasoning off) matches the serial output **256/256** for cohort
8, 16 and 32. Wall time falls by **49.6 %** versus serial (416.3 s → 209.9 s),
with TG rising from 39.4 to 78.1 tok/s and PP from 233.5 to 462.9 tok/s.

The cohort-size curve at 1024x16 (same binary, cap overridden for the test;
measured before the decode-tile ladder, so the cap ordering is directional):

| Cohort cap | Wall | agg TG | agg PP | HTTP |
| ---: | ---: | ---: | ---: | --- |
| 8 | 231.5 s | 70.8 tok/s | 419.5 tok/s | 1024/1024 |
| 16 | 224.8 s | 72.9 tok/s | 432.0 tok/s | 1024/1024 |
| **32** | **215.0 s** | **76.2 tok/s** | **451.7 tok/s** | 1024/1024 |
| 64 | 235.9 s | 69.5 tok/s | 412.0 tok/s | 1024/1024 |
| 128 | 234.6 s | 69.8 tok/s | 414.2 tok/s | 1024/1024 |

32 is the measured optimum; 64 and 128 regress. With 8, 20 and 1 requests
actually open, the scheduler formed cohorts of 8, 20 and no cohort (serial)
respectively, and every response matched the serial reference.

Prefix-cache note: checkpoints created by a batched prefill are not yet
restored by the component cache, so repeated prefixes in a cache-on deployment
may re-prefill. The measurements above use cache off. Batching stays
default-on; the cache lookup is an independent follow-up.

The earlier failure was an arithmetic mismatch, not cross-request
contamination: a focused GPU probe compared `Session::Sync` with
`Model::PrefillBatch` and found the batched frontier logits differed already at
the first logit (`max_abs_diff=10.01`). A split probe (23+7 tokens) reproduced
the serial logit hash exactly (`16641625846117315637`), which located the
missing stable-prefix chunk boundary. With the boundary reproduced, the probe
reports bit-identical logits (`max_abs_diff=0`) for all 48 layers.

An earlier cohort A/B in this document set only `GUFO_PREFILL_COHORT`, which
at that time did not enable batching; the 2/4/8 results (416.3/415.5/421.9 s)
were three serial runs. That conclusion is withdrawn: batching is now
automatic, and the cohort follows the prefilling queue depth.

## Measures 1-21 (measured baseline)

| # | Measure | Bottleneck addressed | Expected effect at 1024 | Effort / risk | Status / evidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | Fix batched-prefill arithmetic and make batching automatic | ~80 % of wall is serial prefill | **-48.5 % wall, +94 % TG, +94 % PP** (measured) | Medium/high: stable-prefix chunking plus executor-wide cohort grouping | Adopted and default-on: 256/256 deterministic outputs for cohort 8/16/32; 1024×16 209.9 s / 78.1 TG / 462.9 PP |
| 2 | Raise the cohort cap above 8 | Fixed read still amortized across more prompts | +4.6 % TG from 16→32; 64/128 regress | Medium: scratch sized for `max_batch` rows | Adopted: cap 32 |
| 3 | Size the cohort from queue depth | Avoid forming small cohorts or waiting for peers | Two-pass optimum at 1024; shrinks automatically at low load | Low | Adopted: `min(32, prefilling queue depth)` |
| 4 | Mixed prefill+decode passes (chunked-prefill packing) | Prefill and decode alternate; each pass reads the experts once | +15-30 % (estimate): decode rows ride on the prefill pass | High: new executor pass shape | Not implemented |
| 5 | Prefill/decode overlap on two streams | Decode idles during prefill | +10-20 % (estimate); bandwidth-limited on one APU | High: ordering, cancellation, bandwidth | Not implemented |
| 6 | Prefix-cache boundary for shared system prompts | Every request prefills the shared prefix | 2-5× on shared-prefix batches; neutral on unique prompts | Medium | Observed `no_compatible_boundary` at `common_prefix_tokens=86` |
| 7 | Coalesce identical greedy requests | Duplicate requests prefill and decode separately | Up to 9.08× on exact duplicates | Low/medium: greedy only | Opt-in `GUFO_REQUEST_COALESCE=1`; measured |
| 8 | 256-row hybrid decode plan | Decode weight read per row | +7.4 % at 1024×32, +10.8 % at 256×128, neutral at 16 output | Low | Adopted |
| 9 | Cohort assembly before decode | Small decode waves | Unlocks `plan=batched-w256` | Low | Adopted for auto sessions |
| 10 | Base-state auto sessions + shared-KV admission | Full-context session charge | 96/150/256 resident sessions at 250k/128k/4k | Low | Adopted |
| 11 | Cache `ResourceClaim` per scheduler turn | `SharedKvPool()` calls `ResourceClaim()`, which runs `hipMemGetInfo` and host-memory reads, several times per turn | No gain measured (422.8 vs 416.3 s) | Low: snapshot once per turn | Tested at 1024×16; reverted |
| 12 | MTP drafting beyond eight rows | Decode tokens per pass | +10-20 % on decode-heavy runs (estimate) | High: acceptance and correctness | Aggregate flat with the 8-row gate |
| 13 | Decode GEMM/attention kernel work | ~125 tok/s decode phase | +3-8 % total at 32 output | High | Needs roofline/ISA profile |
| 14 | GPU-side sampling for wide cohorts | Host staging of 256 logits rows per step | +2-5 % decode | Medium | 256-row logits staging is ~155 MB |
| 15 | KV quantisation (FP8/INT8) | Attention/KV bandwidth | +5-15 % long context; small at 95 tokens | High: quality | Proposal |
| 16 | Phase-aware scheduling | Static interleaving | +3-8 % (estimate) | Low/medium | Not implemented |
| 17 | Event-loop HTTP instead of one thread per connection | 1024 OS threads | +1-3 % if CPU-bound | Medium | GPU is the usual limit |
| 18 | TCP backlog and client pool | Handshake resets | Stable 1024/1024; no GPU gain | Low | Adopted: backlog floor 1024, 10 ms accept poll; 0 errors with host networking |
| 19 | Pin GPU clocks | DPM ramp, compositor | +2-5 % sustained | Low: host root | Server warns `power_profile=auto` |
| 20 | Cold-batch cache policy | Capture/eviction churn | +1-5 % cold | Low | Use `--cache-ram-bytes off` for unique-prompt batches |
| 21 | Decode tile ladder and 8k+1 hold-back | Eight-row tail launches waste a full weight read | +1-5 % at 8k+1; neutral elsewhere | Low: scheduler defers one row one step; matrix kernel up to 48 | Adopted: 9/17/25/33 dips removed; 1024×16 improves to 209.9 s / 78.1 TG |

## Measures 22-52 (next candidates)

The first 21 measures removed the serial-prefill bottleneck and the 8k+1
decode-tile tail. At 1024x16 the default then took 209.9 s / 78.1 TG. The
routed-expert dispatch in #28 removed the duplicated expert-weight reads of
the 8-token GEMV groups and brought the same run to 177.0 s / 92.6 TG; the
decode-dominated 256x128 run dropped from 103.5 s / 114.3 TG to 81.0 s /
145.8 TG with unchanged output lengths. The GDN state writeback in #51 then
removed the scalar bf16 stores and reached 156.5 s / 104.7 TG at 1024x16 and
39.5 s / 103.7 TG at 256x16. The wide Q8 Dense route (#52) then reached
144.4 s / 7.09 req/s / 113.5 TG / 673.2 PP at 1024x16. Further gains come
primarily from decode weight bandwidth, kernel launches and speculation, not
from another prefill split.
The list below is ranked within the new block by expected **total** TG at
1024; workload-dependent items are marked.

| # | Measure | Bottleneck addressed | Expected effect at 1024 | Effort / risk | Status / evidence |
| ---: | --- | --- | --- | --- | --- |
| 22 | Wider decode plans beyond 256 rows | 256-row plan is the current executor ceiling; 1024 resident sessions leave decode width on the table | No total-TG gain measured; p50 improves | High: larger scratch, state capacity | **Tested/rejected (2026-11-07):** 512 sessions fails `request-state claim exceeds state capacity`; 320 rows at 4k loads but 207.6 s / 78.9 TG equals 256 rows (207.6 s / 78.9 TG); p50 155.9 -> 130.1 s. Cap stays 256 |
| 23 | Batched speculative verification across requests | Verification runs per request; a wide verify pass could share expert reads | +5-15 % TG if acceptance holds | High: MTP/DFlash acceptance and exactness | Partial: MTP gates; aggregate flat at the 8-row gate |
| 24 | Prompt-lookup/n-gram drafts in wide decode | Repetitive and agent prompts leave draft budget idle | +2-10 % TG on repetitive prompts | Medium: draft verification; no quality change | Proposal; n-gram cache exists |
| 25 | Decode weight repacking/layout for 256+ rows | Q4_K/Q8 rows and LDS tiles are tuned for 8-64 rows | +3-8 % decode | Medium/high: kernel rewrite and numerics | **Adopted (2026-11-07, tile layout):** wide decode MoE buckets now use the 64-row down and 128-row gate/up tiles instead of 16-row decode tiles. 1024x16 2x2 A/B against the #28 binary: 176.28 -> 174.59 s, 92.95 -> 93.85 TG (16,384 completion tokens both); 256x128: 81.48 -> 80.41 s, 145.0 -> 149.0 TG; 256/256 exact |
| 26 | Fused RoPE + attention wide decode | Separate rope, KV and attention launches per layer | +2-5 % decode | Medium: fused kernel, paged layouts | Assessed/rejected (2026-11-07): PrepareAttention already fuses RMSNorm+RoPE+KV store into one kernel; skipping all 12 full-attention layers at 256x16 saved 0.81 s of 43.97 s wall (~2 %), so an inner fusion is below the 5 % gate |
| 27 | Fused RMSNorm + HC mixer epilogue + residual | Small epilogue kernels and extra residual reads | +1-3 % | Medium: fused mixer arithmetic, bit-exactness | **Tested/rejected (2026-11-07):** routing the decode batch through the existing fused F16 HC combine/mixer path diverged on the 256-prompt gate (0/256 exact); reverted. The batch path already fuses combine+norm and the mixer epilogue in HcCombine/HcMixEpilogue |
| 28 | Fused MoE router/top-k/dispatch + shared-expert epilogue | Small router/dispatch kernels and repeated staging | +18.7 % TG measured (estimate was +1-3 %) | Medium: routing exactness | **Adopted (2026-11-07):** wide decode plans (>32 rows) now take the compacted routed-expert matrix route instead of 8-token GEMV groups, with the deferred epilogue settled before the batch combine. 1024x16: 210.2 -> 177.0 s, 78.0 -> 92.6 TG, 462 -> 549 PP (2x2 A/B); 256x128: 103.5 -> 81.0 s, 114.3 -> 145.8 TG, completion tokens 11,826 -> 11,804; 256/256 deterministic prompts exact |
| 29 | Expert-weight prefetch/double buffering | Expert reads stall between layers | +2-5 % decode | Medium: extra staging memory | Proposal |
| 30 | Hot-expert residency in L2/MALL | Skewed routing re-reads the same experts from DRAM | +2-5 % decode | Medium: cache hints, capacity | Proposal |
| 31 | Q8_1 activation reuse across projections | The same rows are quantized again for QKV and FFN | +1-2 % | Low/medium: shared staging | **Tested/rejected (2026-11-07):** one Q8_1 quantization shared by separate Q/K/V (and MTP Q/K/V) measured 206.73 s / 79.3 TG versus 207.43 s / 79.0 TG at 1024x16 and 104.77 s / 114.0 TG versus 104.20 s / 112.9 TG at 256x128; both inside run noise, 256/256 exact. Reverted |
| 32 | Fused attention output + residual + next norm | Extra roundtrip and launch after attention | +1-2 % | Medium: layer epilogue fusion | Assessed/rejected (2026-11-07): CombineBatch already folds residual+next norm+inject in one kernel next to the attention output projection; all 12 attention layers together cost 0.81 s of 43.97 s at 256x16 (~2 %), below the 5 % gate |
| 33 | Draft-vocabulary subset head | Full 248k-vocab head per row in decode | +1-4 % decode | Medium: draft-vocab exactness | Partial: draft subset exists for MTP |
| 34 | KV block-size tuning | 256-token blocks add fragmentation and read slack at short contexts | +1-3 % | Low/medium: pool layout, migration | **Tested/rejected (2026-11-07):** 128 vs 256-token blocks over a 2x2 A/B at 1024x16: 128 gives 206.78/206.75 s and 79.2/79.2 TG, 256 gives 206.22/206.86 s and 79.5/79.2 TG. Difference is run noise; block stays 128 |
| 35 | KV block defragmentation/contiguous active blocks | Scattered blocks slow paged attention | +0.5-2 % | Medium: pool compaction | Proposal |
| 36 | Runtime shared-prefix block dedup | Duplicate prefix blocks across concurrent requests | 2-5x on shared-prefix workloads; neutral unique | Medium: block sharing, COW | Partial: KV blocks share by reference; batched prefill still recomputes |
| 37 | Prefix-aware admission/grouping | Cohorts mix prefixes and miss shared-prefix reuse | +1-3 % shared-prefix | Low/medium: scheduler affinity | Partial: shared-prefix leader exists; no cohort affinity |
| 38 | Adaptive prefill chunk budget | Fixed chunk competes with active decode | +1-3 % total | Low: budget policy | Partial: budget adapts to KV/decoders |
| 39 | Output-length-aware batch packing | A long request tails a cohort of short ones | +2-5 % tail | Medium: fairness/latency policy | Proposal |
| 40 | Scheduler-side decode-plan/batch-buffer reuse | Plan object, request vector and lease metadata rebuilt every token | +0.5-1 % CPU | Low: scheduler bookkeeping; no KV/page/pool change | Proposal |
| 41 | HIP graph capture of the decode step | Launch overhead at 256 rows | +1-3 % decode | Medium/high: graph safety | Proposal |
| 42 | GPU logit masking for structured output | Host-side constraint/mask work per row | +1-4 % structured | Medium: mask exactness | Partial: constraint vocabulary exists |
| 43 | Structured-output grammar-state cache | Grammar state rebuilt per request/token | +1-2 % structured | Low/medium: state reuse | Partial |
| 44 | Idle-session KV host offload/eviction | Device pool pressure under oversubscription | +1-3 % at 1024; enables more sessions | Medium: transfer latency | Partial: idle spill exists |
| 45 | Async logits/sampling copies with pinned staging | Host synchronization per decode step | +0.5-1 % | Low: staging and ordering | Partial |
| 46 | Zero-copy/batched SSE writes | Per-token HTTP writes and copies | +1-2 % CPU-bound | Low: response buffering | Proposal |
| 47 | Parallel tokenization/detokenization | CPU serialization at high concurrency | +0.5-1 % | Low/medium: thread pool | Proposal |
| 48 | Stop-sequence/UTF-8 streaming filter optimization | Per-token string scans in the output path | +0.5-1 % | Low: filter fast path | Proposal |
| 49 | Build-level PGO/LTO + hipBLASLt plan cache | Branch/launch overhead and plan lookup | +1-3 % | Low/medium: build and cache | Proposal |
| 50 | Lower-bit expert/weight quantization (Q3_K/Q2_K) | Expert weight DRAM bandwidth | +5-15 % decode; quality risk | High: model quality | Proposal |
| 51 | GDN recurrent-state writeback vectorization | 32 scalar 2-byte bf16 state stores per thread at a 64-byte lane stride | +11.7 % TG measured | Low: bit-identical store width | **Adopted (2026-11-07):** four 16-byte bf16 stores per thread (`global_store_b128`); 1024x16 174.8 -> 156.5 s, 93.8 -> 104.7 TG, 555.6 -> 621.0 PP; 256x16 43.7 -> 39.5 s, 93.7 -> 103.7 TG; 256/256 exact |
| 52 | Wide Q8 projections on the W8A8/F16 Dense route | 32-row MMVQ tiles re-read each Q8 weight block eight times at 256 rows | +7 % requests/s measured | Low: route switch, output-identical gate | **Adopted (2026-11-07):** `DenseBatch` sends Q8 rows >48 through `Dense` (wider activation tiles, weight reuse). 1024x16 155.4 -> 145.3 s, 7.05 vs 6.59 req/s, 105.4 -> 112.8 TG, 625 -> 668 PP (2x2); 256x128 64.1 -> 57.1 s, 184 -> 208 TG; 256x16 38.9 -> 37.0 s; 64x128 neutral per token; shipped binary 144.4 s / 7.09 req/s; 256/256 exact |

**Scope note for #40.** This measure is CPU-side scheduler bookkeeping only.
It reuses the `TextExecutionPlan` and the per-step request/lease containers; it
does not pin KV pages, does not reuse cache slots, and does not change the
shared paged KV pool (`kv_pool_blocks_`, `Session::AttentionState::paged`). The
KV/paged measures are #15, #34-36 and #44.

**First ten to evaluate.** In order: #23 batched speculative verification,
#22 wider decode plans, #25 weight repacking/layout, #26 fused RoPE+attention,
#24 prompt-lookup drafts, #29 expert prefetching, #36 shared-prefix block dedup,
#39 output-length-aware packing, #30 hot-expert residency, and #50 lower-bit
expert weights (only behind the quality gate). #34-35/#44 move up for long
contexts, #42-43 for structured output, and #36-37 for shared-prefix traffic.

**Ranking logic.** The measured default is decode-heavy once the cohort prefill
is amortized: 1024x16 is 209.9 s / 78.1 TG, 1024x32 is 326.8 s / 99.9 TG, and
256x128 is 110.0 TG in ~12k generated tokens. Therefore weight bandwidth,
kernel launch/fusion and speculation rank above another prefill change.
Items 36-37 are high only for shared-prefix traffic; 34-35 and 44 rise with
context length; 42-43 only with structured output. All estimates require a
matched 1024-request A/B and the deterministic 256-prompt quality gate before
adoption.

## Cohort principle for TG

Prefill cohorts also synchronize decode waves. At 1024x32, serial prefill
finished one request at a time and the scheduler decoded in ~31-row batches;
automatic prefill cohorts made 32 requests decode-ready together and the
decode plan filled 256 rows.

| 1024x32 | Wall | agg TG | agg PP | TTFT p50 | Decode rows | Decode-rate estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial prefill | 575.0 s | 56.6 tok/s | 169.0 tok/s | 288 s | 31 | ~141 tok/s |
| Automatic cohort | **326.8 s** | **99.9 tok/s** | **297.4 tok/s** | **119 s** | **256** | **~178 tok/s** |

The total TG gain (+76.5 %) is mostly the faster prefill. Fixing the decode
width at 32 under automatic prefill keeps 94.4 TG (345.3 s); the 256-row decode
cohort itself therefore adds **5.8 % TG**. A decode-dominated run (256 requests
x 128-token budget, ~12k generated tokens) gives 68.5 -> **110.0 TG** (+60.6 %) and 142.4 -> **225.1 PP**,
with TTFT 106 -> 25 s and decode rows 48 -> 256.

Per-request decode speed falls (0.7 vs 4.4 tok/s at 1024x32) because 256
streams share the GPU; aggregate TG, wall and TTFT improve. The cohort
principle has a TG face: synchronize decode-ready requests and let the widest
qualified plan fill. The production 250k auto server already sizes that
assembly to its 96 sessions. The "decode-rate estimate" is the server's
per-request `decode_tps` times `batch_width`, not a wall-clock decomposition.

## Decode tile ladder and the 8k+1 tail

The decode GEMMs no longer pre-split a pass at eight rows. The MMVQ dispatcher
already carries 1-8 rows directly, 9-32 rows in 2/3/4 internal eight-row waves,
and 16/32/48 rows through the matrix kernel that reuses one weight block across
the tile. `DenseBatch` therefore issues one launch up to 48 rows (matrix tile
for 33-48, wave path otherwise), and 32-row launches above that.

A ready set of `8k+1` still opens a tile that is almost empty. The scheduler
now leaves that one row due for the next step and runs the current pass with
8/16/24/32 full tiles; the deferred request advances one step later. The
server logs at the 8k+1 test points contain only `batch_width=8/16/24/32`.

| Concurrency | Before | After | Delta |
| ---: | ---: | ---: | ---: |
| 8 | 213.7 PP/s | 213.6 PP/s | noise |
| **9** | **209.9** | **211.7-214.1** | **+1-2 %** |
| 16 | 252.6 | 259.1 | +2.6 % |
| **17** | **249.6** | **259.2** | **+3.8 %** |
| 24 | 267.3 | 271.5 | +1.6 % |
| **25** | **261.9** | **275.9** | **+5.3 %** |
| 32 | 280.5 | 285.3 | +1.7 % |
| **33** | **265.5** | **283.2** | **+6.7 %** |

The 1024x16 reference improves to 209.9 s, 78.1 TG, 462.9 PP, 1024/1024,
and the deterministic 256-prompt check remains 256/256. Widths 40 and 48 use
the same 48-token matrix tile. All ungated 9-48-row batches now take the
matrix kernel; only gated projections keep the eight-row cap, and the
scheduler's 8k+1 hold-back keeps those on full tiles as well.

## Why total throughput plateaus above a full wave

The supplied 1-96 concurrency curve (README) is consistent with **two requests
per concurrency slot and 128 generated tokens**: for concurrency >= 5,
`tot_tg_tps * wall_s / 128` is the request count and equals `2 * conc`. The
full-wave points are 32, 64 and 96 client concurrency:

| conc | requests | wall s | tot TG tok/s | req/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 70.84 | 115.6 | 0.90 |
| 64 | 128 | 105.81 | 154.8 | 1.21 |
| 65 | 130 | 112.04 | 148.5 | 1.16 |
| 96 | 192 | 157.55 | 156.0 | 1.22 |

The 64 -> 96 step adds 50 % requests and 49 % wall time for +0.8 % total TG:
the extra requests form another wave. The 64 -> 65 step adds two requests but
6.2 s of wall and -4.1 % total TG. The marginal throughput makes the fixed wave
cost visible:

- 64 -> 65: +256 generated tokens for +6.23 s = **41 tok/s marginal**, far
  below the 156 tok/s average; the third wave has started but is almost empty.
- 65 -> 66: +256 tokens for +0.90 s = **284 tok/s marginal**; filling the new
  wave is efficient again.

Two quantizations align at 65:

1. **Decode wave width.** The measured run's wave width is 64 requests/rows
   (consistent with a 64-session pool or a 64-row plan): 128 requests are
   exactly two full waves; 130 requests start a third wave that contains two
   requests but still pays a full prefill and decode pass. The total TG of a
   full wave is ~155-156 tok/s; a partial wave adds its fixed cost without its
   full work.
2. **Prefill cohort cap.** The cohort cap is 32. 128 requests are four full
   cohorts; 130 are four cohorts plus two requests, so a fifth expert sweep is
   paid for two requests. At 65 both boundaries move together, which is why the
   dip is larger than at 33 or 49.

The per-request rates fall as the wave is shared (`tg/s req` 29.9 -> 4.1), but
the aggregate is already at the per-wave bandwidth ceiling: the decode step
reads dense weights, routed experts, BF16 recurrent state and KV, and the
prefill pass is serialized on the same stream. The decode profile measured
~200 GB/s of the 240.76 GB/s gather ceiling in a c8 step; the prefill path is
at ~58 GB/s and is kernel-efficiency-bound, not DRAM-bound. More waiting
clients cannot add rows to a full wave; they only add waves and queue time.
That is why `req/s` stays at ~1.2 and `tot pp/s` stays at ~445 while `wall s`
grows.

There are two layers to the answer: the **ceiling** is the per-wave memory
bandwidth (~155-156 TG, ~445 PP in this run); the **staircase/dip** is the
64-request decode wave and the 32-request prefill cohort. The 96 point is not a
new ceiling: it is three full 64-request waves, which return to the same
per-wave throughput while the wall grows.

The fix is not more client concurrency; it is a **wider or faster wave**:
raise the resident pool/plan to the 96/150/256 auto points, use the wide-decode
kernels (#28, #25, #51, #52), and overlap prefill with decode. The current
instance logs `batch_width=96` and `sessions=auto(96)`, so its decode wave is
wider than the 64-request wave in this table; re-running the same load on it
moves the decode knee to 96 requests (client concurrency 48 at two requests per
slot). The per-wave ceiling remains, but a faster kernel raises the ceiling
itself.

## Dynamic prefill cohort

The scheduler derives the cohort from queue depth on every pass: cohort =
`min(32, prefilling queue depth)`, falling back to 1 when nothing else waits.
It only takes requests that are already queued, so the rule adds no wait for
peers. With 1024 queued requests the cap is reached; at low load the cohort
shrinks automatically. `GUFO_PREFILL_COHORT` still overrides the cap for
experiments, and `GUFO_BATCHED_PREFILL=0` is a diagnostics kill switch for
matched serial A/B runs; neither is required to enable batching.

2026-11-07 re-test on the #52 binary (1024x16, 4k/256): cohort cap 64 gives
160.4 s, 48 gives 171.1 s and 24 gives 158.7 s, against 144.4 s at the default
32. Decode assembly 192 and 128 gave 159.0 and 161.0 s against 156.4 s at 256
on the pre-#52 binary. The current optimum remains cohort 32 and assembly 256.
Merging the stable-prefix frontier into one pass is blocked by the frontier
arithmetic (the earlier attempt produced a 10.01 logit mismatch); prefill/decode
overlap needs partitioned scratch and a second stream, so it remains the
structural PP lever.

## Recommended order

1. **Batched prefill (#1).** Done and default-on. It is the only measured
   lever that moves the prefill-bound 1024-request case by tens of percent;
   the deterministic 256-prompt comparison passes for cohort 8/16/32.
2. **Cohort cap and dynamic sizing (#2, #3).** Done: cap 32,
   `min(32, prefilling queue depth)`, no opt-in.
3. **TG cohorts.** Keep the 256-row decode assembly; it adds 5.8 % TG under
   automatic prefill at 1024x32 (and the decode rate estimate rises from ~141
   to ~178 tok/s). The cost is lower per-stream decode speed, not wall time.
4. **Cheap wins:** #11 (one claim snapshot per scheduler turn), #19, #20.
5. **Architecture:** #4 mixed passes, then #5 overlap, once #1 has removed the
   serial-prefill bottleneck and the remaining split is measurable again.
6. **Keep #8-#10 and #18.** They are measured and adopted.

## Measurement rules

Keep prompt count, context, output budget, sessions, cache mode, networking and
power profile identical across A/B runs. Report wall time, aggregate PP/TG,
p50/p95, HTTP status, pool high-water and the cohort events actually logged
(`event=batched_prefill ... cohort=N` at `--log-level debug`). A change is
retained only when total TG
at 1024 requests improves outside run noise **and** the deterministic output
comparison against the serial path passes. Throughput without the quality
check is not evidence.
