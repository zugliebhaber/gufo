# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target
with shared-Q8_0 MTP; adaptive drafting. HTTP, greedy, thinking off.
Gufo `fec4759d` (October 18, 2026); llama.cpp `b11069` AR and `6fcaa16f` MTP
(September 22–23). Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1801.01 | 489.59 | +267.9% | 28.29 | 22.20 | +27.4% |
| 4,096 | 1727.95 | 455.24 | +279.6% | 28.27 | 21.21 | +33.3% |
| 8,192 | 1721.41 | 428.85 | +301.4% | 28.27 | 20.40 | +38.6% |
| 12,288 | 1710.64 | 400.21 | +327.4% | 28.21 | 19.64 | +43.6% |
| 16,384 | 1706.86 | 375.89 | +354.1% | 28.25 | 18.95 | +49.1% |
| 32,768 | 1672.60 | 301.31 | +455.1% | 28.18 | 16.54 | +70.4% |
| 65,536 | 1602.93 | 221.94 | +622.2% | 28.09 | 11.62 | +141.7% |
| 131,072 | 1557.18 | 144.78 | +975.5% | 27.85 | 7.98 | +249.0% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1734.96 ± 0.85 | 468.76 | +270.1% | 35.79 | 31.73 | +12.8% | 116.93 ± 0.10 | 48.12 | +143.0% |
| 4,096 | 1676.83 ± 1.73 | 423.79 | +295.7% | 37.64 | 34.48 | +9.2% | 116.41 ± 0.07 | 45.71 | +154.7% |
| 8,192 | 1669.91 | 394.98 | +322.8% | 36.91 | 32.20 | +14.6% | 114.08 ± 0.01 | 44.31 | +157.5% |
| 12,288 | 1660.54 ± 1.22 | 365.91 | +353.8% | 39.28 | 32.32 | +21.5% | 105.48 ± 0.15 | 43.53 | +142.3% |
| 16,384 | 1635.18 | 341.98 | +378.2% | 40.54 | 32.07 | +26.4% | 113.32 ± 0.23 | 43.66 | +159.6% |
| 32,768 | 1618.63 | 277.61 | +483.1% | 39.97 | 25.41 | +57.3% | 105.38 ± 0.17 | 39.28 | +168.3% |
| 65,536 | 1556.32 | 208.04 | +648.1% | 39.23 | 19.65 | +99.6% | 104.78 ± 0.21 | 27.75 | +277.6% |
| 131,072 | 1481.19 | 135.42 | +993.8% | 39.87 | 14.63 | +172.5% | 108.37 | 20.96 | +417.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.24 | 22.34 | +26.4% |
| 2 | 50.66 | 37.08 | +36.6% |
| 4 | 86.01 | 54.82 | +56.9% |
| 6 | 108.34 | 65.34 | +65.8% |
| 8 | 125.48 | 68.79 | +82.4% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.74 | 31.77 | +12.5% | 116.50 | 46.93 | +148.2% |
| 2 | 61.30 | 42.79 | +43.3% | 165.50 | 52.57 | +214.8% |
| 4 | 88.77 | 48.78 | +82.0% | 218.12 | 48.23 | +352.2% |
| 6 | 109.74 | 52.59 | +108.7% | 231.85 | 48.51 | +377.9% |
| 8 | 127.02 | 61.92 | +105.1% | 245.81 | 58.13 | +322.9% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 16.42 | 117.83 | +617.6% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 88.78 | 84.84 | -4.4% |
| 16K prefix, pp4096 + tg128 | 90.53 | 85.31 | -5.8% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## Batch scaling (November 2026)

Current build (AR, no MTP), `--sessions 32`, `--context 1024`,
`--prefill-chunk 16384`, 336-token unique prompts, 128 output tokens,
`--cache-ram-bytes off`, thinking off, one warmup plus two measured rounds per
level. `req` is the model-reported rate of one request, `agg` the rate over the
whole round (what a client sees), and `decode-phase` the aggregate output rate
divided by the non-prefill share of the round.

<!-- bench:concurrency -->
| Concurrency | req PP tok/s | req TG tok/s | agg PP tok/s | agg TG tok/s | agg total | prefill share | decode-phase | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 643.86 | 28.34 | 66.94 | 25.35 | 92.29 | 10.4% | 28.29 | 0.53 s |
| 2 | 639.19 | 22.12 | 97.82 | 37.15 | 134.97 | 15.3% | 43.86 | 1.03 s |
| 4 | 660.18 | 16.42 | 134.27 | 51.30 | 185.57 | 20.3% | 64.40 | 1.96 s |
| 8 | 669.61 | 11.57 | 173.00 | 66.10 | 239.10 | 25.8% | 89.13 | 3.86 s |
| 16 | 671.78 | 7.29 | 201.46 | 76.44 | 277.90 | 30.0% | 109.18 | 7.86 s |
| 32 | 666.35 | 4.33 | 217.79 | 83.54 | 301.33 | 32.7% | 124.10 | 16.07 s |
<!-- /bench -->

Per-request prefill is flat at 640-672 tok/s across the whole range, because one
prefill pass runs at a time; its share of the round therefore grows from 10.4%
to 32.7% while aggregate decode scales 25.35 -> 83.54 tok/s. Each doubling is
worth less: +46%, +38%, +29%, +16%, +8.4%. Run-to-run variance on the reference
host is 1-3%.

### 2026-11-05 re-measurement with the coalescing build

Same settings as the reference table (32-session server, context 1024,
prefill chunk 16384, 128 tokens, prompt repeat 6, cache off, 2 repetitions).
The request-coalescing path is default-off and the canonical prompts use
unique front-loaded nonces, so this is the unchanged decode path:

| Sessions | PP/s per stream (median, min-max) | PP/s total | TG/s per stream (median, min-max) | TG/s total | Sum | TTFT |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 8 | 783.0 (776-795) | 185.86 | 12.14 (11.99-12.28) | 71.22 | 257.08 | 3.30 s |
| 16 | 789.8 (784-797) | 227.16 | 8.10 (8.03-8.29) | 86.19 | 313.35 | 6.72 s |
| 24 | 788.7 (783-796) | 231.80 | 5.69 (5.63-5.88) | 87.89 | 319.69 | 10.27 s |
| 32 | 793.7 (788-804) | 246.36 | 4.76 (4.67-4.98) | 93.11 | 339.47 | 13.81 s |

Sum deltas: c8 -2.29%, c16 +1.46%, c24 -0.70%, c32 -0.93% - all inside the
documented run noise (c8 is the +/-8% point; c16-c32 are 1-2%). The canonical
unique-nonce workload does not trigger coalescing or prefix reuse. The new
potential for shared-prefix batches is measured separately: the shared-prefix
c32 cache-off vs cache-auto run goes from 94.67 to 119.94 output tok/s
(+26.7%) and from 13.87 s to 0.223 s TTFT, while the physical-token total
falls because cached prefill is logged as zero. Details in
`artifacts/serving/batch-pp-tg-levers.md` §23.

Fresh request-coalescing on/off on the same four levels (sum tok/s):

| Sessions | Off | On | Delta |
| ---: | ---: | ---: | ---: |
| 8 | 255.76 | 258.10 | +0.91% |
| 16 | 305.98 | 308.02 | +0.67% |
| 24 | 321.13 | 323.67 | +0.79% |
| 32 | 342.00 | 343.47 | +0.43% |

The unique nonces mean no group forms; the on-arm difference is within noise
and is not a regression. On a genuinely identical 8/16/24/32 batch, the
coalesced arm is 2.89x / 5.02x / 7.16x / 9.08x faster in wall/output terms;
details and the physical-token caveat are in
`artifacts/serving/batch-pp-tg-levers.md` §24.

### Width curve

Same binary and settings, 336-token prompts:

<!-- bench:width -->
| Batch width | agg total tok/s | change | agg TG tok/s | req TG tok/s | prefill share | decode-phase | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 243.87 | - | 67.71 | 11.38 | 22.5% | 87.34 | 3.29 s |
| 16 | 312.87 | +28.3% | 86.05 | 8.06 | 28.7% | 120.64 | 6.72 s |
| 24 | 323.00 | +3.2% | 88.42 | 5.74 | 29.6% | 125.56 | 10.27 s |
| 32 | 342.76 | +6.1% | 94.01 | 4.82 | 31.4% | 137.05 | 13.82 s |
<!-- /bench -->

Per stream versus total, same run (one stream = one request; each session
carried two requests in two waves):

| Sessions | PP/stream (tok/s) | PP total | TG/stream (tok/s) | TG total | total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 783.9 (780.4-794.7) | 176.2 | 11.38 (10.70-12.11) | 67.7 | 243.9 |
| 16 | 791.1 (765.6-802.5) | 226.8 | 8.06 (7.99-8.21) | 86.1 | 312.9 |
| 24 | 793.0 (787.6-799.9) | 234.6 | 5.74 (5.67-5.94) | 88.4 | 323.0 |
| 32 | 792.0 (787.7-800.8) | 248.8 | 4.82 (4.74-5.05) | 94.0 | 342.8 |

`PP/stream` is the model-reported rate of one prefill pass and is flat across
widths because one pass runs at a time; `PP total` is prompt tokens over the
whole round, so it carries the decode share. `TG/stream` is one request's decode
rate; `TG total` is aggregate output over the round.

A confirmation run on the next morning (same binary, same settings) reproduced
16/24/32 sessions within 1.3% (308.84 / 321.96 / 342.65 total; 7.86 / 5.71 /
4.80 TG per stream; 224.3 / 233.6 / 248.9 PP total). The 8-session point came out
7.9% higher (263.12 vs 243.87, TG per stream 12.31 vs 11.38), so that point is
the least stable of the four; the 16-32 range is the reproducible part.

Post-reboot run with the relinked binary (2026-11-04T18:59). The earlier sweep
in this section measured 231.61 / 277.14 / 294.00 / 305.19 on the same settings;
the host later wedged in an amdgpu SVM deadlock, and the fresh boot is 10-13%
faster at equal settings with identical code paths. The width-32 gain is
therefore reported against a same-binary control rather than against that older
absolute: width 32 versus a c32 two-wave control (16 sessions) is **+11.26%
total** and **+15.48% decode-phase**, with the prefill share 3.07 points worse
(31.38% vs 28.31%), so it still misses the strict gate's second condition.

### Prefill cost model

A prefill pass costs `~0.285 s + t/1450` at context 2048 (`0.275 s + t/1501` at
context 1024): the fixed part is about one full 66 GB expert-weight read at the
240.76 GB/s DRAM ceiling, the per-token part is compute-bound at ~7 TFLOPS
(15% of the 47.9 TFLOPS fp16 peak). Two requests in one experimental cohort
pass measured 2 x 702 tokens in 1157/1077 ms (1213/1303 tok/s) against 945
tok/s for the serial model at that length, i.e. **+28-38% prefill throughput**;
it stays opt-in (`GUFO_BATCHED_PREFILL=1`) because the cohort is not
bit-identical to serial prefill.

Raw artifacts and the full analysis live in
`artifacts/serving/batch-pp-tg-levers.md`:
`pp-tg-concurrency-1-32-ctx1024.json`,
`batch-levers-width-curve-ctx1024.json`, `prefill-c1-r6-ctx2048.json`.

The N1-N9 levers beyond A1-A4/B1-B4, their targets and their per-lever verdicts
are in [`N-LEVERS.md`](N-LEVERS.md) and the machine-readable
`artifacts/serving/n-levers-status.json`.
Width 32 is the only N-lever measured so far; the rest are blocked by the
amdgpu SVM deadlock described there.
