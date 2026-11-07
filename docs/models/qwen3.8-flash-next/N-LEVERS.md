# N1-N9 batch throughput levers on gfx1151

> The levers below are the November 4-5 record. The later wide MoE dispatch
> (#28), wide expert tiles (#25) and GDN state writeback (#51) supersede their
> decode numbers: the same 1024x16 workload reached 144.4 s / 7.09 requests/s /
> 113.5 TG / 673.2 PP. See [BENCHMARKS](BENCHMARKS.md) and
> [BATCH-THROUGHPUT](../../BATCH-THROUGHPUT.md).

The levers below extend the A1-A4/B1-B4 set in
`artifacts/serving/batch-pp-tg-levers.md`
(sections 1-13). Every lever is either measured here or closed with the reason
it cannot be measured yet. The consolidated per-lever verdict lives in
`artifacts/serving/n-levers-status.json`
and its plain-text twin `n-levers-status.txt`; regenerate both with
`python3 artifacts/serving/n_levers_status.py`.

## November 5, 2026 update

- Fresh cold canonical 8/16/24/32: 255.76 / 305.98 / 321.13 / 342.00 tok/s.
  The same-binary coalescing on/off A/B is 342.00 vs 343.47 at c32 (+0.43%,
  noise; unique nonces form no group).
- Request coalescing: 32 identical requests 44.410 -> 4.891 s (9.08x;
  4096 identical output tokens); 8/16/24/32 is 2.89x / 5.02x / 7.16x / 9.08x.
  Canonical unique prompts do not trigger it.
- Shared-prefix cache: output 94.67 -> 119.94 tok/s (+26.7%) and TTFT
  13.87 -> 0.223 s; the physical-token sum falls because cached prefill is
  logged as zero. Report output/TTFT for that workload.
- Details: [BENCHMARKS](BENCHMARKS.md) and
  [ARCHITECTURE](../../ARCHITECTURE.md).

## Current state (November 4, 2026, after the reboot)

**The deadlock is gone.** The host was rebooted at 19:57 local: 0 D-state tasks,
120/125 GB free, `gufo diagnose` PASS, and the release binary relinked in two
steps, so the measurements below are from a working device.

What the deadlock was, because it will recur if a server wedges: three `gufo`
processes sat in D state inside `svm_range_cpu_invalidate_pagetables` (PIDs
3684305, 3692283, 3916108; the largest at 75.4 GB RSS) for about ten hours. It
wedged memory compaction with them (`kcompactd0`/`khugepaged` in D state), so
every 218 MB release link stalled in `folio_wait_bit_common` and leaked another
unkillable `ld`. `kill -9` does not clear a D-state task and the sysfs reset is
root-only; only the reboot cleared it. The classification mattered: it was *not*
the toolchain (a 16 KB hello-world linked in 0.03 s with the same LLD) and *not*
the disk (10.8 GB/s reads, 693 MB/s btrfs writes).

The relinked binary (2026-11-04T18:59) is the first one that contains the N6
code: `gufo diagnose` prints the `power_profile` warn and `gufo serve` logs
`power_profile=auto` before the model opens. The dpm pin itself still needs host
root (`power_dpm_force_performance_level` is root-owned; the write fails with
EACCES on the host and "read-only file system" in the container), so the N6 A/B
stays a runbook item and the numbers here are the honest `auto` baseline.

## Honest c32 total

| configuration | aggregate total | decode-phase | prefill share |
| --- | ---: | ---: | ---: |
| width-32 single batch | **343.19 tok/s** | 136.87 tok/s | 31.38% |
| c32 two-wave control (2 x width 16) | 308.47 tok/s | 118.52 tok/s | 28.31% |

Width 32 gains **+11.26% total** and **+15.48% decode-phase** against the same
binary's two-wave control, but its prefill share rises by **+3.07 points**
(31.38% vs 28.31%), so it still misses the strict lever-2 gate as written ("at
least 10% decode-phase without a worse prefill share"). The total gain is real;
the gate's second condition is not met. 343.19 tok/s is the honest c32 total for
this build: cache off, context 1024, 336-token unique prompts, 128 output
tokens, thinking off, dpm profile `auto` (not pinned).

The width curve on the same run is 243.87 / 312.87 / 323.00 / 342.76 tok/s for
8 / 16 / 24 / 32 sessions. The pre-reboot reference measured 231.61 / 277.14 /
294.00 / 305.19 at identical settings, i.e. the fresh boot is 10-13% faster with
unchanged code paths; that delta is host state (clocks, no memory pressure), not
the N6 code, and it is exactly the effect N6 wants to control for.

Per-request TG at c32 falls to 4.37 tok/s and TTFT rises to 16.1 s; the gain is
aggregate throughput, not latency.

## Lever verdicts

| lever | target (total) | code | verdict |
| --- | --- | --- | --- |
| N1 replica lanes | +20-25% | decision harness ready | blocked: the two-server control that would justify the second lane needs a working device; `PHASES=overlap` now prints the N1 verdict |
| N2 prefill pipelining | +10-15% prefill | not implemented | blocked: needs a double-buffered staging kernel and a profile |
| N3 expert-count sync removal | +3-7% | not implemented | blocked: needs a device-side tile map or a fixed grid, decided by measurement |
| N4 W8A8 expert GEMM | closed | not implemented | **closed by measurement**: the decode MoE already runs at 85-90% of the DRAM ceiling; the +25-52% premise assumed a 238-expert union (real: 113) |
| N5 GPU+CPU split prefill | +3-5% | not implemented | not measured: needs an AVX-512 expert slice and a numerics contract |
| N6 headless + power profile | +5-10% | **landed** | implemented, not measured: the dpm write needs host root |
| N7 trained draft (EAGLE) | +30-50% decode c1-c4 | not implemented | not measured: no draft-training pipeline in this tree |
| Request coalescing (opt-in) | 9.08-9.53x on 32 identical requests | **landed** (`GUFO_REQUEST_COALESCE=1`) | positive on byte-identical greedy batches; canonical bench uses unique nonces and does not trigger |
| N8 prefill kernel fusion | +5-10% prefill | not implemented | not measured: needs a launch trace |
| N9 F16 prefill activations | +3-4% | partly present | not measured: needs a numeric gate |

### N6: what landed in code

The profile decides whether a served measurement sees the driver's ramp-up and
the compositor's share of the iGPU, so the build now reports it without opening
the device:

- `gufo::diagnostics::ReadPowerProfile()` reads
  `/sys/class/drm/card*/device/power_dpm_force_performance_level` and skips the
  connector entries (`card1-DP-1`, ...).
- `gufo diagnose` emits a `power_profile` check: `pass` at `high`, `warn`
  otherwise, with the exact command to pin it.
- `gufo serve llm` logs a `power_profile=` warning before the model opens when
  the level is not `high`, so a throughput number always has its profile.

Unit coverage: `tests/diagnostics/gpu_queues_test.cpp` reads a fake DRM tree and
checks that the card directory wins over its connectors and that a tree without
DRM reports an empty profile. The rebuilt `gpu_queues_test` binary passes
(`gpu_queues_test: all checks passed`).

The `+5-10%` remains unmeasured because the write needs host root: sysfs is
root-owned and the build container's root maps to uid 1000. Headless operation is
a host runbook item for the same reason (no compositor control from here).

The measurement itself no longer needs new code. `lever_matrix.sh` has an `n6`
phase that A/Bs the *same binary* with `dpm=auto` and `dpm=high`: it saves the
current profile, pins `high`, restores the saved value through an `EXIT` trap so a
failed bench cannot leave the driver pinned, and scores the pair with
`artifacts/serving/compare_totals.py`. That comparator reuses the
`compare_decode_width.describe()` metric definitions, matches levels by name,
reports unmatched levels instead of dropping them, and applies the strict gate
(`>=+5%` total and no worse prefill share) per level. Both branches were exercised
against a fake DRM tree: a writable tree runs both arms and prints a verdict, a
read-only tree records the honest baseline and the `needs-host-root` verdict. The
comparator's own arithmetic was checked on synthetic arms — `+6.49%` total with
`-0.04` points of prefill share passes, `+8.13%` total with `+3.08` points fails.

`PHASES=n6` therefore turns N6 into one command after the reboot, with or without
root: without a writable sysfs it still produces the honest `auto` baseline.

## Design notes for the blocked levers

### N1 replica lanes

`TextGenerationScheduler` serializes prefill and decode on one HIP stream. The
cheap evidence for a second lane is two production servers on the same GPU: one
prefill-heavy (`--max-tokens 1`, long prompts), one decode-heavy (short prompts,
long completions). `artifacts/serving/two_server_overlap.py` measures serial vs
concurrent wall time and reports `serial / concurrent - 1`; both servers keep
their own executor, stream, scratch and KV pool, so its gain is the upper bound a
dual-lane in-process executor could reach. The hardware probe
(`artifacts/serving/overlap-probe.txt`) already shows 237 GB/s of read traffic
running concurrently with 49 TFLOPS of WMMA in 36 ms against 46 ms serial, i.e.
44% of the memory work hidden. Run the two-server control first; only a real gain
justifies the second `Executor` over the shared `DeviceModel`, its own scratch and
stream, a per-lane session subset, and the KV pool behind the existing mutex.
`PHASES=overlap` runs exactly that control, and
`artifacts/serving/compare_overlap.py` turns it into the verdict: a run passes
only with a median gain of at least +20% *and* both servers keeping at least 80%
of their solo throughput, so a "gain" bought by starving one arm does not count.
Synthetic artifacts cover pass, target miss, starvation and stream interference.

#### What the code already provides (checked 2026-11-04)

The per-session executor that N1 budgets as "hoch" is already there:
`ContinuationAdapter::CreateSlot` (`src/models/qwen/continuation_adapter.cpp`)
constructs **one `QwenGpuExecutor` per slot**, and `QwenGpuExecutor` owns its own
`QwenGpuArena` and `arena_.stream` (`src/models/qwen/hip/executor.hpp:625`). So
each admitted session already has an independent arena, scratch set and HIP
stream; nothing has to be built for the "eigener Executor/Scratch/Stream" half of
N1.

What is *not* there is concurrency at the scheduling level. `ForwardTokenBatch`
(`src/models/qwen/hip/batched_decode.cpp`) funnels a whole batch into the
**coordinator's** arena and stream, synchronising every other session's stream
first; prefill runs on the request's own slot executor, but the scheduler advances
requests one at a time. N1 therefore reduces to: dispatch two disjoint session
sets as two concurrent advances, each on its own coordinator arena. The remaining
work is thread-safety of the shared weight reads plus per-lane pinned staging —
not a new executor abstraction. That lowers N1's effort estimate from "hoch" to
"mittel".

#### Why c32 at long context is memory-infeasible on the dense path

Because each session owns an arena, the dense path pays `per_request_state_bytes`
(KV + recurrent state) *and* `temporary_scratch_bytes` (FFN/attention staging)
once per session. The admission check in
`src/cli/serve/text_model_runner.cpp` charges both per state, so a 32-session pool
at 2048 context asks for 32 complete scratch sets. Flash-Next avoids that
multiplier differently: it reports `temporary_scratch_bytes = 0` and instead
subtracts `DeferredScratchBytes()` from the capacity figure in `ResourceClaim`
(`src/cli/serve/inference_backend.cpp`), which reserves the shared scratch exactly
once. That is why the wide Flash-Next runs fit while the dense 27B pool at the
same width and context is rejected. A dense wide pool therefore needs either more
GTT headroom or the shared-scratch executor that N1 needs anyway.

### N2 prefill pipelining

A 336-token pass costs 0.499 s: 0.285 s of DRAM (one full expert-weight read at
240.76 GB/s) and 0.232 s of compute (about 7 TFLOPS, 15% of the fp16 peak). At
that size an expert bucket is ~5 rows, so there is little compute to hide the
next expert's load behind; the win grows for longer prompts only if the staging
buffers are double-buffered so expert `e+1` loads while `e` computes. That is a
staging-kernel change and needs a profile of the current staging path.

### N3 expert-count sync removal

`Executor::RouteHints` (`kernels/rocm/executor.cpp`) synchronizes
`counts_ready_` once per layer and builds the routed tile map on the host; the
routed F16 GEMMs read that map. Removing the sync means either building the map
on the device (then the launch geometry still needs the tile count on the host)
or launching a fixed grid of `experts x ceil(tokens/tile_rows)` blocks and letting
each block read its own bucket size and return. The routed kernels already
early-exit on empty column tiles, so the fixed grid is numerically safe, but each
empty block still costs a dispatch and a shared-memory reservation. The trade
(drained pipeline vs empty blocks) is only decidable with a measurement.

### N4 is closed by the measured decode split

The earlier estimate assumed a 32-session decode step touches a 238-expert
union and that the MoE therefore runs at 128 GB/s. `GUFO_LAYER_TIMING` shows
the real union is 105-115 (the benchmark sessions share the prompt and decode
greedily), the MoE is 2.18 ms of a 4.53 ms layer, gate/up reaches 198 GB/s
(85% of the 232 GB/s gather ceiling) and down reaches 174 GB/s DRAM plus
~493 GB/s L2. A bit-identical grouped Q5_1 down kernel that removes the
remaining duplicate slot reads measured -3.0% total and was reverted. An int8
WMMA kernel cannot raise the DRAM ceiling, so N4 is closed rather than
deferred; the remaining expert-side gap is ~2-4% total. The full measurement
is in `artifacts/serving/batch-pp-tg-levers.md` §19.

### Request coalescing (opt-in)

`GUFO_REQUEST_COALESCE=1` coalesces byte-identical greedy requests into one
leader session; followers receive the leader's token stream and own no KV or
runner session. A 32-identical-request batch went from 47.753 s to 5.009 s
(4096 tokens, identical text, 9.53x); the fresh 8/16/24/32 A/B measured
2.89x-9.08x. Per-replica `max_tokens`, follower cancellation and late joins
were tested; details, limitations and the canonical c32 no-trigger result are
in `artifacts/serving/batch-pp-tg-levers.md` §22 and §24. The canonical
`bench_pp_tg` workload gives each request a unique front-loaded nonce, so
coalescing stays opt-in and does not change that benchmark (on/off +0.43% to
+0.91%, noise).

## Re-measurement procedure after the reboot

One command runs the health gate and every implemented phase, then stops on the
first failed gate:

```sh
PHASES=all bash artifacts/serving/lever_matrix.sh all <label>
artifacts/serving/py.sh artifacts/serving/n_levers_status.py --label <label>
```

Two runbook details that cost a whole run if they are missed:

- **`python3` is not on the container's PATH.** The nix store kept the
  interpreter, not the PATH entry, so every analysis step resolves one through
  `artifacts/serving/py.sh` (`PYTHON_BIN` overrides the choice). The matrix
  defaults its `PY` to that resolver; call the status generator through it as
  shown above. Without this, a bench run produces artifacts and then dies in its
  own analysis half.
- **The health gate now really gates.** It takes the status of
  `timeout <n> gufo diagnose` without piping it into `tail` (which returned 0 and
  let a hung device fall through into benches), and it reads D-state tasks from
  `/proc` with `grep`/`cut` only, because the container has neither `ps` nor
  `sed`/`awk`, and its private PID namespace hides host tasks anyway. Against the
  current host it reports `VERDICT health=FAIL gufo processes stuck in D state:
  3684305 3692283 3916108` and `d_state_total_visible=17`, matching the host's
  own view; inside the container it finds the wedged session it can see and says
  so (`d_state_scope=container-private-pid-namespace`). `HEALTH_TIMEOUT` and
  `PROC` exist so both the fail and the pass path are testable without a
  two-minute wait or a real wedge. The gate also refuses to start a run when the
  binary is not executable or the model is not readable, so an unattended run
  fails in seconds instead of minutes.
- **N6's pin is verified by read-back.** `[ -w ]` alone is not proof: a
  container's root can pass the permission test and still be refused by sysfs.
  The n6 phase now writes `high`, reads every attribute back, and falls into the
  honest baseline-only path (`n6_power_writable=yes-but-write-refused`) when the
  level did not change — otherwise the A/B would compare two identical-profile
  arms and read noise as an N6 gain. The `EXIT` trap still restores the saved
  profile; a stub run against a writable fake tree confirms both arms ran and the
  profile came back as `auto`.

`lever_matrix.sh` writes `artifacts/serving/lever-matrix-<label>-verdict.txt` and
aborts immediately if the device still hangs. The N-lever status file records the
per-lever verdict against the targets in this document. `PHASES=n6` runs only the
dpm A/B when N6 is the question and no other lever changed; `PHASES=overlap` runs
only the N1 control and prints its verdict.

The overlap phase no longer leaves the N1 decision to hand comparison: it pipes
the artifact through `artifacts/serving/compare_overlap.py`, which passes a run
only when the median gain reaches +20% **and** both servers keep at least 80% of
their solo throughput while sharing the device. A gain bought by starving one arm
therefore fails. Synthetic artifacts cover all four outcomes: +22.7% with 0.93 /
0.86 retention passes; +9.12% fails the target; +22% with decode retention 0.55
fails as starvation; a negative round minimum fails as stream interference.
