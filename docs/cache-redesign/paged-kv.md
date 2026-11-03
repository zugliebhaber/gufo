# Paged live KV: design and decision

Status: kernels landed, executor pool implemented and **enabled**
(`engine.cpp` sizes `kv_pool_blocks` from the resident sessions so every
session maps its whole context, the same memory envelope as the contiguous
planes it replaces). The paged *snapshot restore* divergence from §10 is fixed:
the cache and the snapshot walker resolve rows through the block table's
physical ids, and blocks are zeroed before a session can preserve rows.
Requests now share physical KV blocks: a capture lends full-attention blocks and
a restore adopts them with one pool reference (§10). Draft for review, 2026-10-30.
Scope: Qwen3.8-27B and Qwen3.8-Flash-Next text serving on gfx1151.
Related: [cache redesign RFC](RFC.md) (its
[open question](RFC.md#open-questions) *whether measured live-memory/cross-slot
copying costs make paged KV necessary*), [cards/06](cards/06-slot-leases-and-mutation.md),
[KV cache](../KV-CACHE.md).

This document answers three questions in order:

1. How much live memory does one session reserve today, and how does that
   couple context capacity to concurrency? (§2)
2. What exactly has to change to make attention read a block table? (§4)
3. Is paging the next performance step, or does the measured batch limit sit
   somewhere else? (§6)

The short answer to (3) is: **paging fixes a capacity coupling, not step
throughput.** The measured decode step does not scale with concurrency on this
target (§2.2), so attribution has to come first. Paging is the right change
once capacity is the binding constraint; it is not a substitute for finding the
step-time limit.

## 1. Decision

Adopt block-managed "live" KV for the full-attention layers of both Qwen text
models, while keeping every non-attention byte (recurrent state, convolution
state, indexer, draft ring, RNG, sampling frontier) in the existing per-session
private region and the existing opaque snapshot format.

Deliver in four independently testable phases (§7). Do not start phase 1 before
phase 0 produces an attribution result.

Explicitly out of scope: DeepSeek V4 Flash. Its engine owns a separate KV
allocator, managed-memory policy and fp8 KV route
(`src/models/deepseek_v4_flash/runtime/rocm_graph.cpp`); it must not inherit
this design.

## 2. Evidence

### 2.1 Live KV per session today

`Executor::SessionBytes` (`src/models/qwen38_flash_next/kernels/rocm/executor.cpp:1474`)
computes the per-session payload as

```
kv    = 2 * max_context * AttentionKvDim * sizeof(half)
index = IndexerCapacity * indexer_head_dim * 4
      + (max_context / compress_ratio + 1) * indexer_head_dim * 2
```

with `AttentionKvDim = num_kv_heads * head_dim = 2 * 256 = 512` and 12
full-attention layers (`full_attention_interval = 4`,
`src/models/qwen38_flash_next/config.cpp:251`). That is **24 KiB per token per
session** in AR, and **48 KiB** in speculative mode, where the mode duplicates
`kv + index` for the draft ring.

| Context capacity | AR per session | MTP per session | 8 sessions, MTP |
| ---: | ---: | ---: | ---: |
| 4,096 | 96 MiB | 192 MiB | 1.5 GiB |
| 32,768 | 768 MiB | 1.5 GiB | 12.0 GiB |
| 65,536 | 1.5 GiB | 3.0 GiB | 24.0 GiB |
| 163,840 | 3.75 GiB | 7.5 GiB | 60.0 GiB |
| 262,144 | 6.0 GiB | 12.0 GiB | 96.0 GiB |

Measured cross-check: `--sessions 8 --context 32768` with MTP reports
`gpu_device_used_mib=97505` of 126976 at load (Q4 target plus MTP, run of
2026-10-30, see §8). The KV share is 12 GiB of that.

The whole allocation is sized by **capacity**, not by use. A request that
prompts 2,048 tokens and generates 128 still holds the full 768 MiB arm, and the
server refuses to start with more sessions than that product allows. This is
the structural difference to a paged engine: vLLM sizes admission from free
blocks, so eight 160k-capacity sessions cost what their tokens actually use.

### 2.2 Measured batch behaviour (gufo 0.20.0)

8 × pp2048/tg128, greedy, context 32,768, 8 sessions, MTP, one warmup and two
repetitions per cell, canonical serving harness (`tools/serving/gufo-serving-bench.py`).

| C | Decode tok/s per request | Σ decode tok/s | ITL p50 | ITL p95 | Prefill tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 73.94 | 73.9 | 13.7 ms | 14.0 ms | 1,070.7 |
| 2 | 56.41 | 112.8 | 18.4 ms | 18.8 ms | 1,050.1 |
| 4 | 36.93 | 147.7 | 32.9 ms | 35.5 ms | 1,042.7 |
| 8 | 22.82 | 182.6 | 58.1 ms | 66.5 ms | 1,048.5 |

Prefill is flat across concurrency, which is the intended chunked-prefill
behaviour. Decode is not: **8× concurrency yields 2.47× aggregate decode** and
ITL grows 4.2×, close to per-request serialisation. A bandwidth-bound batched
step should keep ITL roughly flat, because the weight read is shared.

Two readings are possible and they imply different work:

- If the step is already bandwidth-bound at C=1, extra concurrency cannot help
  and paging only buys capacity.
- If the step is launch- or synchronisation-bound, the batched path is leaving
  bandwidth unused and paging would hide that behind more memory headroom.

§6 makes this the phase 0 gate. Do not assume either answer.

## 3. What paging does and does not buy

Buys:

- Concurrency decoupled from context capacity (the table in §2.1 collapses to
  actual token counts).
- Cross-slot restore stops copying whole prefixes. The RFC's comparison table
  marks this 🔴 today; retained checkpoints can share full blocks by reference
  count instead of duplicating rows.
- Rollback after a rejected draft becomes a length change instead of a
  preservation copy, because the overwritten rows live in blocks the session
  owns.
- Admission can be expressed in blocks, which is the honest currency on a
  unified-memory APU.

Does not buy:

- Faster prefill or decode at fixed concurrency. Attention gains one
  indirection per block; the arithmetic is identical.
- Concurrency beyond what Qwen38's own decode path can sustain. §2.2 suggests
  that limit is reached well before eight sessions on this target.
- Anything for Flash-Next's 36 linear layers. Their state is O(1) per session
  (`conv`, `state`, `ple` in `SessionBytes`) and stays private.

## 4. Current layout and the exact change

### 4.1 Two layouts exist

Production fp16 KV is one plane per (K, V), laid out as

```
[plane][layer][context][kv_heads * head_dim]
```

(`ContinuationAdapter::TransferRows`,
`src/models/qwen/continuation_adapter.cpp:814`). The FP32 reference layout
is head-major, `[layer][head][context][head_dim]`, and is copied with
`hipMemcpy2DAsync` per head.

Attention kernels compute the row base as

```
(layer_idx * max_context + pos) * kv_width
```

(`src/models/qwen/hip/kernels/attention_decode.hip:416` and `:649`). Eight
kernel files reference this stride: `attention_decode.hip`,
`attention_decode_graph.hip`, `attention_decode_verify_wave64.hip`,
`attention_batched.hip`, `attention_tile.hip`, `attention_wmma.hip`,
`fused.hip`, `prefill_rope.hip`.

### 4.2 Block geometry

Measured indirection cost (`tools/bench/kv_paging_bench.hip`, gfx1151, decode
shape, 128-token blocks, permuted table so entries walk the pool backwards,
bit-identical output):

| Context | Contiguous | Table per position | Table hoisted per block |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.33 ms | +5.4% | −0.8% |
| 4,096 | 1.48 ms | +5.5% | −1.0% |
| 16,384 | 10.2 ms | +4.4% | −8.4% |
| 65,536 | 33.4 ms | +3.7% | −1.5% |

A second run repeats the shape: +4.0 to +5.6% per position, −0.7 to −8.4%
hoisted. The rule that follows is binding for the kernels: **resolve the table
once per block, never per position.** A 16-key WMMA tile plus a 128-token block
makes that one lookup per 128 positions, and the paged path then costs nothing
measurable against the contiguous one. The 16 KiB cell is the outlier in both
runs and is not treated as a gain.

One block covers `B` consecutive positions for **all** full-attention layers and
both planes, because every layer uses the same KV head count and head dim. That
is the useful simplification over vLLM, which needs a table per layer: a
session needs exactly one block table.

```
block_bytes = B * 24 KiB        B = 128 -> 3 MiB
table_entry = physical block id (uint32)
```

Row addressing becomes two steps instead of one:

```
block = table[pos / B]                       // one load, hoisted per tile
row   = block * block_bytes
        + plane  * plane_bytes
        + layer  * layer_bytes
        + (pos % B) * params.row_bytes
```

`plane_bytes` and `layer_bytes` are launch constants, so the inner loop over
`pos` keeps its current arithmetic once the block base is resolved. A decode
step hoists one table load per block it touches; a prefill tile stays inside
one or two blocks when `B >= 128` and the scheduled chunk is at least that
large; the default batched-token budget carries 16384 tokens
(`kDefaultDecodeActivePrefillTokens`,
`src/cli/serve/text_generation_scheduler.hpp`).

Block size is a tradeoff, not a constant to copy from another engine:

| B | Block bytes | Worst tail waste | Table at 32k | Notes |
| ---: | ---: | ---: | ---: | --- |
| 64 | 1.5 MiB | 1.5 MiB | 512 entries | finer sharing, more table loads per tile |
| 128 | 3 MiB | 3 MiB | 256 entries | proposed default |
| 256 | 6 MiB | 6 MiB | 128 entries | fewer loads, coarser copy-on-write |
| 512 | 12 MiB | 12 MiB | 64 entries | matches one prefill chunk, poor tail behaviour |

Copy-on-write only ever applies to the **last, partial** block of a shared
prefix. Full blocks are immutable, so extension never copies them: a writer
appends a fresh block instead. With `B = 128` the worst case a fork pays is one
3 MiB copy.

### 4.3 Allocation and lifetime

Implemented in `src/models/qwen/kv_blocks.hpp` and covered by
`tests/models/qwen/kv_blocks_test.cpp` (hosted CPU suite). The pool owns block
ids; the executor owns the device buffer. The table never moves bytes: it
reports a `KvTailDetach` for every copy the caller must make on the device, so
the aliasing rules are tested without a GPU.

- One device pool per model instance, sized at load from
  `--kv-block-pool-bytes`, with a free list of block ids. No compaction, no
  device-side allocation after serving starts.
- A session holds a table with `ceil(context_capacity / B)` uint32 entries in
  its private region; entries are filled as blocks are acquired. Unused entries
  are `kInvalidBlock`, and every kernel loses one bounds check.
- Refcount per block: 1 for a live session, plus one per retained checkpoint
  that references it. Eviction of a checkpoint drops refs; a block returns to
  the free list at zero.
- The mutation guard in [cards/06](cards/06-slot-leases-and-mutation.md) keeps
  its ordering (find readers, preserve or retire, wait, then overwrite) but
  stops preserving rows entirely: it retires or copies **the partial block**,
  then rebinds the table entry.

### 4.4 Snapshot and cache interplay

Snapshots keep their current shape: opaque payload, model-owned
`SaveSnapshot`/`RestoreSnapshot`, versioned payload. Only the attention KV part
changes from inline rows to a block reference list:

- capture writes `{block count, block ids}` plus the private region;
- restore adopts the ids and bumps refcounts instead of copying rows;
- a restored session that continues writing past the captured length copies
  only the partial tail block;
- disk persistence still serialises rows (blocks are device-local), so the
  existing streamed 1 MiB transfer path is unchanged.

This is what turns the RFC's "cross-slot restore without full-prefix copying"
from 🔴 into 🟢, and it is why the change belongs next to the cache work rather
than in a separate allocator project.

## 5. What must not regress

- **Byte-exactness.** Block indirection changes addresses, not values. Every
  phase must compare full-vocabulary logits against the contiguous path on
  identical seeds, per the promotion gates in `docs/PERFORMANCE.md`.
- **Rollback.** Speculative verification relies on preserving borrowed rows
  before mutation. With blocks this becomes a table truncation, which must be
  proven equivalent with the existing state-edge and full-logit rollback tests.
- **Snapshot compatibility.** The payload version moves; every old checkpoint
  is invalidated by the component identity, which the RFC already requires for
  any layout change.
- **Admission fairness.** Block quotas must not let one long prefill starve
  decode-ready sessions. The existing `decode_active_tokens` budget stays
  authoritative; blocks only decide whether a session may start at all.
- **Kernel resources.** Table loads add VGPRs. `kernel_resource` limits in
  `tools/ci/kernel-resources.json` must be re-baselined deliberately, not
  silently.

## 6. Phase 0: attribution (do this before writing kernels)

The measured 2.47× at C=8 has to be attributed before any allocator work.

1. Per-phase kernel time at C=1, 2, 4, 8 on the same history
   (`tools/prof/prof.py`, or `rocprofv3` in the ROCm container).
2. Separate: batched step kernels, per-request draft/verification kernels,
   launch gaps and synchronisation, and any host-side per-request work that
   does not amortise.
3. Report effective bytes/s against the measured bandwidth ceiling
   (`tools/bench/gfx1151_peak.hip`).

Decision rule:

- If the batched step already reaches the bandwidth ceiling at C=2, stop and
  treat paging purely as a capacity feature (still worth doing, lower urgency).
- If it does not, fix the step first: that gains throughput at every
  concurrency, which paging does not.
- If launch gaps dominate, address batching/launch shape first.

## 7. Delivery phases

Each phase is a separate PR with its own measurement, following
`docs/PERFORMANCE.md`.

| Phase | Content | Gate |
| --- | --- | --- |
| 0 | Attribution (§6) | report, decision recorded here |
| 1 | Block pool, table, decode + verify read path for Flash-Next | byte-exact logits, decode p50/p95 at C=1/4/8 |
| 2 | Prefill write path, chunked prefill, rollback truncation | byte-exact logits, prefill tok/s, rollback suites |
| 3 | Snapshot/fork block sharing, copy-on-write tail | restore time and bytes, cache suites |
| 4 | Admission in blocks, `--kv-block-pool-bytes`, retire the fixed product | capacity matrix, C=8 long-context run |

Landed so far:

- Phase 1 groundwork: `KvBlockPool`/`KvBlockTable` with the copy-on-write
  contracts, hosted in the CPU suite.
- Qwen3.8-27B: `KvPageView`/`KvRowOffset` (`src/models/qwen/hip/detail/kv_page.hpp`),
  block-table reads and writes in `WmmaCausalAttention` and the pack kernel, and
  `qwen_kv_page_ops_test`, which packs a permuted three-layer pool and requires
  byte-identical output to the contiguous cache.
- Flash-Next: `StoreKv`, `PrepareAttention`, `WmmaCausalAttention` and the
  per-token `Attention` route take an optional `KvPageView` and `layer`;
  `qwen38_flash_next_kv_page_ops_test` runs both readers against a permuted
  two-layer pool and requires byte-identical output. The sparse route resolves
  the table per selected position, because a sparse tile gathers four-token
  blocks that are not adjacent in the cache.
- Not yet converted: every executor-side allocation, admission, snapshot and
  fork path, and the MTP draft ring (a private ring buffer, not target KV).

Qwen3.8-27B follows phase 2 with the same interfaces; its dense KV means every
layer converts, and its fp32 fallback layout stays contiguous for the
reference path.

## 8. Measurement record

Environment: this host, `AMD RYZEN AI MAX+ 395 / Radeon 8060S`, gfx1151,
20 CUs, 32 logical cores, 128 GiB unified LPDDR5X, Fedora 44 kernel
`7.2.13-200.fc44.x86_64`. GPU runtime taken from
`ghcr.io/gufo-org/toolboxes/gufo-runtime:latest`, which reports `gufo 0.20.0
(0c03748)`; the source tree here is 0.28.0 and was not rebuilt for this run.
Treat the absolute numbers as a 0.20.0 baseline and the scaling shape as the
finding to re-confirm on a 0.28.0 build.

Server:

```sh
podman run -d --name gufo-bench --security-opt label=disable \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ulimit memlock=-1 -p 8090:8080 -v "$HOME/models:/models:ro,Z" \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/<first-shard>.gguf \
  --speculative mtp --mtp-model /models/.../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --context 32768 --sessions 8
```

Client: `tools/serving/gufo-serving-bench.py --concurrency 1,2,4,8
--warmup 1 --repetitions 2 --max-tokens 128 --temperature 0`, artifact at
`artifacts/serving/baseline-0.20.0.json` (not committed).

Load: 42.0 s to ready, `gpu_device_used_mib=97505`,
`host_available_mib=28042` after load.

## 9. Open questions

- Block size: 128 is the proposal; 64 and 256 need one matched run each, because
  the decode step touches one table load per block and the tail waste differs by
  3 MiB per session.
- Whether the last partial block should be pinned (never shared) or copied on
  fork. Pinning is simpler and loses at most one block per retained checkpoint.
- Whether the indexer's compressed buffer (`max_context / compress_ratio`)
  should be paged too. It is 3 MiB per layer at 32k against 64 MiB for KV, so
  the answer is probably no; measure before deciding.
- Whether `--sessions` survives as a preallocation knob or becomes a pure
  admission ceiling derived from the pool.
- Whether the FP32 reference layout should be converted at all, or stay
  contiguous behind a clearly separate test-only path.

## 10. Resolved defect: paged snapshot restore

Two identical greedy requests through a fresh server, 64-token prompt, 48
tokens generated, `--sessions 2`:

| Build | Request 1 | Request 2 (cache hit) |
| --- | --- | --- |
| Contiguous (`kv_pool_blocks = 0`) | sha `e2a6b847d60e2bfb` | sha `e2a6b847d60e2bfb` |
| Paged pool (before the fix) | sha `e2a6b847d60e2bfb` | sha `b3111b1dcd29fe36` |

Both requests reported `cache_n=64, prompt_n=0`, so the checkpoint was found
and restored in both builds. Execution alone is not the cause: request 1, which
prefilled and captured through the paged store, matches the contiguous build
byte for byte.

The defect had two causes, both a logical-versus-physical block confusion:

1. `Executor::WalkSnapshot` emitted one region per logical block but addressed
   the pool by the logical index instead of the table's physical id. A restore
   into blocks with different ids therefore read and wrote another session's
   rows.
2. `ContinuationAdapter::RowSpans`, which is what the serving cache actually
   uses, split a row range per block the same way. It also planned capture
   spans before the session had acquired any block, so the preserved bytes came
   from whatever physical block 0 held.
3. A prefill checkpoint is a lightweight `Session` **view** of the source: it
   copies `attention_` (and therefore the paged table pointer) but leaves
   `kv_blocks_` null. `RowSpans` ran on that view, so the block lookup fell back
   to the logical index exactly for the shared-prefix captures the concurrent
   serving path uses.

The fix reads every paged row span through the owning session's
`KvBlockTable::entry`, zeroes the pool once at creation, and maps a session's
whole context when the session is created. The last step is what makes an early
capture deterministic: a row that is preserved before its first write has a
stable, zeroed physical block. The memory envelope is unchanged, because the
contiguous planes reserved exactly the same bytes per session.

Evidence after the fix, same reproduction (prompt 145 tokens, `--sessions 2`,
`--context 2048`, greedy, 48 tokens, second request arriving after the capture
settles):

| Build | Request 1 | Request 2 (cache hit, `cached_tokens=145`) |
| --- | --- | --- |
| Contiguous | sha `6b00e61ccc2c73ec` | sha `6b00e61ccc2c73ec` |
| Paged pool | sha `6b00e61ccc2c73ec` | sha `6b00e61ccc2c73ec` |

The functional `cache-shared-prefix` and `cache-concurrency` suites
(overlapping prefixes, identical groups, cancelled requests, four concurrent
sessions) pass with the paged pool at an 8k context; `cache-concurrency` failed
before cause 3 was fixed.

**Block sharing between requests is now wired.** A cache capture of full-attention
K/V keeps the source slot's device rows instead of copying them
(`BorrowRowsView` lends one block at a time, because a block is the largest
range that is contiguous in the pool). A restore that receives those exact
device bytes adopts the source's physical blocks: `KvBlockTable::Adopt` takes a
reference on the source block and releases the destination's own, so a follower
holds one copy of a shared prefix instead of two. The partial tail block is
still copied (one block at most), and full blocks are append-only, so a shared
block is never written again.

Safety: the write guard invalidates a table's shares before any overwrite, a
materialized payload has different bytes and is never adopted, and both
`BeginRestore` and slot invalidation remap the table (release all, map the whole
context) so no adopted block survives into a new request. Measured on gfx1151:
`cache-shared-prefix` reported `kv_blocks_shared blocks=26 pool_live=486
pool_free=26`, i.e. 26 blocks (about 78 MiB) returned to the pool per follower,
with the knee throughput unchanged at 164.10 tok/s.
