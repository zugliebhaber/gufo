# Gufo documentation

Start with the [top-level README quickstart](../README.md#quickstart). This
page is the map of the detailed documentation.

## Reading paths

| I want to… | Read |
| --- | --- |
| install and run my first model | [README quickstart](../README.md#quickstart), [models](models/README.md) |
| see what this tree adds over upstream Gufo | [Features](FEATURES.md) |
| understand the engine design | [Architecture](ARCHITECTURE.md), [KV cache](KV-CACHE.md) |
| find a CLI command or setting | [CLI guide](CLI.md), [README settings](../README.md#commands-and-settings) |
| call the HTTP API | [Server contract](SERVER.md) |
| monitor with Prometheus | [Prometheus metrics](PROMETHEUS.md) |
| build Grafana dashboards | [Grafana dashboard](GRAFANA.md) |
| benchmark a model | [Benchmark methodology](BENCHMARKS.md), [performance loop](PERFORMANCE.md) |
| understand the cache | [KV cache](KV-CACHE.md), [cache redesign RFC](cache-redesign/RFC.md) |
| test or contribute | [Testing](TESTING.md), [CI](CI.md), [development](DEVELOPMENT.md) |
| release a build | [Releasing](RELEASING.md) |
| build on Windows | [Windows build](../BUILD-WINDOWS.md) |
| see user-facing changes | [Changelog](../CHANGELOG.md) |

## Architecture and design

- [Features added over upstream Gufo](FEATURES.md): the consolidated inventory
  of serving, cache, decode, API, diagnostics and documentation work, each with
  purpose, usage and evidence.
- [Architecture](ARCHITECTURE.md): process/request flow, prefill/decode,
  MoE experts, memory/cache tiers, coalescing, benchmark flow, ADRs and
  invariants.
- [KV cache](KV-CACHE.md): what is cached, reuse, sharing, retention and
  invalidation.
- [Visual KV cache and pagination](KV-CACHE-VISUAL.md): block pool and page
  tables, address translation, copy-on-write, checkpoint restore, pool sizing.
- [Cache redesign RFC](cache-redesign/RFC.md) and its
  [cards](cache-redesign/cards/README.md): the component cache, ledger,
  checkpoints, transfers and adapters.
- [Paged KV](cache-redesign/paged-kv.md): page/table layout.
- [Server contract](SERVER.md): process/state ownership, HIP execution,
  endpoints, reasoning controls, request model, metrics and troubleshooting.

## Commands, settings and operation

- [CLI guide](CLI.md): text, video, audio, embeddings and the settings that
  map to runtime features.
- [README quickstart](../README.md#quickstart) and
  [commands and settings](../README.md#commands-and-settings): copy-paste
  recipes.
- [Server limits](SERVER.md#lifecycle-and-limits), [authentication and
  exposure](SERVER.md#authentication-and-exposure), [troubleshooting
  logs](SERVER.md#troubleshooting-logs).
- [Prometheus metrics](PROMETHEUS.md): scrape configuration, the complete
  metric catalog, PromQL examples and alert rules.
- [Grafana dashboard](GRAFANA.md): dashboard import, provisioning, panel
  descriptions, customization and troubleshooting.
- [Power/profile diagnostics](ARCHITECTURE.md#adr-7-diagnostics-and-power-profile-are-part-of-the-measurement-contract).

## Models

- [Model index](models/README.md).
- Per model: `README.md` (weights, modes, recipes), `BENCHMARKS.md`
  (retained tables), `QUALITY.md` (independent checks and gaps),
  `EXPERIMENTS.md` (retained/rejected decisions), `ROCM.md` (backend status).
- Qwen3.8 Flash-Next specifically: [batch, speculative and cache
  notes](models/qwen3.8-flash-next/README.md#batching-widths-and-limits),
  [retained benchmarks](models/qwen3.8-flash-next/BENCHMARKS.md),
  [N1-N9 levers](models/qwen3.8-flash-next/N-LEVERS.md).

## Measurement

- [Benchmark methodology](BENCHMARKS.md): drivers, direct vs serving, quality
  matching and artifact retention.
- [Performance loop](PERFORMANCE.md): measurement contract, canonical serving
  benchmark, roofline, profiling and promotion gates.
- [Batch throughput levers](BATCH-THROUGHPUT.md): 52 ranked measures for
  1000 concurrent requests, with measured status and expected effect.
- [Features added over upstream Gufo](FEATURES.md): what changed, why, and the
  retained before/after numbers.
- [Testing](TESTING.md): hosted contract suite, focused GPU/operator/model
  checks and quality tools.

## Documentation checks

The documentation is checked by `tools/ci/check-docs.py`:

```sh
python3 tools/ci/check-docs.py
nix build .#checks.x86_64-linux.docs
```

The first command checks relative links, anchors, JSON code blocks and Mermaid
fences/kinds; the Nix check packages the same validator for CI. For a visual
check, open the file in GitHub or VS Code with a Mermaid preview extension.

## Conventions

- Use relative links for repository files; use absolute `https://` links for
  external services.
- Keep model-specific recipes in the model folder; keep cross-model design in
  `docs/` and `docs/ARCHITECTURE.md`.
- When a benchmark changes, update the model's `BENCHMARKS.md` and link the
  analysis artifact; do not silently replace the retained table.
- Mark opt-in behavior explicitly (environment variable or flag), its scope and
  its quality/determinism contract.
