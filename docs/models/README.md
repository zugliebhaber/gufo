# Models

Linux x86-64 / AMD Strix Halo gfx1151 is the production target. Build with
`nix build`; Python/reference toolchains are development-only. Model weights
are downloaded separately and are not part of the runtime package.

Start with the [documentation index](../README.md), the [architecture
overview](../ARCHITECTURE.md) and the [README
quickstart](../../README.md#quickstart). The [CLI guide](../CLI.md) maps
features to flags; `gufo help <command>` is authoritative.

| Model | Weights / supported inputs | Guide |
| --- | --- | --- |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF; text | [Usage and modes](deepseek-v4-flash/README.md) |
| Qwen3.8 27B | Q4/Q8 GGUF; text/images | [Usage and modes](qwen3.8-27b/README.md) |
| Qwen3.8 Flash-Next | Sharded Q4 GGUF; text/images | [Usage and modes](qwen3.8-flash-next/README.md) |
| Qwen3-ASR 1.7B | BF16 safetensors; audio to text | [Usage](qwen3-asr/README.md) |
| EmbeddingGemma 2 | BF16 safetensors; text/image to embeddings | [Usage](embeddinggemma-2/README.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors; text/reference audio to speech | [Voice modes](qwen3-tts/README.md) |
| Qwen-Image-2.1 | BF16 safetensors; generation and image editing | [Usage](qwen-image-2.1/README.md) |
| MiniMax H3 FL2VA | Pinned safetensors; text to audiovisual output | [Presets and usage](minimax-h3/README.md) |

## Serving recipes

| Goal | Start |
| --- | --- |
| lowest latency / one user | `--sessions 1`; Qwen3.8 27B with DFlash2, Flash-Next with MTP |
| aggregate throughput | `--sessions 8`-`256`; Flash-Next scales the decode plan up to 256 rows |
| long context | `--context N`; Flash-Next above 262144 needs `--enable-yarn` |
| restart-safe continuation | `--cache-disk DIR` with one owner per directory |
| repeated/shared prefixes | leave `--cache-ram-bytes` at auto; report output and TTFT, not only physical tokens |
| duplicate greedy requests | `GUFO_REQUEST_COALESCE=1`; opt-in, identical prompt/sampling only |
| cold reproducible benchmark | `--cache-ram-bytes off` and unique prompts; the canonical table |

The exact files and benchmark settings are in each model's `README.md` and
`BENCHMARKS.md`. For the batch/coalescing/cache trade-offs see the
[architecture decisions](../ARCHITECTURE.md#8-architecture-decisions).

Each model folder contains:

- `README.md`: model card, acquisition, usage and modes.
- `BENCHMARKS.md`: current retained measurements, scope/date and TODO cells.
- `QUALITY.md`: measured quality, reference/scope, replay guarantees and unresolved gaps.
- `EXPERIMENTS.md`: short retained/rejected decisions.
- `artifacts/`: only useful machine-readable results/calibration, when present.

Keep executable fixtures with their tests. New logit/trace dumps and local
profiles belong in ignored top-level `artifacts/`; commit only independent
reference evidence needed by a maintained check. Old experiment records remain
in Git history. See [benchmark methodology](../BENCHMARKS.md).

Quality reports distinguish agreement with an independent model implementation
from consistency between Gufo execution modes. Greedy speculative decoding
should reproduce AR; sampled speculation can use different random draws.
Seeded replay is checked within the same execution configuration.
For numerical comparisons, lower KL divergence, total variation, relative L2
and word-error rate are better; higher cosine similarity and image PSNR are
better. These bounded checks are not broad capability or perceptual scores.
