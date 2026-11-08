# Prometheus metrics

`gufo serve llm` exposes process-wide application metrics on its normal HTTP
listener:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/metrics` | Prometheus text exposition |
| `GET` | `/v1/metrics` | Alias for `/metrics` |

The response uses Prometheus text format:

```text
Content-Type: text/plain; version=0.0.4; charset=utf-8
```

No exporter sidecar and no extra flag are required. The endpoint reports the
loaded text model. It is covered by `--api-key` like every other route, and it
is available on the same address and port as the OpenAI/Anthropic API.

```sh
gufo serve llm --model model.gguf --host 0.0.0.0 --port 8000
curl -sS http://127.0.0.1:8000/metrics | grep '^gufo_'
```

For the Grafana side, see [Grafana dashboard](GRAFANA.md).

## Scrape configuration

```yaml
scrape_configs:
  - job_name: gufo
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8000"]
    # Only when the server runs with --api-key:
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/gufo-token
```

A 15 s interval or shorter is recommended. Token counters advance while a
request is still prefilling or decoding, so their `rate()` reflects live
throughput. Longer intervals smooth away short bursts and make queueing hard
to see.

If Prometheus already uses a different job naming scheme, keep the target and
relabel in the usual way. The endpoint exposes no instance or model labels;
one process serves one text model, so the `job`/`instance` pair from the scrape
config identifies it.

## Metric naming and lifecycle

- `gufo_*` names are the stable Prometheus-native names for new dashboards and
  alerts.
- The `llamacpp:*` families are retained for llama.cpp-compatible dashboards
  and tooling. They carry the same values as their `gufo_*` counterparts and
  are not removed.
- Counters increase monotonically for the life of the process and reset to
  zero on restart. Use `rate()` or `increase()` instead of subtracting values
  across a restart.
- Gauges describe the state at the instant of the scrape.
- Histograms use fixed buckets; see [Latency histograms](#latency-histograms).
- No exemplars, native histograms or OpenMetrics-only features are emitted.

## Request, latency and session metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `gufo_requests_total{status}` | counter | Finished text-generation requests. `status` is `completed`, `cancelled` or `failed`; each recorded request lands in exactly one status. Malformed HTTP requests rejected before generation are not counted. |
| `gufo_ttft_seconds` | histogram | Submission to first generated token, for completed requests. Includes queueing and cache restore. |
| `gufo_request_duration_seconds` | histogram | Submission to last generated token, for completed requests. |
| `gufo_requests_processing` | gauge | Admitted requests holding or preparing a session, including cache cleanup. Matches the busy `/slots` entries. |
| `gufo_requests_deferred` | gauge | Requests waiting for a session. |
| `gufo_sessions_capacity` | gauge | Configured session slots: `--sessions`, or the resolved `auto` count. |
| `gufo_sessions_active` | gauge | Session slots currently processing a request. |
| `gufo_context_tokens` | gauge | Per-session context window in tokens. |

`gufo_requests_total{status="completed"}` is recorded when the text backend
finishes a request, even if the client has already disconnected.
`cancelled` requests do not observe the latency histograms; failed requests
increment only the `failed` status. A request that fails before admission
therefore has no latency sample.

## Token and throughput metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `gufo_prompt_tokens_total` | counter | Prompt tokens processed, excluding cache hits. Advances per prefill chunk. |
| `gufo_prompt_tokens_cached_total` | counter | Prompt tokens reused from the prefix cache. |
| `gufo_tokens_generated_total` | counter | Generated tokens. Advances per generated token. |
| `gufo_prompt_seconds_total` | counter | Prefill time for completed requests. |
| `gufo_tokens_generated_seconds_total` | counter | Decode time for completed requests. |
| `gufo_prompt_tokens_per_second` | gauge | Prompt speed of the latest completed request. |
| `gufo_tokens_per_second` | gauge | Generation speed of the latest completed request. |
| `gufo_max_sequence_tokens` | gauge | Largest observed prompt plus generated sequence. |

`gufo_prompt_tokens_total` excludes cached tokens, matching llama.cpp's
`prompt_tokens_total`. The denominator for a process-wide cache hit rate is
the sum of processed and cached prompt tokens:

```promql
gufo_prompt_tokens_cached_total
  / (gufo_prompt_tokens_total + gufo_prompt_tokens_cached_total)
```

The prebuilt `gufo_prefix_cache_hit_rate` gauge computes the same ratio.

## Cache, KV and speculative metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `gufo_prefix_cache_hits_total` | counter | Completed requests that reused at least one cached prompt token. |
| `gufo_prefix_cache_misses_total` | counter | Completed requests that prefilled their whole prompt. |
| `gufo_prefix_cache_hit_rate` | gauge | Cached prompt tokens over all prompt tokens seen by this process. |
| `gufo_kv_cache_usage_ratio` | gauge | In-flight prompt and generated tokens over sessions times context. Retained cache entries are not counted. |
| `gufo_kv_limited_total` | counter | Prefill chunks whose size was set by free KV capacity instead of the batched-token budget. |
| `gufo_kv_deferred_total` | counter | Requests degraded to their retained host snapshot when the device KV pool refused blocks. |
| `gufo_preemptions_total` | counter | Model states released for RECOMPUTE preemption. |
| `gufo_recompute_tokens_total` | counter | Prompt and generated tokens replayed after RECOMPUTE preemption. |
| `gufo_spec_draft_rounds_total` | counter | Speculative verification rounds. |
| `gufo_spec_draft_tokens_total` | counter | Draft tokens proposed. |
| `gufo_spec_accepted_tokens_total` | counter | Draft tokens accepted. |
| `gufo_spec_lookup_tokens_total` | counter | Prompt-lookup copies proposed. |
| `gufo_spec_lookup_accepted_tokens_total` | counter | Prompt-lookup copies accepted. |

`gufo_kv_cache_usage_ratio` is an approximation of live context occupancy: it
counts the prompt and generated tokens held by active sessions, divided by
`gufo_sessions_capacity * gufo_context_tokens`. It deliberately excludes
retained host checkpoints and the block pool's free-block distribution. Use
the block-pool figures in `usage.gufo` for per-request cache diagnosis and the
KV pressure counters above for alerting.

## Process and device metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `gufo_device_lost_total` | counter | Confirmed GPU context losses. Nonzero means the process needs a restart. |
| `gufo_process_resident_memory_bytes` | gauge | Resident set size of the server process. |
| `gufo_host_available_memory_bytes` | gauge | Host memory available to new allocations. This is the unified-memory headroom Gufo schedules against, not a GPU-only figure. |

Both memory gauges are reported in bytes. A zero reading means the host did
not expose the corresponding `/proc` value; it is not a measurement of zero
memory.

## llama.cpp compatibility families

These families remain for existing llama.cpp-compatible dashboards. Their
values are identical to the `gufo_*` aliases.

| Compatibility family | Prometheus-native alias | Type |
| --- | --- | --- |
| `llamacpp:prompt_tokens_total` | `gufo_prompt_tokens_total` | counter |
| `llamacpp:prompt_tokens_cached_total` | `gufo_prompt_tokens_cached_total` | counter |
| `llamacpp:prefix_cache_hits_total` | `gufo_prefix_cache_hits_total` | counter |
| `llamacpp:prefix_cache_misses_total` | `gufo_prefix_cache_misses_total` | counter |
| `llamacpp:prefix_cache_hit_rate` | `gufo_prefix_cache_hit_rate` | gauge |
| `llamacpp:prompt_seconds_total` | `gufo_prompt_seconds_total` | counter |
| `llamacpp:tokens_predicted_total` | `gufo_tokens_generated_total` | counter |
| `llamacpp:tokens_predicted_seconds_total` | `gufo_tokens_generated_seconds_total` | counter |
| `llamacpp:n_tokens_max` | `gufo_max_sequence_tokens` | gauge |
| `llamacpp:spec_decode_num_drafts_total` | `gufo_spec_draft_rounds_total` | counter |
| `llamacpp:spec_decode_num_draft_tokens_total` | `gufo_spec_draft_tokens_total` | counter |
| `llamacpp:spec_decode_num_accepted_tokens_total` | `gufo_spec_accepted_tokens_total` | counter |
| `llamacpp:prompt_tokens_seconds` | `gufo_prompt_tokens_per_second` | gauge |
| `llamacpp:predicted_tokens_seconds` | `gufo_tokens_per_second` | gauge |
| `llamacpp:requests_processing` | `gufo_requests_processing` | gauge |
| `llamacpp:requests_deferred` | `gufo_requests_deferred` | gauge |
| `llamacpp:preemptions_total` | `gufo_preemptions_total` | counter |
| `llamacpp:recompute_tokens_total` | `gufo_recompute_tokens_total` | counter |
| `llamacpp:kv_limited_total` | `gufo_kv_limited_total` | counter |
| `llamacpp:kv_deferred_total` | `gufo_kv_deferred_total` | counter |
| `llamacpp:kv_cache_usage_ratio` | `gufo_kv_cache_usage_ratio` | gauge |

The legacy `llamacpp:n_tokens_max` family is typed as a counter for llama.cpp
compatibility even though it behaves as a maximum. New dashboards should use
`gufo_max_sequence_tokens`, which is correctly typed as a gauge.

## Latency histograms

`gufo_ttft_seconds` and `gufo_request_duration_seconds` share this fixed bucket
layout, in seconds:

```text
0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5,
7.5, 10, 15, 20, 30, 40, 50, 60, 120, 300
```

A scrape contains `_bucket{le="..."}` for each boundary, `_bucket{le="+Inf"}`,
`_sum` and `_count`. Bucket values are cumulative, as Prometheus requires.
Only completed, non-cancelled requests observe the histograms, so `_count`
tracks `gufo_requests_total{status="completed"}` apart from a scrape that
lands between the two atomic updates.

The histogram layout is deliberately compact: it covers fast interactive
requests through long-context prefills without retaining per-request samples.
If a deployment needs different quantile resolution, compute a recording rule
or add a dedicated benchmark; the endpoint itself does not expose raw samples.

## PromQL examples

Request rate by outcome:

```promql
sum by (status) (rate(gufo_requests_total[5m]))
```

Success and failure ratios:

```promql
sum(rate(gufo_requests_total{status="completed"}[5m]))
  / sum(rate(gufo_requests_total[5m]))

sum(rate(gufo_requests_total{status="failed"}[5m]))
  / sum(rate(gufo_requests_total[5m]))
```

Median and p95 time to first token:

```promql
histogram_quantile(0.50,
  sum by (le) (rate(gufo_ttft_seconds_bucket[5m])))
histogram_quantile(0.95,
  sum by (le) (rate(gufo_ttft_seconds_bucket[5m])))
```

p50 and p95 end-to-end request duration:

```promql
histogram_quantile(0.50,
  sum by (le) (rate(gufo_request_duration_seconds_bucket[5m])))
histogram_quantile(0.95,
  sum by (le) (rate(gufo_request_duration_seconds_bucket[5m])))
```

Aggregate prompt and generation throughput:

```promql
sum(rate(gufo_prompt_tokens_total[1m]))
sum(rate(gufo_tokens_generated_total[1m]))
```

Generated tokens per completed request:

```promql
sum(rate(gufo_tokens_generated_total[5m]))
  / sum(rate(gufo_requests_total{status="completed"}[5m]))
```

Session utilization and KV pressure:

```promql
gufo_sessions_active / gufo_sessions_capacity
gufo_kv_cache_usage_ratio
sum(rate(gufo_kv_limited_total[5m]))
sum(rate(gufo_kv_deferred_total[5m]))
sum(rate(gufo_preemptions_total[5m]))
```

Memory headroom:

```promql
gufo_host_available_memory_bytes / 1024 / 1024 / 1024
gufo_process_resident_memory_bytes / 1024 / 1024 / 1024
```

## Alert examples

Tune thresholds and `for:` durations for the deployment. `GufoDeviceLost` is
critical because the process cannot recover from a lost GPU context in place.

```yaml
groups:
  - name: gufo
    rules:
      - alert: GufoDeviceLost
        expr: increase(gufo_device_lost_total[5m]) > 0
        labels: {severity: critical}
        annotations:
          summary: Gufo lost its GPU context and must restart
      - alert: GufoRequestsFailing
        expr: |
          sum(rate(gufo_requests_total{status="failed"}[5m])) > 0.01
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: Gufo is finishing requests with errors
      - alert: GufoMemoryLow
        expr: gufo_host_available_memory_bytes < 2 * 1024 * 1024 * 1024
        for: 2m
        labels: {severity: warning}
        annotations:
          summary: Less than 2 GiB of host memory is available
      - alert: GufoRequestsDeferred
        expr: gufo_requests_deferred > 0
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: Requests have waited for a session for ten minutes
      - alert: GufoKVPressure
        expr: |
          sum(rate(gufo_kv_deferred_total[5m])) > 0
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: The KV pool is repeatedly forcing host-snapshot degradation
      - alert: GufoTTFTSlow
        expr: |
          histogram_quantile(0.95,
            sum by (le) (rate(gufo_ttft_seconds_bucket[5m]))) > 30
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: p95 time to first token is above 30 seconds
```

## Validation and troubleshooting

Check the endpoint and its type information:

```sh
curl -sS http://127.0.0.1:8000/metrics | head
curl -sS http://127.0.0.1:8000/metrics | grep '^# TYPE'
```

If `promtool` is installed, validate the exposition directly:

```sh
curl -sS http://127.0.0.1:8000/metrics | promtool check metrics
```

Without `promtool`, verify the essentials manually: every `# HELP` family has
one `# TYPE`, every sample is numeric, histogram bucket counts are
non-decreasing, `_bucket{le="+Inf"}` equals `_count`, and `_count` is not
smaller than any bucket.

Common cases:

| Symptom | Cause and remedy |
| --- | --- |
| HTTP 401 | `--api-key` is set and the scrape has no `Authorization: Bearer` header. Add the Prometheus `authorization` block. |
| HTTP 404 | The request is for another path or another server process. Use `/metrics` or `/v1/metrics` on the text server's listener. |
| Counters stay at zero | No completed text requests yet. Start an inference request and scrape again. |
| Counters drop to zero | The process restarted. Counters reset by design; use `rate()`/`increase()` and keep restart annotations in Grafana. |
| Histograms have no samples | No completed requests; cancelled and failed requests do not observe latency. |
| No `gufo_*` names | The scrape hit a non-text service or an older binary. Check the build and `/props`. |
| Scrape times out | The metrics snapshot reads live session state and `/proc`. Keep the interval at 15 s or shorter and avoid very large session counts on a starved host. |

The endpoint is served on the inference listener. If that listener is public,
keep `--api-key` set or place Prometheus on the same private network and expose
only the routes clients need through a reverse proxy.
