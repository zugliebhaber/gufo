# Grafana dashboard

Gufo ships an importable Grafana dashboard at
[`deploy/grafana/gufo-dashboard.json`](../deploy/grafana/gufo-dashboard.json).
It visualizes the Prometheus metrics documented in
[Prometheus metrics](PROMETHEUS.md).

The dashboard is independent of the Gufo process: Grafana queries Prometheus,
and Prometheus scrapes Gufo. A running Gufo server does not need a Grafana
plugin, a separate metrics port or a configuration flag. The asset map is in
[`deploy/grafana/README.md`](../deploy/grafana/README.md).

## Prerequisites

1. Prometheus scrapes `GET /metrics` on the Gufo text server. See the
   [scrape configuration](PROMETHEUS.md#scrape-configuration).
2. Prometheus has at least one successful scrape, so the `gufo_*` series exist.
3. Grafana can reach that Prometheus instance.
4. The Grafana user may create or import dashboards.

Verify the data path before touching Grafana:

```sh
curl -sS http://127.0.0.1:8000/metrics | grep '^gufo_requests_total'
```

If this prints no `gufo_requests_total` line, fix the scrape first. A Grafana
dashboard cannot repair a missing data source.

## Import the dashboard

1. Open Grafana.
2. Select **Dashboards → New → Import**.
3. Upload `deploy/grafana/gufo-dashboard.json`, or paste its contents.
4. Select the Prometheus data source that scrapes Gufo. The dashboard input is
   named `DS_PROMETHEUS`.
5. Select **Import**.

The dashboard UID is `gufo-llm` and its title is **Gufo LLM Server**. Importing
the same file again updates the existing dashboard instead of creating a
duplicate.

The default view is the last hour with a 10 s refresh. Rate and histogram
panels use `$__rate_interval`; gauge panels query the latest value. Changing
the time range or scrape interval therefore does not require editing the
queries.

## Dashboard panels

| Panel | Query | What it shows |
| --- | --- | --- |
| Request rate by status | `sum by (status) (rate(gufo_requests_total[$__rate_interval]))` | Completed, cancelled and failed request rate. |
| Success ratio | `sum(rate(...{status="completed"})) / sum(rate(...))` | Share of finished requests that completed. |
| Sessions | `gufo_sessions_active`, `gufo_sessions_capacity` | Active versus configured session slots. |
| Time to first token | `histogram_quantile(0.50/0.95, ...gufo_ttft_seconds_bucket...)` | p50 and p95 TTFT. |
| Request duration | `histogram_quantile(0.50/0.95, ...gufo_request_duration_seconds_bucket...)` | p50 and p95 end-to-end latency. |
| Token throughput | `sum(rate(gufo_prompt_tokens_total...))`, `sum(rate(gufo_tokens_generated_total...))` | Prompt and generated tokens per second. |
| Requests | `gufo_requests_processing`, `gufo_requests_deferred` | In-flight and waiting requests. |
| Session utilization | `gufo_sessions_active / gufo_sessions_capacity` | Fraction of session slots busy. |
| Prefix cache hit rate | `gufo_prefix_cache_hit_rate` | Cached prompt tokens over all prompt tokens. |
| KV cache usage | `gufo_kv_cache_usage_ratio` | Live context occupancy over capacity. |
| KV pressure | `gufo_kv_limited_total`, `gufo_kv_deferred_total`, `gufo_preemptions_total` rates | How often KV capacity shapes or interrupts work. |
| Memory | `gufo_process_resident_memory_bytes`, `gufo_host_available_memory_bytes` | Process RSS and host headroom in bytes. |
| Device lost | `gufo_device_lost_total` | Confirmed GPU context losses; any value above zero is critical. |
| Latest request speeds | `gufo_prompt_tokens_per_second`, `gufo_tokens_per_second` | Last completed request's prompt and decode speed. |

All panel queries reference only names the text server emits. The dashboard
does not use recording rules.

## Provision the dashboard

For repeatable deployments, Grafana can load the dashboard and datasource from
files instead of using the import dialog. Example files are provided under
[`deploy/grafana/provisioning/`](../deploy/grafana/provisioning/). The
provisioning dashboard copy is
[`deploy/grafana/dashboards/gufo-dashboard.json`](../deploy/grafana/dashboards/gufo-dashboard.json);
it uses the fixed datasource UID `gufo-prometheus` instead of the import-time
`DS_PROMETHEUS` input.

From the repository root, mount the dashboard and provisioning directories into
Grafana:

```sh
podman run -d --name grafana --network=host \
  -v "$PWD/deploy/grafana/dashboards:/var/lib/grafana/dashboards/gufo:ro" \
  -v "$PWD/deploy/grafana/provisioning:/etc/grafana/provisioning:ro" \
  grafana/grafana:latest
```

`deploy/grafana/provisioning/datasources/gufo-prometheus.yaml` defines the
Prometheus datasource. Set `url` to the Prometheus address reachable from the
Grafana container. `deploy/grafana/provisioning/dashboards/gufo.yaml` points
the file provider at `/var/lib/grafana/dashboards/gufo`.

If Grafana runs on the host instead of a container, adjust the provider path in
`gufo.yaml` to the host directory that contains `gufo-dashboard.json`. The
provider scans the directory recursively and reloads changed dashboards every
30 seconds.

The provisioned dashboard uses the datasource UID `gufo-prometheus`. Change it
in the datasource file and in the panel datasource references of
`deploy/grafana/dashboards/gufo-dashboard.json` if that UID is already taken.
Manual import uses the `DS_PROMETHEUS` input instead and does not require a
fixed UID.

## Datasource authentication

If Gufo runs with `--api-key`, Prometheus needs the token in its scrape config,
not in Grafana. Keep the token in a file and reference it from Prometheus:

```yaml
authorization:
  type: Bearer
  credentials_file: /etc/prometheus/gufo-token
```

Only configure a Grafana HTTP header if Grafana queries a Prometheus instance
that itself requires authentication. Do not put the Gufo token into dashboard
JSON.

## Customizing the dashboard

The JSON uses standard Grafana `timeseries` and `stat` panels and has no
dashboard-level template variables. Common changes:

- **Add a panel:** copy an existing panel object, give it a new `id` and
  `gridPos`, and replace `targets[].expr`.
- **Change thresholds:** edit `fieldConfig.defaults.thresholds.steps`.
- **Change units:** request rates use `reqps`, ratios use `percentunit`,
  latency uses `s`, and memory uses `bytes`.
- **Add a recording rule:** query the `gufo_*` families from Prometheus, then
  point the panel at the recorded series. This keeps expensive histogram
  queries out of dashboard rendering.
- **Scope by instance:** add a `instance` filter to each query if one
  Prometheus scrapes several Gufo instances. The endpoint itself does not add
  an instance label.

Keep the dashboard UID stable when updating the shipped file. The file provider
updates the dashboard in place; a changed UID creates a second dashboard.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| **No data** in every panel | Is Prometheus scraping Gufo? Run the `curl` command above. Check the Prometheus **Targets** page and the scrape error string. |
| **Datasource not found** after provisioning | The datasource UID must match `gufo-prometheus` in the dashboard JSON, or update both files. |
| **401 Unauthorized** | The Gufo `--api-key` is missing from the Prometheus scrape config. Grafana should not need the Gufo token. |
| Request rate is flat or empty | Counters start at zero after a restart. Generate a request and check that `gufo_requests_total{status="completed"}` increases. |
| TTFT panel is empty | The histogram only observes completed, non-cancelled requests. Check the request rate and failed ratio. |
| Session utilization shows `0/0` | `gufo_sessions_capacity` is zero for a backend without a session pool, or the scrape hit a non-text service. |
| Memory panel reports 0 | The host did not expose `/proc`; the endpoint emits zero as an unavailable reading. |
| Dashboard shows duplicate panels | The JSON was imported under a different UID; remove the duplicate dashboard. |
| Time range shows gaps | Gufo was restarted, or Prometheus could not scrape during that period. Gaps are expected during startup and shutdown. |

## Validation

Validate the dashboard JSON before importing or publishing:

```sh
python3 -m json.tool deploy/grafana/gufo-dashboard.json >/dev/null
```

Check that every metric referenced by the dashboard exists in a live scrape:

```sh
curl -sS http://127.0.0.1:8000/metrics > /tmp/gufo.metrics
python3 - <<'PY'
import json, re
dash = json.load(open("deploy/grafana/gufo-dashboard.json"))
text = open("/tmp/gufo.metrics").read()
for panel in dash["panels"]:
    for target in panel.get("targets", []):
        for name in re.findall(r"\bgufo_[a-zA-Z0-9_]+", target["expr"]):
            base = re.sub(r"_(bucket|sum|count)$", "", name)
            assert re.search(r"^# HELP " + re.escape(base) + r" ", text, re.M), name
print("dashboard metrics present")
PY
```

The repository documentation checks also validate links and JSON blocks:

```sh
python3 tools/ci/check-docs.py --root .
```
