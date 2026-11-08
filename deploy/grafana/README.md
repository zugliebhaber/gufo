# Gufo Grafana assets

This directory contains the importable dashboard and optional Grafana
provisioning files for Gufo's Prometheus metrics.

| Path | Purpose |
| --- | --- |
| `gufo-dashboard.json` | Manual import. Uses the `DS_PROMETHEUS` input so the import dialog asks for a data source. |
| `dashboards/gufo-dashboard.json` | Provisioning variant. Uses the fixed datasource UID `gufo-prometheus`. |
| `provisioning/datasources/gufo-prometheus.yaml` | Prometheus datasource for file provisioning. Adjust `url`. |
| `provisioning/dashboards/gufo.yaml` | File provider for `/var/lib/grafana/dashboards/gufo`. |

The metric documentation is in [docs/PROMETHEUS.md](../../docs/PROMETHEUS.md);
the Grafana guide is in [docs/GRAFANA.md](../../docs/GRAFANA.md).

## Manual import

Open **Dashboards → New → Import**, upload `gufo-dashboard.json`, select the
Prometheus data source that scrapes Gufo, and import. The dashboard UID is
`gufo-llm`.

## Provisioning

From the repository root, mount both directories into Grafana:

```sh
podman run -d --name grafana --network=host \
  -v "$PWD/deploy/grafana/dashboards:/var/lib/grafana/dashboards/gufo:ro" \
  -v "$PWD/deploy/grafana/provisioning:/etc/grafana/provisioning:ro" \
  grafana/grafana:latest
```

Then set `datasources/gufo-prometheus.yaml`'s `url` to the Prometheus address
reachable from the Grafana container. The provider loads
`dashboards/gufo-dashboard.json` into the **Gufo** folder every 30 seconds.

If Grafana runs outside a container, change the provider `options.path` to the
host path containing `dashboards/gufo-dashboard.json`.

## Validation

```sh
python3 -m json.tool gufo-dashboard.json >/dev/null
python3 -m json.tool dashboards/gufo-dashboard.json >/dev/null
```

See [docs/GRAFANA.md](../../docs/GRAFANA.md#validation) for a live check that
every dashboard query references a metric the server actually emits.
