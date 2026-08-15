# DGX Spark Dashboard

Docker Compose monitoring dashboard for an **NVIDIA DGX Spark (GB10)** with a
single handwritten custom collector that exports **unified-memory** and
**nvidia-smi** GPU metrics directly to Prometheus.

## Why a custom collector?

GB10 uses a *unified-memory* architecture: the GPU has **no discrete VRAM**
(`nvidia-smi` reports memory as `Not Supported` / `N/A`), so every CUDA
allocation shares the host's ~130 GB system RAM. Standard exporters therefore
expose no usable memory figure. This project replaces both `nvidia-gpu-exporter`
and `node-exporter` with one self-contained Python collector that:

- parses `/proc/meminfo` for unified-memory total / used / available
- parses the `nvidia-smi` per-process table for GPU compute + graphics contexts
- reads GPU utilization, temperature, power draw and process count from `nvidia-smi`
- serves all of it over HTTP as Prometheus `/metrics` (no textfile / node-exporter)

The collector installs with the NVIDIA container runtime so `nvidia-smi` works
and sees host GPU workloads (e.g. vLLM).

## Components

| Service | Port | Role |
|---|---|---|
| `dgx-collector` | 9273 | Custom unified-memory + nvidia-smi collector |
| `prometheus` | 9090 | Scrapes the collector |
| `grafana` | 3000 | Dashboard (provisioned automatically) |

## Metrics (`dgx_*`)

- `dgx_unified_memory_total_bytes` / `used_bytes` / `available_bytes`
- `dgx_unified_memory_gpu_used_bytes` — sum across all GPU process contexts
- `dgx_unified_memory_process_used_bytes{pid,type,process_name}` — per process
- `dgx_gpu_utilization_ratio`, `dgx_gpu_temperature_celsius`,
  `dgx_gpu_power_draw_watts`, `dgx_gpu_compute_apps`
- `dgx_gpu_info{name,driver,cuda,uuid}`, `dgx_gpu_pstate`,
  `dgx_gpu_compute_mode`, `dgx_gpu_persistence_mode`
- `dgx_collect_success` — 1 if the last collection succeeded

## Quick start

```bash
docker compose up -d
docker compose ps
```

Then open:

- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090

Grafana auto-provisions the Prometheus datasource and dashboard on first start.

> `NVIDIA_DRIVER_CAPABILITIES=utility`, the NVIDIA device reservation, and the
> `/proc/meminfo` mount are required for the collector to read GPU + memory data.

## Layout

```
docker-compose.yml                      # collector + prometheus + grafana
collector/dgx_collector.py              # handwritten custom collector
prometheus/prometheus.yml               # scrape configs
grafana/provisioning/                   # datasource + dashboard (auto-loaded)
  datasources/prometheus.yml
  dashboards/dashboards.yml
  dashboards/dgx-gpu-smi.json
```