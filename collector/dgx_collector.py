#!/usr/bin/env python3
"""
Custom metric collector for NVIDIA DGX Spark (GB10).

GB10 is a *unified-memory* architecture: the GPU has no discrete VRAM
(nvidia-smi reports memory as "Not Supported" and memory.total = [N/A]),
so every CUDA allocation shares the host's system RAM. This collector turns
that shared pool, plus the nvidia-smi GPU telemetry, into Prometheus metrics
served directly over HTTP on :9273/metrics. No node-exporter, no
nvidia-gpu-exporter, no textfile mechanism — it is fully self-contained.

Metrics exposed:
  dgx_unified_memory_total_bytes         - host RAM available to the GPU (bytes)
  dgx_unified_memory_used_bytes          - total - available (bytes)
  dgx_unified_memory_available_bytes     - available for new allocations (bytes)
  dgx_unified_memory_gpu_used_bytes      - sum of all GPU compute+graphics contexts (bytes)
  dgx_unified_memory_process_used_bytes  - per-process unified memory {pid,type,process_name}
  dgx_gpu_utilization_ratio              - nvidia-smi GPU-Util [0..1]
  dgx_gpu_memory_controller_ratio        - nvidia-smi memory controller util [0..1] (may be N/A)
  dgx_gpu_temperature_celsius            - nvidia-smi temperature
  dgx_gpu_power_draw_watts               - nvidia-smi power draw
  dgx_gpu_info                           - static GPU {name,driver,cuda,uuid}
  dgx_gpu_compute_apps                   - number of processes with a compute context
  dgx_collect_success                    - 1 if the last collection succeeded
"""

import os
import re
import subprocess
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Prometheus exposition text format.
EXPOSITION = "prometheus"
PORT = int(os.environ.get("DGX_EXPORTER_PORT", "9273"))
INTERVAL = int(os.environ.get("COLLECT_INTERVAL", "10"))

# Hold the last rendered metrics text; regenerated every INTERVAL.
_LATEST = "# dgx custom collector starting...\n"


def _parse_meminfo():
    """Parse /proc/meminfo for MemTotal and MemAvailable (kB), return bytes."""
    total = avail = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, val = line.partition(":")
                num = re.search(r"[0-9]+", val)
                if not num:
                    continue
                if key == "MemTotal":
                    total = int(num.group(0)) * 1024
                elif key == "MemAvailable":
                    avail = int(num.group(0)) * 1024
    except OSError:
        pass
    return total, avail


def _parse_nvidia_smi_table():
    """
    Parse `nvidia-smi`'s human-readable process table to capture every GPU
    context (compute C + graphics G). Row format:
        |   0   N/A   N/A   2408    G   /usr/lib/xorg/Xorg    18MiB |
    Returns (gpu_used_bytes, [ (pid, type, name, bytes), ... ]) and
    (gpu_util, mem_util, temp, power, app_count) from the summary header.
    """
    procs = []
    gpu_used = 0
    util = mem_util = temp = power = None
    app_count = 0

    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return gpu_used, procs, util, mem_util, temp, power, app_count

    for line in out.splitlines():
        # Summary header: "| N/A   30C    P8      4W /  N/A  | Not Supported | 0% ... |"
        m = re.search(r"\|\s*N/A\s+(\d+)C\s+\S+\s+([\d.]+)W\s*/\s*\S+\s*\|", line)
        if m:
            temp = float(m.group(1))
            power = float(m.group(2))
        # "| Not Supported          |      0%      Default |"
        m = re.search(r"Not Supported\s*\|\s+(\d+)%", line)
        if m:
            util = int(m.group(1)) / 100.0
        # Process row: "|    0   N/A   N/A            2408      G   /usr/lib/xorg/Xorg      18MiB |"
        m = re.search(
            r"^\|\s+0\s+N/A\s+N/A\s+(\d+)\s+([GC])\s+(.*?)\s+(\d+)(MiB|GiB)\s*\|$", line
        )
        if m:
            pid, ptype, name = m.group(1), m.group(2), m.group(3).strip()
            amount = int(m.group(4))
            if m.group(5) == "GiB":
                amount *= 1024
            proc_bytes = amount * 1024 * 1024
            gpu_used += proc_bytes
            if ptype == "C":
                app_count += 1
            procs.append((pid, ptype, name, proc_bytes))

    return gpu_used, procs, util, mem_util, temp, power, app_count


def _nvidia_driver():
    """
    Static GPU info via nvidia-smi plus host-version env vars.
    Inside the container nvidia-smi sees the driver runtime but reports
    "CUDA Version: N/A" (no CUDA toolkit). docker-compose.yml therefore sets
    DGX_DRIVER_VERSION and DGX_CUDA_VERSION to the host's real values; those are
    authoritative and preferred. nvidia-smi supplies name/uuid/compute_mode/
    persistence_mode/pstate.
    """
    info = {"name": "NVIDIA GB10", "driver": "unknown", "cuda": "unknown", "uuid": "unknown",
            "compute_mode": "unknown", "persistence_mode": "unknown", "pstate": "unknown"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,uuid,compute_mode,persistence_mode,pstate",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        ).stdout
        # values may contain commas? (typically not for these fields)
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 6:
            info["name"] = parts[0]
            info["driver"] = parts[1]
            info["uuid"] = parts[2]
            info["compute_mode"] = parts[3]
            info["persistence_mode"] = parts[4]
            info["pstate"] = parts[5]
    except OSError:
        pass

    # Override driver + CUDA from env vars (authoritative host values set in
    # docker-compose.yml). nvidia-smi may report an empty/placeholder driver in
    # some container setups, and CUDA is always "N/A" here.
    env_driver = os.environ.get("DGX_DRIVER_VERSION", "").strip()
    env_cuda = os.environ.get("DGX_CUDA_VERSION", "").strip()
    if env_driver:
        info["driver"] = env_driver
    if env_cuda:
        info["cuda"] = env_cuda
    return info


def collect():
    """Render the full metrics endpoint text."""
    total, avail = _parse_meminfo()
    used = total - avail
    gpu_used, procs, util, mem_util, temp, power, app_count = _parse_nvidia_smi_table()
    info = _nvidia_driver()

    L = []
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')

    def gauge(name, help_, value, labels=None):
        lbl = ""
        if labels:
            lbl = "{" + ",".join(f'{k}="{esc(v)}"' for k, v in labels.items()) + "}"
        L.append(f"# HELP {name} {help_}")
        L.append(f"# TYPE {name} gauge")
        L.append(f"{name}{lbl} {value}")

    gauge("dgx_unified_memory_total_bytes",
          "Total unified memory (host system RAM) available to the GPU.", total)
    gauge("dgx_unified_memory_used_bytes",
          "Unified memory in active use (total - available).", used)
    gauge("dgx_unified_memory_available_bytes",
          "Unified memory available for new allocations.", avail)
    gauge("dgx_unified_memory_gpu_used_bytes",
          "Unified memory held by active GPU processes (sum of compute+graphics).", gpu_used)

    # Per-process unified memory as a HELP/TYPE block then labeled series.
    L.append("# HELP dgx_unified_memory_process_used_bytes "
             "Unified memory held by one GPU process (compute C or graphics G context).")
    L.append("# TYPE dgx_unified_memory_process_used_bytes gauge")
    for pid, ptype, name, pbytes in procs:
        L.append(f'dgx_unified_memory_process_used_bytes{{pid="{pid}",type="{ptype}",'
                 f'process_name="{esc(name)}"}} {pbytes}')

    def rat(ratio):
        return f"{ratio:.6f}" if ratio is not None else "nan"

    gauge("dgx_gpu_utilization_ratio",
          "GPU compute utilization (nvidia-smi GPU-Util) as a ratio.", rat(util))
    # Mem controller util is not reported on GB10 (Not Supported).
    if mem_util is not None:
        gauge("dgx_gpu_memory_controller_ratio",
              "Memory controller utilization ratio.", rat(mem_util))
    if temp is not None:
        gauge("dgx_gpu_temperature_celsius", "GPU temperature in Celsius.", f"{temp:.1f}")
    if power is not None:
        gauge("dgx_gpu_power_draw_watts", "GPU power draw in watts.", f"{power:.2f}")

    gauge("dgx_gpu_compute_apps",
          "Number of processes with a compute context on the GPU.", app_count)
    gauge("dgx_gpu_info", "Static GPU info.",
          1, {"name": info["name"], "driver": info["driver"],
              "cuda": info["cuda"], "uuid": info["uuid"][:8]})
    gauge("dgx_gpu_pstate", "GPU performance state (NVIDIA P-state).",
          1, {"pstate": info["pstate"]})
    # Compute/persistence modes as both a readable label gauge and a numeric
    # gauge that easy to threshold on.
    gauge("dgx_gpu_compute_mode", "GPU compute mode.",
          1, {"mode": info["compute_mode"]})
    gauge("dgx_gpu_compute_mode_enabled", "1 if compute mode is 'Exclusive_Process'.",
          1 if info["compute_mode"] == "Exclusive_Process" else 0)
    gauge("dgx_gpu_persistence_mode", "GPU persistence mode.",
          1, {"mode": info["persistence_mode"]})
    gauge("dgx_gpu_persistence_mode_enabled",
          "1 if persistence mode is Enabled (0 if Disabled).",
          1 if info["persistence_mode"].lower().startswith("enabled") else 0)
    gauge("dgx_collect_success", "Whether collection succeeded (1) or not (0).", 1)

    return "\n".join(L) + "\n"


def _loop():
    global _LATEST
    while True:
        try:
            _LATEST = collect()
        except Exception as exc:  # keep serving the last good payload
            _LATEST = (
                "# HELP dgx_collect_success Whether collection succeeded (1) or not (0).\n"
                "# TYPE dgx_collect_success gauge\n"
                f"dgx_collect_success 0\n"
                f"# collector error: {exc}\n"
            )
        time.sleep(INTERVAL)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") in ("", "/metrics") or self.path == "/metrics":
            body = _LATEST.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # keep logs quiet


def main():
    import threading
    threading.Thread(target=_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()