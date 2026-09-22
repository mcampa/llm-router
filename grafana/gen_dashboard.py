#!/usr/bin/env python3
"""Generate the LLM Router Overview Grafana dashboard JSON."""
import json

DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
RI = "$__rate_interval"
MODEL = 'model_name=~"$model"'

_id = [0]
def nid():
    _id[0] += 1
    return _id[0]

panels = []

def gp(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}

def tgt(expr, legend=None, refid="A", instant=False):
    t = {"datasource": DS, "expr": expr, "refId": refid}
    if legend is not None:
        t["legendFormat"] = legend
    if instant:
        t["instant"] = True
        t["range"] = False
    return t

def row(title, y):
    return {"type": "row", "title": title, "collapsed": False,
            "gridPos": gp(0, y, 24, 1), "id": nid(), "panels": []}

def stat(title, targets, gpos, unit="none", decimals=None, mappings=None,
         thresholds=None, color_mode="value", graph_mode="area", text_mode="auto"):
    fc = {"defaults": {"unit": unit, "mappings": mappings or [],
                       "thresholds": thresholds or {"mode": "absolute",
                       "steps": [{"color": "green", "value": None}]}},
          "overrides": []}
    if decimals is not None:
        fc["defaults"]["decimals"] = decimals
    return {"type": "stat", "title": title, "datasource": DS, "id": nid(),
            "gridPos": gpos, "targets": targets, "fieldConfig": fc,
            "options": {"colorMode": color_mode, "graphMode": graph_mode,
                        "justifyMode": "auto", "textMode": text_mode,
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "",
                                          "values": False}}}

def gauge(title, targets, gpos, unit="percent", mn=0, mx=100, thresholds=None):
    fc = {"defaults": {"unit": unit, "min": mn, "max": mx,
                       "thresholds": thresholds or {"mode": "absolute", "steps": [
                           {"color": "green", "value": None},
                           {"color": "yellow", "value": 70},
                           {"color": "red", "value": 90}]}},
          "overrides": []}
    return {"type": "gauge", "title": title, "datasource": DS, "id": nid(),
            "gridPos": gpos, "targets": targets, "fieldConfig": fc,
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "",
                        "values": False}, "showThresholdLabels": False,
                        "showThresholdMarkers": True}}

def ts(title, targets, gpos, unit="short", draw="line", stack=False, fill=10,
       legend_calcs=None, decimals=None, mn=None, mx=None):
    defaults = {"unit": unit, "custom": {
        "drawStyle": draw, "lineInterpolation": "smooth", "lineWidth": 2,
        "fillOpacity": fill, "gradientMode": "opacity", "spanNulls": True,
        "showPoints": "never", "pointSize": 5,
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        "axisPlacement": "auto", "axisLabel": ""},
        "color": {"mode": "palette-classic"},
        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
        "mappings": []}
    if decimals is not None:
        defaults["decimals"] = decimals
    if mn is not None:
        defaults["min"] = mn
    if mx is not None:
        defaults["max"] = mx
    return {"type": "timeseries", "title": title, "datasource": DS, "id": nid(),
            "gridPos": gpos, "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"legend": {"displayMode": "table" if legend_calcs else "list",
                        "placement": "bottom", "calcs": legend_calcs or [],
                        "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}

def text(title, md, gpos):
    return {"type": "text", "title": title, "id": nid(), "gridPos": gpos,
            "options": {"mode": "markdown", "content": md}}

UP_MAP = [{"type": "value", "options": {"0": {"text": "DOWN", "color": "red"},
           "1": {"text": "UP", "color": "green"}}}]

def qpct(expr):
    return expr

# ============================ OVERVIEW ============================
y = 0
panels.append(row("🟢 Fleet Overview", y)); y += 1
panels.append(stat("Model", [tgt('max(label_replace(vllm:num_requests_running{%s}, "v", "$1", "model_name", "(.*)")) by (v)' % MODEL, "{{v}}", instant=True)],
                   gp(0, y, 4, 4), text_mode="name", color_mode="none", graph_mode="none"))
panels.append(stat("vLLM", [tgt('up{job="vllm"}', instant=True)], gp(4, y, 2, 4),
                   mappings=UP_MAP, color_mode="background", graph_mode="none",
                   thresholds={"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}))
panels.append(stat("GPU", [tgt('up{job="dcgm"}', instant=True)], gp(6, y, 2, 4),
                   mappings=UP_MAP, color_mode="background", graph_mode="none",
                   thresholds={"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}))
panels.append(stat("Host", [tgt('up{job="gx10-host"}', instant=True)], gp(8, y, 2, 4),
                   mappings=UP_MAP, color_mode="background", graph_mode="none",
                   thresholds={"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}]}))
panels.append(stat("Running", [tgt('sum(vllm:num_requests_running{%s})' % MODEL)],
                   gp(10, y, 3, 4), unit="none",
                   thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 8}, {"color": "red", "value": 32}]}))
panels.append(stat("Waiting", [tgt('sum(vllm:num_requests_waiting{%s})' % MODEL)],
                   gp(13, y, 3, 4), unit="none",
                   thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 1}, {"color": "red", "value": 10}]}))
panels.append(stat("Gen tok/s", [tgt('sum(rate(vllm:generation_tokens_total{%s}[%s]))' % (MODEL, RI))],
                   gp(16, y, 4, 4), unit="none", decimals=1, color_mode="value"))
panels.append(stat("Prompt tok/s", [tgt('sum(rate(vllm:prompt_tokens_total{%s}[%s]))' % (MODEL, RI))],
                   gp(20, y, 4, 4), unit="none", decimals=1, color_mode="value"))
y += 4
panels.append(gauge("KV Cache", [tgt('avg(vllm:kv_cache_usage_perc{%s})*100' % MODEL)], gp(0, y, 4, 4)))
panels.append(gauge("GPU Util", [tgt('avg(DCGM_FI_DEV_GPU_UTIL{job="dcgm"})')], gp(4, y, 4, 4)))
panels.append(stat("GPU Temp", [tgt('max(DCGM_FI_DEV_GPU_TEMP{job="dcgm"})')], gp(8, y, 4, 4),
                   unit="celsius", color_mode="value",
                   thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 70}, {"color": "red", "value": 85}]}))
panels.append(stat("GPU Power", [tgt('sum(DCGM_FI_DEV_POWER_USAGE{job="dcgm"})')], gp(12, y, 4, 4),
                   unit="watt", decimals=1, color_mode="value"))
panels.append(gauge("Host CPU", [tgt('100 - (avg(rate(node_cpu_seconds_total{job="gx10-host",mode="idle"}[%s]))*100)' % RI)], gp(16, y, 4, 4)))
panels.append(gauge("Host RAM", [tgt('(1 - node_memory_MemAvailable_bytes{job="gx10-host"} / node_memory_MemTotal_bytes{job="gx10-host"})*100')], gp(20, y, 4, 4)))
y += 4

# ============================ LITELLM NOTE ============================
panels.append(row("🚪 LiteLLM Gateway", y)); y += 1
note = ("### LiteLLM metrics not wired\n"
        "LiteLLM's built-in Prometheus `/metrics` endpoint is an **Enterprise** feature "
        "(moved out of beta into Enterprise on 2024-09-15, ~\\$250/mo). On this OSS build "
        "`GET /metrics` returns **404** even with the master key, so there is no scrape target.\n\n"
        "**Options to populate this row later:**\n"
        "- Run the free community DB-exporter [`ncecere/exporter-litellm`](https://github.com/ncecere/exporter-litellm) "
        "(reads the LiteLLM Postgres DB → spend, tokens, latency, per-model usage).\n"
        "- Request a free 3-month Enterprise license from BerriAI (beta-migration policy).\n\n"
        "vLLM (the actual inference backend behind the `deepseek→local` fallback) is fully instrumented below.")
panels.append(text("About LiteLLM metrics", note, gp(0, y, 24, 5))); y += 5

# ============================ VLLM ============================
panels.append(row("⚡ vLLM Inference", y)); y += 1
panels.append(ts("Token Throughput", [
    tgt('sum(rate(vllm:generation_tokens_total{%s}[%s]))' % (MODEL, RI), "generation tok/s", "A"),
    tgt('sum(rate(vllm:prompt_tokens_total{%s}[%s]))' % (MODEL, RI), "prompt tok/s", "B")],
    gp(0, y, 12, 8), unit="none", fill=15, legend_calcs=["mean", "max", "lastNotNull"]))
panels.append(ts("Requests: Running vs Waiting", [
    tgt('sum(vllm:num_requests_running{%s})' % MODEL, "running", "A"),
    tgt('sum(vllm:num_requests_waiting{%s})' % MODEL, "waiting", "B")],
    gp(12, y, 12, 8), unit="none", draw="line", fill=20, stack=False,
    legend_calcs=["mean", "max"]))
y += 8
def quant(metric, qs):
    return [tgt('histogram_quantile(%s, sum(rate(%s_bucket{%s}[%s])) by (le))' % (q, metric, MODEL, RI),
                "p%d" % int(float(q)*100), rid) for q, rid in zip(qs, ["A", "B", "C"])]
panels.append(ts("Time To First Token (TTFT)", quant("vllm:time_to_first_token_seconds", ["0.5", "0.95", "0.99"]),
                 gp(0, y, 8, 8), unit="s", legend_calcs=["mean", "max"]))
panels.append(ts("Inter-Token Latency (TPOT)", quant("vllm:inter_token_latency_seconds", ["0.5", "0.95", "0.99"]),
                 gp(8, y, 8, 8), unit="s", legend_calcs=["mean", "max"]))
panels.append(ts("End-to-End Request Latency", quant("vllm:e2e_request_latency_seconds", ["0.5", "0.95", "0.99"]),
                 gp(16, y, 8, 8), unit="s", legend_calcs=["mean", "max"]))
y += 8
panels.append(ts("Queue Time (p50/p95/p99)", quant("vllm:request_queue_time_seconds", ["0.5", "0.95", "0.99"]),
                 gp(0, y, 8, 8), unit="s", legend_calcs=["max"]))
panels.append(ts("Prefill vs Decode Time (p95)", [
    tgt('histogram_quantile(0.95, sum(rate(vllm:request_prefill_time_seconds_bucket{%s}[%s])) by (le))' % (MODEL, RI), "prefill p95", "A"),
    tgt('histogram_quantile(0.95, sum(rate(vllm:request_decode_time_seconds_bucket{%s}[%s])) by (le))' % (MODEL, RI), "decode p95", "B")],
    gp(8, y, 8, 8), unit="s", legend_calcs=["max"]))
panels.append(ts("KV Cache Usage %", [tgt('avg(vllm:kv_cache_usage_perc{%s})*100' % MODEL, "kv cache", "A")],
                 gp(16, y, 8, 8), unit="percent", fill=20, mn=0, mx=100))
y += 8
panels.append(ts("Success by Finish Reason", [
    tgt('sum by (finished_reason) (rate(vllm:request_success_total{%s}[%s]))' % (MODEL, RI), "{{finished_reason}}", "A")],
    gp(0, y, 8, 8), unit="reqps", draw="bars", stack=True))
panels.append(ts("Prefix Cache Hit Ratio", [
    tgt('sum(rate(vllm:prefix_cache_hits_total{%s}[%s])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total{%s}[%s])), 1)' % (MODEL, RI, MODEL, RI), "hit ratio", "A")],
    gp(8, y, 8, 8), unit="percentunit", fill=20, mn=0, mx=1))
panels.append(ts("Preemptions /s", [tgt('sum(rate(vllm:num_preemptions_total{%s}[%s]))' % (MODEL, RI), "preemptions/s", "A")],
                 gp(16, y, 8, 8), unit="none", draw="bars", fill=40))
y += 8
panels.append(ts("Model FLOPs Utilization (MFU)", [
    tgt('sum(rate(vllm:estimated_flops_per_gpu_total{%s}[%s])) / ($peak_tflops * 1e12) * 100' % (MODEL, RI), "MFU %", "A")],
    gp(0, y, 8, 8), unit="percent", fill=15, mn=0))
panels.append(ts("Spec-Decode Acceptance Rate", [
    tgt('sum(rate(vllm:spec_decode_num_accepted_tokens_total{%s}[%s])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total{%s}[%s])), 1)' % (MODEL, RI, MODEL, RI), "accept ratio", "A")],
    gp(8, y, 8, 8), unit="percentunit", fill=15, mn=0, mx=1))
panels.append(ts("Iteration Batch Tokens /s", [tgt('sum(rate(vllm:iteration_tokens_total{%s}[%s]))' % (MODEL, RI), "iter tok/s", "A")],
                 gp(16, y, 8, 8), unit="none", fill=15))
y += 8

# ============================ GPU (DCGM) ============================
panels.append(row("🎮 GPU (DCGM)", y)); y += 1
panels.append(ts("GPU & Memory Utilization", [
    tgt('DCGM_FI_DEV_GPU_UTIL{job="dcgm"}', "gpu{{gpu}} compute", "A"),
    tgt('DCGM_FI_DEV_MEM_COPY_UTIL{job="dcgm"}', "gpu{{gpu}} mem-copy", "B")],
    gp(0, y, 12, 8), unit="percent", fill=15, mn=0, mx=100, legend_calcs=["mean", "max"]))
panels.append(ts("Power Draw", [tgt('DCGM_FI_DEV_POWER_USAGE{job="dcgm"}', "gpu{{gpu}}", "A")],
                 gp(12, y, 12, 8), unit="watt", fill=15, legend_calcs=["mean", "max"]))
y += 8
panels.append(ts("Temperature", [
    tgt('DCGM_FI_DEV_GPU_TEMP{job="dcgm"}', "gpu{{gpu}} core", "A"),
    tgt('DCGM_FI_DEV_MEMORY_TEMP{job="dcgm"}', "gpu{{gpu}} mem", "B")],
    gp(0, y, 8, 8), unit="celsius", legend_calcs=["max"]))
panels.append(ts("SM Clock", [tgt('DCGM_FI_DEV_SM_CLOCK{job="dcgm"}', "gpu{{gpu}}", "A")],
                 gp(8, y, 8, 8), unit="rothz", legend_calcs=["mean", "max"]))
panels.append(ts("Energy Consumption Rate", [
    tgt('rate(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{job="dcgm"}[%s]) / 1000' % RI, "gpu{{gpu}} J/s", "A")],
    gp(16, y, 8, 8), unit="watt", fill=15))
y += 8
panels.append(ts("NVLink Bandwidth", [tgt('rate(DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL{job="dcgm"}[%s])' % RI, "gpu{{gpu}}", "A")],
                 gp(0, y, 6, 8), unit="KiBs"))
panels.append(ts("PCIe Replay /s", [tgt('rate(DCGM_FI_DEV_PCIE_REPLAY_COUNTER{job="dcgm"}[%s])' % RI, "gpu{{gpu}}", "A")],
                 gp(6, y, 6, 8), unit="none", draw="bars"))
panels.append(stat("XID Errors", [tgt('max(DCGM_FI_DEV_XID_ERRORS{job="dcgm"})')], gp(12, y, 6, 8),
                   unit="none", color_mode="background",
                   thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}]}))
panels.append(ts("Encoder / Decoder Util", [
    tgt('DCGM_FI_DEV_ENC_UTIL{job="dcgm"}', "gpu{{gpu}} enc", "A"),
    tgt('DCGM_FI_DEV_DEC_UTIL{job="dcgm"}', "gpu{{gpu}} dec", "B")],
    gp(18, y, 6, 8), unit="percent", mn=0, mx=100))
y += 8

# ============================ HOST ============================
panels.append(row("🖥 Host / Local Resources", y)); y += 1
panels.append(ts("CPU Usage by Mode", [
    tgt('sum by (mode) (rate(node_cpu_seconds_total{job="gx10-host",mode!="idle"}[%s]))' % RI, "{{mode}}", "A")],
    gp(0, y, 12, 8), unit="percentunit", stack=True, fill=30, legend_calcs=["mean", "max"]))
panels.append(ts("Load Average", [
    tgt('node_load1{job="gx10-host"}', "1m", "A"),
    tgt('node_load5{job="gx10-host"}', "5m", "B"),
    tgt('node_load15{job="gx10-host"}', "15m", "C")],
    gp(12, y, 12, 8), unit="short", legend_calcs=["mean", "max"]))
y += 8
panels.append(ts("Memory (Unified)", [
    tgt('node_memory_MemTotal_bytes{job="gx10-host"} - node_memory_MemAvailable_bytes{job="gx10-host"}', "used", "A"),
    tgt('node_memory_Cached_bytes{job="gx10-host"} + node_memory_Buffers_bytes{job="gx10-host"}', "cache+buffers", "B"),
    tgt('node_memory_MemAvailable_bytes{job="gx10-host"}', "available", "C"),
    tgt('node_memory_MemTotal_bytes{job="gx10-host"}', "total", "D")],
    gp(0, y, 12, 8), unit="bytes", legend_calcs=["mean", "max"]))
panels.append(ts("Swap Used", [
    tgt('node_memory_SwapTotal_bytes{job="gx10-host"} - node_memory_SwapFree_bytes{job="gx10-host"}', "swap used", "A")],
    gp(12, y, 12, 8), unit="bytes", fill=15))
y += 8
panels.append(ts("Disk I/O", [
    tgt('sum(rate(node_disk_read_bytes_total{job="gx10-host"}[%s]))' % RI, "read", "A"),
    tgt('sum(rate(node_disk_written_bytes_total{job="gx10-host"}[%s]))' % RI, "write", "B")],
    gp(0, y, 8, 8), unit="Bps", legend_calcs=["mean", "max"]))
panels.append(ts("Network Throughput", [
    tgt('sum(rate(node_network_receive_bytes_total{job="gx10-host",device!="lo"}[%s]))*8' % RI, "rx", "A"),
    tgt('sum(rate(node_network_transmit_bytes_total{job="gx10-host",device!="lo"}[%s]))*8' % RI, "tx", "B")],
    gp(8, y, 8, 8), unit="bps", legend_calcs=["mean", "max"]))
panels.append(ts("Filesystem Used %", [
    tgt('(1 - node_filesystem_avail_bytes{job="gx10-host",fstype!~"tmpfs|overlay"} / node_filesystem_size_bytes{job="gx10-host",fstype!~"tmpfs|overlay"})*100', "{{mountpoint}}", "A")],
    gp(16, y, 8, 8), unit="percent", mn=0, mx=100, legend_calcs=["lastNotNull"]))
y += 8
panels.append(ts("CPU Pressure (stall %)", [
    tgt('rate(node_pressure_cpu_waiting_seconds_total{job="gx10-host"}[%s])*100' % RI, "cpu some", "A"),
    tgt('rate(node_pressure_io_waiting_seconds_total{job="gx10-host"}[%s])*100' % RI, "io some", "B"),
    tgt('rate(node_pressure_memory_waiting_seconds_total{job="gx10-host"}[%s])*100' % RI, "mem some", "C")],
    gp(0, y, 8, 8), unit="percent"))
panels.append(ts("Open File Descriptors", [tgt('node_filefd_allocated{job="gx10-host"}', "allocated", "A")],
                 gp(8, y, 8, 8), unit="short", fill=15))
panels.append(stat("Uptime", [tgt('node_time_seconds{job="gx10-host"} - node_boot_time_seconds{job="gx10-host"}')],
                   gp(16, y, 8, 8), unit="dtdurations", color_mode="value", graph_mode="none"))
y += 8

# ============================ RELIABILITY ============================
panels.append(row("🛡 Reliability", y)); y += 1
panels.append(ts("vLLM Error Rate", [
    tgt('sum(rate(vllm:request_success_total{%s,finished_reason="error"}[%s]))' % (MODEL, RI), "errors/s", "A")],
    gp(0, y, 12, 8), unit="reqps", draw="bars", fill=40))
panels.append(ts("vLLM Process Uptime", [
    tgt('time() - process_start_time_seconds{job="vllm"}', "uptime", "A")],
    gp(12, y, 12, 8), unit="s", fill=10))
y += 8

# ============================ DASHBOARD ============================
dashboard = {
    "uid": "llm-router-overview",
    "title": "LLM Router Overview",
    "tags": ["llm", "vllm", "gpu", "litellm"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"},
    "editable": True,
    "graphTooltip": 1,
    "annotations": {"list": [
        {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
         "enable": True, "hide": True, "name": "Annotations & Alerts",
         "type": "dashboard"},
        {"name": "vLLM restarts", "datasource": DS, "enable": True,
         "iconColor": "red", "titleFormat": "vLLM restart",
         "expr": 'changes(process_start_time_seconds{job="vllm"}[2m]) > 0',
         "step": "60s"}
    ]},
    "templating": {"list": [
        {"name": "DS_PROMETHEUS", "type": "datasource", "label": "Datasource",
         "query": "prometheus", "current": {"text": "Prometheus", "value": "Prometheus"},
         "refresh": 1, "hide": 0},
        {"name": "model", "type": "query", "label": "Model", "datasource": DS,
         "query": {"query": "label_values(vllm:num_requests_running, model_name)", "refId": "A"},
         "refresh": 2, "includeAll": True, "multi": True, "allValue": ".*",
         "current": {"text": "All", "value": "$__all"}, "sort": 1},
        {"name": "peak_tflops", "type": "constant", "label": "Peak TFLOP/s (per GPU)",
         "query": "1000", "current": {"text": "1000", "value": "1000"}, "hide": 2}
    ]},
    "panels": panels,
}

out = "/home/mcampa/llm-router/grafana/llm-router-overview.json"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(dashboard, f, indent=2)
print("panels:", len([p for p in panels if p["type"] != "row"]), "rows:",
      len([p for p in panels if p["type"] == "row"]))
print("wrote", out)
