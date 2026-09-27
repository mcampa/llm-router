#!/bin/bash
# Launch vLLM for qwen38-hibrid48 on the DGX Spark / GB10.
#
#   qwen38-hibrid48 = myllmbox/Qwen3.8-Flash-Next-hibrid48 (125B-A6B: NVFP4 body,
#                     4-bit output head, NVFP4 n-gram table)
#                     on bilikaz's myllmbox image v4 (vLLM 0.30), pinned by digest.
#
# This service is COMMENTED OUT in docker-compose.yml - qwen38-fast is the active
# backend. See the header of that file for how to switch.
#
# Mirrors the `vllm:` and `env:` sections of bilikaz's v4 recipe
#   qwen38-flash-next-recipe/recipe.yaml
# - the kit's single source of truth; its run.sh maps `name: value` to `--name value`.
# Three deliberate departures, all documented in the README:
#   * --served-model-name is qwen3.8-flash-next, not the recipe's Qwen/Qwen3.8-Flash-Next,
#     so both backends present the same model id and config.yaml never changes on a switch;
#   * --kv-cache-memory-bytes is 26e9 rather than the recipe's 27e9, and spelled in full
#     rather than the recipe's `--kv-cache-memory` abbreviation of the same option. The
#     recipe documents 26G for a box that runs other things, and this one runs
#     litellm/db/redis/open-webui plus the two exporters;
#   * --chat-template is passed explicitly. The recipe passes none and takes the model's
#     default; this repo passes a copy of that default with the leading-system-message
#     merge applied (see chat-templates/ and the README).
# Everything else, --block-size 1632 and the K=5 speculative config included, is the
# recipe's, unmodified. None of the fast backend's B12x flags or env apply here: this is
# vLLM 0.30 with its own patched NVFP4 table reader, not Eugr's B12x build.
#
# NOTHING here is RDMA: no RoCE env, no /dev/infiniband, no IPC_LOCK. The table is
# demand-paged from local NVMe by the image's own reader (the MBX_PLE_MMAP* env vars in
# docker-compose.yml), and that is what makes this fit on one Spark.
#
# Mounted read-only at /workspace/vllm-start-hibrid48.sh. Compose sets
# entrypoint: ["/bin/bash", "/workspace/vllm-start-hibrid48.sh"], which *replaces* the
# image's own ENTRYPOINT ["vllm","serve"] - it cannot go in `command:`, or that entrypoint
# would prepend itself and yield argv "vllm serve /workspace/vllm-start-hibrid48.sh".

set -euo pipefail

MODEL_NAME=Qwen3.8-Flash-Next-hibrid48
MODEL=/models/$MODEL_NAME
# Where the image puts its compile caches; the ~31G table map lives beside it under
# /cache. Both must be on the NVMe and writable - see the compose volumes.
CACHE_ROOT=${VLLM_CACHE_ROOT:-/cache/vllm-cache}
WORKSPACE=${FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer-workspace}

die() { echo "FATAL vllm-start-hibrid48: $*" >&2; exit 1; }

# --- 1. the checkpoint ---------------------------------------------------------
# Served from a plain --local-dir tree, not the shared HF cache the other backends use
# (that is what the upstream kit does, and the model is 98 GiB).
[ -d "$MODEL" ] || die "$MODEL is not mounted.
  Download it first (~98 GiB), e.g.:
    sudo mkdir -p /models/qwen38/models /models/qwen38/cache
    sudo chown -R \$(id -u):\$(id -g) /models/qwen38
    hf download myllmbox/$MODEL_NAME --local-dir /models/qwen38/models/$MODEL_NAME
  Host /models/qwen38/models is mounted at /models - see docker-compose.yml."

[ -f "$MODEL/model.safetensors.index.json" ] || die "$MODEL has no model.safetensors.index.json.
  The download is incomplete (28 shards, ~98 GiB), or this is not a full snapshot."

# The 28 shards carry the n-gram table inside them, and the image's table library only
# accepts this release's table - it stops with "invalid quant" on anything else. The
# checkpoint writes the marker that identifies it. Catching it here beats catching it
# several minutes into the weight load, and it catches a mistaken nvidia/... download.
grep -q '"ple_quantization"' "$MODEL/config.json" \
  || die "$MODEL/config.json has no ple_quantization - this is not the hibrid48 checkpoint
  view that image v4 expects (nvidia/Qwen3.8-Flash-Next-NVFP4 and the B12x checkpoints are not
  interchangeable here). See the README."

# --- 2. writable NVMe state ----------------------------------------------------
# The first boot prepares the ~31G table map under /cache (a few minutes, once; the same
# map then serves the uncensored checkpoint too). A read-only or tmpfs /cache fails here
# rather than mid-load.
mkdir -p "$CACHE_ROOT" "$WORKSPACE" \
  || die "cannot create $CACHE_ROOT - is /cache mounted read-write (and not sized 0)?"

# --- 3. serve ------------------------------------------------------------------
# --kv-cache-memory-bytes takes BYTES, and its parser also accepts a human-readable
# suffix: a plain count, or `26g` (26e9, decimal) / `26G` (26 GiB, binary) - which differ
# by ~7% for the same two characters. This script writes the full byte count so there is
# nothing to misread, and uses the option's real name: v4's recipe spells it
# `--kv-cache-memory`, an argparse abbreviation of this same option, and both parse. It is
# also the same option the qwen38-fast script already passes - only the value differs.
# 26e9 of bf16 KV is ~800k tokens, the figure the recipe documents for a box that runs
# other things (it ships 27e9). --gpu-memory-utilization 0.70 does not size the pool once
# this is pinned; it bounds everything else (weights, graphs, activations).
# --block-size 1632 is required at K=5: the boot stops with "QSA ring capacity 12 must
# divide the attention block size 1616" without it.
#
# exec so vLLM becomes PID 1 and receives SIGTERM from `docker stop` directly.
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8000 \
  --engram-config '{"cpu_offload": false}' \
  --gdn-prefill-backend triton \
  --moe-backend marlin \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory-bytes 26000000000 \
  --load-format fastsafetensors \
  --max-model-len 262144 \
  --block-size 1632 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,30,36,42,48,54,60,66,72,78,84,90,96],"compile_ranges_endpoints":[32]}' \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"rejection_sample_method":"block","draft_sample_method":"probabilistic"}' \
  --async-scheduling \
  --chat-template /workspace/chat-templates/qwen38-hibrid48.jinja
