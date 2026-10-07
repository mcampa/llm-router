#!/bin/bash
# Launch vLLM for qwen38-fast on the DGX Spark / GB10.
#
#   qwen38-fast = azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound (125B-A5B)
#                 on Eugr's B12x stack (image vllm-node-b12x, pinned by digest).
#
# Mirrors the `command:` block of the upstream recipe
#   Qwen3.8-Flash-Next-Int4-FAST/qwen3.8-flash-next-int4-b12x-solo.yaml
# with its {placeholders} inlined, so the two stay easy to diff. Two deliberate
# departures, both documented in mods/flashnext-int4-b12x/README.md:
#   * --served-model-name is qwen3.8-flash-next, not the checkpoint's own name, so both
#     backends present the same model id and config.yaml never changes on a switch;
#   * the chat template is this repo's copy of the checkpoint's
#     medium_chat_template.jinja with the leading-system-message merge applied.
#
# Mounted read-only at /workspace/vllm-start-fast.sh. Compose sets
# entrypoint: ["/bin/bash", "/workspace/vllm-start-fast.sh"], which *replaces* the
# image's own ENTRYPOINT ["vllm","serve"] - it cannot go in `command:`, or that
# entrypoint would prepend itself and yield argv "vllm serve /workspace/vllm-start-fast.sh".
#
# Steps 1-3 are the upstream `flashnext-int4-b12x` mod, which Eugr's launcher would
# otherwise run for us. Engine behaviour is env-driven (VLLM_FP8_HYBRID,
# VLLM_PLE_TABLE_MEMORY, ...); see the environment block in docker-compose.yml.

set -euo pipefail

MOD_DIR=/workspace/mod
OUT=/workspace/flashnext
# Same mount as the old stack: host ~/.cache/huggingface -> /cache/huggingface.
HF_HUB=${HF_HUB_CACHE:-/cache/huggingface/hub}
REPO="$HF_HUB/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound"

die() { echo "FATAL vllm-start-fast: $*" >&2; exit 1; }

# --- 1. the checkpoint, from the offline HF cache -------------------------------
# HF_HUB_OFFLINE=1: nothing is fetched at runtime, so an incomplete snapshot must fail
# here rather than halfway through weight loading.
[ -f "$REPO/refs/main" ] || die "model not in the HF cache at $REPO"
REV=$(cat "$REPO/refs/main")
SNAP="$REPO/snapshots/$REV"
[ -d "$SNAP/ple-table" ] || die "snapshot ${REV:0:8} has no ple-table/ (download incomplete?)"
for f in config.json model.safetensors.index.json model-healed-shared-expert.safetensors \
         model_extra_tensors.safetensors model-lmhead-int4.safetensors medium_chat_template.jinja; do
  [ -e "$SNAP/$f" ] || die "snapshot ${REV:0:8} is incomplete: $f is missing (download still running?)"
done

# --- 2. model views: symlinks + small JSON, nothing copied ----------------------
# /workspace/flashnext/model      served model: PLE table files indexed at top level,
#                                 config.json given the b12x ple_embedding_dtype and
#                                 the architecture name this vLLM registers
# /workspace/flashnext/draft-k10  slim MTP draft folder (top-k 10, the head's own
#                                 shared-expert width)
mkdir -p "$OUT"
python3 "$MOD_DIR/build_views.py" "$SNAP" "$OUT" || die "could not build the model views"

# --- 3. patch this container's vLLM --------------------------------------------
# Required: fp8-hybrid quant config, int4 lm_head, index filter for the mixed PLE
# files. Optional: draft x2, MTP cap, penalties, pad rows, GDN fixes (each warns and
# continues). Exits non-zero if a required patch cannot be applied - better to abort
# than to serve a broken model.
python3 "$MOD_DIR/patch_b12x.py" "$MOD_DIR" || die "could not patch vLLM"

# The vendored patch_b12x.py is kept byte-for-byte upstream, where the pad-rows patch is
# *optional*: if its anchor moves, the launch only warns and carries on without it. That is
# the one patch this repo is on this revision for, and its failure is quiet - one stream
# drops to ~1 token/step while its output stays correct, so it reads as unexplained
# slowness rather than as a bug. Assert the container's qsa.py really carries the fix
# instead of trusting the patch script's exit status. Booting without it would be worse
# than not booting: config.yaml falls local -> deepseek, so an aborted launch degrades to
# the cloud route rather than going dark.
# The second alternative accepts an image that fixes this upstream without our marker.
if ! grep -rqE --include=qsa.py \
      'self\._request_ids\[num_mapped_tokens:\]\.fill_\(-1\)|flashnext-int4-b12x:drafter-pad-rows' \
      /usr/local/lib/python3.*/dist-packages/vllm/models; then
  die "drafter-pad-rows fix absent from the container's qsa.py (anchor moved upstream?)"
fi

# --- 4. serve ------------------------------------------------------------------
# --gpu-memory-utilization 0.01 is deliberate: the KV pool is sized by
# --kv-cache-memory-bytes (20g) and the rest of the unified memory is left to the rest of
# the box. KV is fp8, which halves the per-token cost, so that same 20g holds ~1.29M tokens
# rather than ~645k. That buys concurrency and prefix-cache retention, not a longer single
# request - --max-model-len 262144 is that ceiling.
#
# MTP stays at 4 on block-size 16 by choice. The vendored mod's mtp-cap patch does raise QSA's cap
# to 7, but depth 5+ needs the upstream recipe's block_size 1632 (its 12-row QSA ring must divide
# the page), so depths above 4 are not enabled here.
#
# exec so vLLM becomes PID 1 and receives SIGTERM from `docker stop` directly.
exec vllm serve "$OUT/model" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --mamba-cache-mode align \
  --recurrent-checkpoint-policy aligned \
  --linear-backend b12x \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --block-size 16 \
  --load-format fastsafetensors \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.01 \
  --kv-cache-memory-bytes 20g \
  --gdn-decode-kernel b12x \
  --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template /workspace/chat-templates/qwen38-fast-medium.jinja \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"model":"/workspace/flashnext/draft-k10","rejection_sample_method":"block","draft_sample_method":"greedy"}'
