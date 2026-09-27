#!/bin/bash
# Launch vLLM for qwen38 (the full NVFP4 model) on the DGX Spark / GB10.
#
#   qwen38 = local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (125B-A6B)
#            on Eugr's B12x stack (image vllm-node-b12x, pinned by digest).
#
# This service is COMMENTED OUT in docker-compose.yml - qwen38-fast is the active
# backend. See the header of that file for how to switch.
#
# Mirrors the `command:` block of Eugr's current solo recipe
#   spark-vllm-docker/recipes/qwen3.8-flash-next-nvfp4-solo.yaml
# with its {placeholders} inlined. Two deliberate departures:
#   * --served-model-name is qwen3.8-flash-next rather than the repo id, matching the
#     fast backend so config.yaml never changes on a switch;
#   * --chat-template is passed explicitly. Upstream passes none and takes the model's
#     default; this repo passes a copy of that default with the leading-system-message
#     merge applied (see chat-templates/).
#
# Mounted read-only at /workspace/vllm-start-full.sh; compose sets
# entrypoint: ["/bin/bash", "/workspace/vllm-start-full.sh"] to replace the image's own
# ENTRYPOINT ["vllm","serve"], which would otherwise prepend itself to the script path.

set -euo pipefail

MODEL=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
# Same mount as the fast backend: host ~/.cache/huggingface -> /cache/huggingface.
HF_HUB=${HF_HUB_CACHE:-/cache/huggingface/hub}
REPO="$HF_HUB/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4"

die() { echo "FATAL vllm-start-full: $*" >&2; exit 1; }

# This is NOT the nvidia/Qwen3.8-Flash-Next-NVFP4 checkpoint. Eugr's B12x recipe serves
# the 2026-09-16 re-export, which is what --load-format b12x and --quantization
# modelopt_mixed expect. Fail early with the download command rather than in the middle
# of weight loading.
[ -f "$REPO/refs/main" ] || die "$MODEL is not in the HF cache.
  Fetch it first (~124 GiB), e.g.:
    hf download $MODEL
  (host cache: ~/.cache/huggingface/hub, mounted at /cache/huggingface)"

# exec so vLLM becomes PID 1 and receives SIGTERM from `docker stop` directly.
exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --block-size 16 \
  --load-format b12x \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --gdn-decode-kernel b12x \
  --linear-backend b12x \
  --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
  --chat-template /workspace/chat-templates/qwen38-full.jinja \
  --gpu-memory-utilization 0.8
