#!/bin/bash
# Launch vLLM for Qwen3.8-Flash-Next-NVFP4 on the DGX Spark.
#
# Mirrors the `command:` block of the sparkrun recipe
#   ~/sparkrun-recipes/recipes/qwen3.8-flash-next-nvfp4.yaml
# so the two stay easy to diff. The recipe templated ({model}, {max_model_len},
# {splitting_ops}, ...); those placeholders are inlined here. Every argument is
# otherwise byte-for-byte the same as the running server's argv.
#
# Mounted read-only at /workspace/vllm-start.sh. The compose entrypoint is
# ["/bin/bash", "/workspace/vllm-start.sh"], which *replaces* the image's own
# ENTRYPOINT ["vllm","serve"] - it cannot go in `command:`, or that entrypoint
# would prepend itself and yield argv "vllm serve /workspace/vllm-start.sh".
#
# Engine behaviour is env-driven (VLLM_PLE_MMAP, VLLM_FP8_HYBRID, ...); see the
# environment block in docker-compose.yml.

set -euo pipefail

# exec so vLLM becomes PID 1 and receives SIGTERM from `docker stop` directly.
exec vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --load-format safetensors \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  -cc.cudagraph_mode=PIECEWISE \
  -cc.splitting_ops='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]' \
  --no-enable-flashinfer-autotune \
  --kv-cache-dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template /workspace/vllm/chat_template.jinja \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
