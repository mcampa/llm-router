#!/bin/bash
# Launch vLLM for qwen38-myllmbox on the DGX Spark / GB10.
#
#   qwen38-myllmbox = azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound (125B-A5B)
#                     on myllmbox's qwen38-flash-next-recipe v5.2 image.
#
# Mirrors recipe.yaml's `vllm:` section at tag v5.2, with the INT4-AutoRound profile
# `run.sh` applies on top of it (the checkpoint under ./models has a ple-table/, which is
# how run.sh decides). Everything below is transcribed from run.sh, not invented:
#
#   * drops --moe-backend: int4 experts take Marlin anyway, and the draft head's bf16
#     experts refuse it;
#   * injects "model":"/cache/draft-k10" into --speculative-config, and builds that folder;
#   * --kv-cache-memory 27000000000 -> 24000000000 ("27G leaves too little headroom");
#   * adds VLLM_FP8_HYBRID=1 and MBX_MTP_DRAFT_SCALE=2;
#   * points the fp8 n-gram table at the checkpoint's ple-table/, with its own map in
#     /cache/ple-nvme-fp8 (~52G, built on the first boot of this backend).
#
# Two deliberate departures from the recipe, both required to sit behind this repo's own
# LiteLLM and clients rather than myllmbox's front end:
#
#   * --served-model-name is qwen3.8-flash-next, not Qwen/Qwen3.8-Flash-Next, so the
#     `local` alias and config.yaml are unchanged across a switch;
#   * --chat-template qwen38-fast-medium.jinja. The recipe passes none, so the checkpoint's
#     own template applies - which rejects the two leading {role:system} blocks OpenCode and
#     Hindsight emit (HTTP 400) and leaves reasoning effort off `medium`. See the note before
#     the serve line.
#
# Mounted read-only at /workspace/vllm-start-myllmbox.sh. Compose sets
# entrypoint: ["/bin/bash", "/workspace/vllm-start-myllmbox.sh"], which *replaces* the
# image's own ENTRYPOINT - it cannot go in `command:`.

set -euo pipefail

# The HF repo dir is mounted whole (not its snapshot/): snapshot entries are relative
# symlinks into ../../blobs/, so mounting the snapshot alone breaks every weight.
MODEL_ROOT=/models/azampatti
CACHE=/cache
OUT=$CACHE/draft-k10

die() { echo "FATAL vllm-start-myllmbox: $*" >&2; exit 1; }

# --- 1. the checkpoint, from the offline HF cache -------------------------------
[ -f "$MODEL_ROOT/refs/main" ] || die "model not in the HF cache at $MODEL_ROOT"
REV=$(cat "$MODEL_ROOT/refs/main")
SNAP="$MODEL_ROOT/snapshots/$REV"
[ -d "$SNAP/ple-table" ] || die "snapshot ${REV:0:8} has no ple-table/ (wrong checkpoint? run.sh keys the INT4 profile off it)"
for f in config.json model.safetensors.index.json model_extra_tensors.safetensors; do
  [ -e "$SNAP/$f" ] || die "snapshot ${REV:0:8} is incomplete: $f is missing"
done

# --- 2. the MTP draft folder, as run.sh builds it ------------------------------
# Slim folder (mtp. + lm_head. + embed_tokens) of symlinks + a small index, plus the
# checkpoint's own expert counts. Top-k 10 is the model's num_experts_per_tok.
mkdir -p "$CACHE"
python3 - "$SNAP" "/models/azampatti/snapshots/$REV" "$OUT" <<'PY' || die "could not build the draft folder"
import json, os, shutil, struct, sys
snap, csnap, out = sys.argv[1:4]
wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
keep = {k: v for k, v in wm.items() if k.startswith(("mtp.", "lm_head.")) or k.endswith("embed_tokens.weight")}
if os.path.lexists(out):
    shutil.rmtree(out)
os.makedirs(out)
for f in sorted(os.listdir(snap)):
    if f in ("config.json", "model.safetensors.index.json", "ple-table", "fast-fp8", ".cache") or \
       (f.endswith(".safetensors") and f not in set(keep.values())):
        continue
    os.symlink(os.path.join(csnap, f), os.path.join(out, f))
json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
cfg = json.load(open(os.path.join(snap, "config.json"))); t = cfg.get("text_config", cfg)
t["num_experts_per_tok"] = 10
with open(os.path.join(snap, "model_extra_tensors.safetensors"), "rb") as fh:
    h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
w = {v["shape"][0] for k, v in h.items() if k.endswith("mlp.shared_expert.gate_proj.weight")}
if len(w) == 1:
    t["shared_expert_intermediate_size"] = w.pop()
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
PY

# --- 3. dynamic draft depth -----------------------------------------------------
# recipe.yaml mtp_depth. The serve re-reads this file while it runs, so it can be
# changed live without a restart.
printf '{"mode": "dynamic", "min": 3, "window": 48, "promote": [60, 45], "demote": [25, 15], "log": false}\n' \
  > "$CACHE/mbx-depth.json"

# --- 4. serve ------------------------------------------------------------------
# Paths the image reads directly. MBX_PLE_FP8_DIR must be the mounted copy, so it is
# derived here rather than pinned in compose.
export MBX_PLE_FP8_DIR="$SNAP/ple-table"
export MBX_PLE_NVME_DIR="$CACHE/ple-nvme-fp8"

# Third departure from the recipe, and the reason this backend is usable behind this repo's
# harnesses at all: myllmbox passes no --chat-template, so the checkpoint's own template
# applies. Two things follow, and this repo has already fixed both once for qwen38-fast:
#   * OpenCode + Hindsight emit two {role:system} blocks, which the checkpoint's template
#     rejects with "System message must be at the beginning." (HTTP 400);
#   * the checkpoint default is not `medium`, so reasoning effort - and with it the thinking
#     budget - differs from every other backend here.
# qwen38-fast-medium.jinja is this repo's copy of the checkpoint's medium template with the
# leading-system-message merge applied, so it answers both while rendering byte-identically
# to upstream for a single leading system message. Passing it is a deliberate departure from
# the recipe, taken because the alternative is a backend that 400s on this repo's own clients.
#
# --gpu-memory-utilization 0.70 is the recipe's; with --kv-cache-memory pinned it does not
# size the KV pool. exec so vLLM is PID 1 and gets SIGTERM from `docker stop` directly.
exec vllm serve "$SNAP" \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 \
  --port 8000 \
  --engram-config '{"cpu_offload": false}' \
  --gdn-prefill-backend triton \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory 24000000000 \
  --load-format fastsafetensors \
  --max-model-len 262144 \
  --block-size 1632 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --chat-template /workspace/chat-templates/qwen38-fast-medium.jinja \
  --async-scheduling \
  --use-replayssm \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,21,24,25,28,30,32,35,36,40,42,44,45,48,49,50,52,54,55,56,60,63,64,65,66,70,72,75,77,78,80,84,88,90,91,96,98,104,105,112,120,128],"compile_ranges_endpoints":[32]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7,"rejection_sample_method":"block","draft_sample_method":"probabilistic","model":"/cache/draft-k10"}'
