# mod: flashnext-int4-b12x (vendored)

Makes Eugr's `vllm-node-b12x` image able to serve
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`.

Copied verbatim from:

- repo: <https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST>
- commit: `d0b4b59037b813704f4008f3d800b4c6f0223104` (2026-09-26)
- path: `flashnext-int4-b12x/`

| File | Role |
|---|---|
| `build_views.py` | builds `/workspace/flashnext/model` (served view: PLE table indexed, b12x config fields) and `/workspace/flashnext/draft-k10` (slim MTP draft folder, top-k 10) out of symlinks — nothing is copied |
| `patch_b12x.py` | patches the container's vLLM: fp8-hybrid quant config, int4 lm_head, index filter for the mixed PLE files, draft-scale, GDN fixes. Idempotent (`flashnext-int4-b12x:` marker comments), and every edit is checked before it is written |
| `vllm_fp8_hybrid.py` | the fp8-hybrid hook that `patch_b12x.py` installs into site-packages |

The upstream `run.sh` is **not** vendored: it hardcodes the container path
`/root/.cache/huggingface` that Eugr's launcher uses, whereas this stack mounts the HF cache
at `/cache/huggingface`. `vllm-start-fast.sh` performs the same three steps (check the
snapshot, build the views, patch vLLM) against the mount this repo uses.

`vllm-start-fast.sh` also differs from upstream in one more way: upstream's recipe serves the
checkpoint under its own name (`--served-model-name azampatti/Qwen3.8-Flash-Next-125B-A5B-...`).
Here it is served as `qwen3.8-flash-next` so that both backends present the same model id and
LiteLLM's `config.yaml` never changes when you switch.

## Updating

Re-pull the three files at a newer upstream commit, then update the commit hash above and in
the `vllm-start-fast.sh` header. Check them against the running container before trusting a
new revision — `patch_b12x.py` fails loudly (non-zero exit, launch aborted) if an anchor it
depends on has moved, which is the intended failure mode.
