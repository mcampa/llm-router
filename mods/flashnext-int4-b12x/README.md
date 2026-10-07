# mod: flashnext-int4-b12x (vendored)

Makes Eugr's `vllm-node-b12x` image able to serve
`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`.

Copied verbatim from:

- repo: <https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST>
- commit: `73c4fa07bddd5f4ae04c3384973ab5b4c4b61a80` (2026-10-06)
- path: `flashnext-int4-b12x/`

| File | Role |
|---|---|
| `build_views.py` | builds `/workspace/flashnext/model` (served view: PLE table indexed, b12x config fields) and `/workspace/flashnext/draft-k10` (slim MTP draft folder, top-k 10) out of symlinks — nothing is copied |
| `patch_b12x.py` | patches the container's vLLM: fp8-hybrid quant config, int4 lm_head, index filter for the mixed PLE files, draft-scale, MTP cap, request-default penalties, drafter pad rows, GDN fixes. Idempotent (`flashnext-int4-b12x:` marker comments), and every edit is checked before it is written |
| `vllm_fp8_hybrid.py` | the fp8-hybrid hook that `patch_b12x.py` installs into site-packages |

The upstream `run.sh` is **not** vendored: it hardcodes the container path
`/root/.cache/huggingface` that Eugr's launcher uses, whereas this stack mounts the HF cache
at `/cache/huggingface`. `vllm-start-fast.sh` performs the same three steps (check the
snapshot, build the views, patch vLLM) against the mount this repo uses.

`vllm-start-fast.sh` also differs from upstream in one more way: upstream's recipe serves the
checkpoint under its own name (`--served-model-name azampatti/Qwen3.8-Flash-Next-125B-A5B-...`).
Here it is served as `qwen3.8-flash-next` so that both backends present the same model id and
LiteLLM's `config.yaml` never changes when you switch.

## Upstream recipe knobs this repo does not take

The vendored patch also carries two optional patches that require corresponding recipe changes
to become active. This repo has not adopted either, so both are inert here — and taking either
is a recipe change (`vllm-start-fast.sh`), not a mod change:

| Patch | Pairs with | Status here |
|---|---|---|
| `mtp-cap` | `num_speculative_tokens: 5` + `block_size: 1632` | inert: we run MTP 4 on `block_size 16` (the 12-row QSA ring that needs the 1632 page only appears at depth 5+) |
| `default-penalties` | `--override-generation-config '{"presence_penalty":0.5,...}'` | inert: we pass no `--override-generation-config`, so each request keeps its own value |

## The drafter pad-rows fix is asserted at runtime

The reason this repo is on this revision is upstream's `drafter-pad-rows` patch: with 3, 5, 6 or
7 requests in flight right after a prefill, one stream used to lose MTP entirely and generate at
~1 token/step for the rest of the request. Its output stayed correct, so it read as inexplicable
slowness rather than as a bug. It edits `qsa.py` in the container and needs no recipe change.

Upstream ships that patch as *optional*, so a moved anchor only warns and the launch carries on.
Rather than edit the vendored file to make it required — which would cost the byte-for-byte
upstream property this directory is built on — the hard guarantee lives in `vllm-start-fast.sh`:
after patching it greps the container's `qsa.py` for the fix and aborts the launch if it is
absent. The vendor snapshot stays clean (a re-pull stays a re-pull) and a silent miss is still
impossible.

## Updating

Re-pull the three files at a newer upstream commit and update the commit hash above. Because
`patch_b12x.py` is byte-for-byte upstream, that is the whole procedure — the runtime assertion
lives in `vllm-start-fast.sh`, not in the vendored file.

Check a new revision against the running container before trusting it: `patch_b12x.py`'s
*required* patches fail loudly (non-zero exit, launch aborted) if an anchor has moved, while its
*optional* ones fall through with a warning — that is the case that can look like success, which
is why the launcher asserts `drafter-pad-rows` itself. Confirm the markers show up in the launch
log's `patched: ...` list rather than its `WARNING (optional, skipped): ...` list.
