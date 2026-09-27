# llm-router

A local LLM gateway for the DGX Spark. One OpenAI-compatible endpoint on port `4000`
that routes to a self-hosted vLLM model and to DeepSeek's cloud API, with Postgres-backed
model storage and Redis-backed caching.

The self-hosted side runs **Eugr's B12x vLLM stack** on one GB10, and can serve either of
two Qwen3.8-Flash-Next checkpoints. `qwen38-fast` is enabled; `qwen38` (full NVFP4) is
staged in `docker-compose.yml`, commented out.

## Architecture

```
client ──► litellm :4000 ──┬──► vllm :8000        qwen3.8-flash-next, self-hosted
                           │                       (loopback + LAN, see below)
                           └──► api.deepseek.com   deepseek, cloud
              │
              ├──► db     postgres:16   model storage
              └──► redis  redis:7       response cache, rate limits, router state
```

`litellm` publishes `4000`. `db` and `redis` publish nothing. The `vllm` service publishes
`8000` on loopback (for host-side tooling) **and** on `192.168.0.96` so Prometheus on
`192.168.0.40` can scrape `/metrics` — which also exposes the full OpenAI API to the LAN.
That is a deliberate trade-off, chosen over running a metrics-only reverse proxy; see the
comment in `docker-compose.yml` if you want to revisit it.

## Backends

Both backends are the same Compose service name (`vllm`), on the same ports, serving the
same `--served-model-name qwen3.8-flash-next`. LiteLLM reaches them at
`http://vllm:8000/v1` either way, so **nothing outside `docker-compose.yml` changes when
you switch** — not `config.yaml`, not the `local` alias, not any caller.

| | `qwen38-fast` (enabled) | `qwen38` (commented out) |
|---|---|---|
| Checkpoint | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` |
| Quantisation | Intel AutoRound int4 + blockwise-fp8 side layers | NVIDIA ModelOpt NVFP4 (`modelopt_mixed`) |
| Shape | 125B total, ~5 of 10 routed experts per token | 125B total, 10 experts per token |
| Weights on disk | ~119 GiB | ~124 GiB |
| KV cache | pinned `20g` (`--kv-cache-memory-bytes`), ~645k tokens | sized from `--gpu-memory-utilization 0.8`, fp8 |
| Decode (this box) | ~33% faster than the full model | baseline |
| Deterministic at T=0 | no | yes |
| Startup script | `vllm-start-fast.sh` | `vllm-start-full.sh` |
| Extra pieces | the vendored `mods/flashnext-int4-b12x/` | none |

Only one may run at a time — they would fight for the GPU and port 8000.

> `qwen38` uses `local-inference-lab/…`, **not** the `nvidia/Qwen3.8-Flash-Next-NVFP4`
> checkpoint. No current B12x recipe targets the `nvidia` layout; Eugr's recipe serves the
> 2026-09-16 re-export, which is what `--load-format b12x` and `--quantization
> modelopt_mixed` expect. If you have `nvidia/…` in your cache from the old stack, it will
> not be reused.

### Switching to the full NVFP4 model

```bash
# 1. In docker-compose.yml: comment out the "ACTIVE: qwen38-fast" service and
#    delete the leading '#' from the "ALTERNATIVE: qwen38" block.
# 2. Make sure its checkpoint is cached (see Prerequisites).
# 3. Recreate the backend:
docker compose up -d --remove-orphans

# back to fast: reverse step 1 and re-run the same command
```

Both blocks use `container_name: vllm-qwen38-flash`, so the switch is a plain recreate —
no container-name conflicts, and host tooling that targets the container by name keeps
working. Expect ~10–15 minutes of weight loading after either switch.

## Model routes

| Request `model` | Resolves to | Where |
|---|---|---|
| `local` | `qwen3.8-flash-next` | self-hosted vLLM, over the compose network |
| `deepseek` | `deepseek-flash`, then `local` | DeepSeek cloud API; falls back to vLLM |

Both are defined in `config.yaml` as `model_name` aliases. Callers only ever name the
alias. `local` is stable across a backend switch, and the upstream model id
(`qwen3.8-flash-next`) is the same for both backends — that is why a switch needs no
config change.

`deepseek` has a LiteLLM model-group fallback to `local`. If the cloud key is out of
balance, times out, or returns 5xx, the proxy retries on vLLM and the client still sees
`model: deepseek`. After one failure the cloud deployment is cooled down for 300s so later
turns skip the dead round trip. Restart `litellm` after editing `config.yaml`
(`docker compose restart litellm`).

The Admin UI play button on Router Settings → Fallbacks sends `mock_testing_fallbacks`.
That param is gated; `general_settings.dangerously_allow_mock_testing_request_params` must
be true in `config.yaml` (it cannot be flipped from the UI).

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | the services, and the two alternative `vllm` backends |
| `config.yaml` | model routing, Redis cache, router settings |
| `vllm-start-fast.sh` | build the model views, patch vLLM, serve `qwen38-fast` |
| `vllm-start-full.sh` | serve `qwen38` (used only when that block is enabled) |
| `mods/flashnext-int4-b12x/` | vendored upstream mod: model views + vLLM patches for the A5B checkpoint |
| `chat-templates/` | the checkpoints' own chat templates, with one local fix (below) |
| `llm-router.service` | systemd unit (see below) |
| `.env.example` | template for `.env` — copy and fill in |
| `.env` | real secrets. **gitignored, never committed** |

## Setup

### 1. Secrets

```bash
cp .env.example .env
```

Fill in:

- `LITELLM_MASTER_KEY` — `echo "sk-$(openssl rand -hex 32)"`
- `LITELLM_SALT_KEY` — `openssl rand -hex 32`
- `DEEPSEEK_API_KEY` — from the DeepSeek console

> `LITELLM_SALT_KEY` encrypts credentials stored in the database. Changing it later makes
> previously-stored keys unreadable. Treat it as permanent once the stack has run.

`LITELLM_MASTER_KEY` is also the admin UI password and the bearer token for API calls.

### 2. Prerequisites

- Docker with the NVIDIA container runtime (`docker info` should list `nvidia`)
- ~1.6 TB free for the two checkpoints if you want both

The vLLM image is pulled from Docker Hub by `docker compose up`. It is pinned by digest
(not `latest`), because Eugr's `latest` has since moved and the newer build is not a
drop-in for these checkpoints:

```
eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262
```

~10.4 GiB compressed, ~31 GB on disk. To pre-pull it: `docker compose pull vllm`.

The checkpoints live in the host HuggingFace cache at `~/.cache/huggingface`, mounted at
`/cache/huggingface`. The `vllm` services run `HF_HUB_OFFLINE=1`, so nothing is fetched at
runtime and a missing/incomplete snapshot fails fast at container start.

```bash
# qwen38-fast - already cached on this box (~119 GiB)
hf download azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound

# qwen38 (full NVFP4) - only needed if you enable that backend (~124 GiB)
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4
```

The A5B checkpoint ships its n-gram PLE table, an int4 lm_head, a healed shared-expert
shard and its chat templates in the same snapshot; there is nothing else to fetch.

### 3. Start

```bash
docker compose up -d
```

`litellm`, `db` and `redis` come up in seconds. **`vllm` takes 10–15 minutes** to load
weights — it will sit unhealthy until then. See [Operations](#operations).

## systemd

The unit is `llm-router.service` in this repo. It runs
`docker compose up -d --remove-orphans` from the repo root as user `mcampa`.

### Install and start

```bash
sudo cp llm-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start llm-router.service
```

Check it:

```bash
systemctl status llm-router.service
docker compose ps
```

`Type=oneshot` with `RemainAfterExit=yes` means systemd considers the unit "active
(exited)" once compose returns — which happens as soon as containers are *created*, not
healthy. So `systemctl status` saying `active` tells you nothing about vLLM. Verify with
the health checks in [Operations](#operations) instead.

### Start on boot

```bash
sudo systemctl enable llm-router.service
```

Only enable this **after** confirming the stack starts cleanly by hand. With a 10–15
minute vLLM load, a unit enabled on an untested stack is hard to debug at boot.

### Reboot does *not* start the stack unless you ran `enable`

Installing the unit and starting it does not make it survive a reboot — systemd starts
units at boot only if they are enabled.

### Avoid `--wait`

Do not add `--wait` to `ExecStart`. It blocks until every service is healthy, which would
stall the boot sequence for the full vLLM weight load.

### Check for conflicting units

If another unit also starts a vLLM server, both will fight for port `8000` and the GPU. On
this machine that was `sparkrun.service`, which ran a different model
(`qwen3.6-35b-a3b-fp8-mtp`). Disable anything similar before enabling this stack:

```bash
systemctl list-unit-files --state=enabled | grep -iE 'vllm|sparkrun|llm'
sudo systemctl disable --now <unit>
sudo rm /etc/systemd/system/<unit>
sudo systemctl daemon-reload
```

The `vllm` service binds `127.0.0.1:8000`; a conflicting server on `0.0.0.0:8000` covers
loopback too, so the container fails with `EADDRINUSE` rather than picking another port.

## Usage

### API

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}'
```

Swap `"local"` for `"deepseek"` to hit the cloud route.

Thinking is on by default. To skip it, pass
`"chat_template_kwargs": {"enable_thinking": false}`. Because thinking is on, an answer
can sit in the `reasoning` field until the model finishes — an apparently empty response
usually means it is still thinking, not that something broke.

### Admin UI

```
http://<host-ip>:4000/ui
```

Log in as **`admin`** with your `LITELLM_MASTER_KEY` as the password.

### From another machine

Two options:

```bash
# a) open the firewall to your LAN only
sudo ufw allow from 192.168.0.0/24 to any port 4000 proto tcp

# b) or tunnel over SSH - nothing is exposed, and the key stays off the wire
ssh -L 4000:localhost:4000 mcampa@<host-ip>
#    then browse http://localhost:4000/ui
```

Port `8000` publishes on `192.168.0.96`, so another machine on the LAN can reach the model
directly at `http://192.168.0.96:8000/v1` — with no authentication. Reach the local model
through litellm by requesting model `"local"` unless you specifically want that.

## Operations

```bash
docker compose ps                       # service states
docker compose logs -f vllm             # weight loading progress
docker compose logs -f litellm          # proxy logs
curl -s localhost:8000/health           # vLLM engine ready (200 = live)
curl -s localhost:8000/v1/models        # what the backend is serving
curl -s localhost:8000/metrics | head   # Prometheus metrics
docker compose down                     # stop everything
```

`vllm` reports unhealthy for the whole weight load (the healthcheck's `start_period` is 20
minutes). That is normal, not a failure.

## Notes

### The vLLM services

Both backends run Eugr's B12x image and mirror the corresponding upstream recipe's
`command:` block with its template placeholders inlined, so the two stay easy to diff.
Every argument is quoted deliberately: `--speculative-config '{...}'` and
`--compilation-config '{...}'` are single-quoted so the shell hands vLLM one JSON
argument. If you edit the flags, re-check the resulting argv (`bash -x` or the
stub-run technique) rather than assuming.

The image's entrypoint is `["vllm","serve"]`, which is why the serve command lives in a
script driven by `entrypoint:` rather than in `command:` — a `command:` value would have
`vllm serve` prepended to it.

Both services set `security_opt: ["seccomp=unconfined"]`, and that is **required**, not
hardening drift. The B12x PLE reader does its scatter-gather reads through io_uring, and
Docker's default seccomp profile blocks `io_uring_setup`; without this the engine dies ~3
minutes into the weight load with `io_uring initialization failed: Operation not
permitted`. Eugr's launcher runs containers `--privileged` (which also drops seccomp),
which is why upstream never sees it. Unconfining seccomp is the *only* piece of
`--privileged` taken here — the GPU reservation, capabilities, device list and mounts are
all still restricted. It is needed on both backends, since both offload the PLE table to
disk.

`qwen38-fast` additionally runs the vendored `mods/flashnext-int4-b12x/` before serving:
it builds two symlink farms under `/workspace/flashnext` (the served model, with the PLE
table's 131 shards indexed and the config fields B12x needs; and a slim MTP draft folder
with top-k 10) and patches the container's vLLM for this checkpoint. Both steps are
idempotent and re-run on every container start — the image itself is never modified. A
required patch that cannot be applied aborts the launch rather than serving a broken
model. See `mods/flashnext-int4-b12x/README.md`.

### Chat templates

`chat-templates/` holds copies of each checkpoint's **own** template with exactly one
change: consecutive *leading* system messages are merged into a single system block.

This is needed because both checkpoints' templates consume only `messages[0]` as the system
prefix and then raise `System message must be at the beginning.` on a second one — and the
agent harnesses used against this gateway (OpenCode, Hindsight) emit two `{role: system}`
blocks. Nothing else is touched: tool calling, reasoning/thinking tokens, special tokens
and the assistant generation prompt are upstream's, verbatim. A request with a single
leading system message renders byte-identically to the unpatched template (verified by
diffing rendered output over plain, tool-calling, multi-turn, reasoning-off and
empty-system requests). A system message anywhere *other* than the leading run is still
rejected, as upstream does.

- `qwen38-fast-medium.jinja` — from the A5B checkpoint's `medium_chat_template.jinja`. The
  A5B recipe selects this template because the MTP speculative head was trained on it
  (~+5pp acceptance over the stock template). Ours is that file plus the merge.
- `qwen38-full.jinja` — from `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`'s default
  `chat_template.jinja` (byte-identical to the `nvidia/…` one), plus the merge. Unlike
  upstream's recipe, which passes no `--chat-template` and takes the model default, the
  full backend passes this file explicitly.

To update a template, re-derive it from the checkpoint rather than editing by hand, then
re-run the rendering diff. The upstream files are inside the cached snapshots.

### Caching

`cache: true` in `config.yaml` is global: identical requests are served from Redis for
`ttl: 600` seconds. In an agent or dev loop this looks like the model ignoring you. Scope
it per-model if that is not what you want.

### `deepseek-flash`

Verified working against the live API. The id is not in DeepSeek's public documentation
(which lists `deepseek-chat` and `deepseek-reasoner`), so it may be account-gated or new.
If it ever starts returning a 400 about an unknown model, change one line in `config.yaml`:

```yaml
model: deepseek/deepseek-chat
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `vllm` unhealthy for ~15 min | weight loading. Check `docker compose logs -f vllm` |
| `local` route 502s, `deepseek` works | vLLM still loading, or failed — check its logs |
| vLLM exits: `FATAL vllm-start-fast: model not in the HF cache` | checkpoint not downloaded, or still downloading. See [Prerequisites](#2-prerequisites) |
| vLLM exits: `FATAL … could not patch vLLM` | a required `mods/` patch anchor moved — the pinned image and the vendored mod have drifted apart. Re-pull the mod at the commit the image expects, or re-pin the image |
| vLLM dies a few minutes into loading: `io_uring initialization failed: Operation not permitted` | `security_opt: ["seccomp=unconfined"]` is missing from the `vllm` service. See [The vLLM services](#the-vllm-services) |
| vLLM exits: `FATAL vllm-start-full: … is not in the HF cache` | the full backend's checkpoint is a separate ~124 GiB download; see [Prerequisites](#2-prerequisites) |
| `deepseek` route 401s | `DEEPSEEK_API_KEY` placeholder in `.env`; the key is read at proxy startup, so restart `litellm` after editing |
| `deepseek` fails with `Available Model Group Fallbacks=None` | `router_settings.fallbacks` missing or litellm not restarted after the config change |
| `deepseek` still 400s after fallbacks land | LiteLLM treated a 400 as non-retryable; request `local` directly until the DeepSeek balance is topped up |
| UI play button: `mock_testing_fallbacks` disabled | set `general_settings.dangerously_allow_mock_testing_request_params: true` and restart litellm |
| Chat 400s: `System message must be at the beginning.` | a system message arrives somewhere other than the leading run — the merge fix only covers leading ones |
| `local` returns 429 "No deployments available" | a previous call 404'd and put the deployment in cooldown. Check `api_base` ends in `/v1` — the provider appends the path verbatim and vLLM 404s without it |
| `local` returns 404 from vLLM | same cause as above |
| vLLM container exits with `EADDRINUSE` | another server holds port 8000 |
| Repeated cached answers | global `cache: true`; see [Caching](#caching) |
