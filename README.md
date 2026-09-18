# llm-router

A local LLM gateway for the DGX Spark. One OpenAI-compatible endpoint on port `4000`
that routes to a self-hosted vLLM model and to DeepSeek's cloud API, with Postgres-backed
model storage and Redis-backed caching.

## Architecture

```
client ──► litellm :4000 ──┬──► vllm :8000        qwen3.8-flash-next, self-hosted
                           │                       (loopback only - not LAN-reachable)
                           └──► api.deepseek.com   deepseek, cloud
              │
              ├──► db     postgres:16   model storage
              └──► redis  redis:7       response cache, rate limits, router state
```

Only `litellm` is exposed. `db` and `redis` publish no host ports, and `vllm` binds to
`127.0.0.1` only, so the LAN can reach the gateway and nothing else.

## Model routes

| Request `model` | Resolves to | Where |
|---|---|---|
| `local` | `qwen3.8-flash-next` | self-hosted vLLM, over the compose network |
| `deepseek` | `deepseek-flash`, then `local` | DeepSeek cloud API; falls back to vLLM |

Both are defined in `config.yaml` as `model_name` aliases. Callers only ever name the
alias; the upstream model id is an implementation detail.

`deepseek` has a LiteLLM model-group fallback to `local`. If the cloud key is out of
balance, times out, or returns 5xx, the proxy retries on vLLM and the client still
sees `model: deepseek`. After one failure the cloud deployment is cooled down for
300s so later turns skip the dead round trip. Restart `litellm` after editing
`config.yaml` (`docker compose restart litellm`).

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | the four services |
| `config.yaml` | model routing, Redis cache, router settings |
| `vllm-start.sh` | the `vllm serve` invocation |
| `chat_template.jinja` | chat template for the local model |
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
- The `qwen38-flash-dgx:nvfp4` image, built **locally** — it is not on any registry, so
  `docker compose pull` will fail for the `vllm` service. That is expected.
- The model in the HuggingFace cache (`nvidia/Qwen3.8-Flash-Next-NVFP4`), mounted at
  `/cache/huggingface`. The `vllm` service runs `HF_HUB_OFFLINE=1`, so nothing is fetched
  at runtime.

### 3. Start

```bash
docker compose up -d
```

`litellm`, `db` and `redis` come up in seconds. **`vllm` takes 10–16 minutes** to load
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

Only enable this **after** confirming the stack starts cleanly by hand. With a 10–13
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

`localhost:8000` will **always** time out from another machine. vLLM is deliberately
loopback-only; reach the local model through litellm by requesting model `"local"`.

## Operations

```bash
docker compose ps                       # service states
docker compose logs -f vllm             # weight loading progress
docker compose logs -f litellm          # proxy logs
curl -s localhost:8000/health           # vLLM engine ready (200 = live)
docker compose down                     # stop everything
```

`vllm` reports unhealthy for the whole weight load. That is normal, not a failure.

## Notes

### The vLLM service

`vllm-start.sh` reproduces the `command:` block of the sparkrun recipe
`qwen3.8-flash-next-nvfp4.yaml`, with its template placeholders inlined. Its argv was
verified byte-identical to the previously running server. If you edit the serve flags,
re-check that the script's quoting still produces the intended argv — the
`-cc.splitting_ops='[...]'` line is single-quoted deliberately; removing the quotes lets
the shell glob the brackets.

The image's entrypoint is `["vllm","serve"]`, which is why the serve command lives in a
script driven by `entrypoint:` rather than in `command:` — a `command:` value would have
`vllm serve` prepended to it.

### The chat template

`chat_template.jinja` comes from the `@mcampa/fix-qwen3.8-flash-chat-template` sparkrun
mod. It is **not** the same as the `chat_template.jinja` inside the checkpoint snapshot,
and the mod's version is the one that must be used. It is bind-mounted read-only over
`/workspace/vllm/chat_template.jinja`.

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
| `vllm` unhealthy for ~13 min | weight loading. Check `docker compose logs -f vllm` |
| `local` route 502s, `deepseek` works | vLLM still loading, or failed — check its logs |
| `deepseek` route 401s | `DEEPSEEK_API_KEY` placeholder in `.env`; the key is read at proxy startup, so restart `litellm` after editing |
| `deepseek` fails with `Available Model Group Fallbacks=None` | `router_settings.fallbacks` missing or litellm not restarted after the config change |
| `deepseek` still 400s after fallbacks land | LiteLLM treated a 400 as non-retryable; request `local` directly until the DeepSeek balance is topped up |
| `docker compose pull` fails on `vllm` | the image is local-only by design |
| vLLM container exits with `EADDRINUSE` | another server holds port 8000 |
| `local` returns 429 "No deployments available" | a previous call 404'd and put the deployment in cooldown. Check `api_base` ends in `/v1` — the provider appends the path verbatim and vLLM 404s without it |
| `local` returns 404 from vLLM | same cause as above |
| Repeated cached answers | global `cache: true`; see [Caching](#caching) |
