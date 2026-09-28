# llm-router

A local LLM gateway for the DGX Spark. One OpenAI-compatible endpoint on port `4000`
that routes to a self-hosted vLLM model and to DeepSeek's cloud API, with Postgres-backed
model storage and Redis-backed caching.

The self-hosted side runs on one GB10 and can serve any **one of three** Qwen3.8-Flash-Next
backends. `qwen38-fast` (Eugr's B12x stack) is enabled; `qwen38` (full NVFP4, the same B12x
stack) and `qwen38-hibrid48` (bilikaz's myllmbox v4 image, n-gram table demand-paged from
local NVMe) are staged in `docker-compose.yml`, commented out.

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

All three are the same Compose service name (`vllm`), on the same ports, serving the same
`--served-model-name qwen3.8-flash-next`. LiteLLM reaches them at `http://vllm:8000/v1`
either way, so **nothing outside `docker-compose.yml` changes when you switch** — not
`config.yaml`, not the `local` alias, not any caller. This is checked, not just asserted:
the three switch positions render a byte-identical `litellm` service, and every startup
script passes the same `--served-model-name`.

| | `qwen38-fast` (enabled) | `qwen38` (commented out) | `qwen38-hibrid48` (commented out) |
|---|---|---|---|
| Checkpoint | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` | `myllmbox/Qwen3.8-Flash-Next-hibrid48` |
| Image | Eugr B12x (`eugr/spark-vllm-b12x`) | same Eugr B12x image | `myllmbox/qwen38-flash-next-vllm` v4 (vLLM 0.30) |
| Quantisation | Intel AutoRound int4 + blockwise-fp8 side layers | NVIDIA ModelOpt NVFP4 (`modelopt_mixed`) | NVFP4 body + NVFP4 4-bit output head |
| Shape | 125B total, ~5 of 10 routed experts per token | 125B total, 10 experts per token | 48 layers, 10 of 512 routed experts per token |
| Weights on disk | ~119 GiB | ~124 GiB | ~98 GiB |
| PLE / n-gram table | streamed from disk (`VLLM_PLE_TABLE_MEMORY=disk`) | streamed from disk (same) | demand-paged from **local NVMe** by the image's own reader |
| KV cache | pinned `20g` (`--kv-cache-memory-bytes`), ~645k tokens | sized from `--gpu-memory-utilization 0.8`, fp8 | pinned `26000000000` (`--kv-cache-memory-bytes`), bf16, ~800k tokens |
| Concurrency | `--max-num-seqs 8` | `--max-num-seqs 8` | `--max-num-seqs 16` |
| Draft / spec decode | MTP, K=4, greedy draft | MTP, K=4 | MTP, K=5, sampled (`probabilistic`) draft |
| Deterministic at T=0 | no | yes | no (K=5 drafts are sampled) |
| RDMA | no (RoCE env is inert on one box) | no | no |
| Startup script | `vllm-start-fast.sh` | `vllm-start-full.sh` | `vllm-start-hibrid48.sh` |
| Chat template | `qwen38-fast-medium.jinja` | `qwen38-full.jinja` | `qwen38-hibrid48.jinja` |
| Extra pieces | the vendored `mods/flashnext-int4-b12x/` | none | a ~31 GB table map under `/cache`, built on first boot |
| Decode on this box | baseline (~33% faster than the full NVFP4 model) | not measured | **43 tok/s 1-stream, 201 at c=16** — see [First boot](#first-boot-on-this-box-what-it-actually-did-2026-09-28) |

Only one may run at a time: they would fight for the GPU and port 8000, and the box has
unified memory for one of them, not two.

On the `qwen38-hibrid48` row: the checkpoint reports 48 layers and 10 of 512 routed experts
per token, which is the same 125B-A6B body as `qwen38` — but the shipped kit measures
**180.0B counted parameters at 4.35 bpw effective** for it. The extra ~51B is the n-gram
table (320,001,536 rows × 160, itself NVFP4), which is why the weights are ~98 GiB rather
than the ~124 GiB the body alone would suggest, and why the table is the thing this backend
moves to NVMe. Do not quote a parameter count for this checkpoint from the table alone.

> `qwen38` uses `local-inference-lab/…`, **not** the `nvidia/Qwen3.8-Flash-Next-NVFP4`
> checkpoint. No current B12x recipe targets the `nvidia` layout; Eugr's recipe serves the
> 2026-09-16 re-export, which is what `--load-format b12x` and `--quantization
> modelopt_mixed` expect. If you have `nvidia/…` in your cache from the old stack, it will
> not be reused.
>
> The same applies to `qwen38-hibrid48`: it needs its own `myllmbox/…` download. Neither
> B12x checkpoint can be served by this image, and this checkpoint cannot be served by the
> B12x image — `vllm-start-hibrid48.sh` checks for the `ple_quantization` marker in
> `config.json` and refuses to start on anything else.

### Switching between backends

The three blocks live in `docker-compose.yml` under the banner comment above them, one
live and two commented. Switching is: comment out the live one, uncomment the one you
want, recreate. Reversing it switches back — the change is three lines of comment markers,
so `git checkout docker-compose.yml` is the always-available undo.

```bash
# 1. In docker-compose.yml: comment out the live "ACTIVE: qwen38-fast" block, and delete
#    the leading '#' from whichever "ALTERNATIVE" block you want.
# 2. Make sure that backend's checkpoint is in place (see Prerequisites).
# 3. Recreate the backend:
docker compose up -d --remove-orphans

# Roll back: restore the comment markers (or `git checkout docker-compose.yml`) and re-run
# the same command.
```

Check the result before you commit to it — this parses the file without starting anything:

```bash
docker compose config --services          # must list exactly one `vllm`
docker compose config | grep -A2 '^  vllm'
```

All three blocks use `container_name: vllm-qwen38-flash`, so a switch is a plain recreate —
no container-name conflicts, and host tooling that targets the container by name keeps
working. Expect ~10–15 minutes of weight loading after a `qwen38-fast` / `qwen38` start, and
about 4 minutes (plus a few minutes once, for the table map) after a `qwen38-hibrid48` start.

## Model routes

| Request `model` | Resolves to | Where |
|---|---|---|
| `local` | `qwen3.8-flash-next` | self-hosted vLLM, over the compose network |
| `deepseek` | `deepseek-flash`, then `local` | DeepSeek cloud API; falls back to vLLM |

Both are defined in `config.yaml` as `model_name` aliases. Callers only ever name the
alias. `local` is stable across a backend switch, and the upstream model id
(`qwen3.8-flash-next`) is the same for all three backends — that is why a switch needs no
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
| `docker-compose.yml` | the services, and the three alternative `vllm` backends |
| `config.yaml` | model routing, Redis cache, router settings |
| `vllm-start-fast.sh` | build the model views, patch vLLM, serve `qwen38-fast` |
| `vllm-start-full.sh` | serve `qwen38` (used only when that block is enabled) |
| `vllm-start-hibrid48.sh` | serve `qwen38-hibrid48` (used only when that block is enabled) |
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
- ~1.7 TB free for the three checkpoints if you want all of them

#### Images

Both images are pulled from Docker Hub by `docker compose up`. Each is pinned by digest
(not a tag) because the upstream `latest` tags have moved since and the newer builds are
not drop-ins for these checkpoints:

```
eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262
    qwen38-fast, qwen38 - ~10.4 GiB compressed, ~31 GB on disk
myllmbox/qwen38-flash-next-vllm@sha256:51629f438f5ba3f7a96db110826c783d69b91a6851c43fb07d447644f157dcc4
    qwen38-hibrid48 - bilikaz's kit, v4 tag = vLLM 0.30 + the NVFP4 table/output-head patches
```

Only the image for the backend you are running needs to be pulled. To pre-pull the live
one: `docker compose pull vllm`. `docker compose config` prints the digest of whichever
block is uncommented.

#### Checkpoints

`qwen38-fast` and `qwen38` read from the host HuggingFace cache at `~/.cache/huggingface`,
mounted at `/cache/huggingface`. Those services run `HF_HUB_OFFLINE=1`, so nothing is
fetched at runtime and a missing/incomplete snapshot fails fast at container start.

```bash
# qwen38-fast - already cached on this box (~119 GiB)
hf download azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound

# qwen38 (full NVFP4) - only needed if you enable that backend (~124 GiB)
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4
```

The A5B checkpoint ships its n-gram PLE table, an int4 lm_head, a healed shared-expert
shard and its chat templates in the same snapshot; there is nothing else to fetch.

`qwen38-hibrid48` is different: it is a plain `--local-dir` tree on the NVMe, **not** the
shared HF cache (that is the layout bilikaz's kit uses, and the table reader expects it).
It needs **~130 GB on the NVMe**: ~98 GiB of weights plus ~31 GB of table map.

```bash
# 1. ~130 GB on the NVMe. `/` is the NVMe volume on this box; adjust if yours differs.
sudo mkdir -p /models/qwen38/models /models/qwen38/cache
sudo chown -R "$(id -u):$(id -g)" /models/qwen38

# 2. ~98 GiB, 28 shards. The repo is public and ungated, so no token is required
#    (anonymous downloads are rate-limited; `hf auth login` first if that bites).
hf download myllmbox/Qwen3.8-Flash-Next-hibrid48 \
  --local-dir /models/qwen38/models/Qwen3.8-Flash-Next-hibrid48
```

`/models/qwen38/models` is mounted at `/models` and `/models/qwen38/cache` at `/cache`; the
container serves `/models/Qwen3.8-Flash-Next-hibrid48`. Both must be the same NVMe volume:
`/cache` has to be writable and fast, because the first start writes the ~31 GB n-gram
table map there (a few minutes, once, and it is rebuilt if you delete it). Later boots read
that map and allocate almost nothing for the table — which is the whole point of this
backend, and where the memory for the 26G KV pool comes from.

The download is resumable; re-run the same command if it is interrupted. Do not point the
volume at a tmpfs or at a network mount. The container runs as root, so `/models/qwen38/cache`
ends up root-owned once the map is built — `sudo rm -rf /models/qwen38/cache/*` if you ever
need to drop it and start over.

### 3. Start

```bash
docker compose up -d
```

`litellm`, `db` and `redis` come up in seconds. **`vllm` takes 10–15 minutes** to load
weights (about 4 minutes on `qwen38-hibrid48`, plus a few minutes once for its table map) —
it will sit unhealthy until then. See [Operations](#operations).

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

### GB10 smoke test for `qwen38-hibrid48`

**This backend has not been run.** It was added without access to the GB10 — no inference
was executed for this PR, and it has never been switched on. Everything below is the plan
for the first time someone does, and each step has a pass condition you can actually check.
Budget an hour, most of it waiting.

Because `qwen38-fast` is the live backend, do this at a time when the gateway may be down
for ~10 minutes, and finish with a deliberate choice of which block stays uncommented.

**0. Before switching — does it even fit?**

```bash
df -h /models/qwen38                      # want >= 130 GB free, on the NVMe
free -g                                   # want >= 100G available: one backend at a time
ls /models/qwen38/models/Qwen3.8-Flash-Next-hibrid48/model.safetensors.index.json
docker compose config --services          # exactly one vllm
```

**1. Startup.** Switch the block, then `docker compose up -d --remove-orphans`.

- Pass: reaches `/health` 200 in ~4 minutes, plus a few minutes once on the first boot
  while the table map is built.
- Watch: `docker compose logs -f vllm`. Read the KV line it prints at startup and confirm
  it reports the pool you expect (roughly 800k tokens for 26G bf16) — that is the check
  that `--kv-cache-memory-bytes 26000000000` was read as 26e9 bytes and not rejected.
- Confirm the map landed and is reusable: `du -sh /models/qwen38/cache` (~31 GB), then
  `docker compose restart vllm` and check the second boot skips the map build.
- Pass: `curl -s localhost:8000/v1/models` returns `qwen3.8-flash-next` — the same id the
  fast backend serves, which is what keeps the `local` alias working.

**2. A tool call, the way the harnesses actually call it.** Two `{role: system}` blocks —
this is the Yeiberson/OpenCode shape, and the reason `qwen38-hibrid48.jinja` exists:

```bash
curl -s localhost:4000/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' -d '{
  "model":"local",
  "messages":[
    {"role":"system","content":"You are OpenCode, a coding agent."},
    {"role":"system","content":"Hindsight context: the user is editing vllm-start-hibrid48.sh."},
    {"role":"user","content":"Read /etc/hostname and tell me what it says."}
  ],
  "tools":[{"type":"function","function":{"name":"read_file","description":"Read a file",
    "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]}'
```

- Pass: 200, with a `tool_calls` entry named `read_file` and `arguments` as an object.
- Fail if you instead get `System message must be at the beginning.` (the template is not
  the one being used — check the `--chat-template` path) or an empty `tool_calls` with the
  call text left in `content` (the parser is wrong; this checkpoint wants `qwen3_xml`).
- Then repeat through a real agent turn: send the assistant's `tool_calls` back with a
  `role: tool` result and confirm the follow-up is a normal answer. That exercises the
  multi-step-tool path in the template, not just the first hop.
- Also check reasoning parsing: with thinking on (the default) the answer arrives in the
  **`reasoning`** field, with `content` holding only the final answer. Note the field name —
  this image returns `reasoning`, **not** `reasoning_content`, which is why the Usage section
  above and the fast backend's template already read `reasoning`. An answer that never leaves
  `reasoning` at all means `--reasoning-parser qwen3` is not matching this template. (There
  is a related asymmetry worth knowing: all three templates read *prior-turn* reasoning from
  `reasoning_content` only, so multi-turn thinking preservation is a no-op here even though
  the templates request it — the fast template was patched to read `reasoning` for exactly
  this reason. Left as-is to keep this template upstream-plus-merge; see the PR.)

**3. Repeated-prefix TTFT.** Prefix caching is on, and this is where the fast backend's
20 GiB pool and this one's 26 GiB are most comparable. Send the same long system prompt
twice (say ~8k tokens) and compare:

- Pass: the second request's TTFT is a small fraction of the first, and
  `vllm:prefix_cache_hit_rate` (or the hit counters) moves off zero.
- Note the first-request TTFT separately — with the table on NVMe, the first tokens after
  a cold start also pay for page-ins, so do not read request one as steady state.

**4. Decode rate.** Measure the same way for both backends or the numbers are not
comparable: thinking off, a fixed prompt, `stream: true`, and compute
`completion_tokens / (last_token_time - first_token_time)`.

- Pass: within the range the kit publishes on one Spark (73 tok/s single stream, ~288 at 16
  streams) minus what the different seats and this box's other services cost.
- Expect this to be *slower* than `qwen38-fast` at low concurrency and to close or cross
  over as concurrency rises — the fast backend is the one that is ~33% quicker decoding.
  Report it at 1 stream and at `--max-num-seqs` (16) at least.
- **The seats differ**: this backend runs upstream's 16, the other two run 8. That is
  upstream's shipped configuration, kept deliberately, but it makes the concurrency rungs
  an unfair comparison as they stand. To compare like for like, set `--max-num-seqs 8` *and*
  truncate the cudagraph capture sizes to
  `[1,2,4,6,8,12,16,18,24,30,36,42,48]` — they must stay multiples of K+1 up to `seats × 6`,
  or the upper rungs decode without CUDA graphs.
- Also read speculative-decoding acceptance if the build exposes it: at K=5 the kit reports
  ~5.1 of 6 accepted on code. Materially lower means the draft path is not working and the
  speed number is not the one this backend is supposed to produce.

**5. Host memory.** GB10 unified memory is the hard constraint.

```bash
free -g                                   # MemAvailable, before and at full load
curl -s localhost:8000/metrics | grep -E 'kv_cache_usage|num_requests_(running|waiting)'
docker stats --no-stream vllm
```

- Pass: `MemAvailable` stays positive (the kit measures ~2.8–2.9G at 16 streams on a box
  running nothing else; expect less here). If it goes to zero, or the engine is killed,
  drop `--kv-cache-memory-bytes` to `24000000000` and retry — that is the knob, and it is a
  one-line change
  in the serve script.
- Do this while the *other* services are up, since that is the real configuration.

**6. NVMe I/O.** The point of this backend is that the table does not sit in RAM.

```bash
iostat -x 1 30                             # or: watch -n1 'cat /sys/block/nvme0n1/stat'
du -sh /models/qwen38/cache                # the table map
```

- Pass: during decode, the NVMe device shows real read throughput that *persists* across a
  long run rather than a burst while the map is paged in and then nothing. A steady ~0
  read rate during decode means the table ended up resident, which is worth knowing — you
  would be measuring a different thing than the kit measures.
- Watch for the opposite failure too: reads that never settle, or `%util` pinned at 100,
  mean the working set is thrashing the page cache against the 26G KV pool.
- Finally, confirm the map is not being rebuilt on every boot (step 1's restart check).

**7. Leave it in a known state.** Record which block is uncommented, `docker compose ps`
output, and the numbers above. Then either switch back to `qwen38-fast` (restore the
comment markers, `docker compose up -d --remove-orphans`) or state plainly that
`qwen38-hibrid48` is now the live backend — the PR that adds this backend does not switch
it on, so that decision is the operator's.

#### First boot on this box: what it actually did (2026-09-28)

Recorded so the next person has a baseline to compare against. Same box, same
`config.yaml`, `qwen38-fast` stopped for the whole window.

| | measured | kit's published figure |
|---|---|---|
| Boot to `/health` 200 | **~6.3 min** (65.8 s of it loading weights) | ~4 min + first-boot table map |
| Table map under `/cache` | **30 GB**, built on that first boot | ~31 GB, a few minutes, once |
| KV pool | **800,229 tokens**, 24.21 GiB | "26G ≈ 800k" |
| Single stream decode | **43.0 tok/s** mean (48.2 / 39.2 / 41.6) | 73 |
| c=4 aggregate | **110 tok/s** | 154 |
| c=16 aggregate | **201 tok/s** | 288 |
| Repeated-prefix TTFT | **2.4 s warm vs 69.8 s cold (29×)** on a ~10k-token prefix | — |
| Spec-decode acceptance | **0.68** (1570/2325 drafted) | 5.1 of 6 on code (~0.85) |
| NVMe reads during decode | ~4.3 MB/s, ~1600 IOPS at ~4 KB | — |
| **Prefill, uncached** | **~1,300 tok/s** (77k prompt tokens in 60 s under live agent load) | — |
| Prefill, cache hit | ~25,000 tok/s | — |
| Host memory, idle | 1 GB free / 3–4 GB available, page cache 4.3 GB | 5.7 GB available at 1 stream |
| Artifacts | healthy, `RestartCount=0`, no OOM | — |

Every flag was confirmed in the boot log: `v0.30.0`, `quantization=modelopt_mixed`, `num_spec_tokens=5`,
the full v4 cudagraph capture list, `moe_backend=marlin`, the Triton/FLA GDN prefill kernel,
and `PLE: NVFP4 table DEMAND-PAGED from /models/Qwen3.8-Flash-Next-hibrid48`.

**Read the throughput column as a lower bound, not a shortfall.** Three things separate it
from the kit's ladder, and none of them is a fault in the deployment: the kit's figures are
code prompts (acceptance ~0.85; the plain-text prompt used here accepts 0.68, which directly
costs tokens per engine step), the kit excludes prefill from its measurement windows while
these include TTFT, and `vm.compaction_proactiveness` is still **20** on this box — upstream
measures that alone at ~10% plus 4–5 s stalls, and its `run.sh` warns on every launch. Setting
it to 0 (`./tune-host.sh`, needs root) is the single highest-value next step.

#### A real agent on this backend: prefill is the bottleneck, not decode

Measured with an actual agent harness driving the gateway (not a synthetic prompt), which is
the workload this backend exists for. Sampled over 60 s of its traffic:

```
requests finished : 0
decode tokens     : 111    -> 1.8 tok/s aggregate
prefill tokens    : 77,182 -> 1,286 tok/s
```

The aggregate *decode* number is near-useless here, and that is the point: the agent spent that
minute **reading a large prompt, not writing**. On the intervals where it was actually decoding
it ran at 24–59 tok/s, mean ~40 — the same figure as the synthetic single-stream test, so the
backend delivers its decode rate under real load. What an agent spends its wall-clock on is
prefill.

Practical consequence: a ~20k-token context that misses the prefix cache costs ~15 s before the
first token; the same context on a second turn is nearly free (29× TTFT, above). So if an agent
session feels slow here, look at prompt growth and cache hits before blaming decode.

The likely reason prefill is the weak side is structural: GB10 has no native FP4 compute, so
`--moe-backend marlin` dequantizes weight-only (the boot logs a warning saying exactly this).
That is cheap for decode, which is memory-bound, and expensive for prefill, which is
compute-bound. It is upstream's own recommended kernel for this checkpoint on this hardware —
the intended trade-off on a Spark, not a misconfiguration, and not something a flag here can
fix.

Two things that were checked and are *not* problems, having been suspected: the box is not
memory-thrashing (`/proc/pressure/memory` and `/io` are ~0, and the other six containers total
under 1 GB), and the NVMe is not saturated (4 KB random table lookups at ~1600 IOPS, far
below the device). The GPU idles at 611 MHz and holds 923 MHz under load.

## Notes

### The vLLM services

Each backend mirrors the `command:`/`vllm:` block of its upstream recipe with the template
placeholders inlined, so the two stay easy to diff: `qwen38-fast` and `qwen38` against
Eugr's B12x recipes, `qwen38-hibrid48` against bilikaz's `recipe.yaml` (whose `run.sh` maps
`name: value` to `--name value`, which is why that block's flags look plainer). Every
argument is quoted deliberately: `--speculative-config '{...}'` and
`--compilation-config '{...}'` are single-quoted so the shell hands vLLM one JSON
argument. If you edit the flags, re-check the resulting argv rather than assuming — put a
stub `vllm` on `PATH` that prints `"$@"`, run the script's `exec` line, and diff the flags
against the recipe.

The images' entrypoint is `["vllm","serve"]`, which is why the serve command lives in a
script driven by `entrypoint:` rather than in `command:` — a `command:` value would have
`vllm serve` prepended to it.

All three services set `security_opt: ["seccomp=unconfined"]`, and that is **required**, not
hardening drift. Both engines reach their n-gram table through io_uring — the B12x PLE
reader with scatter-gather reads, the v4 image's table library against the NVMe map — and
Docker's default seccomp profile blocks `io_uring_setup`; without this the engine dies a few
minutes into the weight load with `io_uring initialization failed: Operation not permitted`.
Both upstream launchers run containers `--privileged` (which also drops seccomp), which is
why upstream never sees it. Unconfining seccomp is the *only* piece of `--privileged` taken
here — the GPU reservation, capabilities, device list and mounts are all still restricted.

`qwen38-fast` additionally runs the vendored `mods/flashnext-int4-b12x/` before serving:
it builds two symlink farms under `/workspace/flashnext` (the served model, with the PLE
table's 131 shards indexed and the config fields B12x needs; and a slim MTP draft folder
with top-k 10) and patches the container's vLLM for this checkpoint. Both steps are
idempotent and re-run on every container start — the image itself is never modified. A
required patch that cannot be applied aborts the launch rather than serving a broken
model. See `mods/flashnext-int4-b12x/README.md`.

`qwen38-hibrid48` needs none of that: its image carries its own patches, and the script
does three things — check the checkpoint is there, make sure `/cache` is writable, serve.
It is also the odd one out on purpose, so do not "harmonise" it with the other two:

- **No RDMA anything.** No RoCE env, no `/dev/infiniband`, no `cap_add: IPC_LOCK`, no
  `memlock` ulimit. Those come from the B12x recipes (where they are inert on one box
  anyway) and the v4 image does not want them. This is the local-NVMe sibling of the engine
  behind Saren's `magi-v2`, not the RDMA one.
- **KV memory.** `--kv-cache-memory-bytes 26000000000`. This is the *same option* the
  `qwen38-fast` script already passes — the recipe writes it as `--kv-cache-memory`, an
  unambiguous argparse abbreviation of it, and this repo spells it in full so nothing
  depends on prefix matching. Only the value differs. The option takes bytes, and its
  parser also accepts a suffix, where **case matters**: `26g` is 26e9 (decimal) and `26G`
  is 26 GiB (binary), ~7% more for the same two characters — which is why the full byte
  count is written out here. Upstream ships `27000000000` (its "27G", 830,582 tokens); this
  repo pins `26000000000` — `26g` — because the recipe documents 26G for a box that runs
  other things, and this box runs litellm, Postgres, Redis, Open WebUI and both exporters.
  It is a bound, not a measured capacity: read the pool size off the startup log (step 1 of
  the smoke test) and watch `vllm:kv_cache_usage_perc` against the host's `MemAvailable`.
- **`cpuset: "5-9,15-19"`** from the recipe — the GB10's ten 3.9GHz Cortex-X925 cores.
  Worth a couple of percent; drop it if you would rather the scheduler place the container
  freely alongside everything else.
- **`--block-size 1632`** is required at K=5, not a tuning choice: without it the boot stops
  with `QSA ring capacity 12 must divide the attention block size 1616`. Likewise the
  cudagraph capture sizes are multiples of K+1 up to `max-num-seqs × 6`; if you change
  `--max-num-seqs`, change them together or the top rungs decode without CUDA graphs.

### Chat templates

`chat-templates/` holds copies of each checkpoint's **own** template with exactly one
change: consecutive *leading* system messages are merged into a single system block.

This is needed because all three checkpoints' templates consume only `messages[0]` as the
system prefix and then raise `System message must be at the beginning.` on a second one — and the
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
- `qwen38-hibrid48.jinja` — from the `myllmbox/Qwen3.8-Flash-Next-hibrid48` checkpoint's own
  `chat_template.jinja`, plus the merge. The v4 recipe also passes no `--chat-template`;
  this repo passes this file explicitly for the same reason as the full backend.

The hibrid48 template was **not** assumed to match the others. The checkpoint's file was
fetched and compared before reuse: it is byte-identical to the `local-inference-lab/…`
template (8952 bytes, and to the copy embedded in `tokenizer_config.json` too), so
`qwen38-hibrid48.jinja` differs from `qwen38-full.jinja` only in its provenance header, and
the two bodies are byte-identical. If either upstream file moves, re-derive both and re-run
the rendering diff instead of trusting that note.

Tool-call parsing is per-backend and tracks the recipe: `qwen38-fast` uses
`--tool-call-parser qwen3_coder`, while `qwen38` and `qwen38-hibrid48` use `qwen3_xml`. All
three use `--reasoning-parser qwen3`, and all three templates gate thinking on the same
`chat_template_kwargs.enable_thinking`.

To update a template, re-derive it from the checkpoint rather than editing by hand, then
re-run the rendering diff. The upstream files are inside the cached snapshots — and for
hibrid48 it is a single file, at
`https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48/resolve/main/chat_template.jinja`.

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
| `qwen38-hibrid48` exits: `FATAL vllm-start-hibrid48: /models/… is not mounted` | the checkpoint is a separate ~98 GiB `--local-dir` download, not the HF cache; see [Prerequisites](#2-prerequisites) |
| `qwen38-hibrid48` exits: `FATAL vllm-start-hibrid48: … has no ple_quantization` | wrong checkpoint mounted at `/models` — this image serves `myllmbox/Qwen3.8-Flash-Next-hibrid48` only, not the B12x or `nvidia/…` ones |
| `qwen38-hibrid48` exits: `cannot create /cache/vllm-cache` | `/models/qwen38/cache` is missing, read-only, or not mounted. It must be writable: the first boot writes the ~31 GB table map there |
| `qwen38-hibrid48` stops with `invalid quant` | the image's table library rejected the n-gram table. Usually a partial download, or a table from a different release — re-download and re-check `du -sh` against the 28 shards |
| `qwen38-hibrid48` stops with `QSA ring capacity 12 must divide the attention block size 1616` | `--block-size 1632` was dropped. It is required at K=5, not a tuning choice |
| `qwen38-hibrid48` comes up but OOMs under load | `--kv-cache-memory-bytes 26000000000` is too big for a box that is also running everything else — drop it to `24000000000` and re-run the memory step of the [smoke test](#gb10-smoke-test-for-qwen38-hibrid48) |
| A `--kv-cache-memory*` value came out ~7% smaller than intended | the suffix is case-sensitive: `26g` is decimal (26e9), `26G` is binary (26 GiB). Write the byte count out, or use the case you mean |
| `--kv-cache-memory` rejected as ambiguous | another `--kv-cache-memory*` option exists in this build and the abbreviation no longer resolves. Use the full `--kv-cache-memory-bytes` |
| `qwen38-hibrid48` answers but never emits `tool_calls` | the tool call landed in `content` instead — the parser must be `qwen3_xml` for this checkpoint, not the fast backend's `qwen3_coder` |
| Repeated cached answers | global `cache: true`; see [Caching](#caching) |
