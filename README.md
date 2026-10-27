# DeepSeek-V4.1-Flash-1017 abliterated on 4x NVIDIA RTX PRO 6000

We serve an abliterated DeepSeek-V4.1-Flash-1017 with vLLM-Moet r21 and DSpark speculative decoding on one
Compute Engine `g4-standard-192` VM, behind a TLS proxy with a Bearer key. This repository holds the scripts
that create the VM, prepare it, serve the model, restore it after a stop, and test it.

| | |
| --- | --- |
| Model | [`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) at revision `517efab9ccfef7f714dbd7d679f181a86f43ccf9`, served as `DeepSeek-V4.1-Flash-1017-abliterated` |
| Weights | the native 1017 layout, 48 shards, 510.3 GB: FP8 E4M3 dense weights with UE8M0 32x32 block scales and FP4 routed experts (`quant_method: fp8`, `expert_dtype: fp4`) |
| Engine | [vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) `b0dc737` on vLLM nightly `df3bd094` (image `vllm-moet-sm120:dsv41-nightly-df3bd094-r21`, built on the VM), with two tokenizer patches |
| Speculation | DSpark, 5 draft tokens from the MTP layers of the checkpoint |
| KV cache | `fp8_ds_mla` records and an FP8 indexer cache, a pool of 4.19M tokens |
| Batching | 8 sequences, 8,192-token prefill chunks, CUDA graphs up to batch 128 |
| GPUs | 4x RTX PRO 6000 Blackwell Server Edition (96 GB, PCIe, no NVLink), tensor parallel 4 |
| Context | 1,048,576 tokens per request |
| Decode, coding agent | 371 / 375 / 376 tok/s on one stream at 30k / 60k / 120k tokens of context, thinking at max effort |
| First token | 1.40 / 2.95 / 5.99 s at 30k / 60k / 120k; 0.15 to 0.31 s on a follow-up turn over the cached prefix |
| Features | thinking (effort max by default), parallel tool calls, image input and SSE streaming over the Anthropic Messages API; OpenAI chat completions too |
| Host | Flex-start VM in a zone with capacity, $9.00 per hour |

```bash
cp .env.example .env            # set DS41_API_KEY, and VM_NAME for a new deployment
bash gcp/provision.sh           # firewall, Flex-start VM in the first of 46 zones, static IP, r21, TLS
bash scripts/smoke.sh           # 9 checks against https://<static-ip-with-dashes>.sslip.io
```

## Hardware

Cloud Run attaches at most one GPU to an instance, so a 4-way tensor parallel model needs a Compute Engine G4 VM.
`g4-standard-192` has 4 RTX PRO 6000 cards, 192 vCPUs, 720 GB of memory and 16 local Titanium NVMe disks of
375 GB, which are included in the price and must be attached as all 16 or none. The cards talk over PCIe:
`nvidia-smi topo -m` shows two switch pairs, (0,1) and (2,3), with a NUMA hop between the pairs, so the default
GPU order is as good as any other ring.

The VM runs `common-cu129-ubuntu-2404-nvidia-580-v20261019` (Ubuntu 24.04, driver 580.178.04-server-open,
CUDA 13.0), to which `vm/vm-docker.sh` adds Docker and the NVIDIA container toolkit. We stay on the R580 branch,
which Google recommends for G4; the release notes of 580.196, 595, 610 and 615 list no decode or prefill
changes for SM120.

The Server Edition cards run at their default power limit of 600 W, which is also the maximum. During
single-stream decode they draw 280 to 397 W (318 W on average) at 2,400 to 2,415 MHz of a 2,430 MHz SM clock,
with `clocks_event_reasons` at 0x0 in every 1 s sample. The memory clock is fixed at 12,481 MHz and cannot be
overclocked. Decode is limited by neither power nor clocks, so `nvidia-smi -pl` has nothing to raise.

## Checkpoint

We compared four abliterated builds of the 1017 release in its full format. Each card reports refusals on its
own prompt set:

| Build | Edit | Refusals after | KL, harmless | Format | Access |
| --- | --- | --- | ---: | --- | --- |
| **`mrfakename/...-abliterated`** | rank-2 subspace removed from `attn.wo_b` and `ffn.shared_experts.w2` in 40 layers | chat 4/100, thinking 9/100 | **0.067** | native 1017 | public |
| `pqhaz/...-heretic` | one direction per block removed from `attn.wo_b` in blocks 2 to 39 | chat 1/100 | 0.136 | native 1017 | public |
| `dealignai/...-UNCENSORED-FP8` | fine-tune of the base | | | FP8 | public |
| `jcartu/...-abliterated-lossless-CSF` | refusal direction inverted in `attn.wo_b` and `ffn.shared_experts.w2`, layers 4 to 36 | | | LiL CSF container | gated, no redistribution |

We serve the mrfakename build. It keeps the native file layout, which every engine here loads as is, and of
the builds with published numbers it has the lowest KL divergence on harmless prompts. It edits only the two
dense residual writes. The routed experts, embeddings, Engram tables and MTP layers are bit-identical to the
stock 1017, and the edited tensors were requantized to FP8 with their original block scales. The checkpoint
keeps the mixed precision of the release: dense and attention weights in FP8, routed experts in FP4. It answers
a step-by-step lock picking request, which the stock 1017 refuses.

## Serving

`vm/ds41-start-r21.sh` starts the container with host networking, `--ipc host`, `seccomp=unconfined` and an
unlimited `memlock`, and reads the served configuration from `/etc/ds41/r21.env` (`vm/r21.env`):

| Setting | Value |
| --- | --- |
| `--tensor-parallel-size` | 4 |
| `--speculative-config` | `dspark`, 5 tokens, probabilistic drafts, block rejection sampling, drafter on b12x |
| `--kv-cache-dtype`, indexer cache | `fp8_ds_mla`, FP8 |
| `--max-num-seqs` | 8 |
| `--max-num-batched-tokens` | 8192 |
| `--max-cudagraph-capture-size` | 128 |
| `--gpu-memory-utilization` | 0.94 |
| `--max-model-len` | 1048576 |
| `--moe-backend` | b12x |
| `--default-chat-template-kwargs` | `{"thinking": true, "reasoning_effort": "max"}` |
| `--override-generation-config` | temperature 1.0, top_p 0.95 |
| `--attention-config` | `indexer_tp_shard_min_rows` 2048 |
| KV offload | none |
| vLLM-Moet switches | fused b12x MoE, sparse MLA LDG, MXFP8 `lm_head`, P2P all-reduce (fused up to 8 rows), DFlash prep groups, deterministic candidate top-k, SWA decode and draft builds, MoE top-k sum add |
| Prefix caching, usage in streams | on |
| Listen | 127.0.0.1:8000, behind Caddy |

Tool calls need the encoding shipped with the checkpoint, so the start script mounts
`encoding/encoding.py` of the checkpoint over `vllm/tokenizers/deepseek_v41_encoding.py`. That file rejects text containing the
image placeholder token, while the vLLM tokenizer replaces every image part with exactly that token before it
calls the encoder, so `patches/encoding-1017-vllm.patch` disables the two checks (`vm/vm-encoding.sh`).
`patches/deepseek_v41-moet.patch` changes `vllm/tokenizers/deepseek_v41.py` in three ways. It renders the
placeholder token inside text, reasoning and tool output as an ASCII look-alike, so a web page that contains the
token cannot become an extra image slot that shifts every later image by one. It renders each real image as a
unique sentinel first and reads the image order off the encoded prompt, which the 1017 encoder needs because it
takes no image order argument. It sets the default reasoning effort to max. With a 23-message agent request,
the model read the last image in 4 of 4 tries with the placeholder fix and in 0 of 4 without it.
`vm/vm-r21.sh` builds the overlay from the image and keeps a copy on the boot disk.

`/v1/models` answers before warm-up ends, so `vm/ds41-ready.sh` waits for a real `/v1/messages` call to
return 200.

## Endpoint

vLLM listens on 127.0.0.1:8000. Caddy terminates TLS on port 443 with a Let's Encrypt certificate for
`<static-ip-with-dashes>.sslip.io`, a name that resolves to the IP it spells. It forwards a request only when
`Authorization: Bearer <key>` or `x-api-key: <key>` matches, and answers 401 on every other request and path,
including `/health` and `/metrics`. Caddy writes no access log. The firewall rule `FIREWALL_RULE` opens TCP
443 and 80 to the network tag `NETWORK_TAG`; port 80 serves only the ACME challenge and the redirect. A static
address keeps the name stable across stops. The site name is written into `/etc/caddy/Caddyfile` and
`/etc/ds41/deploy.env`, so a new IP needs both updated.

## Evaluation

We measure with three workloads. All count output tokens from `usage`, so thinking tokens are included, and
compute decode speed as output tokens divided by the time from the first token to the last.

| Workload | Requests |
| --- | --- |
| codebench | the coding agent workload: `/v1/messages` with tools, thinking at max effort, 30k / 60k / 120k tokens of context, 4,096 output tokens; C1 is one stream, C2 and C3 run two and three agents at once. A follow-up turn reuses the cached prefix. |
| Fraser bench | the vLLM-Moet benchmark: one fixed 2,678-token code-generation prompt, thinking off, temperature 0, 2,048 output tokens, one stream, a cache-busting line per request |
| bench3 / `scripts/bench.mjs` | a 1,500-word essay with thinking on, the Fraser prompt with thinking on and off, and 8 or 16 concurrent essays, 2,048 output tokens each |

codebench C1 varies by about ±10% per request with the task, so we compare medians of at least six runs and
check the step rate (decode speed over accepted length), and we drop runs during which other clients were
active.

### Engines

Three engines served this checkpoint on the same hardware: Local Inference Lab vLLM (LiL,
`ghcr.io/local-inference-lab/vllm@sha256:fd74019f...` with DSpark k = 3 and adaptive verification at cost scale
0.5), vLLM-Moet r13 (`b522a06`) and vLLM-Moet r21 (`b0dc737`). The first table compares LiL and r13 on one VM
with bench3, reasoning effort high, medians of five runs:

| Workload | LiL, k3 | r13, k3 | r13, k5 |
| --- | ---: | ---: | ---: |
| essay, 1 stream, tok/s | 209.2 | 237.5 | **252.7** |
| code, thinking on, tok/s | 285.3 | 313.6 | **383.9** |
| code, thinking off, tok/s | 346.8 | 395.7 | **513.2** |
| 8 streams, total tok/s | 726.0 | **765.9** | 688.5 |
| 16 streams, total tok/s | 1,018.4 | **1,142.6** | |
| first token, 2,695-token prompt, s | 0.200 | **0.147** | **0.147** |
| first token, 41,555-token prompt, s | 2.504 | 2.041 | **2.032** |
| tokens per step, essay / code | 2.08 / 2.85 | 2.13 / 2.79 | 2.28 / 3.43 |
| KV pool, tokens | **6,560,752** | 3,651,468 | 3,681,614 |

LiL keeps the Engram tables on the local NVMe RAID (`ENGRAM_TABLE_MEMORY=disk`), which leaves 8.05 GiB of KV
per card after 74.42 GiB of weights and gives the largest pool. On its own VM and endpoint, `scripts/bench.mjs`
measured 166 to 211 tok/s on one stream and 710 to 722 tok/s at eight streams, and the smoke test found a needle
in 109,683 tokens in 6.5 s. Our agent traffic is a few sessions at a time with thinking at max effort, so
single-stream speed decides, and both Moet engines with k = 5 lead there.

The second table compares r13 and r21 on one VM with codebench (C1 medians of six runs, first token in seconds):

| Config | C1 30k / 60k / 120k tok/s | First token 30k / 60k / 120k | Follow-up | C2 30k / 60k / 120k | KV pool |
| --- | --- | --- | --- | --- | ---: |
| r13, k5, FP8 KV | 367 / 354 / 349 | 1.40 / 2.96 / 5.99 | 0.15 / 0.20 / 0.30 | 254 / 222 / 193 | 4.19M |
| r21, k7, FP4 KV, 64 GiB offload | 351 / 344 / 363 | 1.52 / 3.23 / 6.59 | 0.21 / 0.25 / 0.37 | 246 / 202 / 212 | 10.57M |
| r21, k5, FP4 KV | 374 / 369 / **393** | 1.52 / 3.22 / 6.57 | 0.20 / 0.25 / 0.36 | 255 / **246** / 213 | **10.75M** |
| **r21, k5, FP8 KV (served)** | 371 / **375** / 376 | **1.40 / 2.95 / 5.99** | **0.15 / 0.20 / 0.31** | **257** / 210 / **218** | 4.19M |

The served configuration decodes 1%, 6% and 8% faster than r13 at 30k, 60k and 120k with the same first-token
latency and the same KV pool. The FP4 KV record holds 2.5 times as many tokens and costs about 8% of prefill
and 0.05 s per follow-up turn. With thinking off, codebench C1 runs at 507 to 516 tok/s on r13. A newer r13
build, `69aa612`, measured within noise of `b522a06` (369 / 360 / 357 tok/s, 102 to 109 steps per second
against 104 to 111). The DeepInfer-derived vLLM changes of October 2026 target SM100 and NVL72 systems; on
this hardware they are worth under 1% of decode (vllm#63293) and a few percent of first-token latency at 120k
(vllm#62817), and confidence-scheduled verification, which vLLM-Moet measured on SM120, costs 14% of
single-stream code decode.

### Speculation and batching

We swept the draft length and the batching knobs on r13 with codebench at 32 sequences (C1 medians of two or
three runs at 30k / 60k / 120k; C2 and C3 total tok/s at 60k):

| Config | C1 tok/s | First token, s | C2 | C3 | Accepted length | KV pool |
| --- | --- | --- | ---: | ---: | --- | ---: |
| k3 | 304 / 322 / 310 | 1.42 / 2.96 / 6.00 | **239** | 175 | 2.83 to 2.94 | 3.68M |
| k4 | 323 / 352 / 342 | 1.41 / 2.96 / 5.99 | 209 | 165 | 3.14 to 3.33 | 3.68M |
| **k5** | 353 / 379 / 380 | 1.42 / 2.96 / 5.99 | 209 | 184 | 3.38 to 3.55 | 3.68M |
| k6 | 346 / 343 / 359 | 1.41 / 2.96 / 6.00 | 194 | 161 | 3.43 to 3.55 | 3.68M |
| k7 | 329 / 356 / 345 | 1.41 / 2.96 / 5.99 | 198 | 159 | 3.47 to 3.61 | 3.68M |
| k5, chunk 16,384 | 353 / 380 / 366 | 1.35 / 2.84 / 5.79 | 213 | 186 | 3.35 to 3.60 | 1.98M |
| k5, chunk 16,384, 4 sequences, graphs to 32 | 357 / 366 / 381 | **1.34 / 2.84 / 5.79** | 207 | 186 | 3.38 to 3.56 | 2.74M |
| k5, chunk 16,384, FP4 KV | **364 / 380 / 387** | 1.37 / 2.90 / 5.90 | 204 | **187** | 3.44 to 3.53 | 2.85M |

Five draft tokens give the best single stream on reasoning text. Longer drafts pay off only on predictable code:
on the thinking-off Fraser prompt, k = 7 reaches an accepted length of 5.45. A 16,384-token chunk saves 4% of
prefill time and halves the KV pool, and a 32,768-token chunk leaves no memory for KV at all, so we keep 8,192.
The FlashInfer b12x MoE backend does not support the FP4 experts. The KV pool depends on the sequence limit:
3.68M tokens at 32 sequences, 4.34M at 4 and 4.19M at 8, the limit we serve.

### Fraser bench

vLLM-Moet reports 680 tok/s (median of 40 runs, 741 best) on its Fraser bench for DeepSeek-V4.1-Flash on 4x RTX
PRO 6000, at 127.8 steps per second and an accepted length of 5.34. Its recipe is r21 with DSpark k = 7, an FP4
KV record (`nvfp4_ds_mla`) with an MXFP4 indexer cache, 64 GiB of host and disk KV offload, and workstation
cards at 400 W with a GDDR7 overclock of +5,000 MHz. We ran the recipe and our configurations on one VM with our
weights and tokenizer patches:

| Config | Fraser tok/s | Steps/s | Accepted length | KV pool |
| --- | ---: | ---: | ---: | ---: |
| r13, k5, FP8 KV | 500 / 517 | 107 | 4.7 | 4.19M |
| r21 recipe: k7, FP4 KV, offload | **555** | 102 | **5.45** | 10.57M |
| r21, k5, FP4 KV | 543 / 537 | **113** | 4.7 | **10.75M** |
| r21, k5, FP8 KV (served) | 534 / 521 | 111 | 4.8 | 4.19M |

Drafts are accepted as well here as on the reference host (5.45 against 5.34); the step rate is 102 against
128 per second. The Server Edition cards already run at full clocks, and their fixed memory clock rules out the
overclock, so 555 tok/s is the Fraser result of this hardware. The Fraser prompt with thinking on
runs at 354 tok/s on r13 and 358 on the recipe, and our agent traffic is close to that case.

### Quality

The served configuration passes the smoke test (9 of 9 over the public endpoint), needles at 110k and 214k
tokens, a streamed thinking request, three parallel tool calls with valid JSON, an image combined with a tool
call, and the effort check (the prompt renders `Reasoning Effort: 100`). GSM8K-100 with thinking scores 96% on
r21 k5 FP8, 96% on r13 and 97% on the FP4 KV recipe. Under greedy decoding on r13, two FP8 runs agree on 99 of 100
final answers, and FP4 KV agrees with FP8 on 97. Of five multimodal tool tests, four pass on every
engine. The fifth sends images inside tool results whose order differs from the calls: the Anthropic adapter of
vLLM moves those images into a separate user message, so the pairing between image and call is lost.

## Capacity and cost

G4 capacity in the US regions was scarce. Every provisioning model we used:

| Provisioning | Result |
| --- | --- |
| On demand, the 6 US zones where we had on-demand quota | 270 attempts over 75 min, all failed: 132 `stockout`, 131 `resource_availability`, 7 `gpu_availability` |
| On-demand specific reservation, same zones, every 10 min | failed with `gpu_availability` and `ZONE_RESOURCE_POOL_EXHAUSTED` |
| Calendar-mode future reservation | G4 is not supported |
| Spot, one US zone | ran for 13 hours, then 7 preemptions, 6 of them within 73 min, followed by stockout |
| Flex-start, one zone at a time | pending for over 2 hours |
| **Flex-start, 46 zones at once** | capacity within minutes, in two or three zones at the same time, on both runs |

Flex-start VMs run up to 7 days without preemption and then stop. They draw on the preemptible GPU quota of
the region and need `--discard-local-ssds-at-termination-timestamp=true` with local SSDs.
`gcp/flex.sh` submits one request per zone, each valid for an hour and resubmitted when it expires. Because
several zones can start a VM within the same minute, a coordinator keeps the first one, stops the loops, and
deletes the other candidates in several passes, since requests already submitted can still start. Boot disks
are created with auto-delete, so a deleted candidate leaves nothing behind, and the disk of the winning VM is switched
to keep.

Cloud Billing catalog prices for `g4-standard-192` (4 GPUs, 192 vCPUs, 720 GB), per hour:

| Region | On demand | Spot | Flex-start |
| --- | ---: | ---: | ---: |
| us-west1 | 18.00 | 6.34 | 9.00 |
| us-central1 | 18.00 | 6.65 | 9.00 |
| us-west3 | 19.80 | 7.84 | 9.00 |
| asia-south2 | 23.40 | 9.71 | 9.00 |
| asia-southeast2 | 23.40 | 10.27 | 9.00 |

Flex-start is 4 x $0.54783 per GPU + 192 x $0.024455 per vCPU + 720 x $0.002935 per GB, the same in every
region, or $1,512 for a full 7-day run. Committed use costs $12.42 per hour for one year and $7.92 for three
years in us-west1 and us-central1. The 16 local disks cost nothing on G4. The 400 GB hyperdisk-balanced boot
disk and the static IP add about $36 per month and remain billed while the VM is stopped.

One Flex-start VM serves this model with r21, k5, FP8 KV and 8 sequences. A second VM with r13 served as
backup until 2026-10-26.

## Cold start and recovery

On a fresh VM, from the log timestamps:

| Phase | Time |
| --- | ---: |
| VM create, driver, Docker, RAID0 over the 16 NVMe disks, hf cli | 1.5 min |
| Weights, 510.3 GB (`hf download`, hf_xet high-performance mode, 48 workers, anonymous) | 261 s, 1.95 GB/s |
| Encoding patch, Caddy and certificate | 1.8 min |
| r21 image build from `Dockerfile.sm120-dsv41-nightly` | 12 min |
| r21 start to ready (weights, kernel compilation, KV pool, CUDA graphs) | 7 min |

Later starts take 6 to 8 minutes. The r21 kernel cache lives in `/data/cache/r21`, mounted as `/root/.cache`
and `/root/.deep_gemm`, with a 7 MB copy in `/var/lib/ds41/r21-cache`.

A stop discards the local disks: weights, Docker images and kernel caches. `vm/ds41-boot.service` runs
`vm/ds41-boot.sh` on every boot. When the server is not up, it rebuilds the RAID, downloads the weights again,
loads the r21 image from `/var/lib/ds41/r21-image.tar.zst` (9.4 GB) on the boot disk, restores the kernel cache
and the patched tokenizer files, and starts the engine named in `/etc/ds41/engine`: `r21`, `moet` (r13) or
`lil`. When the image of that engine is missing, it falls back to the next one, ending with the LiL image, which
`vm/vm-download.sh` pulls from its registry. Its log is `/var/log/ds41-boot.log`.

## Secrets

The scripts read two secrets from the environment, or from `.env`, which git ignores. Everything else is a
committed default in `config.sh`.

| Variable | Read by | Where it goes |
| --- | --- | --- |
| `DS41_API_KEY` | `gcp/provision.sh`, `gcp/bootstrap.sh`, `scripts/smoke.sh`, `scripts/bench.mjs` | over SSH stdin into `/etc/ds41/env` (`VLLM_API_KEY`, root, 0600) and `/etc/ds41/caddy.env` |
| `HF_TOKEN` | `gcp/bootstrap.sh`, optional | `/root/.hf_token` (0600), read by `vm/vm-download.sh` |

On the client side, keep the key in one place, such as the client config, and export it from there.

A new deployment takes a new key, for example `echo "sk-ds41-$(openssl rand -hex 24)"`. The model repository
is public, so the download needs no token.

## Clients

pi takes a `baseUrl` and `headers` per model, so the model sits next to other models of a provider without a
router:

```json
{
  "providers": {
    "rtx": {
      "api": "anthropic-messages",
      "models": [
        {
          "id": "DeepSeek-V4.1-Flash-1017-abliterated",
          "baseUrl": "https://<static-ip-with-dashes>.sslip.io",
          "headers": { "Authorization": "Bearer <DS41_API_KEY>" },
          "contextWindow": 1048576,
          "maxTokens": 262144,
          "reasoning": true,
          "input": ["text", "image"]
        }
      ]
    }
  }
}
```

Any Anthropic client works:

```bash
curl https://<static-ip-with-dashes>.sslip.io/v1/messages \
  -H "Authorization: Bearer <DS41_API_KEY>" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017-abliterated","max_tokens":2000,
       "messages":[{"role":"user","content":"Reply with one word: pong"}]}'
```

Thinking is on by default. The reasoning effort maps low, high and max to 50, 75 and 100 on a 1-100 scale,
there is no xhigh, and a request that names no effort gets max; a client can ask for less with
`output_config.effort`. On the OpenAI endpoint, `chat_template_kwargs: {"thinking": false}` turns thinking off.
The Anthropic adapter accepts a `thinking.budget_tokens` of at least 1,024 and below `max_tokens`, and answers
400 otherwise. The model card asks for `max_tokens` of at least 256K, so the pi entry uses 262144. Usage follows
Anthropic: a prompt splits into `cache_creation_input_tokens`, `cache_read_input_tokens` and the remaining
`input_tokens`. A client that sends one provider-wide key in the `Authorization` header of every model needs
a separate provider for this model.

## Reproduction

```bash
bash gcp/provision.sh            # firewall rule, VM via flex.sh, static IP, then bootstrap.sh
PROVISIONING=SPOT bash gcp/provision.sh       # spot VM via grab.sh; STANDARD for on demand
ZONES="ZONE_A ZONE_B" bash gcp/flex.sh   # only the Flex-start request loop
bash gcp/bootstrap.sh            # copy vm/, write secrets, set up, build r21, start, wait, back up the cache
RESTART=1 bash gcp/bootstrap.sh  # the same, and recreate the container
ENGINE=lil bash gcp/bootstrap.sh # serve with the LiL image
bash scripts/smoke.sh            # auth, models, chat, thinking, tools, stream, image, long, refusal
bash scripts/smoke.sh long       # one check
DS41_URL=https://... node scripts/bench.mjs single   # also c8, c32
bash gcp/teardown.sh stop|start|delete
```

`gcp/provision.sh` leaves an existing `VM_NAME` alone, so a new deployment sets `VM_NAME` to a name that does
not exist yet; with Flex-start the VM is named `<FLEX_PREFIX>-<zone>`, and that name goes into `.env`. The
vLLM-Moet r13 image of the former backup VM is in no registry; `vm/ds41-start-moet.sh` and `vm/moet.env` document its
configuration. On the VM, the files land where `vm/ds41-boot.sh` expects them:

| Repo | VM |
| --- | --- |
| `vm/vm-docker.sh`, `vm-setup.sh`, `vm-download.sh`, `vm-encoding.sh`, `vm-caddy.sh`, `vm-r21.sh`, `Caddyfile`, `r21.env`, `moet.env`, `patches/*.patch` | `/opt/ds41/` |
| `vm/ds41-start-r21.sh`, `ds41-start-moet.sh`, `ds41-start.sh`, `ds41-boot.sh`, `ds41-ready.sh` | `/usr/local/bin/` |
| `vm/ds41-boot.service` | `/etc/systemd/system/` |
| values of `config.sh` (LiL image, model, revision, domain) | `/etc/ds41/deploy.env` |
| `vm/r21.env`, the engine name | `/etc/ds41/r21.env`, `/etc/ds41/engine` |
| weights, Docker data root, kernel caches, logs | `/data` (RAID0, wiped on stop) |
| r21 image, tokenizer overlays, kernel cache backups | `/var/lib/ds41/` |

```bash
gcloud compute ssh VM_NAME --zone ZONE --project PROJECT
sudo docker logs -f ds41              # server log
sudo /usr/local/bin/ds41-start-r21.sh # recreate the container
sudo /usr/local/bin/ds41-ready.sh     # 200 when it serves
```

## Credit

The serving engine is [vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet), whose README defines the
Fraser bench. The LiL image, the `serve-ds41-flash.sh` recipe and the first 4x RTX PRO 6000
settings come from [Local Inference Lab](https://huggingface.co/local-inference-lab) and the public card of
[`jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF`](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF).
The abliterated weights are
[`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated),
an edit of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MIT). The single-card companion of this deployment is
[`Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000`](https://github.com/hanxiao/Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000).
