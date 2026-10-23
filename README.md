# DeepSeek-V4.1-Flash-1017 abliterated · FP8 · vLLM + DSpark · 4x NVIDIA RTX PRO 6000

| | |
| --- | --- |
| Model | [`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) at revision `517efab9ccfef7f714dbd7d679f181a86f43ccf9`, served as `DeepSeek-V4.1-Flash-1017-abliterated` |
| Weights | the native 1017 layout: FP8 E4M3 dense weights with UE8M0 32x32 block scales, FP4 routed experts, 48 shards, 510.3 GB |
| Engine | vLLM of Local Inference Lab, `ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920`, with `CHECKPOINT=original` and a two-line patch to the 1017 encoding (`patches/`) |
| Speculation | DSpark, 3 draft tokens from the 3 MTP layers of the checkpoint, adaptive verification at cost scale 0.5 |
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) in one `g4-standard-192` VM, tensor parallel 4 |
| Memory | 74.42 GiB of weights per GPU; the Engram tables stay on a 5.9 TB RAID0 of 16 local NVMe disks |
| Context | **1,048,576** tokens per request; a 6,560,752-token KV pool, 6.26 requests at full context |
| Decode, one stream | **166 to 211 tok/s** over four runs, first token after 0.26 to 0.33 s |
| Decode, eight streams | 710 and 722 tok/s in total over two runs, 93 to 95 tok/s per stream |
| Long context | needle found in 109,683 tokens, answer after 6.5 s |
| Features | thinking, tool calls (parallel), SSE streaming and image input over the Anthropic Messages API; OpenAI chat completions too |
| Refusals | answers a step-by-step lock picking request, which the stock 1017 refuses |
| Download | 510.3 GB from Hugging Face in 261 s, 1.95 GB/s, anonymous |
| Start | 16.5 min on a fresh VM (12 min of kernel tuning), 3 min with the persisted kernel cache |
| Cost | $6.34 per hour on spot, $18.00 on demand |

All decode numbers come from `scripts/bench.mjs`, which streams a 1,500-word essay with
`max_tokens` 2048 over HTTPS through the TLS proxy. Thinking is always on, so the 2,048 output
tokens include the reasoning. Decode speed is the output tokens divided by the time from the first
delta to the last byte.

```bash
cp .env.example .env            # then set DS41_API_KEY (HF_TOKEN is optional)
bash gcp/provision.sh           # firewall, spot VM, static IP, RAID, weights, server, TLS: ~25 min
bash scripts/smoke.sh           # 9 checks against https://<static-ip>.sslip.io
DS41_URL=https://<static-ip>.sslip.io node scripts/bench.mjs c8
```

## GPU

Cloud Run attaches at most one GPU to an instance, and its fractional RTX PRO 6000 accepts only 0.5
or 1. A 4-way tensor parallel model therefore needs a Compute Engine G4 VM. `g4-standard-192` has
4 RTX PRO 6000 cards, 192 vCPUs, 720 GB of memory and 16 local Titanium NVMe disks of 375 GB, which
are included in the price and must be attached as all 16 or none. `nvidia-smi topo -m` shows two
PCIe switch pairs, (0,1) and (2,3), with a NUMA-node hop between the pairs, so the default GPU order
is as good as any other ring.

The VM runs `common-cu129-ubuntu-2404-nvidia-580-v20261019` (Ubuntu 24.04, driver 580.178.04),
to which `vm/vm-docker.sh` adds Docker (29.9.2 on this VM) and the NVIDIA container toolkit
(1.19.1). The cards draw on two quotas. Spot VMs use `PREEMPTIBLE-NVIDIA-RTX-PRO-6000-GPUS-per-project-region`, 32 GPUs in every
region of the project. On-demand VMs use `GPUS-PER-GPU-FAMILY-per-project-region` with
`gpu_family=NVIDIA_RTX_PRO_6000`, which the project holds only in us-west1 (10) and us-central1;
elsewhere the create call fails with `Quota 'GPUS_PER_GPU_FAMILY' exceeded`. On 2026-10-22,
`gcp/grab.sh` tried on-demand in the six zones of those two regions for 42 rounds over 75 minutes.
All 270 attempts failed, 132 with `stockout`, 131 with `resource_availability` and 7 with
`gpu_availability`. A spot request found capacity in us-west3-a, so spot is the default here.

## Checkpoint

We compared four abliterated builds of the 1017 release in its full format. Each card reports its
refusals on its own prompt set:

| Build | Edit | Refusals after | KL, harmless | Format | Access |
| --- | --- | --- | ---: | --- | --- |
| **`mrfakename/...-abliterated`** | rank-2 subspace removed from `attn.wo_b` and `ffn.shared_experts.w2` in 40 layers | chat 4/100, thinking 9/100 | **0.067** | native FP8 | public |
| `pqhaz/...-heretic` | one direction per block removed from `attn.wo_b` in blocks 2 to 39 | chat 1/100 | 0.136 | native FP8 | public |
| `dealignai/...-UNCENSORED-FP8` | fine-tune of the base | | | FP8 | public |
| `jcartu/...-abliterated-lossless-CSF` | refusal direction inverted in `attn.wo_b` and `ffn.shared_experts.w2`, layers 4 to 36 | | | LiL CSF container | gated, no redistribution |

We serve the mrfakename build. It keeps the native file layout, so the generic loader of the LiL
image reads it with `CHECKPOINT=original`. It edits only the two dense residual writes, and the
routed experts, embeddings, Engram tables and DSpark layers are bit-identical to the stock 1017.
The edited tensors were requantized to FP8 with their original block scales. Of the builds with
published numbers, it has the lowest KL divergence on harmless prompts. The CSF build needs a
manual approval and the `CHECKPOINT=csf` loader; on the same image its card reports 846.8 tok/s at
eight streams against 824.8 for the plain 1017 checkpoint.

## Memory

The checkpoint holds 475 GiB of tensors and the four cards 4x 94.97 GiB. With
`ENGRAM_TABLE_MEMORY=disk` the Engram tables stay on the local RAID and only their scales stay
resident (`ENGRAM_DISK_RESIDENT_SCALES=true`); the startup log reports 5.72 GiB of packed tables in
mapped host memory across the four ranks. The rest fits:

| Per GPU | GiB | Log line |
| --- | ---: | --- |
| Card, usable | 94.97 | `Free memory on device (93.43/94.97 GiB)` |
| Budget at `GPU_MEMORY_UTILIZATION=0.95` | 90.22 | `Desired GPU memory utilization is (0.95, 90.22 GiB)` |
| Weights | 74.42 | `Model loading took 74.42 GiB memory` |
| KV cache | 8.05 | `Available KV cache memory: 8.05 GiB` |
| CUDA graphs | 2.81 | `CUDA graph pool memory: 2.81 GiB (actual)` |

The KV pool holds 6,560,752 tokens over the four cards, so one request can use the full
1,048,576-token context while others run.

## Serving

`vm/ds41-start.sh` is the deployed command: `serve-ds41-flash.sh -- --max-cudagraph-capture-size
128` in the LiL image, with host networking, `--shm-size 128g`, `seccomp=unconfined` and an
unlimited `memlock`. The environment is:

| Variable | Value | Effect |
| --- | --- | --- |
| `CHECKPOINT`, `MODEL` | `original`, `/model` | plain safetensors layout |
| `TP` | `4` | tensor parallel over the four cards |
| `GPU_MEMORY_UTILIZATION` | `0.95` | |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | prefill chunk |
| `DSPARK_TOKENS` | `3` | draft tokens per step |
| `DSPARK_ADAPTIVE_VERIFICATION`, `..._COST_SCALE` | `true`, `0.5` | |
| `ENGRAM_TABLE_MEMORY`, `ENGRAM_DISK_RESIDENT_SCALES` | `disk`, `true` | Engram tables on NVMe |
| `VLLM_ENABLE_PCIE_ALLREDUCE`, `VLLM_PCIE_ALLREDUCE_BACKEND` | `1`, `b12x` | one-shot PCIe all-reduce |
| `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE`, `VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE` | `96KB`, `96KB` | |
| `VLLM_PCIE_DMA_MIN_BYTES` | `6MB` | |
| `B12X_MHC_PDL`, `B12X_PCIE_ONESHOT_PDL` | `1`, `1` | |
| `VLLM_MXFP8_LM_HEAD` | `1` | MXFP8 `lm_head` |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | reachable only from the VM |
| `HF_HUB_OFFLINE` | `1` | |
| `VLLM_API_KEY` | from `/etc/ds41/env` | vLLM checks the key too |

The image vendors a DeepSeek encoding older than 1017, and tool calls need the encoding shipped
with the checkpoint, so the start script mounts `encoding/encoding.py` of the checkpoint over
`vllm/tokenizers/deepseek_v41_encoding.py`. That file rejects text containing the image
placeholder token, while the deepseek_v41 tokenizer of vLLM replaces every image part with
exactly that token before it calls the encoder, so every request with an image returned 400. `patches/encoding-1017-vllm.patch` disables the two checks. `vm/vm-encoding.sh`
applies it to the downloaded file, whose SHA-256 is `13231c59...`, and the result is
`ee0dc68d...`.

Turning those checks off has a cost: the encoder can no longer tell a real image from the
placeholder token written as text. A web page or file that contains the token then becomes an
extra image slot, and every real image after it shifts by one. This README contains the token in
its first version, so an agent that had read it saw no image, or the previous one, on every later
screenshot (jdh run d43e80382f05). `patches/deepseek_v41-placeholder.patch` fixes this in vLLM's
`tokenizers/deepseek_v41.py`: text, reasoning and tool output have the token replaced with an
ASCII look-alike before images are flattened. With the same 23-message request, the model read the
last image in 4 of 4 tries after the patch and in 0 of 4 before it. `vm/vm-encoding.sh` builds the
file from the image and `ds41-start.sh` mounts it.

`/v1/models` answers before warm-up ends, so `vm/ds41-ready.sh` waits for a real `/v1/messages`
call to return 200.

## Endpoint

vLLM listens on 127.0.0.1:8000. Caddy terminates TLS on port 443 with a Let's Encrypt
certificate for `<static-ip-with-dashes>.sslip.io`, a name that resolves to the IP it spells. It
forwards a request only when `Authorization: Bearer <key>` or `x-api-key: <key>` matches, and
answers 401 on every other request and path, including `/health` and `/metrics`. Caddy writes no
access log. The firewall rule `allow-ds41-https` opens TCP 443 and 80 to the network tag
`ds41-serve`; port 80 serves only the ACME challenge and the redirect. The static address
`ds41-flash-ip` keeps the name stable across stops and preemptions.

## Cold start

From VM creation to the first token on a fresh spot VM, according to the timestamps of the logs:

| Phase | Time |
| --- | ---: |
| VM create, driver, Docker, RAID0, hf cli | 1.5 min |
| Weights, 510.3 GB (`hf download`, hf_xet high-performance, 48 workers), image in parallel | 261 s |
| Encoding patch, Caddy and certificate | 1.8 min |
| Container start to weights loaded (17.4 s, about 18 GB/s from the RAID) | 1.3 min |
| b12x and FlashInfer kernel compilation and tuning, 3,029 kernels | 12 min |
| KV pool, CUDA graphs, warm-up | 3 min |
| **Server start to ready** | **16.5 min** |
| **VM create to ready** | **24 min** |

The kernel cache lives in `/data/cache/lil-cache` (792 MB), mounted as `/cache` in the container.
`gcp/bootstrap.sh` copies it to `/var/lib/ds41/lil-cache` on the boot disk once the server is
ready. With the cache, a restart of the container takes 3 min to ready, 114 s of which the engine
spends on its pools and graphs.

## Preemption

A spot preemption stops the VM, and a stop discards the contents of the local disks: weights,
Docker image and kernel cache. `vm/ds41-boot.service` runs `vm/ds41-boot.sh` on every boot. When
the server is not up, it rebuilds the RAID, re-downloads the weights, re-pulls the 44.1 GB image,
restores the kernel cache and the patched encoding from the boot disk, and starts the container.
Its log is `/var/log/ds41-boot.log`. From the phases above this takes about 15 minutes. The path
has not run on a real preemption yet; on a running server the script exits with `already serving`.

## Cost

Cloud Billing catalog prices for us-west3, which match Oregon, on 2026-10-22:

| | per hour | per month (730 h) |
| --- | ---: | ---: |
| Spot: 192 x $0.01898 + 720 GB x $0.002281 + 4 x $0.2625 | **$6.34** | **~$4,630** |
| On demand: 192 x $0.04891 + 720 GB x $0.00587 + 4 x $1.09565 | $18.00 | ~$13,140 |
| Boot disk, 400 GB hyperdisk-balanced, and the static IP | | ~$36 |

The 16 local disks cost nothing on G4. `bash gcp/teardown.sh stop` stops the VM and its billing
except the boot disk and the IP.

## Secrets

The scripts read two secrets from the environment, or from `.env`, which git ignores. Everything
else is a committed default in `config.sh`.

| Variable | Read by | Where it goes |
| --- | --- | --- |
| `DS41_API_KEY` | `gcp/provision.sh`, `gcp/bootstrap.sh`, `scripts/smoke.sh`, `scripts/bench.mjs` | over SSH stdin into `/etc/ds41/env` (`VLLM_API_KEY`, root, 0600) and `/etc/ds41/caddy.env` |
| `HF_TOKEN` | `gcp/bootstrap.sh`, optional | `/root/.hf_token` (0600), read by `vm/vm-download.sh` |

On the Mac that runs the clients, the key is stored only in `~/.pi/agent/models.json`, and this
loads it into the shell:

```bash
export DS41_API_KEY=$(jq -r '.providers["jina-rtx"].models[]
  | select(.id|startswith("DeepSeek")) | .headers.Authorization | sub("^Bearer ";"")' ~/.pi/agent/models.json)
```

A new deployment takes a new key, for example `echo "sk-ds41-$(openssl rand -hex 24)"`. The model
repository is public, so the download needs no token.

## Clients

pi takes a `baseUrl` and `headers` per model, so the model sits next to other models of a
provider without a router:

```json
{
  "providers": {
    "jina-rtx": {
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
  -H "Authorization: Bearer $DS41_API_KEY" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017-abliterated","max_tokens":2000,
       "messages":[{"role":"user","content":"Reply with one word: pong"}]}'
```

The model always thinks, and `thinking: {"type": "disabled"}` in a request has no effect. Its one
thinking knob is the reasoning effort: low, high and max map to 50, 75 and 100 on a 1-100 scale, and
there is no xhigh. A `budget_tokens` from the client is ignored: at effort high, one short math
question thought until it hit a 30,000-token `max_tokens`. The image defaults to high. The patch
above also moves that default to max, so a request that names no effort gets 100; a client can still
ask for less with `output_config.effort`. The model card asks for `max_tokens` of at least 256K, so
the pi entry below uses 262144. Usage follows Anthropic: a prompt splits into `cache_creation_input_tokens`,
`cache_read_input_tokens` and the remaining `input_tokens`, in blocks of 256 tokens. In OpenClaw, a provider with `authHeader: true`
replaces the `Authorization` header of every model with the provider key, so this model needs its
own provider there.

## Reproduction

```bash
bash gcp/provision.sh            # firewall rule, VM via grab.sh, static IP, then bootstrap.sh
PROVISIONING=STANDARD bash gcp/provision.sh   # on demand, zones of us-west1 and us-central1
bash gcp/grab.sh                 # only the create loop: every zone once per round, 45 s apart, 90 min
bash gcp/bootstrap.sh            # copy vm/, write secrets, set up, start, wait, back up the cache
RESTART=1 bash gcp/bootstrap.sh  # the same, and recreate the container
bash scripts/smoke.sh            # auth, models, chat, thinking, tools, stream, image, long, refusal
bash scripts/smoke.sh long       # one check
DS41_URL=https://... node scripts/bench.mjs single   # also c8, c32
bash gcp/teardown.sh stop|start|delete
```

On the VM, the files land where `vm/ds41-boot.sh` expects them:

| Repo | VM |
| --- | --- |
| `vm/vm-docker.sh`, `vm-setup.sh`, `vm-download.sh`, `vm-encoding.sh`, `vm-caddy.sh`, `Caddyfile`, `patches/encoding-1017-vllm.patch` | `/opt/ds41/` |
| `vm/ds41-start.sh`, `ds41-boot.sh`, `ds41-ready.sh` | `/usr/local/bin/` |
| `vm/ds41-boot.service` | `/etc/systemd/system/` |
| values of `config.sh` (image, model, revision, domain) | `/etc/ds41/deploy.env` |
| weights, Docker data root, kernel cache, logs | `/data` (RAID0, wiped on stop) |
| patched encoding and kernel cache backups | `/var/lib/ds41/` |

```bash
gcloud compute ssh ds41-flash-rtx-spot --zone us-west3-a --project jinaai-dev
sudo docker logs -f ds41          # server log
sudo /usr/local/bin/ds41-start.sh # recreate the container
sudo /usr/local/bin/ds41-ready.sh # 200 when it serves
```

## Credit

The serving image, the `serve-ds41-flash.sh` recipe and the 4x RTX PRO 6000 settings come from
[Local Inference Lab](https://huggingface.co/local-inference-lab) and the public card of
[`jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF`](https://huggingface.co/jcartu/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF).
The abliterated weights are
[`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated),
an edit of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MIT). The single-card companion of this deployment is
[`Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000`](https://github.com/hanxiao/Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000).
