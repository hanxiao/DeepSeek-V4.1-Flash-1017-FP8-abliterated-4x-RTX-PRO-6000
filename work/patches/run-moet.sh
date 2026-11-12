#!/bin/bash
# Serve the abliterated DeepSeek-V4.1-Flash-1017 on 4x RTX PRO 6000 with the vLLM-Moet sm_120 image
# (vLLM nightly + SM120 decode kernels). Same contract as ds41-start.sh: vLLM on 127.0.0.1:8000,
# Caddy in front on :443, weights at /data/models/ds41-abl.
# Overrides live in /etc/ds41/moet.env (KEY=VALUE lines).
set -uo pipefail
M=/data/models/ds41-abl
NAME=ds41
PORT=8000
IMAGE=${MOET_IMAGE:-vllm-moet-sm120:ds41-r13}
CACHE=/data/cache/moet
mkdir -p "$CACHE/dot-cache" "$CACHE/deep_gemm" /data/logs

[ -f /etc/ds41/env ] && set -a && . /etc/ds41/env && set +a

# optional overrides
[ -f /etc/ds41/moet.env ] && set -a && . /etc/ds41/moet.env && set +a

SPEC_TOKENS=${SPEC_TOKENS:-7}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.94}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
INDEXER_TP_SHARD_MIN_ROWS=${INDEXER_TP_SHARD_MIN_ROWS:-2048}
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-64}
INDEXER_KV_DTYPE=${INDEXER_KV_DTYPE:-mxfp4}
KV_RECORD=${KV_RECORD:-nvfp4}
SERVED_NAME=${SERVED_NAME:-DeepSeek-V4.1-Flash-1017-abliterated}

ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}")
[ "${SPEC_TOKENS}" = "0" ] && ARGS=()
case "$KV_RECORD" in
  nvfp4) KV_DTYPE=nvfp4_ds_mla ;;
  fp8_ds_mla) KV_DTYPE=fp8_ds_mla ;;
  *) KV_DTYPE=fp8 ;;
esac
ARGS+=(--attention-config "{\"indexer_kv_dtype\":\"${INDEXER_KV_DTYPE}\"}")

# Our patched 1017 encoding + placeholder fix (image/vLLM paths), when the image has them.
MOUNTS=()
for pair in "/data/models/encoding-1017-vllm.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41_encoding.py" \
            "/data/models/deepseek_v41-moet.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41.py"; do
  src=${pair%%:*}
  [ -f "$src" ] && MOUNTS+=(-v "$pair:ro")
done

docker rm -f "$NAME" >/dev/null 2>&1
for region in /dev/shm/vllm_offload_*.mmap; do
  [ -e "$region" ] || continue
  grep -qs -- "$region" /proc/[0-9]*/maps || rm -f "$region"
done

docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  -v "$M":/model:ro \
  -v "$CACHE/dot-cache":/root/.cache -v "$CACHE/deep_gemm":/root/.deep_gemm \
  ${MOUNTS[@]+"${MOUNTS[@]}"} \
  -e VLLM_API_KEY="$VLLM_API_KEY" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
  -e VLLM_MOET_B12X_MOE_FUSED=1 -e VLLM_MOET_SPARSE_MLA_LDG=1 -e VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 \
  -e INDEXER_TP_SHARD_MIN_ROWS="$INDEXER_TP_SHARD_MIN_ROWS" \
  -e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=16 \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" \
  --model /model --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --kv-cache-dtype "$KV_DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE" \
  --moe-backend b12x \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  "${ARGS[@]}" \
  --host 127.0.0.1 --port "$PORT"
docker logs -f "$NAME" > "/data/logs/moet-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
echo started
