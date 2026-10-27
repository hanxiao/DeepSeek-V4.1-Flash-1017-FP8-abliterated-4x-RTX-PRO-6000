#!/bin/bash
# Start DeepSeek-V4.1-Flash-1017-abliterated on the vLLM-Moet sm_120 image (TP4, 4x RTX PRO 6000).
# vLLM-Moet r13 (b522a06), the engine of the backup VM. Same contract as vm/ds41-start.sh (vLLM on
# 127.0.0.1:8000, Caddy in front, weights at /data/models/ds41-abl, API key from /etc/ds41/env, served
# name from /etc/ds41/deploy.env). Its image is in no registry; /etc/ds41/moet.env (vm/moet.env) pins the knobs.
#   PORT=8001 NAME=ds41-moet ds41-start-moet.sh     second instance for a blue/green check
# Files it needs:
#   /data/models/encoding-1017-vllm.py   checkpoint 1017 encoding.py + encoding-1017-vllm.patch (same as LiL)
#   /data/models/deepseek_v41-moet.py    nightly vllm/tokenizers/deepseek_v41.py + deepseek_v41-moet.patch
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
# Persistent engine settings (chosen SPEC_TOKENS etc.). Lines use ": "${VAR:=value}"" so an explicit
# environment override (e.g. SPEC_TOKENS=6 ds41-start-moet.sh) still wins.
[ -f /etc/ds41/moet.env ] && . /etc/ds41/moet.env
M=/data/models/ds41-abl
NAME=${NAME:-ds41}
PORT=${PORT:-8000}
MOET_IMAGE=${MOET_IMAGE:-sha256:a576e4e797d7a787570d80d3f901472529e7a510b6ddbf32f51249c15a9d3d73}   # vllm-moet-sm120:ds41-r13 (vLLM-Moet b522a06, a local build in no registry)
SPEC_TOKENS=${SPEC_TOKENS:-3}               # 3: best C8 (+6%), 5: best single stream (+21% prose, +35% code) but C8 -5%
REJ=${REJ:-block}                    # DSpark rejection sampling: block | standard
ADAPTIVE=${ADAPTIVE:-false}          # DSpark adaptive verification (LiL runs true + cost scale 0.5)
COST=${COST:-}                       # adaptive_verification_cost_scale (not accepted by this vLLM nightly)
SPEC_EXTRA=""; [ -n "$COST" ] && SPEC_EXTRA=",\"adaptive_verification_cost_scale\":$COST"
KV_DTYPE=${KV_DTYPE:-fp8_ds_mla}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.94}
INDEXER_KV_DTYPE=${INDEXER_KV_DTYPE:-fp8}
CG=${CG:-128}
EXTRA_ENV=(${EXTRA_ENV:-})          # e.g. EXTRA_ENV="-e VLLM_MOET_CANDIDATE_DET_TOPK=1"
# Default reasoning effort, passed as --default-chat-template-kwargs; moet.env sets max (100).
EFFORT=${EFFORT:-high}
SHARD=${SHARD:-2048}                 # indexer TP row sharding from this many rows (vllm#61647)
# indexer cache: fp8 is vLLM's default (no key); mxfp4 is lossy
AC="\"indexer_tp_shard_min_rows\":$SHARD"
[ "$INDEXER_KV_DTYPE" = mxfp4 ] && AC="$AC,\"indexer_kv_dtype\":\"mxfp4\""
ATTN=(--attention-config "{$AC}")
CACHE=/data/cache/moet
SP=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers
mkdir -p "$CACHE/dot-cache" "$CACHE/deep_gemm" /data/logs
for f in /data/models/encoding-1017-vllm.py /data/models/deepseek_v41-moet.py; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
# docker rm -f gives up when the GPU teardown takes >10 s, so stop first and retry the removal.
docker stop -t 60 "$NAME" >/dev/null 2>&1
while docker inspect "$NAME" >/dev/null 2>&1; do docker rm -f "$NAME" >/dev/null 2>&1 || sleep 3; done
docker run -d --name "$NAME" --restart unless-stopped \
  --network host --gpus all --ipc host --shm-size 32g \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --env-file /etc/ds41/env \
  -v $M:/model:ro \
  -v /data/models/encoding-1017-vllm.py:$SP/deepseek_v41_encoding.py:ro \
  -v /data/models/deepseek_v41-moet.py:$SP/deepseek_v41.py:ro \
  -v "$CACHE/dot-cache":/root/.cache -v "$CACHE/deep_gemm":/root/.deep_gemm \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e OMP_NUM_THREADS=1 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
  -e VLLM_MOET_B12X_MOE_FUSED=1 -e VLLM_MOET_SPARSE_MLA_LDG=1 -e VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 \
  -e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=16 ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  -e HF_HUB_OFFLINE=1 \
  "$MOET_IMAGE" \
  --model /model --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --default-chat-template-kwargs "{\"thinking\":true,\"reasoning_effort\":\"$EFFORT\"}" \
  --kv-cache-dtype "$KV_DTYPE" \
  ${ATTN[@]+"${ATTN[@]}"} \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len 1048576 \
  --max-num-seqs "${MNS:-8}" \
  --max-num-batched-tokens "${MNBT:-8192}" \
  --max-cudagraph-capture-size "$CG" \
  --moe-backend b12x \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-force-include-usage --enable-request-id-headers \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"${REJ}\",\"enable_adaptive_verification\":${ADAPTIVE}${SPEC_EXTRA}}" \
  --host 127.0.0.1 --port "$PORT"
docker logs -f "$NAME" > "/data/logs/$NAME-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
echo started
