#!/bin/bash
# Start DeepSeek-V4.1-Flash-1017-abliterated on vLLM-Moet r21 (TP4, 4x RTX PRO 6000): the production engine.
# vLLM on 127.0.0.1:8000 behind Caddy, API key from /etc/ds41/env, served name from /etc/ds41/deploy.env.
# The defaults below are the vLLM-Moet README recipe; /etc/ds41/r21.env (vm/r21.env) pins the served values.
# Knobs (env): SPEC_TOKENS (7) KV_DTYPE (nvfp4_ds_mla | fp8_ds_mla) INDEXER_KV_DTYPE (mxfp4 | fp8)
#   MNS (8) MNBT (4096) CG (64) MAXLEN (1048576) OFFLOAD_GIB (64, 0=off) EFFORT (max) AR_ROWS (8)
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
[ -f /etc/ds41/r21.env ] && . /etc/ds41/r21.env
M=/data/models/ds41-abl
NAME=${NAME:-ds41}
PORT=${PORT:-8000}
R21_IMAGE=${R21_IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r21}
SPEC_TOKENS=${SPEC_TOKENS:-7}
KV_DTYPE=${KV_DTYPE:-nvfp4_ds_mla}
INDEXER_KV_DTYPE=${INDEXER_KV_DTYPE:-mxfp4}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.94}
MNS=${MNS:-8}
MNBT=${MNBT:-4096}
CG=${CG:-64}
MAXLEN=${MAXLEN:-1048576}
OFFLOAD_GIB=${OFFLOAD_GIB:-64}
EFFORT=${EFFORT:-max}
AR_ROWS=${AR_ROWS:-8}
SHARD=${SHARD:-2048}
REJ=${REJ:-block}                     # block | standard | fly | synthetic
DRAFT_METHOD=${DRAFT_METHOD:-probabilistic}
ADAPTIVE=${ADAPTIVE:-false}
FLY_WIN=${FLY_WIN:-}                  # fly_window_size, empty = vLLM default min(6, k-1)
FLY_ENT=${FLY_ENT:-}                  # fly_entropy_threshold
FLY_TOPK=${FLY_TOPK:-}                # fly_entropy_top_k
TOPP=${TOPP:-0.95}                    # sampling top_p of the override generation config
PROF=${PROF:-0}                       # 1 = enable the torch profiler endpoint (POST /start_profile)
PROF_DIR=${PROF_DIR:-/profiles}
TOK_OVERLAY=${TOK_OVERLAY:-/data/models/deepseek_v41-r21.py}
ENC_OVERLAY=${ENC_OVERLAY:-/data/models/encoding-1017-vllm.py}
EXTRA_ENV=(${EXTRA_ENV:-})
EXTRA_ARGS=(${EXTRA_ARGS:-})   # extra vLLM CLI args, e.g. --async-scheduling or --engram-config '{"use_thp":true}'
AC="\"indexer_tp_shard_min_rows\":$SHARD"
[ "$INDEXER_KV_DTYPE" = mxfp4 ] && AC="$AC,\"indexer_kv_dtype\":\"mxfp4\""
CACHE=/data/cache/r21
SP=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers
mkdir -p "$CACHE/dot-cache" "$CACHE/deep_gemm" /data/logs /data/kvcache
MOUNTS=(-v $M:/model:ro -v "$CACHE/dot-cache":/root/.cache -v "$CACHE/deep_gemm":/root/.deep_gemm)
[ -n "$TOK_OVERLAY" ] && MOUNTS+=(-v "$TOK_OVERLAY":$SP/deepseek_v41.py:ro)
[ -n "$ENC_OVERLAY" ] && MOUNTS+=(-v "$ENC_OVERLAY":$SP/deepseek_v41_encoding.py:ro)
PROF_ARGS=()
if [ "$PROF" = 1 ]; then
  mkdir -p /data/prof; MOUNTS+=(-v /data/prof:"$PROF_DIR")
  PROF_ARGS=(--profiler-config.profiler torch --profiler-config.torch_profiler_dir "$PROF_DIR" --profiler-config.torch_profiler_with_stack false)
fi
# OVERLAYS: space separated host:container pairs, each mounted read-only over an image file.
# Used for engine patches (e.g. a patched speculator.py) without rebuilding the image.
for pair in ${OVERLAYS:-}; do MOUNTS+=(-v "${pair%%:*}":"${pair#*:}":ro); done
# MOUNT_DIRS: space separated host directories to mount read-write at the same path in the container.
for d in ${MOUNT_DIRS:-}; do MOUNTS+=(-v "$d":"$d"); done
OFF=()
if [ "$OFFLOAD_GIB" != 0 ]; then
  FS=/data/kvcache/$KV_DTYPE-$INDEXER_KV_DTYPE
  if [ "$(docker image inspect --format '{{index .Config.Labels "com.vllm-moet.v41-kv-records"}}' "$R21_IMAGE" 2>/dev/null)" = interleaved ]; then FS=$FS/v41-interleaved; fi
  mkdir -p "$FS"; MOUNTS+=(-v /data/kvcache:/data/kvcache)
  for region in /dev/shm/vllm_offload_*.mmap; do [ -e "$region" ] || continue; grep -qs -- "$region" /proc/[0-9]*/maps || rm -f "$region"; done
  OFF=(--kv-offloading-size "$OFFLOAD_GIB" --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$FS\",\"n_read_threads\":16,\"n_write_threads\":16,\"locality\":\"LOCAL\"}],\"offload_prompt_only\":false}}")
fi
SPEC=()
if [ "$SPEC_TOKENS" != 0 ]; then
  SJ="{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"${DRAFT_METHOD}\",\"rejection_sample_method\":\"${REJ}\",\"enable_adaptive_verification\":${ADAPTIVE},\"moe_backend\":\"b12x\""
  [ -n "$FLY_WIN" ] && SJ="$SJ,\"fly_window_size\":${FLY_WIN}"
  [ -n "$FLY_ENT" ] && SJ="$SJ,\"fly_entropy_threshold\":${FLY_ENT}"
  [ -n "$FLY_TOPK" ] && SJ="$SJ,\"fly_entropy_top_k\":${FLY_TOPK}"
  SPEC=(--speculative-config "$SJ}")
fi
docker stop -t 60 "$NAME" >/dev/null 2>&1
while docker inspect "$NAME" >/dev/null 2>&1; do docker rm -f "$NAME" >/dev/null 2>&1 || sleep 3; done
docker run -d --name "$NAME" --restart unless-stopped \
  --network host --gpus all --ipc host --shm-size 32g \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --env-file /etc/ds41/env \
  "${MOUNTS[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e OMP_NUM_THREADS=1 \
  -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS \
  -e VLLM_MOET_B12X_MOE_FUSED=1 -e VLLM_MOET_SPARSE_MLA_LDG=1 -e VLLM_MOET_TARGET_LM_HEAD_MXFP8=1 \
  -e VLLM_MOET_P2P_AR=1 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=$AR_ROWS -e VLLM_MOET_P2P_AG=0 \
  -e VLLM_MOET_DFLASH_PREP_GROUPS=1 -e VLLM_MOET_CANDIDATE_DET_TOPK=1 -e VLLM_MOET_SWA_DECODE_BUILD=1 \
  -e VLLM_MOET_MOE_TOPKSUM_ADD=1 -e VLLM_MOET_SWA_DRAFT_BUILD=1 ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  -e HF_HUB_OFFLINE=1 \
  "$R21_IMAGE" \
  --model /model --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --override-generation-config "{\"temperature\":1.0,\"top_p\":$TOPP}" \
  --default-chat-template-kwargs "{\"thinking\":true,\"reasoning_effort\":\"$EFFORT\"}" \
  --kv-cache-dtype "$KV_DTYPE" \
  --attention-config "{$AC}" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens "$MNBT" \
  --max-cudagraph-capture-size "$CG" \
  --moe-backend b12x \
  "${OFF[@]}" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --enable-force-include-usage --enable-request-id-headers \
  "${SPEC[@]}" \
  ${PROF_ARGS[@]+"${PROF_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --host 127.0.0.1 --port "$PORT"
docker logs -f "$NAME" > "/data/logs/$NAME-r21-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
echo started
