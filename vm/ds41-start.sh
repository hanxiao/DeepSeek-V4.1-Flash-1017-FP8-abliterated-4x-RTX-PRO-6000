#!/bin/bash
# Start the DeepSeek-V4.1-Flash-1017-abliterated vLLM server (LiL image, TP4, 4x RTX PRO 6000).
# vLLM binds 127.0.0.1:8000 only; Caddy terminates TLS on :443 and checks the Bearer key.
# /etc/ds41/env holds VLLM_API_KEY (root, 0600). /etc/ds41/deploy.env holds IMAGE and the model name.
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
M=/data/models/ds41-abl
mkdir -p /data/cache/root-cache /data/logs
docker rm -f ds41 >/dev/null 2>&1
docker run -d --name ds41 --restart unless-stopped \
  --network host --gpus all --shm-size 128g --ipc host \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit nofile=1048576 \
  --env-file /etc/ds41/env \
  -v $M:/model:ro \
  -v /data/models/encoding-1017-vllm.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41_encoding.py:ro \
  -v /data/models/deepseek_v41-vllm.py:/opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41.py:ro \
  -v /data/cache/root-cache:/root/.cache -v /data/cache/lil-cache:/cache \
  -e CHECKPOINT=original -e MODEL=/model -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  -e HOST=127.0.0.1 -e PORT=8000 \
  -e TP=4 -e OMP_NUM_THREADS=1 -e MAX_NUM_BATCHED_TOKENS=8192 -e GPU_MEMORY_UTILIZATION=0.95 \
  -e DSPARK_TOKENS=3 -e DSPARK_ADAPTIVE_VERIFICATION=true -e DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE=0.5 \
  -e ENGRAM_TABLE_MEMORY=disk -e ENGRAM_DISK_RESIDENT_SCALES=true \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=96KB -e VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=96KB \
  -e VLLM_PCIE_DMA_MIN_BYTES=6MB -e B12X_MHC_PDL=1 -e B12X_PCIE_ONESHOT_PDL=1 -e VLLM_MXFP8_LM_HEAD=1 \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" serve-ds41-flash.sh -- --max-cudagraph-capture-size 128
docker logs -f ds41 > "/data/logs/ds41-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
echo started
