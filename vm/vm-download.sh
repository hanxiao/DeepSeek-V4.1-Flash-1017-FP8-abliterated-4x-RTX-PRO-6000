#!/bin/bash
# Download the weights to /data/models/ds41-abl and pull the serving image in parallel.
# 510.3 GB took 261 s (1.95 GB/s) on g4-standard-192, anonymous, hf_xet high-performance mode.
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
DST=/data/models/ds41-abl
mkdir -p $DST /data/logs
( docker pull "$IMAGE" > /data/logs/pull.log 2>&1; echo "PULL_EXIT=$?" >> /data/logs/pull.log ) &
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=64
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
[ -s /root/.hf_token ] && HF_TOKEN=$(cat /root/.hf_token) && export HF_TOKEN
T0=$(date +%s)
echo "START $T0" > /data/logs/dl.log
/opt/hf/bin/hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir $DST --max-workers 48 >> /data/logs/dl.log 2>&1
RC=$?
T1=$(date +%s)
BYTES=$(du -sb $DST | cut -f1)
echo "END $T1 rc=$RC secs=$((T1-T0)) bytes=$BYTES MBps=$((BYTES/1000000/(T1-T0+1)))" >> /data/logs/dl.log
wait
echo ALL_DONE >> /data/logs/dl.log
