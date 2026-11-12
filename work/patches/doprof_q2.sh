#!/bin/bash
set -uo pipefail
R=/var/lib/ds41/x/prof4; mkdir -p $R/prof
OVL="/data/patches/b12x_moe_fused_sm120.cu:/opt/vllm-moet/dsv41_sm120/b12x_moe_fused/b12x_moe_fused_sm120.cu /data/patches/b12x_moe_fused_sm120.py:/opt/vllm-moet/dsv41_sm120/b12x_moe_fused/b12x_moe_fused_sm120.py /data/patches/w4a8_compact_micro_q2.py:/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py /data/patches/w4a8_compact_projection_q2.py:/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_projection.py /data/patches/w4a8_phase1_q2.py:/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_phase1.py"
M=/data/models/ds41-abl-q2b MOUNT_DIRS=/data/models OVERLAYS="$OVL" EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=1 /usr/local/bin/ds41-start-r21.sh
for i in $(seq 1 200); do
  [ "$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break
  sleep 5
done
echo "ready $(date -u +%T)"
python3 /usr/local/bin/alprobe.py 2000 math warm >/dev/null 2>&1
python3 /usr/local/bin/profdecode.py $R/prof 3000 3 120
cp -a /data/prof/. $R/prof/ 2>/dev/null; rm -f /data/prof/*.json.gz
python3 /usr/local/bin/traceparse.py $R/prof/dec_*.pt.trace.json.gz --top 12 > $R/top.txt 2>&1
echo "DONE $(date -u +%T)"
