#!/bin/bash
# VM side, root. A/B the PUSH_FLAGS_AT_END push kernel (one system fence + 3 releases per boundary
# instead of 3 fences + 3 releases), same probe sequence as the abfinal controls.
set -uo pipefail
R=/var/lib/ds41/x/push; mkdir -p $R
PD=/opt/vllm-moet/dsv41_sm120/p2p_ar
run(){ local TAG=$1 OVL=$2
  echo "=========== $TAG $(date -u +%T) ovl=$OVL"
  OVERLAYS="$OVL" EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  local F=$(ls -t /data/logs/ds41-r21-*.log | head -1)
  grep -a -m2 -E "P2P all-reduce|one-shot push" "$F" | tee -a $R/$TAG.log
  python3 /usr/local/bin/alprobe.py 1500 math warm >/dev/null 2>&1
  for r in 1 2 3 4; do python3 /usr/local/bin/alprobe.py 3000 long120 "$TAG-r$r" 2>&1 | tee -a $R/$TAG.log; done
}
run pushnew "/data/patches/p2p_allreduce_sm120.cu:$PD/p2p_allreduce_sm120.cu /data/patches/p2p_allreduce_sm120.py:$PD/p2p_allreduce_sm120.py"
echo "PUSHTEST DONE $(date -u +%T)"
