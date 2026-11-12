#!/bin/bash
# VM side, root. Round-I: close out the config space (the 4 knobs left) on the round-68 config,
# then check the clocks under decode load.
set -uo pipefail
R=/var/lib/ds41/x/i; mkdir -p $R
run(){ local TAG=$1 EXTRA=$2
  echo "=========== $TAG $(date -u +%T) extra=$EXTRA"
  EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1 $EXTRA" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /usr/local/bin/alprobe.py 1500 math warm >/dev/null 2>&1
  ( sleep 6; nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader >> $R/$TAG.clocks ) &
  for r in 1 2 3; do python3 /usr/local/bin/alprobe.py 3000 long120 "$TAG-r$r" 2>&1 | tee -a $R/$TAG.log; done
  wait
}
while IFS=';' read -r ARM EXTRA; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$EXTRA"
done <<LIST
ictl;
inooverlap;-e VLLM_MOET_MHC_OVERLAP=0
igemvv2;-e VLLM_MOET_GEMV_IMPL=v2
ibf16v2;-e VLLM_MOET_BF16_CHAIN=v2
imhcproj;-e VLLM_MOET_MHC_PROJ=fused
LIST
echo "ARMS-I DONE $(date -u +%T)"
