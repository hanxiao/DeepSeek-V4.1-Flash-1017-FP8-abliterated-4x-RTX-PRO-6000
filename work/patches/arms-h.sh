#!/bin/bash
# VM side, root. Round-H: the remaining unmeasured zero-code knobs, on the round-68 config.
set -uo pipefail
R=/var/lib/ds41/x/h; mkdir -p $R
run(){ local TAG=$1 EXTRA=$2
  echo "=========== $TAG $(date -u +%T) extra=$EXTRA"
  EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1 $EXTRA" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /usr/local/bin/alprobe.py 1500 math warm >/dev/null 2>&1
  for r in 1 2 3; do python3 /usr/local/bin/alprobe.py 3000 long120 "$TAG-r$r" 2>&1 | tee -a $R/$TAG.log; done
}
while IFS=';' read -r ARM EXTRA; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$EXTRA"
done <<LIST
hctl;
mhcproj;-e VLLM_MOET_MHC_PROJ=fused
gemvv2;-e VLLM_MOET_GEMV_IMPL=v2
mhcnoverlap;-e VLLM_MOET_MHC_OVERLAP=0
bf16v2;-e VLLM_MOET_BF16_CHAIN=v2
LIST
echo "ARMS-H DONE $(date -u +%T)"
