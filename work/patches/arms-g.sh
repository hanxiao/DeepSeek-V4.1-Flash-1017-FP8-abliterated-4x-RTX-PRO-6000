#!/bin/bash
# VM side, root. Round-G: the b12x fused-MoE tuning knobs (never measured) + two opt-in sites.
# Metric: step from the verify-step count on the fixed 120k prompt, 2 reps (reproduces to +-0.1%).
set -uo pipefail
R=/var/lib/ds41/x/g; mkdir -p $R
run(){ local TAG=$1 EXTRA=$2
  echo "=========== $TAG $(date -u +%T) extra=$EXTRA"
  EXTRA_ENV="$EXTRA -e VLLM_MOET_DET_TOPK=1" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /usr/local/bin/alprobe.py 800 math warm >/dev/null 2>&1
  for r in 1 2; do python3 /usr/local/bin/alprobe.py 3000 long120 "$TAG-r$r" 2>&1 | tee -a $R/$TAG.log; done
}
while IFS=';' read -r ARM EXTRA; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$EXTRA"
done <<LIST
ctl;
pf2;-e VLLM_MOET_B12X_MOE_FUSED_PF=2
pf8;-e VLLM_MOET_B12X_MOE_FUSED_PF=8
spin0;-e VLLM_MOET_B12X_MOE_FUSED_SPIN_NS=0
spin128;-e VLLM_MOET_B12X_MOE_FUSED_SPIN_NS=128
sites4;-e VLLM_MOET_BF16_GEMM_SITES=indexer,wk,router,compressor
mhcproj;-e VLLM_MOET_MHC_PROJ=fused
LIST
echo "ARMS-G DONE $(date -u +%T)"
