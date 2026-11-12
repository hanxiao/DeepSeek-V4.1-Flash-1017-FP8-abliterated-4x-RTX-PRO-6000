#!/bin/bash
# VM side, root. Round-B arm queue. Each arm = one container start + one bench.
export PATH=/usr/local/bin:$PATH
run(){ local TAG=$1 ENVF=$2 PLAN=$3 EXTRA=$4 OVL=$5
  echo "=================== ARM $TAG $(date -u +%FT%TZ) ovl=$OVL"
  OVERLAYS="$OVL" THINK=1 GATE=0 bash /usr/local/bin/exp.sh "$TAG" "$ENVF" "$PLAN" "$EXTRA"
}
SPECD=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
while IFS=';' read -r ARM ENVF PLAN EXTRA OVL; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$ENVF" "$PLAN" "$EXTRA" "$OVL"
done <<LIST
twoshot;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=0;
ag1;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8;
dls;/etc/ds41/base5.env;120:1x2,30:1x2;;/data/patches/dspark_speculator.py:$SPECD
LIST
echo "=================== ARMS-B DONE $(date -u +%FT%TZ)"
