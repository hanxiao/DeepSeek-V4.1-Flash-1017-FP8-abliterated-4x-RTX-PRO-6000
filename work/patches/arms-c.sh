#!/bin/bash
# VM side, root. Round-C/D arm queue: remaining zero-code config A/Bs + the proposal-shaping sweep.
export PATH=/usr/local/bin:$PATH
SPECD=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
run(){ local TAG=$1 ENVF=$2 PLAN=$3 EXTRA=$4 OVL=$5 PMODE=$6
  echo "=================== ARM $TAG $(date -u +%FT%TZ) ovl=$OVL prof=$PMODE extra=$EXTRA"
  OVERLAYS="$OVL" PROF="$PMODE" THINK=1 GATE=0 bash /usr/local/bin/exp.sh "$TAG" "$ENVF" "$PLAN" "$EXTRA"
}
while IFS=';' read -r ARM ENVF PLAN EXTRA OVL PMODE; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$ENVF" "$PLAN" "$EXTRA" "$OVL" "${PMODE:-0}"
  if [ "$ARM" = sweep3 ]; then
    CTX=120 REPS=2 SPECS="base|1.0|0;s085|0.85|0;s070|0.70|0;s115|1.15|0;k512|1.0|512;k2048|1.0|2048" bash /usr/local/bin/sweep2.sh
  fi
done <<LIST
cfg1;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8 -e VLLM_MOET_DET_TOPK=1 -e VLLM_MOET_MOE_PREQUANT=1;;
kv4;/etc/ds41/kv4.env;120:1x2,30:1x2;;;
sweep3;/etc/ds41/base5.env;120:1x1;;/data/patches/dspark_speculator3.py:$SPECD;0
pf2;/etc/ds41/base5.env;120:1x2;-e VLLM_MOET_B12X_MOE_FUSED_PF=2;;
pf8;/etc/ds41/base5.env;120:1x2;-e VLLM_MOET_B12X_MOE_FUSED_PF=8;;
spin0;/etc/ds41/base5.env;120:1x2;-e VLLM_MOET_B12X_MOE_FUSED_SPIN_NS=0;;
spin128;/etc/ds41/base5.env;120:1x2;-e VLLM_MOET_B12X_MOE_FUSED_SPIN_NS=128;;
prof;/etc/ds41/base5.env;120:1x1;;;2
LIST
echo "=================== ARMS-C DONE $(date -u +%FT%TZ)"
