#!/bin/bash
# VM side, root. Round-E: attribute + confirm the config wins with 3 runs (a full task cycle) per cell.
export PATH=/usr/local/bin:$PATH
run(){ local TAG=$1 ENVF=$2 PLAN=$3 EXTRA=$4 PMODE=$5
  echo "=================== ARM $TAG $(date -u +%FT%TZ) extra=$EXTRA prof=${PMODE:-0}"
  PROF="${PMODE:-0}" THINK=0 GATE=0 bash /usr/local/bin/exp.sh "$TAG" "$ENVF" "$PLAN" "$EXTRA"
}
while IFS=';' read -r ARM ENVF PLAN EXTRA PMODE; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$ENVF" "$PLAN" "$EXTRA" "${PMODE:-0}"
done <<LIST
base6;/etc/ds41/base5.env;120:1x3;;
oneshot;/etc/ds41/base5.env;120:1x3;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8;
bundle;/etc/ds41/base5.env;120:1x3;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8 -e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_DET_TOPK=1 -e VLLM_MOET_MOE_PREQUANT=1;
prof;/etc/ds41/base5.env;120:1x1;;;2
LIST
echo "=================== ARMS-E DONE $(date -u +%FT%TZ)"
