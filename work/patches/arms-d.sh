#!/bin/bash
# VM side, root. Round-D: attribute the ag1 win (one-shot push vs P2P all-gather) and confirm with 6 runs.
export PATH=/usr/local/bin:$PATH
run(){ local TAG=$1 ENVF=$2 PLAN=$3 EXTRA=$4
  echo "=================== ARM $TAG $(date -u +%FT%TZ) extra=$EXTRA"
  THINK=0 GATE=0 bash /usr/local/bin/exp.sh "$TAG" "$ENVF" "$PLAN" "$EXTRA"
}
while IFS=';' read -r ARM ENVF PLAN EXTRA; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$ENVF" "$PLAN" "$EXTRA"
done <<LIST
oneshot;/etc/ds41/base5.env;120:1x3;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8
agonly;/etc/ds41/base5.env;120:1x3;-e VLLM_MOET_P2P_AG=1
oneshot_nofuse;/etc/ds41/base5.env;120:1x3;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8 -e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=0
oneshot64;/etc/ds41/base5.env;120:1x3;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=64
LIST
echo "=================== ARMS-D DONE $(date -u +%FT%TZ)"
