#!/bin/bash
# VM side, root. Round-F: clean step-time comparison. Each arm = restart, then the fixed-length probe
# (step from the verify-step count, independent of accepted length) plus one codebench cell.
export PATH=/usr/local/bin:$PATH
R=/var/lib/ds41/x; mkdir -p $R
run(){ local TAG=$1 ENVF=$2 EXTRA=$3
  echo "=================== ARM $TAG $(date -u +%FT%TZ) extra=$EXTRA"
  THINK=0 GATE=0 bash /usr/local/bin/exp.sh "$TAG" "$ENVF" "30:1x1" "$EXTRA" >/dev/null 2>&1
  local F=$(ls -t /data/logs/ds41-r21-*.log | head -1)
  mkdir -p $R/$TAG
  for rep in 1 2 3; do
    python3 /usr/local/bin/alprobe.py 3000 math,code,long "$TAG-r$rep" 2>&1 | tee -a $R/$TAG/probe.log
  done
  grep -a -E "GPU KV cache size" "$F" | tail -1 >> $R/$TAG/probe.log
}
while IFS=';' read -r ARM ENVF EXTRA; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  run "$ARM" "$ENVF" "$EXTRA"
done <<LIST
pbase;/etc/ds41/base5.env;
poneshot;/etc/ds41/base5.env;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8
pbundle;/etc/ds41/base5.env;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8 -e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_MOE_PREQUANT=1
pkv4;/etc/ds41/kv4.env;-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8 -e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_MOE_PREQUANT=1
LIST
echo "=================== ARMS-F DONE $(date -u +%FT%TZ)"
