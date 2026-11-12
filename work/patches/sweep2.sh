#!/bin/bash
# VM side, root. Sweep proposal-shaping knobs on a RUNNING server (patch v3 overlay).
# usage: sweep2.sh  -> reads specs from $SPECS (semicolon-separated "label|scale|topk")
set -uo pipefail
D=/var/lib/ds41/ecc; R=/var/lib/ds41/x/sweep2; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY CB_API_KEY=$KEY
idle(){ local n=0; while [ "$(curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')" != 0 ]; do n=$((n+1)); [ $((n%20)) = 1 ] && echo busy; sleep 5; done; }
CTX="${CTX:-120}"; REPS="${REPS:-2}"
SPECS="${SPECS:-base|1.0|0;s070|0.70|0;s085|0.85|0;s115|1.15|0;s130|1.30|0;k512|1.0|512;k2048|1.0|2048}"
for rep in $(seq 1 "$REPS"); do
 for spec in ${SPECS//;/ }; do
  L=${spec%%|*}; rest=${spec#*|}; SC=${rest%%|*}; TK=${rest#*|}
  echo "$SC" > /dev/shm/ds41_draft_scale; echo "$TK" > /dev/shm/ds41_draft_topk; sleep 1
  idle
  python3 $D/codebench.py http://127.0.0.1:8000 $R/${L}-r${rep} "${CTX}:1x1" > $R/${L}-r${rep}.log 2>&1
  echo "rep$rep $L scale=$SC topk=$TK :: $(grep -a 'C1 n=' $R/${L}-r${rep}.log)"
 done
done
echo "SWEEP2 DONE"
