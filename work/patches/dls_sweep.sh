#!/bin/bash
# VM side, root. Sweep the draft-side logit scale on a RUNNING server (needs the overlay that
# reads /dev/shm/ds41_draft_scale). Metric: accept (= 1 + accepted/drafts) and decode tok/s.
set -uo pipefail
D=/var/lib/ds41/ecc; R=/var/lib/ds41/x/dls; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY CB_API_KEY=$KEY
idle(){ local n=0; while [ "$(curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')" != 0 ]; do n=$((n+1)); [ $((n%20)) = 1 ] && echo "busy"; sleep 5; done; }
SCALES="${SCALES:-1.0 0.85 0.7 0.55 1.15 1.35}"
CTX="${CTX:-120}"; REPS="${REPS:-2}"
for rep in $(seq 1 "$REPS"); do
 for s in $SCALES; do
  echo "$s" > /dev/shm/ds41_draft_scale; sleep 1
  idle
  python3 $D/codebench.py http://127.0.0.1:8000 $R/s${s}-r${rep} "${CTX}:1x1" > $R/s${s}-r${rep}.log 2>&1
  echo "rep$rep scale=$s $(grep -a 'C1 n=' $R/s${s}-r${rep}.log)"
 done
done
echo "SWEEP DONE"
