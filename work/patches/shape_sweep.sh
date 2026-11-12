#!/bin/bash
# VM side, root. Proposal-shaping sweep measured with the fixed-length AL probe (low variance).
# SPECS: semicolon-separated "label|scale|topk", scale/topk may be 5 comma-separated per-position values.
set -uo pipefail
SPECS="${SPECS:-base|1.0|0;s115|1.15|0;s130|1.30|0;s150|1.50|0;s200|2.00|0}"
REPS="${REPS:-2}"; PROMPTS="${PROMPTS:-math,code}"; MAXTOK="${MAXTOK:-4000}"
for rep in $(seq 1 "$REPS"); do
 for spec in ${SPECS//;/ }; do
  L=${spec%%|*}; rest=${spec#*|}; SC=${rest%%|*}; TK=${rest#*|}
  echo "$SC" > /dev/shm/ds41_draft_scale; echo "$TK" > /dev/shm/ds41_draft_topk; sleep 1
  python3 /usr/local/bin/alprobe.py "$MAXTOK" "$PROMPTS" "r${rep}-${L}" 2>&1 | sed "s/^/[$L] /"
 done
done
echo "SHAPE SWEEP DONE"
