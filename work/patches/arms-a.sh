#!/bin/bash
# VM side, root. Round-A arm queue: zero-code env/config A/Bs, each = one container start + one bench.
# usage: arms-a.sh [tag ...]   (no args = run all)
export PATH=/usr/local/bin:$PATH
RUNDIR=/var/lib/ds41/x
mkdir -p $RUNDIR
run(){ local TAG=$1 ENVF=$2 PLAN=$3 EXTRA=$4
  echo "=================== ARM $TAG $(date -u +%FT%TZ)"
  THINK=1 GATE=0 bash /usr/local/bin/exp.sh "$TAG" "$ENVF" "$PLAN" "$EXTRA"
}
want(){ [ $# -eq 0 ] && return 0; local t; for t in "$@"; do [ "$t" = "$ARM" ] && return 0; done; return 1; }
while IFS=';' read -r ARM ENVF PLAN EXTRA; do
  [ -z "$ARM" ] && continue
  case "$ARM" in \#*) continue;; esac
  want "$@" || continue
  run "$ARM" "$ENVF" "$PLAN" "$EXTRA"
done <<'LIST'
base5;/etc/ds41/base5.env;120:1x2,30:1x2;
twoshot;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=0
ag1;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8
std;/etc/ds41/base5.env;120:1x2,30:1x2;
dls085;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_DRAFT_LOGIT_SCALE=0.85
dls115;/etc/ds41/base5.env;120:1x2,30:1x2;-e VLLM_MOET_DRAFT_LOGIT_SCALE=1.15
LIST
echo "=================== ARMS DONE $(date -u +%FT%TZ)"
