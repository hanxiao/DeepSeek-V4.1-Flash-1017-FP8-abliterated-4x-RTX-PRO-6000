#!/bin/bash
# VM side, root: per-position acceptance on the round-68 config (thinkbench), the c vector every AL
# decision depends on.
set -uo pipefail
R=/var/lib/ds41/x/cvec; mkdir -p $R
EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=0 /usr/local/bin/ds41-start-r21.sh
for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
echo "ready $(date -u +%T)"
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY CB_API_KEY=$KEY
python3 /var/lib/ds41/ecc/thinkbench.py $R/think.jsonl 1.0 all 2 6000 > $R/think.log 2>&1
tail -20 $R/think.log
echo "CVEC DONE $(date -u +%T)"
