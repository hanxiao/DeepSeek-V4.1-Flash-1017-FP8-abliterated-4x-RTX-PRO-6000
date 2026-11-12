#!/bin/bash
# VM side, root. Full GSM8K (1319 questions) on the FP4 baseline and on the 2-bit requant, same harness.
set -uo pipefail
export PATH=/usr/local/bin:$PATH
R=/var/lib/ds41/x/q2full; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY
arm(){ local TAG=$1 M=$2
  echo "=========== $TAG $(date -u +%T) model=$M"
  M="$M" MOUNT_DIRS=/data/models EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /opt/bench/evalq.py gsm8k /opt/bench/gsm8k.jsonl 1319 --port 8000 --out $R/$TAG-gsm.jsonl --think 1 --tag $TAG > $R/$TAG-gsm.log 2>&1
  grep -a "RESULT" $R/$TAG-gsm.log | tail -1
}
arm full-base /data/models/ds41-abl
arm full-q2 /data/models/ds41-abl-q2
echo "Q2FULL DONE $(date -u +%T)"
