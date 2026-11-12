#!/bin/bash
# VM side, root. Quality A/B of the 2-bit requantised experts against the exact baseline.
# Same harness, same 200 GSM8K questions, same config otherwise.
set -uo pipefail
export PATH=/usr/local/bin:$PATH
R=/var/lib/ds41/x/q2; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY
arm(){ local TAG=$1 M=$2
  echo "=========== $TAG $(date -u +%T) model=$M"
  M="$M" MOUNT_DIRS=/data/models EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /usr/local/bin/alprobe.py 1500 math warm >/dev/null 2>&1
  for r in 1 2 3; do python3 /usr/local/bin/alprobe.py 3000 long120 "$TAG-r$r" 2>&1 | tee -a $R/$TAG.log; done
  python3 /opt/bench/evalq.py gsm8k /opt/bench/gsm8k.jsonl 200 --port 8000 --out $R/$TAG-gsm200.jsonl --think 1 --tag $TAG > $R/$TAG-gsm200.log 2>&1
  grep -a -oE "RESULT[^\n]*" $R/$TAG-gsm200.log | tail -1
  cd /opt/bench && for i in 1 2; do python3 mmtests.py --port 8000 --tag $TAG-n$i 2>&1 | grep -a -E "needle|image-order" | head -3; done
}
arm base /data/models/ds41-abl
arm q2 /data/models/ds41-abl-q2
echo "Q2EVAL DONE $(date -u +%T)"
