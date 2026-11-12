#!/bin/bash
# VM side, root: round-68 final numbers on base5 + VLLM_MOET_DET_TOPK=1.
set -uo pipefail
export PATH=/usr/local/bin:$PATH
R=/var/lib/ds41/x/final2; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY CB_API_KEY=$KEY
EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=0 /usr/local/bin/ds41-start-r21.sh
for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
echo "ready $(date -u +%T)"
idle(){ local n=0; while [ "$(curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')" != 0 ]; do n=$((n+1)); sleep 5; done; }
python3 /usr/local/bin/alprobe.py 1000 math warm >/dev/null 2>&1
echo "--- 30k probe"; for i in 1 2 3; do python3 /usr/local/bin/alprobe.py 3000 long "s30-r$i"; done
echo "--- 120k probe"; for i in 1 2 3; do python3 /usr/local/bin/alprobe.py 3000 long120 "s120-r$i"; done
echo "--- codebench 120k x3"; idle
python3 /var/lib/ds41/ecc/codebench.py http://127.0.0.1:8000 $R/cb120 "120:1x3" 4096 > $R/cb120.log 2>&1
grep -a "C1 n=" $R/cb120.log
echo "--- aggregate 30k C1/C2/C4/C8"; idle
python3 /var/lib/ds41/ecc/codebench.py http://127.0.0.1:8000 $R/agg "30:1x1,30:2x1,30:4x1,30:8x1" 4096 > $R/agg.log 2>&1
grep -a "C. n=" $R/agg.log
echo "--- gate"; idle
bash /opt/bench/gate.sh 8000 "r68d" > $R/gate.log 2>&1
cp /var/lib/ds41/results/gate-r68d.log $R/gate-full.log 2>/dev/null || true
echo "gate PASS=$(grep -ac PASS $R/gate-full.log 2>/dev/null) FAIL=$(grep -ac FAIL $R/gate-full.log 2>/dev/null)"
echo "--- gsm8k100"; idle
python3 /opt/bench/evalq.py gsm8k /opt/bench/gsm8k.jsonl 100 --port 8000 --out $R/gsm8k100.jsonl --think 1 --tag r68d > $R/gsm8k100.log 2>&1
grep -a -oE "accuracy[^\n]*|RESULT[^\n]*" $R/gsm8k100.log | tail -2
echo "FINAL2 DONE $(date -u +%T)"
