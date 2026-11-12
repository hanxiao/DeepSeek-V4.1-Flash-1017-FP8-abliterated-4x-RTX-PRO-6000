#!/bin/bash
# VM side, root: final round-68 numbers on the served (exact) config: quality gate + aggregate C1/C2/C4/C8.
set -uo pipefail
export PATH=/usr/local/bin:$PATH
R=/var/lib/ds41/x/final; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY CB_API_KEY=$KEY
PROF=0 /usr/local/bin/ds41-start-r21.sh
for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
echo "ready $(date -u +%T)"
idle(){ local n=0; while [ "$(curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')" != 0 ]; do n=$((n+1)); sleep 5; done; }
# 1. aggregate: per-stream rate at 1/2/4/8 streams, 30k context, one full task cycle each
idle; python3 /var/lib/ds41/ecc/codebench.py http://127.0.0.1:8000 $R/agg "30:1x1,30:2x1,30:4x1,30:8x1" 4096 > $R/agg.log 2>&1
echo "AGG done $(date -u +%T)"; grep -a "C. n=" $R/agg.log
# 2. quality gate: smoke (9) + mm (image order, needle) + openai order
idle; bash /opt/bench/gate.sh 8000 "r68" > $R/gate.log 2>&1
grep -a -cE "PASS" /var/lib/ds41/results/gate-r68.log 2>/dev/null || true
cp /var/lib/ds41/results/gate-r68.log $R/gate-full.log 2>/dev/null || true
# 3. GSM8K-100 thinking
idle; python3 /opt/bench/evalq.py gsm8k /opt/bench/gsm8k.jsonl 100 --port 8000 --out $R/gsm8k100.jsonl --think 1 --tag r68 > $R/gsm8k100.log 2>&1
grep -a -oE "accuracy[^\n]*|RESULT[^\n]*" $R/gsm8k100.log | tail -2
echo "FINAL DONE $(date -u +%T)"
