#!/bin/bash
# VM side, root. usage: ecc.sh bench TAG [cbplan] | ecc.sh gate TAG | ecc.sh state TAG
# Results under /var/lib/ds41/ecc/TAG
M=$1; TAG=$2; CBPLAN=${3:-"30 60 120"}
D=/var/lib/ds41/ecc; R=$D/$TAG; mkdir -p $R
L=$D/orch.log
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY; export CB_API_KEY=$KEY
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
running(){ curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} /^vllm:num_requests_waiting\{/{s+=$NF} END{print s+0}'; }
wait_idle(){ local n=0; while [ "$(running)" != 0 ]; do n=$((n+1)); [ $((n%20)) = 1 ] && log "busy, waiting"; sleep 5; done; }
state(){
  nvidia-smi --query-gpu=index,ecc.mode.current,ecc.mode.pending,memory.total,memory.used,clocks.max.memory,clocks.mem --format=csv > $R/nvsmi.csv
  nvidia-smi -q -d ECC > $R/ecc_q.txt 2>&1
  nvidia-smi -q -d MEMORY > $R/mem_q.txt 2>&1
  dmesg 2>/dev/null | grep -i -E 'xid|nvrm.*error' > $R/xid.txt; true
  F=$(ls -t /data/logs/ds41-r21-*.log 2>/dev/null | head -1); echo "$F" > $R/serverlog.txt
  grep -a -m1 'GPU KV cache size' "$F" >> $R/serverlog.txt
  grep -a -m4 -E 'Available KV cache memory|Model loading took' "$F" >> $R/serverlog.txt
  cat $R/nvsmi.csv $R/serverlog.txt | tee -a $L
}
case $M in
state) state ;;
bench)
  log "BENCH $TAG start"
  state
  cd $D
  python3 $D/thinkbench.py $R/think.jsonl 1.0 all 2 6000 > $R/think.log 2>&1
  log "THINKBENCH $TAG done rc=$?"
  for s in $CBPLAN; do for k in 1 2; do
    wait_idle
    python3 $D/codebench.py http://127.0.0.1:8000 $R/cb-$s-$k "$s:1x3" > $R/cb-$s-$k.log 2>&1
    log "CB $TAG ctx$s set$k: $(grep -a 'C1 n=' $R/cb-$s-$k.log)"
  done; done
  state
  log "BENCH $TAG done"
  ;;
gate)
  log "GATE $TAG start"
  wait_idle
  bash /opt/bench/gate.sh 8000 ecc-$TAG
  cp /var/lib/ds41/results/gate-ecc-$TAG.log $R/gate.log
  wait_idle
  python3 $D/gate2.py http://127.0.0.1:8000 /opt/bench/test-image.png > $R/gate2.log 2>&1
  wait_idle
  python3 /opt/bench/evalq.py gsm8k /opt/bench/gsm8k.jsonl 100 --port 8000 --out $R/gsm8k100.jsonl --think 1 --tag ecc-$TAG > $R/gsm8k100.log 2>&1
  grep -a -E 'PASS|FAIL|RESULT' $R/gate.log $R/gate2.log $R/gsm8k100.log | tee -a $L
  log "GATE $TAG done"
  ;;
cb1)
  wait_idle
  python3 $D/codebench.py http://127.0.0.1:8000 $R/cb-30-1 "30:1x3" > $R/cb-30-1.log 2>&1
  log "CB $TAG ctx30: $(grep -a 'C1 n=' $R/cb-30-1.log)"
  ;;
esac
