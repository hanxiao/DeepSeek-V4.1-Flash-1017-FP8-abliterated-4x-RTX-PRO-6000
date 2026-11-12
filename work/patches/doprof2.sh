#!/bin/bash
# VM side, root: profiled decode with the DET_TOPK switch on, to see whether the indexer's
# persistent_topk (45.6 us x 8 per step in the base profile) is replaced by det_topk.
set -uo pipefail
R=/var/lib/ds41/x/prof3; mkdir -p $R/prof
EXTRA_ENV="-e VLLM_MOET_DET_TOPK=1" PROF=1 /usr/local/bin/ds41-start-r21.sh
for i in $(seq 1 200); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break
  sleep 5
done
echo "ready $(date -u +%T)"
python3 /usr/local/bin/alprobe.py 2000 math warm >/dev/null 2>&1
python3 /usr/local/bin/profdecode.py $R/prof 3000 3 120
cp -a /data/prof/. $R/prof/ 2>/dev/null; rm -f /data/prof/*.json.gz
python3 /usr/local/bin/traceparse.py $R/prof/dec_*.pt.trace.json.gz --top 45 > $R/top.txt 2>&1
echo "DOPROF2 DONE $(date -u +%T)"
