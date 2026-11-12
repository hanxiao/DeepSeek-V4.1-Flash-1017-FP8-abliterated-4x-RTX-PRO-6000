#!/bin/bash
# VM side, root: start the served config WITH the torch profiler endpoint, then profile decode only.
set -uo pipefail
R=/var/lib/ds41/x/prof2; mkdir -p $R/prof
PROF=1 /usr/local/bin/ds41-start-r21.sh
for i in $(seq 1 200); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break
  sleep 5
done
echo "ready $(date -u +%T)"
python3 /usr/local/bin/alprobe.py 2000 math warm >/dev/null 2>&1
python3 /usr/local/bin/profdecode.py $R/prof 3000 3 120
cp -a /data/prof/. $R/prof/ 2>/dev/null; rm -f /data/prof/*.json.gz
ls -la $R/prof/ | head -5
echo "DOPROF DONE $(date -u +%T)"
