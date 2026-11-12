#!/bin/bash
# VM side, root: A/B the DET_TOPK switch on the 120k step, 3 probes each, restart in between.
set -uo pipefail
probe(){ for i in 1 2 3; do python3 /usr/local/bin/alprobe.py 3000 long120 "$1-r$i" 2>&1; done; }
for arm in base det; do
  if [ "$arm" = det ]; then E="-e VLLM_MOET_DET_TOPK=1"; else E=""; fi
  echo "=========== arm $arm $(date -u +%T) extra=$E"
  EXTRA_ENV="$E" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /usr/local/bin/alprobe.py 1000 math warm >/dev/null 2>&1
  probe $arm
done
echo "CMP120 DONE $(date -u +%T)"
