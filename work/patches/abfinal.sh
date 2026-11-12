#!/bin/bash
# VM side, root. Interleaved A/B/A/B of VLLM_MOET_DET_TOPK, identical probe sequence every arm
# (warm, then 4 fixed-length 120k probes); the arms alternate so machine drift cannot favour one.
set -uo pipefail
R=/var/lib/ds41/x/abfinal; mkdir -p $R
arm(){ local TAG=$1 E=$2
  echo "=========== $TAG $(date -u +%T) extra=$E"
  EXTRA_ENV="$E" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  python3 /usr/local/bin/alprobe.py 1500 math warm >/dev/null 2>&1
  for r in 1 2 3 4; do python3 /usr/local/bin/alprobe.py 3000 long120 "$TAG-r$r" 2>&1 | tee -a $R/$TAG.log; done
}
arm b1 ""
arm d1 "-e VLLM_MOET_DET_TOPK=1"
arm b2 ""
arm d2 "-e VLLM_MOET_DET_TOPK=1"
echo "ABFINAL DONE $(date -u +%T)"
