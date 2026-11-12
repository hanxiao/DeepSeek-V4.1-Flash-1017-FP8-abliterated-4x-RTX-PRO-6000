#!/bin/bash
# VM side, root: does the image-order-reordered-tool-results check fail on the exact config WITHOUT
# VLLM_MOET_DET_TOPK too? (i.e. is it a DET_TOPK regression or pre-existing in the exact config)
set -uo pipefail
export KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env)
for arm in base det; do
  if [ "$arm" = det ]; then E="-e VLLM_MOET_DET_TOPK=1"; else E=""; fi
  echo "=========== arm $arm $(date -u +%T)"
  EXTRA_ENV="$E" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  echo "ready $(date -u +%T)"
  cd /opt/bench && for i in 1 2 3; do python3 mmtests.py --port 8000 --tag ord-$arm-$i 2>&1 | grep -a "image-order-reordered"; done
done
echo "ORDERCHK DONE $(date -u +%T)"
