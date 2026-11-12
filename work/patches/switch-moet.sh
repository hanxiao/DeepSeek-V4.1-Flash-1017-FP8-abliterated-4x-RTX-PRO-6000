#!/bin/bash
# Switch the serving VM from the LiL vLLM container to the vLLM-Moet sm_120 image, with a fallback:
# if the new server is not serving within 25 min, go back to the LiL container.
set -uo pipefail
. /etc/ds41/env
LOG=/data/logs/switch-moet.log
exec >> "$LOG" 2>&1
echo "=== switch $(date -u +%FT%TZ)"

ready() { [ "$(curl -s -o /dev/null -w '%{http_code}' -m 60 http://127.0.0.1:8000/v1/messages \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017-abliterated","max_tokens":16,"messages":[{"role":"user","content":"say OK"}]}')" = 200 ]; }

wait_ready() {  # $1 = seconds
  local end=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    ready && return 0
    sleep 15
  done
  return 1
}

echo "stopping LiL container"
docker stop -t 60 ds41 >/dev/null 2>&1; docker rm -f ds41 >/dev/null 2>&1

echo "starting Moet container"
bash /usr/local/bin/run-moet.sh

if wait_ready 1500; then
  echo "MOET READY $(date -u +%FT%TZ)"
  echo ok > /etc/ds41/engine
  exit 0
fi

echo "Moet did not become ready; reverting to the LiL container"
docker logs ds41 2>&1 | tail -40
docker rm -f ds41 >/dev/null 2>&1
bash /usr/local/bin/ds41-start.sh
if wait_ready 900; then
  echo "REVERTED AND READY $(date -u +%FT%TZ)"
  echo lil > /etc/ds41/engine
  exit 0
fi
echo "FALLBACK ALSO NOT READY $(date -u +%FT%TZ)"
exit 1
