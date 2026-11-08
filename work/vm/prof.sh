#!/bin/bash
# VM side, root. Profile exactly one decode iteration with vLLM's torch profiler.
# Requires the container started with PROF=1 (--profiler-config ... --torch_profiler_dir /profiles,
# host dir /data/prof mounted there).
# usage: prof.sh TAG [PLAN] [DELAY_ITERS]
set -uo pipefail
TAG=${1:?tag}; PLAN=${2:-"30:1x1"}; DELAY=${3:-10}
R=/var/lib/ds41/x/$TAG; mkdir -p "$R/prof"
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env)
H=(-H "Authorization: Bearer $KEY" -H 'content-type: application/json')
curl -s "${H[@]}" -X POST http://127.0.0.1:8000/start_profile \
  -d "{\"profile_prefix\":\"$TAG\",\"delay_iterations\":$DELAY,\"max_iterations\":1}" \
  -o "$R/prof/start.txt" -w 'start=%{http_code}\n'
sleep 1
CB_API_KEY=$KEY python3 /var/lib/ds41/ecc/codebench.py http://127.0.0.1:8000 "$R/cb" "$PLAN" 4096 > "$R/cb.log" 2>&1
echo "codebench rc=$? :: $(grep -a 'C1 n=' "$R/cb.log")"
sleep 2
curl -s "${H[@]}" -X POST http://127.0.0.1:8000/stop_profile -o "$R/prof/stop.txt" -w 'stop=%{http_code}\n'
sleep 2
cp -a /data/prof/. "$R/prof/" 2>/dev/null
rm -f /data/prof/*.json.gz /data/prof/*.json 2>/dev/null
ls -la "$R/prof"
