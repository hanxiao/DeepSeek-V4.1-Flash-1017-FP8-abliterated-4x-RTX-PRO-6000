#!/bin/bash
# VM side, root. One experiment = one container start + one bench battery.
# usage: exp.sh TAG [ENVFILE] [PLAN] [EXTRA]
#   TAG      results land in /var/lib/ds41/x/TAG
#   ENVFILE  r21.env-style file (default: /etc/ds41/r21.env, i.e. the served config)
#   PLAN     codebench plan (default "30:1x3,120:1x2")
#   EXTRA    extra DOCKER ARGS for the container, e.g. "-e VLLM_MOET_DET_TOPK=0" or
#            "-e VLLM_TORCH_PROFILER_DIR=/tmp/prof" (note the -e, they are appended to docker run)
# Env knobs for exp.sh itself: THINK=1 also runs thinkbench (per-position acceptance),
#   GATE=1 also runs the smoke gate, gate2 and GSM8K-100.
#   PROF=1 runs prof.sh (profiles an iteration), PROF=2 runs profdecode.py (profiles decode only),
#   PROFPLAN overrides the profiled plan.
# Writes TAG/meta.txt with the env, KV pool, model load time and GPU clocks.
set -uo pipefail
TAG=${1:?tag}; ENVFILE=${2:-/etc/ds41/r21.env}; PLAN=${3:-"30:1x3,120:1x2"}; EXTRA=${4:-}
R=/var/lib/ds41/x/$TAG; mkdir -p "$R"
L=/var/lib/ds41/x/orch.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY CB_API_KEY=$KEY
ready(){ curl -s -o /dev/null -w '%{http_code}' -m 20 http://127.0.0.1:8000/metrics; }
wait_idle(){ local n=0; while [ "$(curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')" != 0 ]; do n=$((n+1)); [ $((n%20)) = 1 ] && log "busy, waiting"; sleep 5; done; }

log "EXP $TAG start env=$ENVFILE plan=$PLAN extra=$EXTRA think=${THINK:-0} prof=${PROF:-0}"
[ "$ENVFILE" = /etc/ds41/r21.env ] || cp "$ENVFILE" /etc/ds41/r21.env
{ echo "TAG=$TAG"; echo "ENVFILE=$ENVFILE"; echo "PLAN=$PLAN"; echo "EXTRA=$EXTRA";
  echo "--- r21.env"; cat /etc/ds41/r21.env; } > "$R/meta.txt"

STAMP=$(date -u +%FT%TZ)
EXTRA_ENV="$EXTRA" /usr/local/bin/ds41-start-r21.sh
T0=$(date +%s)
for _ in $(seq 1 180); do
  [ "$(ready)" = 200 ] && break
  sleep 5
done
T1=$(date +%s)
log "EXP $TAG ready=$([ "$(ready)" = 200 ] && echo yes || echo no) after $((T1-T0))s"
F=$(ls -t /data/logs/ds41-r21-*.log 2>/dev/null | head -1); echo "serverlog=$F since=$STAMP" >> "$R/meta.txt"
grep -a -m2 -E 'GPU KV cache size|Maximum concurrency' "$F" >> "$R/meta.txt" 2>/dev/null
grep -a -m3 -E 'Model loading took|Loading weights took|Graph capturing finished' "$F" >> "$R/meta.txt" 2>/dev/null
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,memory.used,temperature.gpu --format=csv >> "$R/meta.txt"
[ "$(ready)" = 200 ] || { log "EXP $TAG NOT READY"; tail -30 "$F" >> "$R/meta.txt"; exit 1; }

PMODE=${PROF:-0}; unset PROF
if [ "$PMODE" = 1 ]; then
  wait_idle
  bash /usr/local/bin/prof.sh "$TAG" "${PROFPLAN:-30:1x1}" >> "$R/prof.log" 2>&1
  log "EXP $TAG prof rc=$? :: $(ls "$R/prof" 2>/dev/null | tr '\n' ' ')"
elif [ "$PMODE" = 2 ]; then
  wait_idle
  python3 /usr/local/bin/profdecode.py "$R/prof" 3000 3 30 >> "$R/prof.log" 2>&1
  cp -a /data/prof/. "$R/prof/" 2>/dev/null; rm -f /data/prof/*.json.gz
  log "EXP $TAG profdecode rc=$? :: $(ls "$R/prof" 2>/dev/null | tr '\n' ' ')"
fi
wait_idle
python3 /var/lib/ds41/ecc/codebench.py http://127.0.0.1:8000 "$R/cb" "$PLAN" 4096 > "$R/cb.log" 2>&1
log "EXP $TAG codebench rc=$? :: $(grep -a 'C1 n=\|C2 n=\|C3 n=' "$R/cb.log" | tr '\n' ' | ')"
if [ "${THINK:-0}" = 1 ]; then
  wait_idle
  python3 /var/lib/ds41/ecc/thinkbench.py "$R/think.jsonl" 1.0 all 2 6000 > "$R/think.log" 2>&1
  log "EXP $TAG thinkbench rc=$?"
fi
if [ "${GATE:-0}" = 1 ]; then
  wait_idle
  bash /opt/bench/gate.sh 8000 "x-$TAG" > "$R/gate.log" 2>&1
  python3 /opt/bench/gate2.py http://127.0.0.1:8000 /opt/bench/test-image.png > "$R/gate2.log" 2>&1
  wait_idle
  KEY=$KEY python3 /opt/bench/evalq.py gsm8k /opt/bench/gsm8k.jsonl 100 --port 8000 --out "$R/gsm8k100.jsonl" --think 1 --tag "x-$TAG" > "$R/gsm8k100.log" 2>&1
  log "EXP $TAG gate :: $(grep -a -c PASS "$R/gate.log" 2>/dev/null) pass, gsm8k $(grep -a -oE 'RESULT[^\n]*|accuracy[^\n]*' "$R/gsm8k100.log" 2>/dev/null | tail -1)"
fi
nvidia-smi --query-gpu=index,clocks.sm,power.draw,memory.used --format=csv >> "$R/meta.txt"
log "EXP $TAG done"
