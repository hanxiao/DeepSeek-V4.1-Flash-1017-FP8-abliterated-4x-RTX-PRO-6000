#!/bin/bash
# Fine-tune the DSpark MTP head of DeepSeek-V4.1-Flash-1017-abliterated and measure accepted length.
#
# Run on the test VM (ds41x-*), as root, with this directory copied to the VM (e.g. /data/mtp).
# Every stage is idempotent and can be run on its own:
#
#   bash run.sh                          # gate0 dump validate cache train export eval
#   STAGES=smoke bash run.sh             # 2 optimizer steps on the real cache (~3 min): do this first
#   bash run.sh gate0                    # cheap correctness gates only (~10 min)
#   STAGES="dump cache" TOKENS=40000000 bash run.sh
#
# All Python runs inside the serving image (it already has torch 2.14, transformers 5.20.2,
# safetensors), so the host needs no Python environment at all.
#
# Time on g4-standard-192 (4x RTX PRO 6000): gate0 ~10 min, dump ~5 h for 40 M tokens, cache ~30 min,
# train ~1-2 h, export ~15 min, eval ~40 min (two container starts at ~6 min each).
set -euo pipefail

# ---------------------------------------------------------------- knobs
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-$(docker image inspect --format '{{.RepoTags}}' vllm-moet-sm120:dsv41-nightly-df3bd094-r21 2>/dev/null | tr -d '[]"' | cut -d' ' -f1)}
IMAGE=${IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r21}
MODEL=${MODEL:-/data/models/ds41-abl}                 # read-only checkpoint
OVERLAY=${OVERLAY:-/data/models/ds41-abl-mtp-v2}      # served after training (built by stage export)
ROOT=${ROOT:-/data/mtp}                               # scratch
RAW=$ROOT/raw                                         # engine dump
CACHE=$ROOT/cache                                     # training cache
RUNS=$ROOT/runs
EXPORT=$ROOT/export
ENGINE=$ROOT/engine                                   # patched speculator for the container
TOKENS=${TOKENS:-40000000}                            # in-domain tokens to generate
REQUESTS=${REQUESTS:-$((TOKENS / 20000))}             # ~20 k generated tokens per request
CONCURRENCY=${CONCURRENCY:-8}
MAXLEN=${MAXLEN:-8192}                                # training sample length
TOPK=${TOPK:-256}                                     # stored target support per position
BLOCK=${BLOCK:-9}                                     # trained draft block (== k we intend to serve)
K=${K:-9}                                             # num_speculative_tokens for the eval
MAX_STEPS=${MAX_STEPS:-}                              # 2 = smoke-test the real cache, empty = full run
START=${START:-/usr/local/bin/ds41-start-r21.sh}
STAGES=${STAGES:-${*:-gate0 dump validate cache train export eval}}
mkdir -p "$ROOT" "$RAW" "$ENGINE" "$RUNS" "$EXPORT"
log() { echo "$(date -u +%FT%TZ) [mtp] $*" | tee -a "$ROOT/run.log"; }

# run a python entry point inside the serving image, with /data and this directory mounted
py() {
  docker run --rm -i --gpus all --ipc host --network host --shm-size 32g \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v /data:/data -v "$HERE":/work -w /work -e PYTHONPATH=/work \
    --entrypoint python3 "$IMAGE" "$@"
}
torchrun_in_image() {
  docker run --rm --gpus all --ipc host --network host --shm-size 32g \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v /data:/data -v "$HERE":/work -w /work -e PYTHONPATH=/work \
    -e NCCL_P2P_DISABLE=0 -e NCCL_P2P_LEVEL=SYS -e OMP_NUM_THREADS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint torchrun "$IMAGE" --nproc_per_node 4 --standalone "$@"
}
wait_ready() {
  local n=0
  while [ $n -lt "${1:-900}" ]; do
    curl -sf -m 10 -o /dev/null http://127.0.0.1:8000/metrics && return 0
    sleep 5; n=$((n + 5))
  done
  return 1
}
wait_idle() {
  local n=0
  while [ "$(curl -s -m 10 http://127.0.0.1:8000/metrics | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')" != 0 ]; do
    n=$((n + 1)); [ $((n % 24)) = 1 ] && log "engine busy, waiting"; sleep 5
  done
}
stop_engine() { docker stop -t 60 ds41 >/dev/null 2>&1 || true; docker rm -f ds41 >/dev/null 2>&1 || true; }
api_key() { sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env 2>/dev/null || true; }

# the launcher hardcodes the model path; make it overridable once so the overlay can be served
preflight() {
  grep -q '^M=${M:-' "$START" || sed -i 's|^M=/data/models/ds41-abl|M=${M:-/data/models/ds41-abl}|' "$START"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "image $IMAGE not found"; exit 1; }
  local free; free=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
  log "preflight: image $IMAGE, /data free ${free} GiB, stages [$STAGES]"
  [ "$free" -gt 400 ] || log "WARNING: less than 400 GiB free on /data; the raw dump is ~40 kB/token"
}

# ---------------------------------------------------------------- stage 0: gates
stage_gate0() {
  log "stage gate0: quantisation conventions, then checkpoint -> training model"
  py quantize.py selftest
  py tools/gate0.py --model-dir "$MODEL" --block-size "$BLOCK" --device cuda
  log "gate0 done"
}

# ---------------------------------------------------------------- stage 1: dump
stage_dump() {
  log "stage dump: $TOKENS tokens with the patched speculator recording hidden states"
  py dump/make_patch.py --src /work/vendor/vllm/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py \
    --out "$ENGINE/speculator.py"
  if [ ! -s "$ROOT/context.txt" ]; then
    find /usr/local/lib/python3.12/dist-packages/vllm -name '*.py' -size -80k 2>/dev/null | head -400 \
      | xargs cat 2>/dev/null | head -c 400000 > "$ROOT/context.txt" || true
    [ -s "$ROOT/context.txt" ] || echo "no repository context available" > "$ROOT/context.txt"
  fi
  stop_engine
  rm -f "$RAW"/dump-*.bin
  log "starting the engine with VLLM_MTP_DUMP (container start ~6 min)"
  # MNS=32: the dump run wants aggregate tokens/s, not single-stream latency
  M="$MODEL" SPEC_TOKENS=5 MNS=${MNS:-32} \
    OVERLAYS="$ENGINE/speculator.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py" \
    MOUNT_DIRS="$ROOT" \
    EXTRA_ENV="-e VLLM_MTP_DUMP=$RAW/dump-0.bin" bash "$START"
  wait_ready 1800 || { log "engine did not come up"; tail -40 "$(ls -t /data/logs/ds41-r21-*.log | head -1)"; return 1; }
  log "engine ready; generating traces (this is the long stage)"
  # PROMPTS: real prompts from Han's own agent logs (dump/real_prompts.py). When present, the
  # hidden states are dumped on the distribution of his actual work instead of synthetic prompts.
  PROMPT=${PROMPTS:-$ROOT/real_prompts.jsonl}
  PROMPT_ARGS=()
  [ -s "$PROMPT" ] && PROMPT_ARGS=(--prompts-jsonl "$PROMPT") && log "generating from $PROMPT"
  py dump/gen_traces.py --url http://127.0.0.1:8000 --key "$(api_key)" \
    --requests "$REQUESTS" --concurrency "$CONCURRENCY" --max-tokens 20000 \
    --context-file "$ROOT/context.txt" "${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}" 2>&1 | tee -a "$ROOT/gen.log"
  wait_idle
  stop_engine
  du -sh "$RAW" | tee -a "$ROOT/run.log"
  log "dump done"
}

# ---------------------------------------------------------------- stage 2: cache
stage_validate() {
  log "stage validate: does the dump's target distribution predict the next token?"
  py dump/reassemble.py validate --raw $(ls "$RAW"/dump-*.bin) --model-dir "$MODEL" \
    --out "$ROOT/cache" --device cuda --limit 40000 2>&1 | tee -a "$ROOT/validate.log"
  log "validate done (a score below 0.25 means the dump's tokens or the norm are wrong: re-dump)"
}

stage_cache() {
  log "stage cache: reassemble sequences, recompute the target top-$TOPK distribution"
  if [ -s "$CACHE/meta.json" ]; then log "cache exists, skipping"; return 0; fi
  py dump/reassemble.py build --raw $(ls "$RAW"/dump-*.bin) --model-dir "$MODEL" \
    --out "$CACHE" --max-length "$MAXLEN" --topk "$TOPK" --device cuda \
    --collapse "${COLLAPSE:-best}" 2>&1 | tee -a "$ROOT/cache.log"
  py -c "import json;m=json.load(open('$CACHE/meta.json'));print(f\"cache: {m['num_samples']} samples, {m['total_tokens']/1e6:.1f} M tokens\")"
  log "cache done"
}

# ---------------------------------------------------------------- stage 3: train
stage_train() {
  log "stage train: block $BLOCK, TVP beta 0->1 over 80% of the run${MAX_STEPS:+, MAX_STEPS=$MAX_STEPS}"
  stop_engine
  local out=${TRAIN_OUT:-$RUNS/block$BLOCK}
  torchrun_in_image train_mtp.py --config config/dspark_v41_block9.py --model-dir "$MODEL" \
    --cache "$CACHE" --out "$out" --block-size "$BLOCK" \
    ${MAX_STEPS:+--max-steps "$MAX_STEPS"} 2>&1 | tee -a "$ROOT/train.log"
  log "train done: $(ls -d "$out"/step_* | sort | tail -1)"
}

# ---------------------------------------------------------------- stage 4: export
stage_export() {
  log "stage export: requantise to the native format and build the overlay checkpoint"
  local step; step=$(ls -d "$RUNS"/block"$BLOCK"/step_* | sort | tail -1)
  py tools/export_mtp.py --state "$step/model.safetensors" --model-dir "$MODEL" \
    --out "$EXPORT/mtp-block$BLOCK.safetensors"
  rm -rf "$OVERLAY"
  py quantize.py build --model "$MODEL" --weights "$EXPORT/mtp-block$BLOCK.safetensors" --out "$OVERLAY"
  du -sh "$OVERLAY"
  log "export done: $OVERLAY"
}

# ---------------------------------------------------------------- stage 5: eval
stage_eval() {
  log "stage eval: serve $OVERLAY at k=$K, measure accepted length on the codebench workload"
  stop_engine
  local tag="mtp-block$BLOCK-k$K"
  # exp.sh starts the container itself; M/SPEC_TOKENS are inherited by ds41-start-r21.sh
  M="$OVERLAY" SPEC_TOKENS="$K" THINK=1 GATE=1 \
    bash /usr/local/bin/exp.sh "$tag" /etc/ds41/r21.env "30:1x3,120:1x2" "" || true
  local r=/var/lib/ds41/x/$tag
  log "results in $r"
  grep -a -h 'C1 n=\|C2 n=\|C3 n=' "$r"/cb.log 2>/dev/null | tee -a "$ROOT/run.log" || true
  py tools/report_think.py "$r" 2>&1 | tee -a "$ROOT/run.log" || true
  log "eval done. rollback: serve with M=$MODEL; nothing in the image or /model changed"
}

preflight
for s in $STAGES; do
  case "$s" in
    gate0)  stage_gate0 ;;
    dump)   stage_dump ;;
    validate) stage_validate ;;
    cache)  stage_cache ;;
    train)  stage_train ;;
    export) stage_export ;;
    smoke)  MAX_STEPS=${MAX_STEPS:-2} TRAIN_OUT=${TRAIN_OUT:-$RUNS/smoke} stage_train ;;   # 3-min real-cache check
    eval)   stage_eval ;;
    *) log "unknown stage '$s'"; exit 2 ;;
  esac
done
log "all requested stages done"
