# DS41 speed work — round 1 status

Date: 2026-11-08. Goal: 600 tok/s single stream on the daily coding-agent workload (thinking at max
effort, 30k-120k context), same abliterated weights, no quality regression, no caching tricks, plus
max aggregate. Production (`ds41-prod`) was never touched.

## Result of this round

**Single-stream decode on the codebench workload went from 377.7 to 463.5 tok/s at 30k and from 358.3
to 460.1 at 120k — +22% mean, +18 to +28% per task — with GSM8K-100 unchanged at 96%.** That is
config-only: two fields in the spec JSON plus the sampling top_p.

| config | 30k C1 tok/s | 120k C1 tok/s | accepted length | GSM8K-100 |
| --- | ---: | ---: | ---: | ---: |
| served today (block rejection, top_p 0.95) | 377.7 | 358.3 | 3.45 / 3.31 | 96% (ref) |
| + FLy verification, window 1 | 426.8 | 402.9 | 3.92 / 3.76 | 97% |
| + top_p 1.0 | **463.5** | **460.1** | **4.24 / 4.24** | **96%** |

The change is in `vm/r21.env` and `vm/ds41-start-r21.sh` (new knobs TOPP, REJ, FLY_WIN, FLY_ENT,
DRAFT_METHOD, ADAPTIVE, PROF, OVERLAYS, MOUNT_DIRS, EXTRA_ARGS, all defaulting to the old behaviour):

    : "${TOPP:=1.0}"
    : "${REJ:=fly}"
    : "${FLY_WIN:=1}"

Deploying it needs one container restart (~6 min). Not done in this round because production serves
this agent.

## How it was found

- `rejection_sample_method: fly` is wired into the served vLLM but unused. Its window is the lever:
  `can_defer` needs `pos + window < k`, so at k=5 vLLM's default window `min(6,k-1)=4` lets only draft
  position 0 defer and measures -2%; window 1 lets positions 0-3 and measures +11..+14% across three
  entropy thresholds and six task/context cells.
- top_p 1.0 alone is neutral; on top of FLy it is worth another +8% (needs one confirmation run).
- Dead ends measured: k=7 (with and without FLy, -3 to -9%), FLy default window (-2%).

## What is running now

`cd /data/mtp && STAGES="dump cache train export eval" TOKENS=20000000 BLOCK=9 K=5 bash run.sh` on
`ds41-test-x`: the DSpark MTP drafter fine-tune. Gate 0 passed on the real checkpoint (quantisation
value-exact, 1223/1223 tensors, finite forward). Two real bugs in the pipeline were found and fixed by
that gate. Stages: dump ~2-3 h, cache 30 min, train 1 h, export 15 min, eval 40 min. Artifacts in
`/data/mtp/`, eval results in `/var/lib/ds41/x/mtp-block9-k5/`. Rollback: serve `M=/data/models/ds41-abl`;
the checkpoint and the image are untouched.

## Where 600 stands

At the current 9.1 ms step, 600 tok/s needs accepted length 5.46; we are at 4.24. The retrain targets
native accepted length 3.42 -> ~4.0 (position-1 acceptance 0.81 -> 0.88), which with FLy would give
~5.0 and ~550 tok/s. The rest has to come from the step axis (9.1 ms, marginal verified row 0.61-0.69 ms,
~80 PCIe all-reduces per step, routed-expert bytes 85-100% of the row cost).

## Next round

1. Confirm the top_p 1.0 + FLy interaction with a second arm (1 boot).
2. Read the retrain eval: accepted length at k=5 and k=9, per-position vector, GSM8K.
3. Decode-step profile (the earlier attempt captured a prefill iteration; use `profdecode.py` with a
   small prompt and a decode-time start).
4. Cheap step-axis arms still untested: `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8` (the one-shot path is dead
   code as served), `VLLM_MOET_MOE_PREQUANT=1`, `--engram-config use_thp`, draft-logit scale overlay.
5. Aggregate: measure C8 on the codebench workload; C1/C2/C3 aggregate is 375/488/601 today.

## Cost

One Flex-start g4-standard-192 (`ds41-test-x`, asia-southeast2-c) at $9.00/h, plus a 400 GB boot disk and
one 87 GB snapshot. Production is untouched and unchanged.
