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
`ds41-test-x`: the DSpark MTP drafter fine-tune, running since 21:30 UTC at 933 tok/s per stream x 8
streams (~7.5 k tok/s aggregate), so the 20 M-token dump is ~45 min, then cache ~30 min, train ~1 h,
export 15 min, eval 40 min; expected done ~00:50 UTC. Artifacts in `/data/mtp/`, eval results in
`/var/lib/ds41/x/mtp-block9-k5/`. Rollback: serve `M=/data/models/ds41-abl`; the checkpoint and the
image are untouched.

Three bugs were found and fixed on the way, all worth knowing:

1. The FP8/FP4 scale suffix was appended instead of replacing `.weight`, so no scale was ever found:
   FP4 experts loaded at half width (visible shape error) and FP8 dense weights loaded as raw bytes
   cast to bf16 (silent, correct shapes, garbage values). Fixed in `quantize.py` / `tools/load_mtp.py`;
   gate 0 now prints `value-exact on 1177 quantised tensors` and `1223 model tensors filled, 0 missing,
   0 unexpected`.
2. The container does not mount `/data`, so the hidden-state dump could not open its output file. The
   launcher gained a `MOUNT_DIRS` knob (space separated host dirs mounted read-write at the same path)
   and the dump stage passes `MOUNT_DIRS=$ROOT`.
3. `dump/gen_traces.py` read `args.context` but only defined `--context-file`, so the dump generated 0
   tokens and the cache/train stages ran on nothing. Fixed by setting `args.context = context`.

## Round 2 in flight: the DSpark MTP retrain

The 6.1 M-token dump finished (729 GB, 950 tok/s aggregate at 8 streams, 2 h), then three pipeline
bugs had to be fixed before the cache could be built:

1. The dump's float payload is **7 x 5120 int16 per row, not 4** - the hook records the model's full
   HC stream (4 copies) because with MTP on the model keeps `hidden_states` at [n, 4*5120] and only
   afterwards collapses it with the last layer's per-token `pre_mix` (model.py:1117, `hc_collapse_triton`
   at :1130). `pre_mix` is not in the dump, so the target distribution is recomputed from the **mean over
   the 4 HC copies**: exact whenever the pre-mix weights are equal, and RMSNorm makes it a rotation
   rather than a scale error. Row size is now auto-detected so this cannot recur.
2. The reader pinned the whole 729 GB in RAM as numpy views; rewritten as a bounded-memory two-pass
   reader (header walk + `pread`).
3. A truncated tail record (the dump was cut at 6.1 M tokens) now stops cleanly.

Cache built in 40 min (layout auto-detected as 3 aux + 4 HC blocks). **Train then failed on three
data-shape bugs**, all in the reassembler's sample layout rather than in the model: the dump's
contiguous runs are only ~6-9 tokens (each record holds the 6 draft-query rows of one verify step, and
the next step starts at the next accepted position), so `_slice` emitted **216,991 samples of 19 tokens
each instead of ~1,245 of 8,192**, which made the run 108,492 steps (30 h) and then OOM'd on an 18.7 GB
per-sample tensor and hit a zero-length input in `context_kv`. The sample layout is now fixed (916 samples of 6,016 tokens, 5.51 M tokens, gaps marked in
`loss_mask` with true positions in `positions.bin`), the OOM is gone, but the training step still fails
on a shape bug in the training model's context-KV path (`wkv` receives a 1-D 1,503-element `main_x`).
That is the third bug the real cache has exposed; each cycle costs a GPU-hour, so the next fix comes
with a one-step test on the real cache. The 729 GB dump and the 136 GB cache are both reusable, so the
resume is `STAGES="train export eval"` and about 1 h, not another dump. Eval lands in
`/var/lib/ds41/x/mtp-block9-k5/`; rollback is `M=/data/models/ds41-abl`.

Two known weaknesses of this first dump, for the next one: the rows are the *draft query* rows, so the
aux hidden states at the noise positions are computed from draft tokens rather than from the accepted
sequence, and the contiguous runs are only ~6-20 tokens. Dumping with `SPEC_TOKENS=0` would give clean,
contiguous single-token steps (half the tokens per hour, all of them usable).

## Where 600 stands

At the current 9.1 ms step, 600 tok/s needs accepted length 5.46; we are at 4.24. The retrain targets
native accepted length 3.42 -> ~4.0 (position-1 acceptance 0.81 -> 0.88), which with FLy would give
~5.0 and ~550 tok/s. The rest has to come from the step axis (9.1 ms, marginal verified row 0.61-0.69 ms,
~80 PCIe all-reduces per step, routed-expert bytes 85-100% of the row cost).

## Round 2, second stream: the step-axis env arms

Four zero-code arms are queued and running (`work/vm/manifest8.txt`): the fused mHC consumer off
(`P2P_AR_FUSED_MAX_ROWS=0` - the served fused path moves 420 KB per boundary against the two-shot's
180 KB and was never A/B'd at 6 rows), the one-shot push on (`P2P_AR_ONESHOT_MAX_ROWS=8` - dead code as
served), the P2P logits all-gather (`P2P_AG=1`) and the deterministic indexer top-k (`DET_TOPK=1`).
Together they are worth 1-3% of the step; the big step items (2-bit experts -1.5 ms, mHC push-from-
producer -0.12..-0.20 ms over 2-4 days) are for later rounds.

## Next round

1. Fix the retrain sample layout and run it (cache rebuild 40 min, train 155 steps, export, eval).
2. Read the step-arm results and fold the winners into the served config.
3. Confirm the top_p 1.0 + FLy interaction with a second arm (1 boot).
2. Read the retrain eval: accepted length at k=5 and k=9, per-position vector, GSM8K.
3. Decode-step profile (the earlier attempt captured a prefill iteration; use `profdecode.py` with a
   small prompt and a decode-time start).
4. Cheap step-axis arms still untested: `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8` (the one-shot path is dead
   code as served), `VLLM_MOET_MOE_PREQUANT=1`, `--engram-config use_thp`, draft-logit scale overlay.
5. Aggregate: measure C8 on the codebench workload; C1/C2/C3 aggregate is 375/488/601 today.

## Cost

One Flex-start g4-standard-192 (`ds41-test-x`, asia-southeast2-c) at $9.00/h, plus a 400 GB boot disk and
one 87 GB snapshot. Production is untouched and unchanged.

## Trial endpoint for Han (round 3)

The +22% config is served on the test VM for a hands-on trial, production untouched:

    https://<static-ip-with-dashes>.sslip.io/v1        (vLLM OpenAI API, Bearer RTX_API_KEY, TLS via Caddy)
    model id: DeepSeek-V4.1-Flash-1017-abliterated
    config:   k=5 DSpark, fp8 KV, --max-num-seqs 8 (8 parallel slots), 1M context,
              server-side thinking on + reasoning_effort max + temperature 1.0 + top_p 1.0,
              rejection_sample_method=fly with fly_window_size=1

DSH profile `~/.dsh/profiles/desktop/cordis.patch.yml` has a second provider `rtx-fast`
(baseURL https://<static-ip-with-dashes>.sslip.io/v1) whose model shows up as `ds41-fast (+22%, FLy)`. The
default model is unchanged, so this session keeps running on production.

To open 443 the VM needed the `ds41-serve` network tag (the firewall rule is `allow-ds41-https` with
that tag; the flex launcher's `dsv41-https` tag matches nothing). The Caddyfile for the test VM has the
API key inline because `{$DS41_KEY}` substitution did not take on that box.
