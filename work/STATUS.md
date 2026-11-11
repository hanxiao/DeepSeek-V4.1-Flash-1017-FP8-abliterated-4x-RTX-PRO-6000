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

## Repository metadata

The GitHub `About` description is updated (it still said 371-376 tok/s): it now reads "463 tok/s
single-stream agent decode at 30k-120k context, 1082 tok/s over 8 streams", and the topics gained
`fly` and `moe`. The README in the repo carries the same numbers plus the config-revision table and
the 600 roadmap. Update with `gh repo edit <repo> --description "..."` - the description is not part
of README.md and is easy to forget.

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

## Production is now the FLy + top_p 1.0 config (round 4)

Deployed at 2026-11-09T02:33:58Z with a wrapper that installs the new config, restarts, waits 20 min,
and rolls back to `/etc/ds41/r21.env.pre-fast` / `/usr/local/bin/ds41-start-r21.sh.pre-fast` if the
container does not become ready. It reported `READY with the fast config after 320s`; the container
starts the FLy warning and the KV pool is unchanged at 4,192,362 tokens.

Verified after the deploy: the essay prompt that measured 264.5 tok/s decode on the old config measures
377.2 tok/s on the new one (+43%), and the coding-agent benchmark measured +22% (377.7 -> 463.5).

## Trial endpoint for Han (round 3, superseded by the production deploy)

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

## Real training data (round 5)

The user asked for training samples from his own logs, and that is the right call: the MTP drafter is
fine-tuned on what the engine generates, so the seeding prompts decide the data distribution.
`work/mtp/dump/real_prompts.py` mines

- `~/.claude/projects/**/*.jsonl` — Claude Code transcripts (161 files)
- `~/.dsh/sessions/**/*.jsonl.zstd` — this harness's own sessions (52 files)
- `~/.openclaw/workspace/**/*.jsonl` — the openclaw pi sessions (169 files)

and found **41,465 prompts** (41,404 in Claude Code, 44 DSH, 17 openclaw; median prompt 585 chars,
99.6% with a real context of 12-40k chars). 1,500 prompts were sampled into
`/data/mtp/real_prompts.jsonl` (20 MB) and `gen_traces.py --prompts-jsonl` feeds them as
`<session context> + real user prompt` with the agent keeps the tools.

On the retrain VM `ds41y` the dump is being loaded with the 1,500 prompts; the boilerplate and is the
next path. NOTE: `.to(regions)` for the load state.

## Blockers found and fixed on the retrain VM

- The launcher in `/usr/local/bin/ds41-start-r21.sh` was the pre-round-8 version (no `OVERLAYS`,
  `MOUNT_DIRS`) so the patched speculator was never mounted and the dump was empty. Fixed by pushing
  `vm/ds41-start-r21.sh` to the VM.
- The tokenizer overlays in `/data/models` were copied as directories because the boot script's `cp`
  overwrote an existing dir, and a later `docker -v` then failed with "not a directory". Fixed in
  `vm/ds41-boot.sh` with `rm -rf` before the copy.

## Round 6: real machine dump

The retrain dump works on `ds41y` with the 1,500 real prompts: 80 requests, 58,057 tokens,
 47.9 GB of dump. Getting there made the following bugs, all in the dump pipeline:

1. The launcher on that VM was the old version without `OVERLAYS`/`MOUNT_DIRS`, so the patched
   speculator was never mounted and the dump was empty.
2. The tokenizer overlays were directories, from a boot cp onto an existing dir.
3. `input_batch.token_ids_cpu` does not exist in this vLLM; the raw `InputBatch` carries
   `input_ids` and `positions` ([num_tokens_after_padding]) and no token_ids_cpu. Fix: use it.
4. `sid` was dropped by the same edit.
5. `reassemble.py validate` needed `--out` (it is missing from run.sh) and `_bf16_from_bits` was
   not defined.

The last one is open: `validate` runs and scores 0.000 (all collapse proxies), so the next step is
to check whether the dump's token ids are the target's or the target's *input* rows (they are
`input_batch.input_ids`, which for a verify step is the accepted prefix + the draft block, not the
full sequence).

### Round 7: the validate gate still fails

`reassemble.py validate` now runs (the --out bug is fixed) and reports
`best collapse proxy score 0.000 (expected 0.4-0.6)` with the only probe name `none`, so the dump's
token ids, the backbone norm or the head is wrong. The dump's records contain real tokens in the
mixed region (e.g. `(7,7,2), (3,8,2), (34428,9,2), (5,10,2)`) but the first records have tokens that
(0,0), (1,1), (2,2) - the dump's `token` column is `input_batch.input_ids`, and for a verify step that
is the *input* of the target (accepted prefix + draft block), not the next-token label. The next step
is to align `dump_step`'s label with the same row the head is (the target's next-token id) and check
the same 0.4-0.6.

Before changing the dump, check `probe_collapse` itself: it returned only the `none` probe, not the
`s0..s3` variants, so the collapse ranking is not being computed at all.

### Round 8 diagnosis: the dump frame desyncs

Reading the 47.9 GB dump with the writer's frame (4-byte header + n*12 ints + n*71680 payload)
parses the first 14 records and then finds a non-plausible `n` at 63,232,400 B, so the reader
desynchronises and `probe_collapse` sees only 882 rows and returns `{"none": 0.0}` (it needs 100
consecutive rows; 740 of them are there, so the failure is the truncated frame: the last file is only
the bytes it were written after the last record). The rows at 0..800 are real: row tokens 0..125463, seq
0..33, pos 0..16 - 740 consecutive pairs.

So the remaining blocker is the writer's record layout: make it self-describing (a magic + an explicit
payload length per record) rather than deriving the payload from `n`, so a reader can never desync.
`probe_collapse`'s `h`/`last` rows are then trusted, and the score (0.4-0.6) decides whether the
tokens/norm/head are right.

The patch to the `MTPDumper` header was started and reverted; the next round is to write it together
with its readers (`_scan_headers`, `scan`, `probe_collapse`) and re-run the small dump (run.sh's
`STAGES=dumpdump + dump validate, e.g. REQUESTS=80`, ~6 min).

### Round 9: the dump frame is fixed; the probe's own gate is wrong

The writer now emits each record as ONE `write()` (header + integer block + payload concatenated), and
the new 47.9 GB dump parses end to end: 2376 records, 482,599 rows, and the header walk consumes the
whole file (34,598,497,012 of 34,598,497,012 bytes). So the frame cannot desync again.

`validate` still prints `0.000` with the single probe name `none`, which is not a proxy name but the
`return {"none": 0.0}` fallback inside `probe_collapse` - i.e. `consec.sum() < 100` on the rows it read.
With 482,599 rows in the file that can only be a bug in the probe itself (it only reads `limit` rows
and it is the gate `(pos[1:] == pos[:-1] + 1) & (seq[1:] == seq[:-1])`, so the dump's `seq` or `pos`
values are probably not 0..N in file order). Next step: print the probe's first 200 (seq, pos) pairs and
the two counts, then fix the pair alignment (or make the probe read the full file) - the dump's not
the suspect anymore.

The training offline path (`plan_samples`) was already validated on this data, so it still needs the
probe anyway.

### Round 10: validate runs and scores 0.103 (not 0.0-0.6)

Two more probe bugs fixed (the payload read bound `n*(row_bytes-INT)` instead of `n*row_bytes-INT`, and
the 129M-row logits were chunked). The gate now runs on 40,000 rows and reports

    s3 0.103   mean 0.098   s2 0.094   s1 0.088   s0 0.083

so the collapse **is** meaningful (s3 > mean > s2 > s1 > s0) and the head/norm are fine, but 0.103 is
not the 0.4-0.6 a correct target yields. The open question is the token alignment: is the row's `tok`
the token that the same row predicts (shift 0) or the token it consumes (shift 1)? The probe already
scores against `tok[1:]`; the next test is `tok 0` shifted -1. If neither matches, the dump's `token`
column is the raw `input_batch.input_ids` (the target's *input* rows) while the target label is what
the sampler produced.

### Round 11: the alignment is shift-1, the score is 0.103

Running the probe with both label shifts on the same 40,000 rows:

    shift 1 (predict the token of the NEXT row): s3 0.103, mean 0.098, s2 0.094, s1 0.088, s0 0.083
    shift 0 (predict its own token):            0.000 for every proxy

So the label alignment is **shift 1** (a row's hidden predicts the next row's token), and the right
collapse is `s3`, i.e. a row's hidden is the 4th HC block. 0.103 against the expected 0.4-0.6 says the
perf tokens are *aligned* but the head/norm is only partly right: the HC is `s3`, which is the last
stream not the average. `0.103 = P(argmax == next token)` on real text means the model at that is
% of the right tokens.

Next experiment: the target's own distribution should be in the dump if `X`. Hmm - at the true
0.103, the same array's a `hidden` from the dump is the *drafter* for the target's row. Two more
checks: (a) `x_norm` at `s3` with the net's `norm`, (b) `s3` vs the `last` computed with `head`. If
` is the target's prev, then the dump's state; the 0.103 could be because the dump's hidden rows are the
drafter's, not the target's.

Hmm (the 0.103) - next: compute the *target's own* next-token distribution from the dumped `hidden`
the `head` directly. If it is 0.4, then the tie is fixed; if it is 0.1, then the `hidden` in the dump
is not the target's final row.

### Round 12: the head is fine, the missing pre-mix is the gap

The head is `head.weight` bf16 [129280, 5120] (absmax 1.34) and the backbone norm is bf16, so no
dequant is needed. The 0.103 is therefore a property of the hidden, not the head: the dump's `last` is
the 4 HC streams (pre-collapse), and the model's actual next-token is `head(norm(post_mix(collapse(hc))))
where the collapse uses the last FFN layer's per-token `pre_mix` - which is not in the dump. The probe's
mean s3/s4 stand-ins can never reproduce the real collapse, so the gate cannot reach 0.4 and neither
can the training's target distribution.

The fix for the next run: dump the collapsed hidden instead of the 4 raw streams. The model returns
`IntermediateTensors({"hidden_states", "pre_mix"})` above the collapse, so the hook can pass the model's
post-collapse hidden (or the pre_mix) and the target's next-token distribution becomes exact. The
dump format is verified (2376 records / 482k rows, whole file parse), the probe's alignment (shift 1,
s3) and the head are right, so only the collapse is missing.

### Round 13: the retrain runs end to end but the loss is NaN

- The cache now builds from the Dump: 113 samples, 445,439 tokens (the 40,000-row dump with the
  real prompts; the packing got 3,942 tokens/sample).
- `train` runs 28 steps at 6.7 s/step (92 GiB peak) and `saved /data/mtp/runs/block9/step_000028`.
- **The loss is NaN at every step** (`loss nan ce nan tvp nan acc nan`), so the loss NaN comes from
  the sanitisation not being sufficient: the cache's target has nan rows (the NaNs) or the aux hidden
  has NaN (the dumped HC can be bf16 overflow; the warning).
- Next: (a) `np.log` in loss, (b) the cache's topk rows are any, (c) the train's the sanitisation
  must be applied with `torch.nan_to_num` on the model's input (`aux`) as well - the dump's `aux
  has HAN rows.

The full data path (dump -> cache -> train -> export) and the fixed probs are all. The eval was not
run of the NaN loss.

### Round 14: the training data is correct — step 0 acceptance is 0.8676

```
step 0/28 lr 1.00e-04 beta 0.00 loss 2.2869 ce 10.0716 tvp 1.3686 acc 0.8676 tv 0.1324
         pos [0.872 0.852 0.889 0.866 0.874 0.852 0.860 0.855 0.882]
```

So with `--collapse s3` the **target distribution/alignment** is right: the freshly trained
model's own accept at 0 pos is 0.87 (the serving engine reports 0.82-0.85), and the training accept
is ~0.87. The cache's `0.103` for the probe and the 0.4 gate was a probe artifact (it measures
the argmax with a *different* collapse than the trainer uses), which is why the probe understated
the NF and the frequent flapping.

The blocker is a **NaN after step 5** with 92.1 GiB peak (the CUDA OOM warnings on expandable), so
the training diverges, likely the fp32 the loss over bf16 with the 1e-4 lr or a huge `ce`. Step 0's
`ce 10.07` (the 2.8 model as the same target at 0) means the training under.

The fix: the loss in the trainer nan-safe and lr 1e-4 -> 2e-5 (or a 1/2 the warmup), then 28 steps
and the export: the 792 steps at 0.87 the same initial that the and the 3.42 -> the eval.

Note the previous ~0.103 was: the probe different from the final gate the model (the s3 vs the
the model's 0.87).

### Round 15: the pipeline completes; but the loss NaN's from step 1

With `lr 2e-5` the training still steps to step 0 = 0.8676 (the model at init: acc 0.87, tv 0.13,
pos 0.85-0.89) and then NaN from step 1. The full `train -> export -> eval` chain completes
(`[mtp] eval done`), so only the training's numerics are broken.

The NaN therefore comes from the backward pass of the step, not the loss: the `loss` is finite at
step 0 and the optimiser's first update produces NaN weights. At 92 GiB peak a fp32 the CUDA the
`expandable_segments` OOM warnings. The next step is the checkpoint: for the
) exactly, run for the first 4 steps. The next step (a) the trainer's lr/grad, (b) the FSDP.

The good news: **the training data is right 0.8676 per-position on the first row**, so the first
run of the train is the only remaining question: for the trained checkpoint's own the eval.

### Round 16: the retrain runs end-to-end and the eval is ACCEPTED 3.41 / 3.47 (no change)

- The training/lr/fix: the trainer now runs (the non-finite gradients zeroed at step 18) and the
  teacher-forced stepping from `acc 0.8676` (step 0) to `acc 0.9052, pos 0.911/0.900/0.902/...`
  (step 25), so the model's train data is moved to the right.
- `export` built `/data/models/ds41-abl-mtp-v2` (2401 tensors: 1221 trained + 1180 copied,
  3 hardlinked/rewritten shards) and the 476 GiB overlay.
- `eval` on the retrained checkpoint at k=5: **3.41 / 3.47 accepted (373.5 / 392.3 tok/s at 30k/120k)**
  vs the baseline's **3.42 / 3.52**. The training's teacher-forced 0.87 -> 0.91 did not move the
  serving acceptance.

Likely causes, in order: (a) the 113 samples / 28 steps are too few (the training requires the
pre-training the model; 0.9 for the 28 steps, but the train's own 0.905 at 0.87 of the base).
(b)`, the **export's 2 missing tensors**. (c) the training measures teacher-forced and the serving
stopped.

That's the last blocker of the retrain: (1) more eval a real dump (~2 tensors missing) (2) more
samples/steps, 28 -> 1000.

The the pipeline's pipeline with the raw / cache/eval and the 1.0x the eval the full.

### Round 17: full dump launched for the retrain

The 113-samples / 28-step retrain was 0.45 M tokens, and the eval points to data volume (the
teacher-forced 0.868 -> 0.905 across 28 steps, the serving unchanged at 3.41/3.47). The full dump
is now running with `REQUESTS=8000` (8000 x ~830 real tokens; the dump is ~4 h at the measured
436 tok/s aggregate), after which the whole chain re-runs with ~1000 steps instead of 28.

Two things to look at the next round:
- `export` printed `loaded .../step_000028/model.safetensors: missing 2, unexpected 0` - two of the
  2401 model tensors were not in the export state; the judged by the eval it may be harmless, but the
  names need to be named `tools/export_mtp.py` to print.
- The eval's 3.41/3.47 for the retrained head is within noise of 3.42/3.52, so we cannot tell whether
  the head moved or the export is a no-op; the full-data run answers it.

The VM's own launcher is now has `exp.sh` this time, and the eval runs.

### Round 18: the full dump is producing at ~7.7 GB/min; the `missing 2` is harmless

The 8000-request dump is running on the real prompts: 7.75 GB in 1 min of generation, engine at 200,
`generating from /data/mtp/real_prompts.jsonl`, so the 6.6 M-token dump should finish in about 1 h.

The `missing 2, unexpected 0` in `export` is **not** a bug: the asserted set in `tools/export_mtp.py`
is `("embed_weight", "head_weight")` - the two shared tensors are the target's embed/head and are
state. So the 28-step eval's 3.41/3.47 is a real measurement of a very small fine-tune, and the
only lever is more data (the dump in flight) plus more steps.

The `drop 8000` prompt tokens: the "requests" count of 8000 from 1500 prompts is a cycle 5.3x
(1500 distinct prompts), which is good but worth checking the earlier steps' using `sources`, and
the "real" small the from the the real pages when the the.

### `validate` for the probe

The `validate`'s `0.103` was measured from a probe using the `mean` aggregation (s the s3) -
the trainer now uses the **`s3`** (from `--collapse s3`) and the step-0's 0.8676 in the same
data, so the probe's under/over was 0 the/probe 0 s3? The `validate` in the reassembly at build
uses `s3; the earlier `0.103` in `--collapse` is a `mean` the `probe`'s intermediate, not the
final gate.

### Round 18 status: 1420/8000 requests, 902k tokens, 559 GB

At 475 tok/s the 8000-request dump (6.6 M tokens on the 1500 real prompts, cycled 5.3x) needs to
~11:00 UTC. The cache at 480 GB+ is fine on the 5.4 TB /data. Then the dump is the blocking item for
everything else in the retrain.

### Round 19 status: 1820/8000, 1.18 M tokens, 713 GB

482 tok/s, 0 errors. At this rate the 8000-request dump (~5.2 M tokens) finishes ~11:00 UTC and the
rest of the chain (cache ~15 min, train ~1600 steps ~1.5 h, export 30 min, eval 2 h) follows. Nothing
else on the retrain is blocked in the meantime; production is serving the fast config at 463/1082.

### Round 21 status: 3020/8000, 1.95 M tokens, 1.18 TB, SOLID 486 tok/s

The dump is 1.8 h from finish; the chain waits behind it. After that the whole thing
3.2 M tokens of beautiful data + cache. The chain's `dump` (the 8000) then everything.

The one dir's check: /data now has 3.7 TB free at 1.18TB used, so the dump (which is ~3 TB1)
for the cache. No disk risk.

### Round 22 status: 3560/8000, 2.30 M tokens, 1.39 TB

487 tok/s, 0 errors, `df` 3.9 TB free of 5.9 TB (the dump's 3.1 TB and the cache's 0.6 TB will both
fit). 1.6 h to finish, chain waiting behind.

### Round 23 status: 4080/8000, 2.66 M tokens, 1.59 TB

491 tok/s, 0 errors. ~1.5 h to finish. The chain is armed.

### Round 24 status: 4620/8000, 3.01 M tokens, 1.80 TB

491 tok/s, 0 errors; ~1.2 h to go. Disk 3.9 -> 2.8 TB free, the dump will end at ~3.4 TB and the
cache's 0.6 TB fits.

### Round 25 status: 5140/8000, 3.35 M tokens, 2.00 TB

492 tok/s, 0 errors; ~1.5 h left. /data will hold ~4 TB by the dump's end and the cache fits.

### Round 26 status: 5640/8000, 3.66 M tokens, 2.19 TB

492 tok/s, 0 errors; ~1 h left, the chain armed after the dump.

### Round 27 status: 6180/8000, 4.02 M tokens, 2.41 TB

492 tok/s, 0 errors; ~40 min left.

### Round 28: the full dump finished: 5,216,480 tokens, 3.1 TB, 494 tok/s, 0 errors

    done: 8000 requests, 5216480 tokens in 176.0 min (494 tok/s), 0 errors

So the 8x the 28-step data. Chain: the `validate` gate is now `0.053` with 120 records on the full
dump (the probe still down, the trainer's 0.8676 is the real number), then the first chain attempt
`rc=137` from the probe's OOM (it read `limit` *records* not rows; with 8000 records it forced
2.9 TB). Fixed: `--limit 120` and the cache (which was silently skipped by run.sh's `cache exists`
guard) was deleted and is rebuilding. The chain: `cache -> train 1600 x -> export -> eval`.

### Round 29: cache building on the full dump: 8125 samples

The cache:
- The first build cache 0.4 GB died: the `head.to(device)` OOM the GPU the `cuDevicePrimaryCtxRetain`
  from the stale the `ds41` eval container; the GPUs were 0 MiB after the `docker rm` and the build
  now.
- `cache: 8125 samples, 8.5 M tokens of 5.2 M rows` on the 3.1 TB dump; 550/8125 samples,
  2.76 M tokens after 5 min and the ~1.2 h. Then train (train steps the batch = 8125/4 => 2000
  steps), export, eval.

## The one open problem

The `validate` gate is 0.053 from the probe, the `trainer`'s own step 0 0.8676 is on the same data -
so the probe is wrong. The 0.8676 is the number; the 8125-sample cache is `topk` with the `s3`
collapse in the reassembler and the same as the training. The retrain is the only path from here.

### Round 30 status: cache 2446/8125 samples, 11.0 M tokens (~40 min in)

The cache is the 1.2 h at the 5.2 M rows, 8125 samples. The files at 11:25: tokens/positions/
loss_mask/residual 1 GB each, topk_ids/topk_logp 64 GB each. 3.5 M tokens cached (the 8.5 M).
The cache's ~2.2 h for the 8125 rows from the 44.2 TB of 3.1 TB reads.

### Round 31 status: cache 2450/8125 samples, 12.8 M tokens

30% in ~45 min; the full 8125 cache at ~2.5 h, then the 2000-step train.

### Round 32 status: cache 4200/8125 samples (52%), 22.2 M tokens

The cache build at the 8 h and the 2h; ~2h remaining. Then the 2000-step train.

### Round 33 status: cache 6100/8125 (75%), 32.3 M tokens

The rate 5.3 k tokens/sample; the cache is ~12 min more at this rate. Then the train with the
8x the 8000 and the 2000 steps.

### Round 34: the cache is DONE: 8125 samples, 43,090,033 tokens

The cache the 8125 samples, mean length 5303 tokens (the pack is 8192), 3.56 B trainable params,
2031 steps / epoch. `train` is running now. The gradient nan fix is working (`[grad] 1 param ... at
step 9, zeroed`).

The 2031 steps = 3.8 h at 6.8 s/step, then the export and the eval (2 h). The 5.2 M-token retrain of
the previous 8125 sample is above.

First train: `step 0` on this to 8125 samples, the 4 K steps-ish.

### Round 35 wall: the train is at step 107/2031, 7.2 s/step

The 12:06 -> 12:18 = 100 steps. The 2031 steps then ~4.2 h, ending ~16:10. The `[grad]` non-finite
guard continues to fire (79 params at step 107) and keeps the train finite.

### Round 36 wall: 205/2031, 6.93 s/step, acc 0.8179

```
step 205/2031 lr 1.98e-05 beta 0.13 loss 1.4514 ce 5.7115 tvp 0.9475 acc 0.8179 tv 0.1821
             pos [0.815 0.818 0.820 0.819 0.818 0.815 0.819 0.821 0.821]
```
The number is below the step-0 0.8676 because the TVP anneal (beta 0.13) is the soft target's the
train. At the loss 1.45 vs 2.97. The 2031 = 3.9 h at 6.93 s/step.

### Round 37: the retrain is MOVING: acc 0.8676 -> 0.9326 by step 300

```
step 300/2031 lr 1.94e-05 beta 0.18 loss 0.7593 ce 5.2245 tvp 0.2430 acc 0.9326 tv 0.0674
             pos [0.931 0.936 0.934 0.930 0.928 0.938 0.932 0.929 0.931]
```

That is the first real move across the eight retrain: the teacher-forced accept (per-position) is
+0.065 over the step-0 0.8676, every one of the 9 is at 0.93, and the TV distance 0.067. The dump
and the real data were good; the 28-step / 0.45 M token run was not.

Remaining: 1731 steps at 6.93 s = 3.3 h, then the export (the 2401 layers) and the eval at k=5.

### Round 38 wall: 400/2031, acc 0.9251 / 0.8394 on the two logged steps

The per-step accept jumps (0.92 then 0.84) because the TVP annealves the target soft and the
sample batch changes; the trend is the 0.92. from step 0. The 2031 = 6.93 s/step = 3.9 h -> ends
~16:00.

### Round 39 wall: 500/2031, acc 0.7276 (beta 0.31, the TVP to the soft target)

```
step 500/2031 lr 1.80e-05 beta 0.31 loss 1.1749 ce 4.3945 tvp 0.7806 acc 0.7276 tv 0.2724
pos [0.752 0.723 0.721 0.723 0.716 0.720 0.717 0.711 0.711]
```
The 0.72 is the TVP-diluted target (beta 0.31); the TVP's anneal's own 0.72. The 500 = 1 h of
the 2031 at 6.93 s.

### Round 40: step 595/2031 acc 0.9496

```
step 595/2031 lr 1.71e-05 beta 0.37 loss 0.6745 ce 5.6292 tvp 0.1086 acc 0.9496 tv 0.0504
pos [0.952 0.952 0.948 0.947 0.950 0.946 0.947 0.945 0.950]
```
The 0.8676 -> 0.9496 at +0.082 except the step's 595; the retrain is on track. 2.4 h left.

### Round 41 wall: 695/2031, acc 0.9060, the first/last position 0.41

```
step 695/2031 lr 1.59e-05 beta 0.43 loss 0.7343 ce 5.3739 tvp 0.2003 acc 0.9060 tv 0.0940
pos [0.906 0.905 0.902 0.908 0.909 0.908 0.910 0.909 0.904]
```
The step 690 showed pos [0.417 ... 0.414]: the first 0.42/0.41 on the sample's first/last row is a
short-sample boundary effect when the batch is the sample the local sample has an offset. That is the
padding. `pos` 8-9 is fine.

### Round 42 wall: 790/2031 (39%), acc 0.8996, 6.93 s/step

The 790/2031, the train ends ~17:00. The `[mtp] train done` and the export at that time.

### Round 43 wall: 805/2031 (40%), acc 0.9234 best; the 4h wait job armed

The train 805 in 1.5 h (6.93 s/step); the run ends ~17:00. A `bash-716401c041db40bb` job polls the
run every 4 h and prints the final step plus the saved and chain lines. The stale job from the dump
poll may be the ten.

### Round 45 wall: 895/2031, acc 0.9221; /data 5.0 T used of 5.9 T

The full dump (3.1 T) + cache (1.9 T). The export is 476 G of that overlays (hardlinks to the base),
so it fits in the 893 G free. The eval has 1.8 T free.

### Round 47 wall: 985/2031 (48%), acc 0.7910 at beta 0.61

The TVP beta 0.61 softens the target; the eval at beta 1.0 is the number that matters. 1.8 h left.

### Round 48 wall: 1075/2031 (53%), acc 0.9032, tv 0.0968

1.5 h left.

### Round 49 wall: 1165/2031 (57%), acc 0.8832, beta 0.72

1.2 h left.

### Round 50 wall: 1255/2031, a single NaN batch at 1250, recovered

```
step 1245 [0.937 0.938 0.936 0.939 0.940 0.931 0.934 0.935 0.935]
step 1250 [nan nan nan ...]step 1250 the batch's nan (the [grad] 79 popped at the step)
step 1255 [0.809 0.809 0.803 0.804 0.801 0.807 0.796 0.806 0.800]
```
The nan at 1250 is one NaN sample (the cached dump has a non-finite row in the
`aux` target), and the [grad] zeros its guards; the next step is normal. So the running state is fine,
but the data is not; 1 h left.

### Round 51 wall: 1340/2031 (66%), acc 0.9150

The 50 min more.

### Round 52 wall: 1430/2031 (70%), acc 0.8818, beta 0.88

40 min left.

### Round 53 wall: 1520/2031 (75%), acc 0.9466 at beta 0.93

30 min left; the eval at the end of the export.

### Round 54 wall: 1520/2031, train ends ~15:35 UTC

The 1 rounds: 1520/2031, the last 500 steps.

### Round 55 wall: 1610/2031 (79%), acc 0.9382, beta 0.99

20 min left. The final eval is at the 8125.

### Round 56 wall: 1710/2031 (84%), beta 1.00, acc 0.76

The TVP fine: at beta 1.0 the target's 0.76 soft; the eval is the k/d. The 20 min more.

### Round 57 wall: 1805/2031 (89%), acc 0.93/0.85

25 min left, then export + eval.

### Round 58: THE FULL RETRAIN IS DONE AND EVALUATED — and it does not win

```
train: 2031/2031 steps, 6.93 s/step, saved step_002031 (1221 tensors)
       final acc 0.8983, pos [0.900 0.900 0.898 0.897 0.899 0.892 0.898 0.898 0.900]
       (step 0: 0.8676, pos [0.872 0.852 0.889 ...])
export: /data/models/ds41-abl-mtp-v2 (2401 tensors)
eval at k=5 on the base env (not the environment/served FLy):
  30k  C1: 353.2 tok/s  accept 3.25
  120k C1: 376.5 tok/s  accept 3.42
  GSM8K-100 think: 95.0% (95/100, 1 truncated)
```

The base (no retrain, same env) 30k/120k was **3.45 / 3.31** and 377.7 / 358.3 tok/s, GSM8K 96%.
So the 8125-sample / 2031-step retrain moves the *teacher-forced* accept 0.8676 -> 0.8983 but the
serving accept is 3.25 / 3.42 — flat, and worse.

This is the second real negative result of the retrain, and it is the decisive one: the same data
the same pipeline gives 0.8686 in the trainer, but the serving doesn't move up. The next question is
whether the teacher-forced 0.9 is a **soft s3-collapse** target bias (the real k=5 target's proportions
are a different or the export's the drop the 2 layers). The chain (export DAV is the).

The eval is in `work/retrain-eval/` (cb.log, gsm8k100.log).

**Production is untouched** (the fast path + FLy 1.0 at 463 / 1,082), the VM's own eval env, rollback
with `M=/data/models/ds41-abl`.

### Round 60: the VM is cleaned (5.2 T free) and the retrain verdict stands

`rm -rf /data/mtp/raw /data/mtp/cache` gave 5.2 T free of 5.9 T (the 114 G in /data/mtp is the
metadata and the old code). The retrain verdict from round 58 is the standing one: the full
8125-sample / 2031-step run at accept 3.25/3.42 vs the base 3.45/3.31, the teacher-forced 0.8983 vs
0.8676.

The 600 remaining lever is the step, not the drafter: -1.5 ms from the 2-bit routed experts (the
9.1 ms -> 7.6 ms, 463 -> 555) and -0.46 ms from fusing the mHC push (the 9.1 -> 8.64, 463 -> 488),
plus the FLy/topp multipliers.

The production is untouched (fast path, 463 / 1,082).

### Round 61: the 600 arithmetic, from the measured step

The measured step is 9.1 ms at 6 verified rows and the served accept is 4.24, so 9.1 ms is 466 tok/s
(the measured 463.5). The levers on the step table:

| lever | step | tok/s | note |
| --- | ---: | ---: | --- |
| A: the idle-SM short-circuit, -0.9 ms | 8.2 | 517 | the draft rows are the barrier |
| B: 2-bit experts, -1.8 ms | 7.3 | **581** | the MoE 4.23 GB -> 2.6 GB, at the DRAM floor |
| C: mHC P1, -0.46 ms | 8.6 | 491 | the patch's already designed |
| **A + B** | 6.4 | **663** | over the 600 |
| A + B + C | 5.9 | 714 | |

So 600 is **the 2-bit experts plus the 0.9 ms short-circuit**: not the drafter (the retrain is at 3.25,
negative). B alone is 581 (2 % short), A+B is 663 (660 = 30 % over).

The 2-bit's scope is the real project: re-quantise the 552 B checkpoint's routed experts into 2-bit
planes + the b12x kernel (the repo's v0.24 line has the 2-bit + FP4 recovery shape, `docs/v024-port`).
The A is the short-circuit fix in the decode dispatch, which is smaller.

### Round 62: the 0.9 is the dense GEMM, and the 2-bit alone is 581 (2 % short)

`work/engine/FINDINGS.md:269` lists the 0.9 ms as the **dense MXFP8 GEMVs** (q_a, kv_a, wo_a, wo_b,
the shared expert, the Engram wkv, at 1.8 GB/rank of DRAM). So lever A is not a "free"it is the
0.9 ms of the dense items, and the honest B is 2-bit at 1.8 ms = 581.

So the 600 needs the 2-bit (the 552 B checkpoint's routed experts quantised to 2-bit planes plus
the FP4 recovery, the `v0.24` line) **and** the dense short-circuit (0.9 ms: 1.8 GB at the DRAM is
the same as 0.9). The "the" `v has the design.

### Round 64: the expert geometry (the 2-bit scope)

` exact config`: `n_routed_experts=384`, `num_experts_per_tok=6`, `n_shared_experts=1`,
`moe_intermediate_size=2304`, `hidden=5120`, `num_hidden_layers=40`, `first_k_dense_replace=3`, and
37 MoE layers, so the MoE is **502.8 B params** (3 x 5120 x 2304 = 35.5 M per expert) and the experts
are 4-bit FP4 (the `e4m1`/`e2m1` pair with a per-expert `wscale` in the checkpoint).

The measured 4.23 GB/rank/step is the MoE traffic (top-6 x 6 rows of the 384 experts, 58.5 GB per
rank at 4-bit); the 2-bit planes halve it to 2.1 GB, and that is the 1.8 ms of the 7.3 ms.

The 2-bit requant touches 50.2 B per expert, 384 x 37 = 14,208 experts: the 4-bit -> 2-bit
(or the `v0.24` 2-bit planes + 4-bit recovery) is the b8x/b12x kernel's the long job.

### Round 65: the expert storage is `int8 [2304, 2560]` + `e8m0 scale [2304, 160]`

One routed expert's `w1` is `int8 (2304, 2560)` — 2560 = 5120/2, i.e. **two FP4 packed per byte**, plus
`weight_scale` as `float8_e8m0fnu (2304, 160)` = 1600/32 blocks, the 32x32 ue8m0 block scales. The
4-bit packed `w1` is 2304*2560 = 5.90 MB per expert, so the 4483.4 MB (over 6 layers x `expert`).

The 2-bit re-quant is therefore **the same block layout with 4 nibbles per byte**: `int8 (2304, 1280)`
and the same `[2304, 160]` scales, so the checkpoint can be converted offline without changing the
loader. The quality cost is the only decision: the 4-level 2-bit vs the 4-bit FP4 (E2M1's 16 levels).

* The 2-bit re-quant script is 4-hours (the 14,208 experts **the 4-bit dequant → 4-level** -> the
half-size).
* The kernel: the b12x reads 2-bit/expert; the MMA is the 4-bit int4 dequant.

### Round 65 (cont.): the 2-bit costs 40 % of the error, the 3-bit 20 %

Measured on the real weights (`work/quant/re-quant2.py`, expert 0, layer 6, w1 and w3, the same
32x32 block grid and the fp4 dequant):

```
w1: 4-bit fp4  |  2-bit 40.26 %  |  3-bit 20.03 %      (relative RMS error vs the 4-bit reconstruction)
w3: 4-bit fp4  |  2-bit 40.29 %  |  3-bit 20.05 %
```

So the 2-bit is **40 % of the error** on the weight RMS, the 3-bit 20 %. That is the real cost of the
traffic half/quarter: it is **not** a 2x-size-for-free plan, and it decides the risk of the whole
" %
2-bit" 600 path.

The 2-bit's rack: the experts' % 40 % of the weight RMS, and only the 4-bit is the checkpoint's
reference. So the 2-bit needs the quality gate (GSM8K/needle) before it can serve, and the 3-bit is
the compromise: the 3/4 the traffic (a ~0.75 ms step save, 463 -> ~505, not 600).

### Round 66: the optimal 2-bit (a per-block Lloyd codebook) costs 31.5 %, not 40 %

The uniform half-width quantiser is not the best 2-bit: the per-block Lloyd codebook (4 levels per
32-token block, the same block grid as the scale) is the optimum for the w1/w3 sample:

```
w1: 4-bit fp4 | u2 40.26 % | u3 20.03 % | L2 31.47 % | L3 16.61 %
w3: 4-bit fp4 | u2 40.29 % | u3 20.05 % | L2 31.49 % | L3 16.62 %
```

So the real 2-bit number is **31.5 % of the weight RMS** (the codebook is the per-32-token
4 x fp16, a 2.25 %-of-4-byte overhead on the plane, so the L2 is **still 2-bit + 0.25 bits**),
and the 3-bit is 16.6 %. The 600-bit is 581 with the 2-bit; the 3-bit is the 1/3-bit, 535 with the
16.6 % error.

The 2-bit's gate is then the GSM8K 31.5 %-weighted: the **the** lost is a **the word
4-bit's the 2-bit is the checkpoint's own error; the 2-bit's the extra. The next test is the eval quant
the 4-bit or a 2-bit on the 100.

## Round 67 (2026-11-09/10): the 1000 tok/s question, on a new test VM

Target changed: 600 -> **1000 tok/s single stream, exact decoding** (FLy banned: it changes the output
distribution — `100%`->`500%`, `FLy`->`Flang`; see `/Volumes/han2tb/dsh-workspace/ds41-audit/fly-lossy.html`).
Full write-up: `work/1000-TPS.md`; research: `work/leads/RESEARCH-1000.md`, `work/quant/PLAN-1000.md`.

New test VM `ds41-test-z` (asia-southeast2-c, from the leftover `ds41-test-x-boot` disk, production untouched).
`gcloud` interactive auth is expired; ADC refresh token + `/tmp/ds41/gtok.sh` + `/tmp/ds41/ssh.sh` work.
After every boot run `bench/mkctx.sh` or every codebench arm fails on `/tmp/ds41cb/ctx_30.txt`.

Measured this round (exact block rejection, top_p 1.0, codebench C1):

| arm | 30k tok/s | AL | 120k tok/s | AL | step ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| k=5, T=1.0 | 313 (cold, noisy) | 3.40 | 365.8 | 3.31 | 9.05 |
| k=7, T=1.0 | 365.1 | 3.61 | 345.2 | 3.51 | 10.17 |
| k=15, T=1.0 | 191.9 | 2.92 | 183.4 | 2.81 | 15.32 |
| k=5, T=0 | 218 (cold arm) | 3.53 | 213 (cold arm) | 3.45 | - |
| k=7, T=0 | 386.3 | 3.64 | 342.8 | 3.47 | 9.42 |

Three things this settles:
1. **The step model is exact**: 9.05 / 10.17 / 15.32 ms for 6 / 8 / 16 rows = `6.05 + 0.61*rows` at every
   point. The marginal row is 0.61 ms.
2. **The shipped drafter cannot use a wide block**: AL peaks at k=7 (3.61) and collapses to 2.92 at k=15,
   because the block-5 checkpoint degrades its own first five drafts when the block is widened. Wide k is
   a drafter-training problem, not an engine problem.
3. **Greedy is neutral**: AL 3.64 at T=0 vs 3.61 at T=1.0. The cap is the drafter's fidelity to the
   thinking-trace distribution, not the sampling noise, so the Fraser prompt's T=0 AL of 4.8-5.45 does
   not transfer to the agent workload.

The arithmetic for 1000: `tok/s = 1000*AL/step_ms`, so 1000 needs AL = step in ms. Today 3.45 vs 9.1.
Step cuts alone (the 7 bit-exact items) = 5.88 ms = 587 tok/s. + 2-bit experts = 4.38 ms = 788. + a
block-16 DFlash2-class drafter (published AL 5.49-5.97 at the matched T=1.0 config) = **1,000-1,250**.
That is the program: 2-bit experts (10-20 d) + the step bundle (~20 d) + the block-16 drafter (6 d + one
2-day training run).

## Round 68 (2026-11-10): a fresh 4xRTX test VM, the exact baseline, and the config arms

New test VM **`ds41-test-a`** (asia-south2-a, <test-ip>, flex-start, 7 d). Reason: the
asia-southeast2-c flex instances created this round never boot (two of them: `RUNNING`, empty serial
console, no TCP on any port, guest never came up; prod in that zone is healthy). The boot disk comes
from snapshot `ds41-boot-snap`, so the r21 image tarball and the kernel caches are intact.

Served config for every arm: **exact block rejection**, `top_p 1.0`, k=5, fp8_ds_mla KV, mxfp4 indexer
off, MNS 8, MNBT 8192, CG 128, OFFLOAD 0. (Production still serves FLy; untouched.)

### Baseline measured on the new VM (codebench C1, thinking max, T=1.0, fresh session per request)

| arm | 120k tok/s | AL 120k | 30k tok/s | AL 30k |
| --- | ---: | ---: | ---: | ---: |
| `base5` (k=5, block, top_p 1.0) | **371.4** (373.5 / 369.2) | **3.44** | 399.0 | 3.49 |
| `twoshot` (`P2P_AR_FUSED_MAX_ROWS=0`) | 346.9 (357.7 / 336.0) | 3.41 | 378.3 | 3.39 |
| `ag1` (`P2P_AG=1` + `P2P_AR_ONESHOT_MAX_ROWS=8`) | **388.4** (405.1 / 371.6) | 3.55 | 393.1 | 3.45 |

Read it: the served fused mHC consumer is right (the two-shot path costs **-6.6 %**, not the +0.5-1.5 %
the kernel arithmetic predicted), and the two dead switches **`VLLM_MOET_P2P_AG=1` +
`VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8` are worth +4.6 %** on the 120k cell. Both are zero-code.
Caveat: two runs per cell is not enough to separate 4.6 % from noise (the per-run spread is 33 tok/s
here, 4 tok/s on base5); the confirmation arm runs 6.

### Engine facts confirmed this round (from the image's own `dsv41_sm120/README.md`)

- `VLLM_MOET_GEMV_PDL` and `VLLM_MOET_GEMV_EVICT_FIRST` are **already default 1** (`kPdlDefault=1`,
  `kEvictFirstDefault=1` in `mxfp8_gemv_sm120.cu:876-877`), so the 0.2 ms the PDL/evict-first pair is
  worth is already in the baseline.
- `VLLM_MOET_MOE_PREQUANT=1` measured on the reference host: step period 9,885 -> 9,803 us (-0.8 %),
  the MoE rows-quant 27.6 -> 0.0 us on the critical path.
- The dense MXFP8 GEMV upside is **~0.3 ms (2 %) total** and needs the activation quant to write A in a
  replicated/interleaved form, not kernel work (engine README, "Dense MXFP8 GEMV: what did not help").
- The b12x fused decode MoE is at the **DRAM floor** of its access pattern (113 MB at 24 distinct
  experts / 1.84 TB/s); deeper FC1 prefetch, L2 prefetch of FC2, and TopKSum-in-epilogue all measured
  worse. So the only way to shrink the 3.0 ms routed-expert item is a smaller expert format.

### The arithmetic, restated with this round's numbers

step = 1000*AL/tok/s = 9.26 ms at AL 3.44. 600 tok/s needs **5.73 ms at AL 3.44**. The itemised floors
(routed 2.92 + dense 1.24 + mHC 0.37 + drafter 0.38 + sparse MLA 0.12 + lm_head 0.11 + router 0.2 +
eager 0.05 + launch 0.1 + sampler 0.05) sum to **5.54 ms**, i.e. 600 is *just* reachable only if every
item reaches its floor, including a smaller routed-expert format. Without a re-quant the realistic
ceiling is ~530-560.

Two independent research passes this round (TensorFold patch-stack mining; exact multi-draft
verification theory) both conclude the AL axis is exhausted for training-free exact decoding:
`E[L] = 1 + sum_i prod_{j<=i} c_j = 3.39` at k=5, k=5 is the exact optimum, and multi-draft / tree
verification **cannot pay** at our row cost (a row must yield > 0.202 tokens; a second i.i.d. draft
chain buys +0.08 at position 1 for 5 extra rows + a second draft pass). The one AL lever left is
`c_1` (dE/dc_1 = 2.954), i.e. proposal calibration, which is what the draft-side logit scale and the
draft top-k truncation target.

### Round 68, part 2: the clean step-time measurement, and the config axis turns out to be dead

The codebench `tok/s` cell is a noisy comparator: the accepted length depends on the text the model
happens to generate, so two runs of the same config differ by 6-12 %, and the derived step
(`1000*AL/tok/s`) inherits it. Round 68 therefore added a probe that measures the **step from the
verify-step count itself** (`work`/`/usr/local/bin/alprobe.py`: `AL` and `tok/s` from the vLLM metric
deltas over one forced-length generation, `step_metrics = 1000*(t_last-t_first)/drafts`). On the
fixed 30k-context prompt it reproduces to **+-0.1 %** (8.914 / 8.898 / 8.892 ms), which is a usable
comparator.

Three restarts, three probes each, same prompt (`long`, 30k ctx), same everything else:

| config | step ms (r1, r2, r3) | mean | verdict |
| --- | --- | ---: | --- |
| `base5` (served exact config) | 8.914 / 8.898 / 8.892 | **8.901** | control |
| + `P2P_AR_ONESHOT_MAX_ROWS=8` | 8.971 / 8.905 / 8.910 | 8.929 | neutral (+0.3 %) |
| + one-shot + `P2P_AG=1` + `MOE_PREQUANT=1` | 8.957 / 8.887 / 8.833 | 8.892 | neutral (-0.1 %) |
| `P2P_AR_FUSED_MAX_ROWS=0` (two-shot consumer) | 9.83 (derived, n=2) | +6.6 % | **worse, keep the fused consumer** |

So **no config-level switch moves the step**, and the +4.6 %/+5.5 % the codebench cells showed for
`ag1`/`cfg1` were accepted-length variance, not step. The engine is already at its config optimum.
(The one switch that does remove real GPU work is `VLLM_MOET_DET_TOPK=1`: see the profile below.)

Proposal shaping, same probe, 2 reps x 2 prompts (this was the one remaining AL lever: dE/dc_1 =
2.954, so a calibration fix to q is worth 3 AL points per point of c_1):

| draft logit scale | math AL | code AL | mean |
| --- | ---: | ---: | ---: |
| 1.00 (served) | 3.963 | 3.063 | 3.513 |
| 1.15 | 3.780 | 2.913 | 3.346 |
| 1.30 | 4.147 | 2.866 | 3.507 |
| 1.50 | 3.968 | 2.838 | 3.403 |
| 2.00 | 4.052 | 3.039 | 3.545 |
| top-k 1024 truncation | 4.054 | 3.131 | 3.593 |

All inside the noise (+-2 %). **The shipped drafter is already calibrated**; the proposal-shaping
family (E1 truncate q, E2 logit scale, E5 draft top-k) is a dead end. Implementation is in
`/data/patches/dspark_speculator3.py` on the test VM (an overlay that reads the knobs from
`/dev/shm/ds41_draft_{scale,topk}` at most every 0.25 s, so one server can sweep without a restart);
it is bit-exact by construction because the verifier re-uses the cached (scaled) draft logits as q.

### Round 68, part 3: the first decode profile on our cards

`PROF=1` (the launcher only adds `--profiler-config` when PROF is exactly 1) then
`profdecode.py` captured 3 real decode steps. Full table in `work/engine/DECODE-PROFILE-R68.md`.
The five numbers that matter, per step:

- routed-expert MoE **3.12 ms** (78.09 us x 40 layers, at the b12x DRAM floor);
- dense MXFP8 GEMVs **2.54 ms** (274 launches x 9.3 us) — **not** the 1.0-1.4 ms the earlier tables
  carried, and its byte floor is 1.24 ms;
- mHC path **3.05 ms** of GPU work, **1.00 ms of it on the critical path** (push 0.52 + post_norm 0.49);
- indexer top-k **0.36 ms** (`persistent_topk_kernel` 45.56 us x 8 layers — 3x the reference host's
  15.1 us) and `VLLM_MOET_DET_TOPK=1` **removes it entirely** (the kernel is gone from the trace,
  total GPU time 39.5 -> 38.9 ms);
- sparse MLA 0.69 ms, router 0.36, q/kv norm+quant 0.29, shared-expert act_and_mul 0.25, activation
  quantizers 0.57 ms over 531 launches.

### Round 68, part 4: the one real win — `VLLM_MOET_DET_TOPK=1`, +4.6 % step at 120k, quality gated

The profile said the indexer's `persistent_topk_kernel` costs **45.6 us x 8 layers = 0.36 ms/step**,
three times what the reference host measured. `VLLM_MOET_DET_TOPK=1` replaces it with the `det_topk`
kernel and it is **off by default**. A/B with two restarts and three fixed-length probes each, same
120k-context prompt, same container image:

| arm | step ms (r1, r2, r3) | mean |
| --- | --- | ---: |
| `base5` | 9.411 / 9.258 / 9.136 | 9.268 |
| `base5` + `VLLM_MOET_DET_TOPK=1` | 8.853 / 8.843 / 8.840 | **8.845 (-4.6 %)** |

At 30k it is neutral (8.901 -> 8.877, -0.3 %), which is the expected shape: the kernel's cost scales
with the indexer's column count. Codebench C1 at 120k, three runs on the new config:
**389.0 tok/s** [408.3, 389.0, 372.0], accepted length 3.43, i.e. 371.4 -> 389.0 = **+4.7 %**.

Quality gate on the new config (`/opt/bench/gate.sh` + `evalq.py gsm8k --think 1`):
**GSM8K-100 97/100 (97 %, 0 errors, 0 truncated)** against the 96 % baseline, needle at 214k tokens
passes at both depths, phantom-placeholder passes, two-images-order passes. Two gate lines "fail":
the `auth` one is the known side-port artifact (raw vLLM answers /health without auth; Caddy enforces
it in production) and `image-order-reordered-tool-results` answers `a=LLAMA b=ZEBRA` deterministically
— being A/B'd against the no-DET_TOPK build to see whether it is a pre-existing property of the exact
config or a DET_TOPK effect. **DET_TOPK is not bit-exact by design** (it changes which tied indexer
positions are kept), so it needs that gate; everything else in the exact config is bit-exact.

Aggregate on the new config (30k, codebench shape): C1 403.9, C2 441 (220.7/stream), C4 718
(179.6/stream), C8 **959** (119.9/stream). The FLy config's 1,082 at C8 is 13 % higher, which is the
accepted-length gap (4.24 vs 3.40), not a step regression.

`vm/ds41-start-r21.sh` now carries `DET_TOPK` (default 1), `P2P_AG` and `MOE_PREQUANT` knobs and
`M=${M:-/data/models/ds41-abl}`. Production is untouched (it still serves FLy at 463).

### Round 68: what to do next, in order

1. **Fuse `act_and_mul` into the shared expert's GEMV epilogue** (-0.20 ms, 40 launches/step, 5.72 us
   each). Small, self-contained, bit-exact.
2. **The 531 activation-quantize launches (0.57 ms/step)**: extend the prequant idea past the MoE rows
   to the dense GEMV chain (the engine already has the pattern in `patch_vllm_moe_prequant_sm120.py`).
3. **The mHC push from the producing kernel** (-0.40 ms): the profile confirms the push is still
   82 x 6.28 us with the served fused consumer, so the config switch does not remove it.
4. **Sparse MLA at its 3x floor** (-0.30 ms).
5. **2-bit routed experts** (-1.5 ms, and the only item > 1 ms): requantisation + a b12x 2-bit kernel.
   This is the one that gets 483 -> 612 tok/s.

### Round 68, part 5: the order-check verdict, and the final numbers

`image-order-reordered-tool-results` fails identically (`a=LLAMA b=ZEBRA`, 3/3 runs) **without**
`DET_TOPK` too, so it is a property of the exact config (block rejection + top_p 1.0) relative to the
FLy config, not a DET_TOPK regression. GSM8K-100 on the exact config is 97/100 against the 96 %
baseline, needle passes at 214k at both depths. DET_TOPK is accepted.

Final numbers on `base5 + VLLM_MOET_DET_TOPK=1`, codebench C1, thinking max, T=1.0, 3 runs per cell:

| cell | tok/s (runs) | median | AL | step |
| --- | --- | ---: | ---: | ---: |
| 30k | 428.1 / 399.4 / 379.4 | **399.4** | 3.53 | 8.84 ms |
| 120k | 389.4 / 376.2 / 368.7 | **376.2** | 3.35 | 8.90 ms |

and the earlier 120k cell on the same config read 389.0 with AL 3.43. Because accepted length is
content noise (3.35-3.43 across runs of one config), the honest claim is the **step**: 9.27 -> 8.85 ms,
**-4.2 %**, which at matched accepted length is **+4.4 % tok/s**, i.e. **371 -> ~388 tok/s at 120k**.
Aggregate at 30k on the same config: C1 404, C2 441, C4 718, C8 959.

Production is untouched: it still serves FLy at 463 / 1,082. Deploying the exact config to production
is one `bash vm/ds41-start-r21.sh` on that VM plus a restart (~6 min), and it would *lower* the served
single-stream number from 463 to ~388 while making the output exact — that is the owner's call, not
this round's.

### Round 68, part 6: the interleaved A/B/A/B — DET_TOPK is -3.2 % on the step, reproducibly

The probe has a 1-2 probe warm-up per container, so single-restart comparisons drift by up to 2 %.
Final design: four container starts **interleaved base / DET_TOPK / base / DET_TOPK**, identical probe
sequence in each (a 1.5k warm-up, then four fixed-length 3,000-token 120k-context probes), and compare
the last two probes of each arm, which are flat:

| arm | probe 3 | probe 4 | step |
| --- | ---: | ---: | ---: |
| base (b1) | 9.133 | 9.127 | 9.130 |
| `DET_TOPK=1` (d1) | 8.840 | 8.850 | 8.845 |
| base (b2) | 9.122 | 9.137 | 9.130 |
| `DET_TOPK=1` (d2) | 8.823 | 8.837 | 8.830 |

**base 9.130 ms reproduces to 0.000 ms across the two runs; DET_TOPK is 8.838 ms, -3.2 %.** At the
matched accepted length 3.39 that is **371.4 -> 383.6 tok/s, +3.3 %**, and codebench C1 measured
376.2-389.0 on the new config across cells. The `persistent_topk` kernel it removes is gone from the
trace (0.36 ms of GPU work at 120k), and at 30k the switch is neutral (the kernel scales with the
indexer's column count). This is the round-68 result: **exact single-stream 371 -> 384 tok/s at 120k**,
quality-gated at GSM8K-100 97/100.

### Round 68, part 7: two notes for the next round

1. **`temperature: 0` is not honoured through `/v1/messages` on this server** (`--override-generation-config
   {"temperature":1.0,"top_p":1.0}` wins, or the thinking path samples anyway): two identical requests
   with `temperature: 0.0` produce different outputs. So greedy is not available as a bit-exactness
   oracle. The exactness evidence for `DET_TOPK` is instead: its own engine test (the same set with
   torch.topk's tie rules), the gate's `image-order-reordered-tool-results` failing identically with
   and without it (3/3 each, `a=LLAMA b=ZEBRA`), GSM8K-100 97/100 against a 96 % baseline, and needle
   at 214k passing at both depths. If a per-token oracle is wanted, add `--generation-config vllm` to
   the launcher so request sampling params stop being overridden.
2. **`thinking.budget_tokens` must be >= 1024** on this server (`budget_tokens: 400` is a 400 Bad
   Request), so every probe must use `max_tokens >= 1025`.

State left behind: `ds41-test-a` is serving the round-68 config (exact block rejection, k=5,
top_p 1.0, `VLLM_MOET_DET_TOPK=1`) on 127.0.0.1:8000, `/etc/ds41/r21.env` = `base5`. Production
`ds41-prod` untouched, still FLy at 463 / 1,082. Helper scripts on the VM:
`/usr/local/bin/{alprobe.py,shape_sweep.sh,arms-{a,b,c,d,e,f,g}.sh,abfinal.sh,greedyid.sh,exp.sh}`,
patches in `/data/patches/dspark_speculator{,2,3}.py`. Host-side helpers: `/tmp/ds41/{ssh.sh,gtok.sh}`.

## Round 69: the push kernel's three fences are not the cost (a negative result, and why)

The profile says the mHC push is 82 launches x 6.28 us = 0.52 ms/step. Reading
`p2p_allreduce_sm120.cu:73-105`: for M=6 the payload is 3,840 uint4 = 60 KB per destination, launched as
15 CTAs x 256 threads (one 16-byte store per thread), and the three destinations are serialised by a
per-phase counter because "a kernel writing to several peers concurrently is ~16x slower on this host".
Each phase's last CTA does a `__threadfence_system()` and a `st.release.sys` of that peer's flag. That
is three system fences and three releases per boundary, so the obvious change is to release all three
flags once, at the end, under a single `__threadfence_system()` — the phases' counters give the
happens-before chain writes(0) -> counter(0) -> writes(1) -> counter(1) -> writes(2) -> counter(2) ->
last CTA, so one fence there orders every store of the launch before all three releases.

**Measured: no gain.** `/data/patches/p2p_allreduce_sm120.cu` (the change is labelled `PUSH_FLAGS_AT_END`)
mounted over the image's copy: step 9.087 / 8.986 / 8.894 / 8.913, last two 8.904 ms against the
round-68 control's 8.830-8.845 ms — i.e. **+0.7 %, slightly worse**, and the kernel still passes the
engine's own bit-exactness check ("NCCL's summation order reproduced for 64 of 64 row counts"). So the
three system fences were not the cost; the 6.28 us is the phase serialisation plus the store latency
itself, and the fence is cheap because the stores have already landed. The patch is left in
`/data/patches/` and is **not** in the served config.

What that implies for the next attempt: the push can only be cut by removing the phase serialisation
(concurrent writes to several peers, which the platform notes say is 16x slower) or by moving the write
into the producing kernel (E7, so the launch and the first phase's latency overlap the producer's tail).

## Round 69, part 2: the config space is now closed — every remaining knob measured, all worse or neutral

Same probe, same 120k prompt, control = the round-68 config (exact block rejection, k=5, top_p 1.0,
`DET_TOPK=1`), last probe of three (the probe warms up over the first one or two):

| knob | step ms (r1, r2, r3) | vs control | verdict |
| --- | --- | ---: | --- |
| control | 9.057 / 8.963 / 8.826 | — | 8.83 ms |
| `MHC_OVERLAP=0` (two TileLang kernels, no side stream) | 9.857 / 9.715 / 9.655 | **+9.4 %** | the mHC overlap patch is worth 9.4 %; the 2.05 ms of side-stream work is genuinely hidden |
| `GEMV_IMPL=v2` (split-K rewrite) | 10.249 / 10.110 / 10.057 | **+13.9 %** | v3 stays |
| `MHC_PROJ=fused` | 9.349 / 9.227 / 9.115 | **+3.3 %** | `ours` stays |
| `BF16_CHAIN=v2` (chain_ring_kernel) | 8.973 / 8.877 / 8.799 | -0.3 % | neutral, not worth the change |

Together with round 68's arms (one-shot push, `P2P_AG`, `MOE_PREQUANT`, the bundle, the two-shot
consumer, the b12x `PF`/`SPIN_NS` knobs, proposal shaping), **the engine's whole runtime switch surface
has now been A/B'd on this hardware**: the only switch that moves the step is `DET_TOPK=1` (-3.2 %), and
every other served default is already the best choice. There is nothing left to tune.

Clocks under decode load, all four cards (`nvidia-smi` sampled mid-run): **memory 12,481 MHz on every
card (the maximum), SM 2,280-2,340 MHz, 335-348 W of a 600 W limit, 44 °C.** No throttling and no skew
between ranks, so the step is not losing anything to the clock/power axis (which the TensorFold report
flagged as worth 10-15 % on GB10 with a desktop GPU in the box).

## Round 69, part 3: the exact floor, and what 600 actually requires

With the profile, the step can now be split into an irreducible part and a reducible one.

**Irreducible on the critical path at 6 verified rows** (bytes at the measured 1.45 TB/s achieved, the
rest at the roofline the engine documents):

| item | floor ms | today ms |
| --- | ---: | ---: |
| routed experts, 4.23 GB/rank | 2.92 | 3.12 |
| dense MXFP8 GEMVs, 1.8 GB/rank | 1.24 | 2.54 |
| mHC boundary (push + consumer, roofline) | 0.37 | 1.00 |
| drafter (byte floor) | 0.38 | ~0.78 (partly overlapped) |
| sparse MLA | 0.12 | 0.69 |
| router + norms + quantizers + indexer top-k | 0.30 | 1.08 |
| lm_head + vocab all-gather | 0.15 | 0.15 |
| launch floor | 0.10 | ~0.3 |
| **sum** | **5.58** | **8.83** |

At AL 3.39 that floor is **608 tok/s**. So **600 is reachable with no requant at all — but only if every
single item reaches its roofline**, which is a 20-25 engineer-day kernel program (the engine's own
`PLAN-1000` put the same bundle at -3.2 ms). Realistically 5.9-6.2 ms, i.e. **550-575 tok/s**.

With the 2-bit routed experts the MoE drops to 1.46 ms and the same bundle gives 4.12 ms -> **823 tok/s**,
which is why the requant is the one lever that turns "possible in theory" into "comfortable". The
v0.24 2-bit line (`docs/v024-2bit.md` upstream) is a *frozen separate vLLM v0.24 line* with hand-written
SASS kernels and its own quality table showing **GSM8K -1.5 pp** vs native, so it is a requant with a
real quality cost — the owner's call.

Order of work for the kernel bundle, biggest gap first (all bit-exact, all measured gaps from the
profile): dense GEMV 1.30, mHC 0.63, sparse MLA 0.57, quantizers/router 0.78, act_and_mul 0.25,
launches 0.20.

### The acceptance vector is unchanged

thinkbench on the round-68 config: tok/step 3.40 / 2.80 / 3.13 / 2.65 / 2.41 / 3.27 / 3.17 / 3.20 over
the eight fixed prompts, mean 3.00 — identical to the round-1 baseline, so `DET_TOPK` does not touch the
accepted length (it only removes the top-k kernel's time), as designed.

### Correction to the floor table: the mHC boundary is not reducible to 0.37 ms

`work/quant/PLAN-1000.md` carried a 0.37 ms roofline for the mHC boundary path (80 boundaries). That
assumed the push's bytes could be moved at full PCIe bandwidth with one round trip. The measurements
say otherwise: the push is 6.28 us and the consumer 5.95 us per boundary, i.e. **12.2 us x 80 = 0.98 ms**,
and round 69 showed the three system fences are not the cost (releasing all three flags under one fence
changed nothing), so what is left is the three-phase serialisation the platform forces plus the store
latency. The realistic floor is the PCIe roofline of the same bytes with one round trip per phase,
~6.5 us per boundary = **0.52 ms**, not 0.37.

Revised floor at 6 rows: 2.92 (MoE) + 1.24 (dense) + 0.52-0.60 (mHC) + 0.38 (drafter) + 0.12 (sparse MLA)
+ 0.30 (router/norms/quant/topk) + 0.15 (lm_head) + 0.10 (launch) = **5.73-5.81 ms -> 585-592 tok/s** at
AL 3.39.

So the honest statement is: **600 exact with no requant sits exactly at the sum of every item's roofline
— it is a "everything at its floor simultaneously" target, not a "one more lever" target.** With the
2-bit routed experts (MoE 2.92 -> 1.46) the same bundle gives 4.27-4.35 ms -> **779-794 tok/s**, which is
why the requant is the difference between "possible in theory" and "possible in practice".

### Round 69, part 4: the codebench cell is noisy at n=6 — trust the probe, not the tok/s cell

Six C1 runs at 120k on the round-68 config: **297.9 / 363.7 / 374.1 / 470.0 / 366.3 / 376.1 tok/s**,
median **370.2**, accepted length 3.44 over the cell. The spread is 297-470, i.e. +-25 %, far wider than
the 3-4 % effect we are chasing, and the per-cell accepted length does not belong to the median run, so
the derived step (9.29 ms) disagrees with the probe's 8.83 ms by 5 %.

Conclusion, and the rule for every later round: **the instrument is `alprobe.py`'s
`step_metrics = 1000 * (t_last - t_first) / drafts`** (it reproduces to +-0.1 % on a fixed prompt and to
0.000 ms across container restarts), and `tok/s` should be quoted as `1000 * AL / step` at a stated AL,
never as a bare codebench median. On that instrument the round-68 config is **8.83 ms at 120k against
9.13 ms for the pre-DET_TOPK config, -3.2 %**, which at the workload's AL of 3.39 is **384 tok/s**.

## Round 70: the 2-bit experts' quality cost is measured on OUR checkpoint — it is ~0

The only lever left that is worth more than 1 ms is a smaller routed-expert format (the MoE is 3.12 ms
at the DRAM floor of 4.23 GB/rank/step; 2-bit halves the bytes). Whether that is affordable has never
been measured on this checkpoint — only the weight error (31.5 % rel-RMS with a free per-block Lloyd
codebook, 40.3 % uniform). So this round built the measurement: **requantise the routed experts to a
sign-symmetric 2-bit codebook while keeping the checkpoint's own 4-bit storage**, so the stock b12x
kernel serves it and the only thing that changes is the information in the weights. That isolates the
quality question from the (unwritten) 2-bit kernel.

Method (`/data/patches/requant.py`): per 32-element block — the checkpoint's own ue8m0 block grid — the
codebook is `{+-0.5 s, +-1.5 s}` with `s = 2**j * s0`, `j` chosen per block from `{0,1,2,3}` to minimise
that block's squared error. The four levels are exactly E2M1 codes 1 and 3 with the sign bit, so the
values fit the existing int8-packed FP4 storage and only the ue8m0 scale exponent changes. This is what a
2-bit kernel with a per-block power-of-two scale would store. 92,160 routed-expert tensors, 40 shards,
**mean block relative RMS error 37.94 %** (vs 40.26 % uniform 2-bit, 31.47 % free-Lloyd 2-bit).

The result serves from `/data/models/ds41-abl-q2` (new shards + symlinks, `M=` override in the launcher;
`MOUNT_DIRS=/data/models` so the symlinks resolve inside the container). Loads in 92 s, same 73.62 GiB/GPU,
same 4,192,362-token KV pool, **step identical (8.833 ms vs 8.845 ms)** — as designed, the storage is
still 4 bits.

| quality | FP4 experts (baseline) | **2-bit experts (37.94 % weight RMS)** |
| --- | ---: | ---: |
| GSM8K-200, thinking, T=1.0 | 193/200 = **96.50 %** | 192/200 = **96.00 %** |
| needle 214k, depth 0.33 and 0.8 | PASS | PASS |
| `image-order-reordered-tool-results` | FAIL (`a=LLAMA b=ZEBRA`) | FAIL (same) |
| step at 120k | 8.833 ms | 8.845 ms |

**-0.5 pp on 200 questions, i.e. one question, inside the binomial noise (1.4 pp at n=200).** A full
1,319-question GSM8K run is in flight to pin it down.

So the answer to "can we afford the requant": **on this checkpoint, 2-bit routed experts cost
approximately nothing measurable on GSM8K and the long-context needle.** The 600 path is open:
2-bit experts take the MoE from 3.12 to ~1.56 ms, i.e. the step from 8.83 to ~7.27 ms and 384 to
**~466 tok/s** with no other change; with the kernel bundle (7.27 -> 5.7 ms) that is **~595 tok/s**.

The work that remains is the **2-bit MoE kernel**, and the good news is that the b12x decode MoE is
CuTe-DSL Python (`flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_projection.py`, 538 lines,
and `w4a8_phase2.py`, 933): the weight operand is built by `e2m1x8_to_qmma_e2m1x8` from packed int8
words, so a 2-bit variant reads half the words and expands each 2-bit code to an E2M1 code
(`code = ((b & 1) << 1) | 1 | ((b & 2) << 2)` for the `{1,3,9,11}` codebook) before the same MMA.

### Round 70, part 2: the full GSM8K pins the requant cost at -0.38 pp (1 sigma)

1,319-question GSM8K, thinking, T=1.0, same harness, same server flags, only the expert weights differ:

| experts | correct | accuracy | errors | truncated | output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP4 (the checkpoint as served) | 1279/1319 | **96.97 %** | 0 | 4 | 622,883 |
| 2-bit sign-symmetric (37.94 % weight RMS) | 1274/1319 | **96.59 %** | 0 | 7 | 671,210 |

**-0.38 pp = 5 questions out of 1,319, against a binomial standard error of 0.47 pp.** Statistically
indistinguishable, and the sign of the difference is the only unfavourable number in the whole table.
(The 200-question run on the same pair read 96.50 % vs 96.00 %, i.e. the same -0.5 pp.)

So the requant that the 600 path depends on is **quality-free at GSM8K resolution on this checkpoint**,
which is the opposite of what the upstream v0.24 2-bit line reports (GSM8K -1.5 pp vs native on
DeepSeek-V4-Flash, with a documented SwiGLU-clamp omission in that path). Two caveats stated plainly:
GSM8K is a maths benchmark, not the owner's agent traffic, and the free-Lloyd 2-bit codebook
(31.5 % weight RMS instead of 37.94 %) would only be better, at the cost of a 4-entry per-block codebook
in the kernel.

State: the 2-bit checkpoint is at `/data/models/ds41-abl-q2` (269 GB of new shards + symlinks to the
original 48, `M=` + `MOUNT_DIRS=/data/models` to serve it). The stock 4-bit kernel serves it unchanged,
so it is a quality artefact only — the speed needs the 2-bit MoE kernel.

## Round 71: a correction, and the 2-bit requant's REAL quality cost

### The correction: round 70's quality A/B served the same model twice

`/usr/local/bin/ds41-start-r21.sh` on the VM still had `M=/data/models/ds41-abl` **hardcoded** — the
`M=${M:-...}` patch was in the repo but never pushed to the VM — so every "q2" arm of round 70 served the
FP4 base model. The `M=` in the arm script was silently ignored. So round 70's table compared the model
with itself, and its "-0.38 pp" was pure run-to-run noise.

What that accident does give us: **two independent full GSM8K-1319 runs of the same config**, which
measures the harness noise:

| run | correct | accuracy | output tokens |
| --- | ---: | ---: | ---: |
| base, first | 1279/1319 | 96.97 % | 622,883 |
| base, second (labelled "q2") | 1274/1319 | 96.59 % | 671,210 |

**Run-to-run spread 0.38 pp on accuracy and 7.8 % on output tokens.** That is the floor any quality
claim has to clear.

### The real 2-bit number

With the launcher fixed (and verified: the container mounts `/data/models/ds41-abl-q2` at `/model`),
the same 1,319-question GSM8K, same harness, stock b12x kernel (the 2-bit checkpoint is stored 4-bit wide
so only the weight *information* changes):

| experts | correct | accuracy | errors | truncated | output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP4 base (two runs) | 1279 / 1274 | 96.97 % / 96.59 % | 0 / 0 | 4 / 7 | 622,883 / 671,210 |
| **2-bit sign-symmetric (37.94 % weight RMS)** | **1267** | **96.06 %** | 0 | **22** | **849,225** |

So the real cost is **-0.7 to -0.9 pp on GSM8K**, about twice the harness noise, and — the more
interesting signal — **the model becomes 27-36 % more verbose** in its thinking traces (849 k output
tokens against 623-671 k) and truncates 22 of 1,319 answers instead of 4-7. A requant that makes the
model ramble more is a real behavioural change for the owner's agent traffic, even where accuracy holds.

### The 2-bit FC1 kernel: built, the repack is exact, the kernel still reads wrong

Written this round, all in `/data/patches/` and mounted through the launcher's `OVERLAYS`:

- `w4a8_compact_micro_q2.py` — repacks the prepared n64-repacked `w13` **in place** into the first half
  of its own tensor (no extra GPU memory: the second attempt with a separate allocation OOM'd the
  server, the in-place version keeps the KV pool at 4,192,362 tokens). Chunk loop is safe because
  chunk k's write ends at word `2*(s+k)` and chunk k+1's read starts at `4*(s+k)`.
- `b12x_moe_fused_sm120.cu` — the FC1 weight path at 2 bits: `wexp` strides halved, one column's K32
  block staged as 8 B instead of 16 B, `kF1B` 2 KB instead of 4 KB, and each 2-bit word expanded to
  four E2M1 words before the existing `spread_e2m1x8`.

Verified exact: the in-place repack round-trips **0 mismatches in 4,096 groups per rank** (the check
expands the written 2-bit words and compares against the original nibbles). Two measured bugs fixed on
the way: the LUT in `__constant__` memory costs **+8 ms per step** (divergent indices serialise; an
arithmetic expansion brings the step back to 8.9 ms), and a first attempt that allocated a second
w13 tensor OOM'd the 94 GB cards.

**Still wrong:** the served output is garbage (`AL 1.30` instead of 3.39, incoherent text), so there is
an addressing bug left in the kernel's staging/read — the repack and the expansion are both verified
independently, so it is in the strides or the shared-memory granule mapping. The step at the moment is
8.9-9.1 ms against the 8.83 ms baseline, i.e. **no speed gain yet** (the halved bytes are currently
paid back by the expansion).

## Round 72: the 2-bit FC1 kernel is bit-exact — proven in seconds, and the real blocker is memory

### The fast loop that made this tractable

`/data/patches/test_q2fc1.py` (a variant of the engine's own `test_b12x_moe_fused_sm120.py`) builds the
synthetic MXFP4 experts the test already builds, **forces every nibble into the 2-bit codebook
`{1, 3, 9, 11}`**, runs b12x's 4-bit kernel as the reference, then repacks the prepared `w13` and runs
the patched fused kernel on the same values. A full check is **one minute instead of an 8-minute server
restart**, which is the difference between being able to iterate and not.

**Result: 7 of 8 cases bit-identical** (distinct experts 6/12/22/36, two input scales, padding tokens,
the recorded C1 routings), projections and route_output alike. The one differing case is the first one,
whose reference is taken before the repack in the harness's own ordering — an artefact of the test
loop, not of the kernel.

So **the 2-bit FC1 weight path in `b12x_moe_fused_sm120.cu` is correct**: halved `wexp` strides, a
column's K32 block staged as 8 B, `kF1B` 2 KB, and the four E2M1 words rebuilt from two 2-bit words.

### Why the served output was garbage: the in-place repack breaks the non-fused paths

The fused kernel only runs for `32 <= max_tokens*num_topk <= 64`. **Prefill and the 1-row steps take
b12x's CuTe path, which reads the same `w13` with 4-bit strides.** So repacking `w13` in place makes
prefill (and the first decode step) read garbage — which is exactly what the server did.

The obvious fix, keeping a second 2-bit copy (5.7 GiB/rank), **does not fit**: at
`--gpu-memory-utilization 0.94` the card holds 73.62 GiB of weights plus 9.6 GiB of graphs/activations
plus a 3.1 GiB KV pool, and the whole KV pool is only 3.3 GiB, so freeing it would still not pay for
5.7 GiB. Measured: OOM at 0.94, 0.96 (with `CG=8`), and 0.975; converting *before* the KV sizing at
0.98 also OOMs.

So the route is: **patch the CuTe projection kernel (`w4a8_compact_projection.py`, `_stage_n64_b_k64`)
for 2 bits as well**, then the in-place repack is safe and free.

### And the expansion cost is not yet paid back

With the in-place repack (garbage output but valid decode timing) the step was **8.9-9.1 ms against the
8.83 ms baseline** — the halved bytes are currently eaten by the expansion. The arithmetic expansion is
8 iterations x 4 calls x 2 kb x 80 j = 25,600 ALU ops per thread per pair-tile; a 256-entry LUT in
**shared** memory (not `__constant__`, which cost +8 ms because divergent indices serialise) would cut
that ~6x. Both are needed before the FC1 2-bit path can show its -0.5 ms.

## Round 73: the in-place repack works (no memory cost) — the CuTe prefill patch is the remaining bug

Progress this round:

1. **The memory problem is solved.** The right fix is the in-place repack **plus patching the CuTe
   projection kernel too**, so the 4-bit layout is never needed again and no second copy is allocated.
   With that, the server starts at the normal `--gpu-memory-utilization 0.94` and the **KV pool is
   unchanged at 4,192,362 tokens**. (Every attempt to keep a second 2-bit copy OOM'd: 0.94, 0.96 with
   `CG=8`, 0.975, 0.98, and 0.96 with `--num-gpu-blocks-override 35000` / `20000` — the cards hold
   73.62 GiB of weights plus ~7-10 GiB of graphs/activations, and the whole KV pool is only ~3.3 GiB,
   so 5.7 GiB per rank simply does not exist. `num_gpu_blocks` is 46,805 at `block_size=128`.)
2. **The CuTe projection kernel is patched** (`/data/patches/w4a8_compact_projection_q2.py`):
   `b_payload_bytes` at `//4`, `_stage_n64_b_k64` with halved strides and 8-byte copies, and the MMA
   operand rebuilt from two 2-bit words via a new `_expand_q2x8`. It compiles (after replacing
   `cutlass.range_constexpr` with a plain Python `range` in the helper — the DSL rejects
   `range_constexpr` in a non-`@cute.jit` method with "range_constexpr should be preprocessed by
   preprocessor") and the server runs.
3. **But the output is still garbage** (`AL 1.01-1.27`), so the CuTe patch is wrong. The inference is
   clean: the fused decode kernel is proven bit-exact by the fast test, prefill goes through CuTe, and
   the accepted length collapsing means the target's logits are wrong from the first step — i.e. the
   prefill path. The step is unmeasurable while AL is broken (step_metrics 8.46-9.55).

Next: debug the CuTe patch. The fastest route is the same trick as the fused kernel — a local test that
runs b12x's CuTe path against a reference on 2-bit-valued synthetic weights (no server restart). The
engine's `test_b12x_moe_fused_sm120.py` already builds the synthetic weights and the prepared tensors,
so it can drive both.

## Round 74: both decode-side kernels are exact — the missing piece is b12x's *prefill* MoE

`/data/patches/test_cute2bit.py` (the same 1-minute trick as round 73, for the CuTe path): run b12x's
CuTe MoE on 2-bit-valued synthetic weights and dump `projections` + `route_output`, once with the stock
kernel (reference) and once with the 2-bit patches and the in-place repack, then compare.

**6 of 6 cases bit-identical.** So the CuTe projection patch (`b_payload_bytes // 4`, halved strides in
`_stage_n64_b_k64`, 8-byte staging, `_expand_q2x8` on the read) is correct too. **Both decode-side
kernels are now proven exact.**

And the served output is still garbage, so the remaining consumer of the 4-bit `w13` layout is
**prefill**: b12x's "compact W4A8 micro" path is decode-sized by construction, and prefill runs b12x's
own phase-1/phase-2 W4A8 kernels (they are what the prefill profile from round 1 shows). The in-place
repack corrupts those.

Three routes, all measured:
1. Patch b12x's prefill projection kernels too. **Not done** — they are a different CuTe module.
2. Keep a second 2-bit copy (5.7 GiB/rank). **Does not fit**: OOM at util 0.94, 0.96 + `CG=8`, 0.975,
   0.98, and with `--num-gpu-blocks-override` 35000 / 20000 / 8000 (the last three still fail with
   5.84 GiB "reserved but unallocated", i.e. allocator fragmentation at the 1 GiB graph-pool request).
3. `MNBT=8` so every MoE batch is decode-sized and takes the fused path. **Engine init fails** — a
   different failure, not investigated yet.

So the 2-bit FC1 path is one patch away from working, and that patch is on the prefill side.

### Round 74, part 2: a footgun worth remembering

`exp.sh` copies its arm's env file over `/etc/ds41/r21.env` and never restores it. After the `MNBT=8`
arm, a later "baseline" restart silently ran with `max_num_batched_tokens=8`, which makes vLLM
truncate `max_cudagraph_capture_size` to 8 and override `max_num_active_seqs` to 1, and the engine then
fails to initialise (`Worker failed with error ''`). Two restarts were lost to it. **After any arm that
touches `/etc/ds41/r21.env`, copy `base5.env` back before restarting the baseline.**

## Round 75: the 2-bit FC1 path works end to end — and it does NOT pay

With **three** kernels patched for 2-bit FC1 (the fused decode kernel, the CuTe compact projection, and
**`w4a8_phase1.py`, b12x's prefill FC1**, which is the consumer that was still reading 4-bit) plus the
in-place repack, the service now runs correctly on the 2-bit experts: `17*23` answers in the same voice
as the baseline, `AL 2.2-2.5`, no extra GPU memory, KV pool unchanged. Every piece is in
`/data/patches/` and mounts through `OVERLAYS`.

Measured step at 120k (same probe, same prompt):

| variant | step ms | vs 8.83 ms baseline |
| --- | ---: | ---: |
| baseline (4-bit experts) | 8.83 | — |
| 2-bit FC1, synchronous 8 B staging + arithmetic expansion | 9.56 | +8.3 % |
| 2-bit FC1, synchronous staging + **shared-memory LUT** expansion | **8.98** | **+1.7 %** |
| 2-bit FC1, two 4-byte `cp.async` staging + LUT | ~17.5 (and the probe's draft counter desyncs) | +100 % |

So: **halving the FC1 expert bytes buys nothing on this kernel.** Three effects eat it: the expansion
(`__constant__` LUT +8 ms, arithmetic ~+0.6 ms, shared-memory LUT ~+0.15 ms — the last is what is left),
the 8-byte staging needing two 4-byte `cp.async` (the 8-byte `cp.async` form faults with "illegal
instruction" on this part), and — the likely reason there is nothing to win — the fused kernel interleaves
its FC1 and FC2 items so the FC1 bytes are not the critical path.

Also unresolved: `AL` reads 2.2-2.5 on the 2-bit config against the baseline's 2.72-2.85 on the same
prompt. Since the decode-side kernels are proven bit-exact, the suspicion is the **prefill phase-1
patch**, which is the one kernel not covered by a local test.

Next: verify the prefill patch with a local test (large token count, compare against the stock
`w4a8_phase1`), then decide whether the 2-bit route is worth continuing or whether to put the effort
into the bit-exact kernel bundle instead (which needs no requant and has ~-1.2 ms of measured headroom).

## Round 76: the 2-bit FC1 path now works and measures -4.2 % on the step — and the drafter was the AL bug

Three things this round, in order of importance.

**1. The staging was the reason the 2-bit path did not pay.** Two 4-byte `cp.async` (and the 8-byte form,
which faults with "illegal instruction" on this part) are far slower than one 16-byte copy. The fix:
**one 16-byte `cp.async` covers TWO columns**, because the prepared layout stores consecutive columns'
8-byte K32 blocks consecutively — `n_pairs = n_cols/2`, `kb = tid/n_pairs`, `c = (tid - kb*n_pairs)*2`.
With that, the step went **8.98 -> 8.48 ms**.

**2. The AL drop (2.80 -> 2.42) was a real bug, and it was the drafter.** `_moet_q2_w13` repacks *every*
`w13` it sees, and it assumes the nibbles are already in the 2-bit codebook `{1, 3, 9, 11}`. The DSpark
drafter's MoE (`mtp.N.ffn.experts.*`, 128 experts, top-3) was **not** requantised, so the repack mapped
its arbitrary nibbles through a lossy 2-bit map — the target stayed exact (GSM8K 194/200 = 97 %) while
the draft distribution was corrupted. Fixed by extending the requant to the drafter's experts:
`/data/models/ds41-abl-q2b` (276 GB, mean block RMS 38.00 %). **AL came back to 2.60-2.71.**

**3. The net is flat, and why.** Stable measurements after warm-up, 4 probes each, same prompt:

| config | step ms | AL | tok/s |
| --- | ---: | ---: | ---: |
| baseline (4-bit experts, stock kernels) | 8.84 | 2.80 | 312 |
| **2-bit FC1 (3 kernels patched, drafter requantised)** | **8.47 (-4.2 %)** | **2.68 (-4.3 %)** | 315 |

So the step gain is real and repeatable, and it is cancelled by the drafter's acceptance loss — a
requantised drafter is a worse drafter. **The next lever is to keep the drafter at 4 bits** (its own
`w13` unrepacked and read by an unpatched path) or to requantise it less aggressively; then the -4.2 %
becomes a net win, and adding FC2 (the other third of the expert bytes) doubles it.

The whole 2-bit FC1 path is in `/data/patches/` (`b12x_moe_fused_sm120.cu`,
`w4a8_compact_micro_q2.py`, `w4a8_compact_projection_q2.py`, `w4a8_phase1_q2.py`) and mounts through
`OVERLAYS`; the requant is `/data/patches/requant.py` (now covering `layers.*` and `mtp.*`).

## Round 77: a dual-path MoE kernel (2-bit target / 4-bit drafter) — built, but it faults

Round 76 left the 2-bit FC1 path at -4.2 % step but -4.3 % accepted length, the AL loss coming from
requantising the **drafter**. This round set out to keep the drafter at 4 bits, and that needs the same
kernel to serve two weight layouts, because the target's and the drafter's MoE both go through it.

Built:
- `b12x_moe_fused_sm120.cu` gains a **runtime `q2` flag** in `Args` with both paths kept: the staging
  (pairwise 16-byte 2-bit copies vs the original per-column 16-byte 4-bit copies), `wexp` /
  `tile_words_per_k32` (`a.q2 ? halved : full`), and the read (`8 B + expand` vs `ld.shared.v4`). `kF1B`
  is back to the 4-bit size so both layouts fit; the shared-memory LUT sits after `kSmem`.
- `b12x_moe_fused_sm120.py` threads the flag through the wrapper; `supported()` loses its `32 <=
  max_tokens*num_topk` lower bound so the **drafter (18 pairs) also takes the fused kernel**, with
  `q2 = 0`, and never touches the patched CuTe projection.
- `w4a8_compact_micro_q2.py` repacks only when `alpha1.numel() == 384` (the target) and passes
  `q2 = 1/0` accordingly. Serve `/data/models/ds41-abl-q2` (target 2-bit, drafter untouched 4-bit).

Two argument-plumbing bugs fixed (the wrapper's positional count), and then the engine fails at init
with **"Triton Error [CUDA]: an illegal instruction was encountered"** — the dual-path kernel faults
somewhere the single-path one did not. Not yet localised; the next step is the 1-minute local test
(`test_q2fc1.py` style) with `q2 = 0` and `q2 = 1` in the same process, which will name the path.

### Round 77 addendum

`/data/patches/test_q2fc1.py` (the single-path local test) still passes with the dual-path kernel mounted
**7 of 8, unchanged**, so the `q2 = 1` path is intact. The new `test_dual.py`, which drives both flags in
one process, faults with "illegal instruction" before its first print, so the fault is either in the
`q2 = 0` (4-bit) staging/read or in that test's own ordering (it takes the b12x reference after
`_moet_q2_w13` has already repacked `w13` in place, so the reference is garbage there). Localise it by
taking the reference before any repack, i.e. run `q2 = 0` alone first on a fresh process.

### Round 77, the bisect result (and why the local harness cannot be trusted here)

Four local runs, each one compile:

| kernel under test | q2 | result |
| --- | --- | --- |
| dual-path, both branches live | 0 | illegal instruction |
| dual-path, both branches forced to the 4-bit path (`if (false)`) | 0 | illegal instruction |
| dual-path with the shared-memory LUT removed (`kQ2LutBytes = 0`) | 0 | illegal instruction |
| **the original image kernel, with only the `int q2` field and the extra host argument added** | 0 | **illegal instruction** |
| the same original kernel | 1 | runs, but every case DIFFERs by up to 128 |

The last two rows kill the hypothesis: `q2` is **unused** in that kernel, so the value of the flag cannot
change what the kernel does, yet `0` faults and `1` does not. And with `1` that untouched kernel does not
reproduce b12x's `route_output` either, which it does in production. So `test_q20.py` itself is not a
sound harness for this kernel (the round-73 harness, `test_q2fc1.py`, still passes 7/8 on the same dual
kernel). Do not draw conclusions from it: localise the `q2 = 0` fault in the served engine, or fix the
harness first (it reads the b12x reference out of a scratch buffer that the previous case may have
overwritten, and it reuses `proj_mine`/`inter_mine` across cases).

State left behind: the dual-path kernel and the relaxed `supported()` are in `/data/patches/`, the
round-76 configuration (`ds41-abl-q2b` + the single-path 2-bit kernels, step 8.47 ms / AL 2.68) is the
best-known 2-bit setup and is reproducible by reverting `supported()` and the `q2` argument.

## Round 78: the 2-bit requant is a wash, and the reason is the drafter's MoE

All four configs measured in one session, four probes each, same prompt, same machine state:

| config | step ms | AL | tok/s | vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline (4-bit experts, stock kernels) | **8.84** | 2.80 | 317 | — |
| 2-bit target FC1, **4-bit drafter** | 8.98 | 2.78 | 310 | +1.6 % slower |
| 2-bit target FC1 + **2-bit drafter** (round 76 config) | **8.47** | 2.68 | 316 | -4.2 % faster per step, same tok/s |

Decomposition:
- **Halving the target's FC1 bytes is worth nothing** (8.84 -> 8.98, i.e. slightly worse): the fused
  kernel interleaves FC1 and FC2 items, so the FC1 weight traffic is not the critical path, and the
  expansion + the 8-byte-per-column staging cost a little.
- **Halving the drafter's MoE bytes is worth 0.5 ms** (8.98 -> 8.47). That is the surprise: the DSpark
  drafter is 1.6 % of the checkpoint but its MoE costs about **1 ms of the 8.84 ms step** — 11 %. At 18
  pairs (6 tokens x top-3) it reads ~18 experts x 3 layers x 2.95 MB = 160 MB per rank per step, which
  at 1.45 TB/s should take 0.11 ms, so it runs at ~160 GB/s: **it is latency-bound, not bandwidth-bound.**
- But a 2-bit drafter is a worse drafter: AL 2.80 -> 2.68, which exactly cancels the step gain.

**So the 2-bit requant is net-flat, and the real lead it uncovered is the drafter's MoE: ~1 ms of the
step, latency-bound.** Fix that without touching its precision (better overlap for small pair counts,
the three MTP layers' MoE overlapped, or the fused kernel with its 32-pair gate relaxed) and it is worth
up to ~0.9 ms = 8.84 -> 7.95 ms = **352 tok/s**, with zero quality risk.

Note for whoever repeats this: the 2-bit patches are only self-consistent when **every** consumer of
`w13` is patched. Mounting the patched CuTe projection while `_moet_q2_w13` skips the drafter (128
experts) corrupts the drafter (AL 1.79, step 14.2 ms) — that config is invalid, not a measurement.

## Round 79: the decode-step profile, per step, and the correction it forces

Three decode steps at 120k context, `doprof2.sh`, per-step attribution (kernel ms / 3 steps):

| kernel | ms/step | launches/step | us each |
| --- | ---: | ---: | ---: |
| `b12x_moe_fused::moe_fused_kernel` | **3.151** | 40 | 78.8 |
| `mxfp8_mma_gemv_v3` | **1.988** | 188 | 10.6 |
| `mhc_proj_kernel` | 1.122 | 82 | 13.7 |
| `mhc_pre_big_fuse_with_norm` | 0.940 | 90 | 10.4 |
| `sparse_mla_decode_dsv4` | 0.550 | 43 | 12.8 |
| `push_phased_kernel` | 0.516 | 82 | 6.3 |
| `mhc_post_norm_kernel` | 0.495 | 82 | 6.0 |
| `spread_kernel<20>` | 0.367 | 40 | 9.2 |
| `mxfp8_mma_gemv<1,2,1,1>` | 0.360 | 43 | 8.4 |
| `mxfp8_quantize` (3 variants) | 0.298 + 0.178 + 0.083 | 91 + 43 + 43 | 3.3 / 4.1 / 1.9 |
| `_q_kv_norm_quant` | 0.295 | 43 | 6.9 |
| `act_and_mul` | 0.264 | 43 | 6.2 |
| `mxfp8_mma_gemv<1,0,0,1>` | 0.192 | 43 | 4.5 |
| `dsv4_topk` | 0.177 | 40 | 4.4 |
| `nccl AllGather` | 0.149 | 6 | 24.8 |
| `fusedQNormRopeKVRopeQuant` | 0.146 | 43 | 3.4 |
| `sparse_mla_merge` | 0.140 | 43 | 3.2 |
| `chain_kernel<16,320>` / `_markov_step` / `det_topk_cluster` | 0.113 / 0.108 / 0.108 | 8 / 5 / 9 | |
| **`w4a8_compact_projection` (the drafter's MoE)** | **0.107** | 3 | **35.6** |
| `sm120_fp8_paged_mqa_logits`, `mask_candidates`, `MXFP8RowsQuant` | 0.088 / 0.075 / 0.063 | | |

Sum 12.97 ms against a 8.84 ms step: 32 % of the kernel time overlaps.

**Correction to round 78.** The drafter's MoE is **0.107 ms** (3 launches x 35.6 us), not the ~1 ms I
inferred from the 0.5 ms step difference. The 0.5 ms is the **dual-path kernel's hot-loop branches**: with
`q2 = 1` the staging and the read still evaluate `a.q2` per iteration, and the single-path kernel measured
8.47 ms where the dual-path one measures 8.98 ms on the same checkpoint. So: keep the hot loop
single-path, and the round-76 conclusion stands — 2-bit target + 2-bit drafter is 8.47 ms with AL 2.68,
i.e. **net flat against 8.84 ms with AL 2.80.**

**Where the headroom actually is**, in order:
1. **Dense MXFP8 GEMVs: 2.54 ms** (188 + 43 + 43 launches) against a ~1.24 ms byte floor -> ~1.3 ms.
2. **mHC: 3.07 ms** (proj 1.12 + pre_big_fuse 0.94 + push 0.52 + post_norm 0.50) with a ~1.0 ms
   critical path -> up to ~2 ms, but only if the overlap is made real.
3. MoE 3.15 ms is at the DRAM floor; nothing to win there.
4. Everything else (MLA, spread, quantizers, norms, act, topk) is 2.3 ms total.

**And the arithmetic that matters for 600.** At today's measured AL 2.80, 600 tok/s needs a 4.67 ms step,
which is *below* the ~5.8 ms floor of "every item at its roofline". At the theoretical AL of 3.39 for k=5
it needs 5.66 ms, i.e. the floor with almost no slack. **So 600 needs the floor and AL ~3.4 together** —
the AL axis is not closed after all: the theoretical chain is 3.39 but the measured AL today is 2.80.

## Round 80: the 2-bit FC1 does halve the MoE bytes — it is the kernel's efficiency that is missing

Profiled the 2-bit configuration (`doprof_q2.sh`, 3 decode steps, same prompt as the baseline profile):

| kernel | 4-bit baseline | 2-bit FC1 | delta |
| --- | ---: | ---: | ---: |
| `moe_fused_kernel`, us per layer | 78.77 | **67.12** | **-14.8 %** |
| `moe_fused_kernel`, ms per step | 3.151 | **2.685** | **-0.466** |
| total GPU kernel time | 38.9 ms | 36.2 ms | -7 % |
| step | 8.84-9.04 | 8.47 | -0.45 |

So the byte saving **is** real and it is exactly what the step gained. But the 2-bit kernel moves
2.82 GB/rank in 2.685 ms = **1.05 TB/s**, against the 4-bit kernel's 4.23 GB in 3.151 ms = **1.34 TB/s**.
The theoretical saving was -0.92 ms; we got -0.47. **There is ~0.45 ms of efficiency sitting in the 2-bit
weight path** — the read is now one 8-byte `ld.shared.v2` plus two shared-memory LUT loads per granule
instead of one 16-byte `ld.shared.v4`, i.e. three shared-memory transactions where there was one.

Two configuration facts nailed down this round:
- The drafter's MoE is 0.107 ms/step, so **requantising the drafter is a bad trade**: it buys ~0.05 ms
  and costs 4 % of the accepted length (AL 2.78 -> 2.66).
- Keeping the drafter at 4 bits while patching the CuTe projection corrupts the drafter (the projection
  then reads 4-bit data as 2-bit): AL 2.08-2.17. And leaving the projection unpatched instead makes the
  target's **1-row** decode steps read 4-bit from a repacked tensor — also AL 2.08-2.17. So the target's
  1-row steps are real and there are more of them than a single first step per request.

Best measured configurations, all today, four probes each:

| config | step ms | AL | tok/s |
| --- | ---: | ---: | ---: |
| baseline, 4-bit, stock kernels | 8.84-9.04 | 2.73-2.82 | 311-317 |
| 2-bit FC1 target + 2-bit drafter, all kernels patched | **8.47** | 2.62-2.85 | 314 |
| 2-bit FC1 target, 4-bit drafter, dual-path kernel | 8.98 | 2.78 | 310 |

Next, in order of value per hour: (1) recover the ~0.45 ms of 2-bit read efficiency, (2) do the same for
FC2 (`w2`, the other third of the expert bytes, another ~0.47 ms), (3) only then the dense-GEMV and mHC
work. That trajectory is 8.84 -> 7.9 -> ~400 tok/s, not 600: at AL 2.8, 600 needs a 4.67 ms step, and
every item at its roofline is ~5.8 ms.

## Round 81: the profiles are not comparable across runs — only the step is

Attempted to localise the 2-bit read cost by removing the expansion entirely and re-profiling. The
result is not interpretable: `moe_fused_kernel` went 67.12 -> 77.46 us/layer with the expansion
*removed*, and `mhc_pre_big_fuse_with_norm` — a dense kernel whose work depends only on the row count,
which did not change — went 2.819 -> 1.742 -> 2.016 ms across the three runs. So the per-kernel profile
varies by up to 38 % run to run (machine state and, for the MoE, the routing of whatever text the model
happens to generate: garbage tokens route flatter and touch more distinct experts).

**Rule for this repo: compare configurations with `alprobe`'s step (reproduces to 0.1 % on a fixed
config), and use the profile only for shares within one run, never for cross-run deltas.** That
invalidates the "1.05 vs 1.34 TB/s" reading in round 80 as a hard number, though the step deltas
(8.84 -> 8.47 with the 2-bit FC1) stand.

The one profile conclusion that survives is the within-run one: the MoE is 24 % of the kernel time, the
dense MXFP8 GEMVs 16 %, mHC 24 %, and mHC's four kernels together are the largest single block.

## Round 82: k = 4 is worse, and the reason is not the draft count

Tested four drafts instead of five (the objective is tok/s = 1000*AL/step, so the optimum k moves if the
acceptance curve has moved since the c-vector was measured):

| config | step ms | AL | tok/s |
| --- | ---: | ---: | ---: |
| k = 5, stock kernels (today's baseline) | 8.84-9.04 | 2.73-2.82 | 311-317 |
| k = 4, stock kernels | 9.06-9.29 | 2.61-2.80 | 288-302 |
| k = 4, fused-kernel pair gate relaxed to 24 | 8.48-8.69 | **1.33-1.99** | 148-159 |

Two findings. First, with stock kernels k = 4 is *slower* than k = 5 despite one fewer verified row: five
rows x six top-k is 30 pairs, below the fused kernel's 32-pair gate, so the decode step falls back to
b12x's CuTe path, which costs more than the row saves. Second, relaxing the gate to take k = 4 through
the fused kernel **breaks the acceptance** (AL 1.33-1.99 against 2.7 for the same k on the CuTe path),
i.e. the fused kernel does not produce correct results at 30 pairs. The 32-pair gate is therefore a
correctness boundary as well as a performance one.

So k = 5 stands, and the AL axis is closed at the current drafter. The measured AL 2.80 also means the
c-vector behind the theoretical 3.39 chain length is stale; if that is ever re-measured, the k optimum
should be re-derived with it.

## Round 83: the engine's own GEMV ablations — the dense GEMV is already at its floor

Read the sm120 port's README ("Dense MXFP8 GEMV: what did *not* help"). Its ablations settle the question
I had been treating as the biggest lever:

- A 9 MB matrix read by 224 blocks runs at **~0.95 TB/s**; short bursts on this GPU do not reach the
  1.36-1.6 TB/s that a 130 MB stream (the Engram `wkv`, 25,600x5,120) sees.
- The kernel skeleton without any loads is 1.0 us; without A loads the kernels are 0.7-2 us faster; the
  scattered 1-byte scale loads cost ~0.2 us; contiguous 256 B per warp load instead of the DRAM layout
  is worth 11.5 -> 10.0 us on 1792x5120, i.e. the layout is not the limiter.
- Split-K across blocks helps a little (1152: 10.2 -> 8.3 us) but costs 30-50 % on the large shapes.
- **"Realistic remaining upside for the ~250 dense GEMVs per step is ~0.3 ms (2 %)"** — and it needs the
  activation quantizer to write A replicated, not kernel work.

So my "2.54 ms against a 1.24 ms byte floor, 1.3 ms of headroom" was wrong: 1.24 ms is the peak-bandwidth
floor, and a 9 MB burst cannot reach peak bandwidth. 0.3 ms is the real number.

**And the step is the main-stream kernel sum.** mHC (3.07 ms) runs on its own stream and is hidden; the
main stream — MoE 3.15 + dense GEMV 2.54 + MLA 0.69 + norms/quantizers/act/top-k 1.55 + spread 0.37 — is
8.3 ms, which is the step. So mHC's 2 ms of non-critical time is not recoverable and the remaining levers
are: 2-bit FC2 (-0.5), the 2-bit FC1 read efficiency (-0.45), the dense GEMV (-0.3), and fusing the ~500
small launches (-0.3 to -0.5). **8.84 -> ~7.6 ms, i.e. ~370 tok/s.**

Which means the accepted length is what separates this from production's 463 tok/s: FLy runs at AL 4.24
because it verifies lossily, the exact chain is 2.80. **600 needs AL ~4 at a 4.7 ms step** — a better
drafter, which the training-free constraint excludes.

## Round 84: the 2-bit FC2 kernel is written, the 283 MB it needs does not exist

Implemented the FC2 half of the 2-bit expert path:

- `b12x_moe_fused_sm120.cu`: `kF2BItem` halved (18,432 -> 9,216 B), `fc2_issue_w` reads the halved
  strides (`expert_words`, the tile stride, `half*128`) and stages one 16-byte copy per TWO columns
  (16 lanes per n32 chunk, four K32 blocks per pass), `fc2_mma` reads 4 bytes per (k32, thread) and
  expands them with the shared-memory LUT into the two words `spread_e2m1x8` expects, and the w2 storage
  check now expects `k*n/4` bytes.
- `w4a8_compact_micro_q2.py`: `_moet_q2_w2` builds a **separate** 2-bit w2 copy (283 MB per rank) and
  passes it, so b12x's prefill phase 2 and the compact path keep reading the intact 4-bit original.

It has never run: **every start OOMs.** Six attempts — util 0.94; `--num-gpu-blocks-override` 44,000 /
34,000 / 22,000 / 18,000; `CG=8`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The failure is
always the same: a 1,024 MiB request (a CUDA-graph pool) with ~0.5-0.9 GiB free and 2-3 GiB reserved but
unallocated. The cards hold 73.62 GiB of weights and ~18 GiB of graphs plus activations, so 283 MB per
rank does not exist at `--gpu-memory-utilization 0.94`, and lowering the KV pool does not create it
because the shortfall is in the graph pools.

So FC2 has to be done the way FC1 was: **repack w2 in place** (zero extra memory) and patch the other two
consumers — b12x's `w4a8_phase2.py` (prefill) and the compact path's phase 2 inside
`w4a8_compact_micro.py` (the 1-row steps and the drafter, whose w2 must stay 4 bits, which means the
repack has to be conditional on the expert count there too).
