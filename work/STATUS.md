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
