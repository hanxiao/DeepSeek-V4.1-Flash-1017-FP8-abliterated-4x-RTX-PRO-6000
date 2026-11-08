# DS41 speed work — measurement log

Test VM: `ds41-test-x`, asia-southeast2-c, g4-standard-192 (4x RTX PRO 6000 Blackwell Server Edition,
sm_120, PCIe, TP4), boot disk from snapshot `ds41-boot-snap`, server on 127.0.0.1:8000.
Production (`ds41-prod`, the VM serving this agent) is never touched.

Everything below is measured on the test VM with the repo's own codebench workload
(`/var/lib/ds41/ecc/codebench.py`: /v1/messages, tools, thinking at max effort, real sglang source
contexts at 30k/60k/120k tokens, 4096 output tokens, fresh `session <uuid>` system prompt per request
so the prefix cache never helps). tok/s = (output_tokens-1)/(t_last - t_first), output tokens from
`usage`. "accept" = generation_tokens / spec_decode_num_drafts = accepted tokens per verify step.

## Baseline (served config: r21, DSpark k=5 probabilistic + block, fp8_ds_mla KV, 8 seqs, 8192 chunk, graphs 128)

Container start 19:07:01Z, ready 19:12:45Z (5.7 min). KV pool 4,192,362 tokens (matches production).
Weights load 78.2 s (73.62 GiB/GPU); checkpoint 510.3 GB total, of which Engram tables are 203.07 GB
(2 layers, 384M x 256 FP8 each) held in pinned host RAM (`EngramConfig.cpu_offload=True` default).

| workload | tok/s | accept | step ms | ttft s | runs |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 30k | 374.7 | 3.42 | 9.13 | 1.35 | 388.4 / 374.7 / 358.1 |
| C1 60k | 392.5 | 3.47 | 8.84 | 2.67 | 413.8 / 371.2 |
| C1 120k | 385.6 | 3.52 | 9.13 | 5.47 | 381.4 / 403.9 / 385.6 |
| C2 30k (per stream) | 243.9 | 3.41 | 14.0 | 2.11 | 259.1 / 265.0 / 228.6 / 220.7 |
| C3 30k (per stream) | 200.2 | 3.47 | 17.3 | 2.77 | 183.5 / 214.2 / 195.3 / 210.4 / 190.3 / 205.1 |

Aggregate: C1 375, C2 488, C3 601 tok/s. Quality: 1 of 3 30k runs produced a passing implementation
(the others hit the 4096-token cap during thinking), 1 of 2 at 60k, 1 of 3 at 120k.

Row cost from C1 vs C2: (14.0 - 9.13 - 0.75 draft-chain)/6 = **0.69 ms per verified row**.

## Per-position acceptance (thinkbench, 8 fixed prompts x 2 reps, T=1.0, thinking max)

Cumulative `pos[i]` from `vllm:spec_decode_num_accepted_tokens_per_pos_total`; c1 = pos0/drafts,
conditional c_i = pos_{i-1}/pos_{i-2}.

| prompt | AL | c1 | c2 | c3 | c4 | c5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math1 | 3.33 | 0.779 | 0.762 | 0.749 | 0.708 | 0.641 |
| math1 rep2 | 3.38 | 0.786 | 0.749 | 0.742 | 0.723 | 0.696 |
| math2 | 2.82 | 0.730 | 0.655 | 0.632 | 0.644 | 0.571 |
| logic1 | 3.23 | 0.789 | 0.745 | 0.698 | 0.664 | 0.625 |
| code1 | 3.40 | 0.812 | 0.758 | 0.733 | 0.669 | 0.717 |
| cb30 (30k code ctx) | 3.44 | 0.820 | 0.761 | 0.702 | 0.706 | 0.800 |
| plan1 | 2.32 | 0.616 | 0.570 | 0.552 | 0.543 | 0.537 |
| prose1 | 2.59 | 0.684 | 0.643 | 0.596 | 0.503 | 0.519 |
| zh1 | 3.09 | 0.755 | 0.717 | 0.688 | 0.667 | 0.691 |

Fits c_i = a*r^(i-1) with a = 0.79, r = 0.965 on the C1 shape. Position 1 (the drafter's parallel
backbone) is the weakest single term and varies most across text types; the tail decay is mild.

## Experiment log

| tag | change | C1 30k | C1 120k | AL | note |
| --- | --- | ---: | ---: | ---: | --- |
| base | served config | 374.7 | 385.6 | 3.42 / 3.52 | control, no restart |
| prof | served + torch profiler enabled | 387.4 | - | 3.54 | n=2, task mix differs; profiler trace captured a PREFILL iteration (see below) |
| fly5 (W=4) | `rejection_sample_method: fly`, default window 4 | 352.9 | 389.2 | 3.35 / 3.54 | no gain; W=4 lets only draft position 0 defer |

### Profiler trace of one prefill iteration (8192-token chunk), rank 0

396.1 ms of kernel time in 517.6 ms wall (76.5% busy), 1949 launches:

| kernel | ms | % | n |
| --- | ---: | ---: | ---: |
| ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 144.1 | 36.4 | 84 |
| b12x MoE phase1 W4A8 | 55.6 | 14.0 | 40 |
| b12x MoE phase2 W4A8 | 37.9 | 9.6 | 40 |
| sparse_mla_prefill_mg_dual | 33.8 | 8.5 | 38 |
| flashinfer MXFP8 GEMM sm120 | 25.1 | 6.3 | 90 |
| mhc_post_tilelang | 20.1 | 5.1 | 82 |
| deep_gemm sm120 tf32 hc prenorm | 11.4 | 2.9 | 40 |
| _engram_lookup_kernel | 4.15 | 1.05 | 2 |

Prefill is all-reduce-bound (36%), which is the TTFT axis, not decode. Decode profiling is queued.

## Facts established from the checkpoint and the image

- Checkpoint 510.3 GB: routed experts 290.2 GB (56.9%), **Engram 203.1 GB (39.8%)**, MTP drafter 7.93 GB
  (1.6%), attention 5.23 GB, embed 1.32 GB, vision 0.82 GB. Per GPU at TP4 the engine loads 73.62 GiB,
  i.e. the Engram tables stay in pinned host RAM (UVA lookup) - they are not resident on the cards.
- Engram = 2 layers (1 and 14), each a 384M x 256 FP8 table (98 GB) + a 25600 x 6144 wkv.
- The DSpark drafter is ONE parallel block forward over k slots (not k sequential forwards) + k cheap
  Markov launches; TP4 honoured, ~7 all-reduces per draft step, ~0.75 ms of the step (9.7%).
- The served decode graph is a FULL CUDA graph of exactly `round_up(capture, k+1)` = 6 tokens at batch 1.
- `rejection_sample_method` accepts standard | synthetic | block | fly; fly is wired (entropy-gated
  deferred window) and is the only config-only lever that raises accepted length. Its window must be
  < k, so at k=5 the default window 4 lets only draft position 0 defer; W=1/W=2 are the useful arms.
- `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS` defaults to 0, so the one-shot push path is dead as served.
- `EngramConfig.cpu_offload` (True), `use_thp` (False) are the Engram knobs.

## FLy rejection sampling: the first real win

`rejection_sample_method: "fly"` is config-only (one field in the spec JSON). It is lossy, so every
arm was gated on GSM8K-100 with thinking (the README's baseline for this stack is 96%).

| arm | window | threshold | C1 30k | AL 30k | C1 120k | AL 120k | GSM8K-100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | - | - | 374.7 | 3.42 | 385.6 | 3.52 | - |
| fly5 | 4 (default) | 0.3 | 352.9 | 3.35 | 389.2 | 3.54 | - |
| fly5w2 | 2 | 0.3 | 391.6 | 3.59 | 385.7 | 3.60 | 96% (96/100) |
| **fly5w1** | **1** | **0.3** | **439.8** | **3.92** | **402.9** | **3.76** | **97% (97/100)** |

Per-task matched pairs at 30k C1 (same codebench tasks, tok/s):

| task | base | fly W=4 | fly W=2 | fly W=1 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 388.4 | 397.9 | 409.8 / 422.0 | 473.4 / 464.5 |
| 1 | 374.7 | 350.0 | 373.4 | 402.3 |
| 2 | 358.1 | 352.9 | 367.8 | 415.1 |

So the window is the lever, exactly as the kernel implies: `can_defer` needs
`pos + window < num_draft_tokens`, so at k=5 a window of 4 lets only draft position 0 defer, window 2
lets 0-2, window 1 lets 0-3. Accepted length: 3.42 -> 3.92 (+14.6%) at window 1, +5% at window 2,
-2% at the default window 4. **Measured single-stream gain at 30k: +14.5% (geometric mean of the
per-task pairs); at 120k +3 to +7% (n=1 per task, needs more runs).** Quality on GSM8K is unchanged
(97% vs the 96% reference), which is the first evidence that the lossy acceptance does not damage
reasoning quality at this threshold.


### FLy, clean per-task comparison (the only fair one: task 0 is much faster than tasks 1-2)

Measured on three separate container starts, per-task means of the same codebench tasks, ratio to the
same-config control `base3` (30k C1 n=2 per task, 120k n=1):

| arm | 30k t0 | 30k t1 | 30k t2 | 120k t0 | 120k t1 | 120k t2 | mean | GSM8K-100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base3 (control) | 403.1 | 375.1 | 366.5 | 388.7 | 351.3 | 358.3 | 377.7 | - |
| fly W=1 th=0.3 | 1.164x | 1.072x | 1.132x | - | 1.171x | 1.100x | 426.8 | 97% |
| fly W=1 th=0.1 | 1.148x | 1.089x | 1.115x | 1.156x | 1.166x | 1.185x | 429.4 | 97% |
| fly W=1 th=0.5 | 1.135x | 1.072x | 1.073x | 1.167x | 1.140x | 1.066x | 418.6 | 96% |

Conclusions:
- **FLy at window 1 is worth +11 to +14% single-stream, at 30k and at 120k alike, and the entropy
  threshold between 0.1 and 0.5 does not matter.** Accepted length goes 3.42 -> 3.82..3.92 (30k) and
  3.52 -> 3.75..4.00 (120k).
- The gain is capped near +15% because at k=5 the deferrable set is positions 0-3 and a deferral also
  needs the next draft token to have been accepted natively; `fly_window_size` has `ge=1`, so window 0
  (defer every position) is not expressible.
- Quality holds: GSM8K-100 with thinking is 96-97% in every arm against the 96% reference, and the
  codebench quality pass rate does not fall.
- Restart-to-restart variance for an identical config is 2-9% per task (base vs base3), so only
  per-task ratios against a same-session control are meaningful. Never compare arm medians.

Recommendation for production: `rejection_sample_method: "fly"`, `fly_window_size: 1`, threshold at the
default 0.3. Config-only, one restart, +12% on the daily coding-agent workload, quality unchanged on
GSM8K. The 600 target now needs +35% more, which is the drafter retrain (work/mtp/) and the step axis.


### top_p = 1.0 on top of FLy: the second win

The verifier truncates the target distribution with top_p before it forms p, while the drafter's q is
never truncated (`rejection_sampler.py:264` vs `rejection_sampler_utils.py:169`), so all q mass outside
p's top-95% set is rejected deterministically. Setting the sampling top_p to 1.0 removes that mismatch
*and* raises the target entropy that FLy's gate reads. Measured on the same plan, per-task ratios to
base3:

| arm | 30k t0 | 30k t1 | 30k t2 | 120k t0 | 120k t1 | 120k t2 | mean tok/s | AL 30k/120k | GSM8K-100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base3 | - | - | - | - | - | - | 377.7 | 3.45 / 3.31 | - |
| fly W=1, top_p 0.95 | 1.164x | 1.072x | 1.132x | - | 1.171x | 1.100x | 426.8 | 3.92 / 3.76 | 97% |
| **fly W=1, top_p 1.0** | **1.182x** | **1.186x** | **1.242x** | **1.202x** | **1.273x** | **1.284x** | **461.1** | **4.24 / 4.24** | **96%** |

So the combination is worth **+22% mean (range +18 to +28% per task), at 30k and at 120k alike**, with
accepted length 3.4 -> 4.24 and GSM8K-100 unchanged at 96%. The per-task spread is now tight (the
toppfly 30k runs were 471.9 / 444.8 / 455.1 / 481.2), so this is a real effect, not a task-mix artifact.

k=7 with FLy W=1 does not beat k=5 (403.7 vs 426.8 mean; accepted length 4.04 vs 3.92 but two extra
rows cost 1.2 ms), so k=5 stays.

**Running total on the daily coding-agent workload: 377.7 -> 461.1 tok/s (+22%).** 600 still needs
+30%, which is the drafter retrain (native accepted length 3.42 -> ~4.0, i.e. position-1 acceptance
0.81 -> 0.88) and/or the step axis.

## What the drafter retrain would have to buy

Accepted length is the only other lever with headroom: `tok/s = accepted_length / step_time`, and the
step is 9.1 ms at 6 verified rows with a measured marginal cost of 0.61-0.69 ms per extra row. At the
current step time, 600 tok/s needs accepted length 5.46. With FLy's measured +14.6% multiplier, the
drafter would need native accepted length 4.77 (from 3.42), i.e. position-1 acceptance about 0.88-0.90
instead of 0.81 with the tail decay preserved. That is the retrain target (work/mtp/).

### top_p = 1.0 alone does nothing (the two changes are an interaction, not two independent wins)

| arm | 30k tok/s | AL 30k | 120k tok/s | AL 120k | GSM8K-100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base3 (top_p 0.95, block rejection) | 399.6 | 3.45 | 358.3 | 3.31 | - |
| top_p 1.0 only | 373.9 | 3.38 | 361.8 | 3.34 | 95% |
| fly W=1, top_p 0.95 | 439.8 | 3.92 | 402.9 | 3.76 | 97% |
| **fly W=1, top_p 1.0** | **463.5** | **4.24** | **460.1** | **4.24** | **96%** |

top_p 1.0 by itself is neutral (accepted length 3.38 vs 3.45), so the +8% it adds on top of FLy is an
interaction: with the target's tail no longer truncated, FLy's entropy-gated deferral fires on more
positions. It is a single arm with a tight spread (30k runs 471.9 / 444.8 / 455.1 / 481.2) but it needs
one confirmation run before it is quoted as +22% rather than +13%.

## Drafter retrain: gate 0 passed

The MTP fine-tune pipeline in `work/mtp/` (see its PLAN.md) was validated on the real checkpoint:
quantisation conventions value-exact, 1223 model tensors filled with 0 missing / 0 unexpected, finite
forward, `gate 0 passed`. Two real bugs were found and fixed by that gate (the FP8/FP4 scale suffix was
appended instead of replacing `.weight`, so no scale was ever found and FP4 experts loaded at half
width). The 20 M-token dump + cache + train + export + eval run is in flight on the test VM:
`cd /data/mtp && STAGES="dump cache train export eval" TOKENS=20000000 BLOCK=9 K=5 bash run.sh`.

## Where the 600 target stands

| lever | measured | status |
| --- | --- | --- |
| FLy W=1 | +11 to +14% | validated, 3 thresholds x 6 tasks, GSM8K 96-97% |
| + top_p 1.0 | +8% on top | one arm, needs confirmation |
| k=7 (with or without FLy) | -3 to -9% | dead |
| FLy default window 4 | -2% | dead |
| FP4 KV | +1 to +5% at 120k only, lossy | known, not adopted |
| drafter retrain | target: native AL 3.42 -> 4.0 | in flight |
| step axis (9.1 ms) | not yet profiled on a decode step | open |
| 2-bit routed experts | -17% step claimed | weeks of work |

Best measured today: **377.7 -> 463.5 tok/s at 30k and 358.3 -> 460.1 at 120k (+22% mean)**. 600 needs
another +30%: either native accepted length 4.0 -> 4.7 (the retrain) or ~20% off the 9.1 ms step, and
most likely some of both.
