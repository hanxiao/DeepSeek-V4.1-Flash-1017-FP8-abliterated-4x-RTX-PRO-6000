# Leads: single-stream decode tok/s, DS-V4.1-Flash-1017 on 4x RTX PRO 6000 (r21)

Owner: leads teammate (task-2). Live file: rewrite the table, append detail. Notes in `work/leads/notes/`.
Baseline (ours, measured): **371 / 375 / 376 tok/s** single stream at 30k / 60k / 120k ctx, step
**8.6-9.0 ms** (mean 8.8; the engine's 8.77 and the k-sweep's 9.09 measure different things and both are
real), accepted length **3.40-3.52** tok/step (position-1 acceptance 0.81, geometric decay r = 0.965 on C1),
FP8 KV, 8 seqs, 8192 batched tokens, graphs to 128, DSpark k=5, block rejection, served T=1.0 / top_p=0.95.
Target: **600 tok/s single stream**, same weights, no quality regression, no caching tricks.

Last updated: 2026-11-08 13:05 PT.

---

## 0. What 600 tok/s actually requires

`tok/s = accepted_length / step_time`. Step time grows with the number of verified rows, so the honest
target is a (k, acceptance-curve) pair, not one number.

**Measured acceptance curve (drafter teammate, from the Lead's baseline `work/bench/x-base/think.jsonl`,
13 requests, k=5, T=1.0; the two codebench-like requests = our C1 workload, 1,962 draft steps):**

```
unconditional s_i = [0.8124, 0.6162, 0.4495, 0.3022, 0.2192]
conditional   c_i = [0.8124, 0.7585, 0.7295, 0.6723, 0.7251]
fit c_i = a * r^(i-1):  a = 0.793, r = 0.965   (C1; hard prose: a = 0.63-0.64, r = 0.93-0.96)
AL = 3.40, matching the C1 medians 3.416 / 3.468 / 3.522 at 30k / 60k / 120k
```

So the loss is **not** at position 1 (0.81 is the highest of the five) and there is no collapse: it is a
geometric decay of ~3.5%/position on C1 and steeper on prose. The flat-alpha model is only good for the
first two positions; use the geometric fit.

**The row economics decide everything.** A verified row costs **0.61 ms** (measured from the k-sweep pair;
the mechanism estimate - 22.5 distinct experts/layer x 4.7 MB/rank x 40 layers at 1.45 TB/s = 0.39 ms - is
the lower bound the MoE alone would allow). **A row pays only if it adds more than 0.24 accepted tokens**
(`dA > A * dT / T` = 3.45 x 0.61 / 8.8). At the measured curve, rows 6-7 add 0.05-0.16 because the block is
trained at 5 (`dspark_block_size: 5`) and positions 6+ are out of distribution - that is why k>5 loses today.

Decay-corrected frontier (drafter teammate's numbers, dT = 0.61 ms, base step 8.8 ms), and with FLy on top
(the multiplier is shape-invariant, see below):

| retrained drafter | best k | tok/s | + FLy W=2 (1.27-1.33x) |
|---|---:|---:|---:|
| a=0.88, r=0.99 | 7 | 492 | **610-650** |
| a=0.90, r=0.99 | 8 | 524 | **650-700** |
| a=0.90, r=0.995 | 8 | 557 | 700-740 |
| a=0.92, r=0.99 | 8 | 578 | 730-770 |

**So the recipe for 600 is retrain to a = 0.88 with r = 0.99 and block 7, plus FLy at window 2 - and nothing
else is a prerequisite.** The row-cost work (dT 0.61 -> 0.45), r = 1.0, the 2-bit experts and the ~9% of
step-time shavings are all upside once that lands. The binding constraints are (i) FLy holding quality at
GSM8K >= 96 and (ii) the retrain actually landing a = 0.88 on thinking prose; the arithmetic is no longer the
constraint.

**The drafter's original FLy matrix** (their PLAN.md v4, T=9.1 ms, dT=0.61) applied a flat +8 to +14%
multiplier and concluded FLy replaces one of two requirements. That matrix was the **W=3-4 case** (vLLM's
default window). With the window as the lever (W=1-2) the multipliers are 1.27-1.43x and the conclusion
changes: FLy multiplies with the retrain, so 600 is reachable on the retrain + FLy alone.

**The FLy multiplier is roughly shape-invariant, so FLy and the retrain multiply** (same simulator, g=0.9):
W=1 gives 1.43x on today's shape and 1.38-1.41x on the retrained shapes (a=0.88/r=0.99/k7,
a=0.90/r=0.99/k8, a=0.90/r=0.995/k8); W=2 gives 1.27x today and 1.32-1.33x on the retrained shapes. Higher
alpha removes rejections (multiplier down) while every repair extends a longer chain (multiplier up), and the
two cancel. Consequence: a=0.88/r=0.99/k7 (AL 5.09) at 1.3-1.4x is AL 6.6-6.8, which at 10.3 ms is ~660 tok/s -
**600 arrives with margin from a=0.88 rather than a=0.90, if E1 validates the multiplier and quality holds.**

**Where the loss is, in the drafter's words**: position 1 is the backbone's clean measurement (c1 = 0.81 on
C1, 0.64 on hard prose) and the backbone dominates the tail too, so the retrain objective is **backbone
per-token accuracy, not Markov sharpening** (the rank-256 first-order Markov bias cannot fix a base-logit
error). The tail ceiling is +47-49% (AL = 1 + 5*c1 if every later position were perfect), and raising c1
0.78 -> 0.88 is +9% AL (dAL/dc1 = (AL-1)/c1 ~ 3.0), not +12-15%. Cheap c1 levers, in order: draft-logit
shaping (L7, acts on c1 directly - measure pos[0]/drafts as the primary metric, AL dilutes the effect ~3x);
FLy (does not raise c1, converts rejections); a 4th aux feature = the post-norm LM-head input in the retrain
(targets position 1 by construction, zero extra GPU cost).

**Calibration against the published DSpark numbers, all at T=1.0** (arXiv 2607.05147 Table 1, Qwen3-4B/8B/14B
with DeepSeek's own drafters): accepted length is 3.29-3.72 on open-ended chat, 4.86-5.52 on code, 4.89-6.21
on math. **Our 3.40-3.52 on thinking prose at T=1.0 is exactly the published chat level**, so the drafter is
not broken - prose at T=1.0 is the hard case, and our target is code-level acceptance on prose. The one
reason to think it is reachable: the published drafters are out-of-domain for thinking mode (DeepSpec's
checkpoints are trained on open-perfectblend non-thinking data and its README warns to fine-tune for
thinking), while ours would be trained on the served model's own thinking traces.

Step-time work cannot substitute for acceptance: even the hard floor (4.3 ms DRAM + 1.1 ms PCIe + 0.5 ms
attention = 5.9 ms = 169 steps/s) yields **583 tok/s at today's 3.45**.

Three consequences that prune the lead list:

- **Temperature is not the lever people assume.** The Fraser prompt *with thinking on* runs 358 tok/s at
  the r21 recipe's 102 steps/s = **3.5 tok/step at temperature 0**, against our 3.45 at T=1.0 on the agent
  workload (our README, Fraser section). Prose acceptance is flat in temperature; it is the text that is
  hard to draft. A sweep is still worth 20 minutes with `bench/thinkbench.py`, but expect single digits.
- **Tree drafting (PCTree) is dead here for the same row economics.** Its published gains cost nodes: at
  B=7 it verifies N=32 nodes for +0.93 accepted tokens on Qwen3-4B GSM8K (0.037 accepted per node). Our bar
  is 0.24 per node, and a 4B dense model's node is nearly free while ours is 0.61 ms.
- **Anything that shrinks the step multiplies with the drafter work, so do both.** L5-L10 total ~9%; that
  alone is worth as much as a perfect drafter is worth alone (433 vs 493-578).

---

## 1. Ranked leads

Expected is single-stream, at the served workload, relative to 371-376. Cost is test-VM minutes (or
engineering days) to a decision; a decision can be a negative result. Sorted by expected value.

| # | Lead | Mechanism | Exact knob / commit | Evidence (exact number, workload) | Expected | Cost | Status |
|---:|---|---|---|---|---|---:|---:|---|
| L1 | **FLy rejection sampling (lossy) - and the window size is the lever, not the threshold** | accepts a would-be-rejected draft when the target entropy is above a threshold AND the next W draft tokens are natively accepted; it repairs a position only if the chain after it is right | `"rejection_sample_method":"fly"` + **`fly_window_size` 1-2, NOT the default 4**; `fly_entropy_threshold` 0.3-0.5; `fly_entropy_top_k` 3 | available in the served image (verified on the box); kernel `vllm/v1/spec_decode/fly.py` (deferral requires `pos + W < num_draft_tokens`, so at k=5 W=4 allows only position 0). Simulation on the measured c_i = [0.81,0.76,0.73,0.67,0.73]: W=1 -> **+23 to +44% AL**, W=2 -> +15 to +27%, W=3 -> +9 to +16%, W=4 (default) -> +4 to +7% | **+23 to +44% at W=1, +15 to +27% at W=2** (lossy; verify quality) | 40/arm | untested, Lead informed |
| L2 | **Better MTP drafter (train): a 0.88-0.90, r 0.99, block 8-9** | position-1 rate is fine (0.81), the loss is the decay r 0.965 -> 0.99 and the out-of-distribution block tail | task-3; code deepseek-ai/DeepSpec (MIT, `block_size` is a config field); loss TVP arXiv 2611.00588; SpecOPD arXiv 2611.02846 | DeepSpec config: block 7, 5 draft layers, 0.1 CE + 0.9 TV, lr 6e-4, batch 512, 10 epochs. Frontier in §0: a=0.90/r=0.99 -> 525 tok/s at today's step, 566 with the step work | +40 to +50% | GPU-days + target cache | task-3 |
| L3 | **Step work bundle (env vars + backports)** | MOE_PREQUANT off the critical path; P2P all-gather; DSpark fused-Markov inverse CDF; sampler micro-opts | `-e VLLM_MOET_MOE_PREQUANT=1 -e VLLM_MOET_P2P_AG=1`; backport vLLM `3d18166d4`, `a512f0e72`, `339017c3e` | Moet Dockerfile step 34: "27.6 us per step leave the critical path, period 9,885 -> 9,803 us"; step 25: logits AG "54.6 vs 62.1 us"; our Markov loop ~88 us/step | +1.2% | 30 for the env vars, 90 for the backports | untested |
| L4 | **Move target lm_head + logits AG + sampler into the FULL graph** | ~0.2-0.25 ms/step of eager launch-latency work today | custom; they are outside every graph | `work/engine/FINDINGS.md` §2.4: lm_head 104-136 us, AG 55-62 us, sampler 30-60 us | +2.5% | 1-2 eng-days | untested |
| L5 | **Fuse the mHC push into the producing kernel** | 80 boundaries x 5.7 us push is its own launch; issue it from wo_b / the MoE epilogue | custom kernel | `work/engine/FINDINGS.md` §0 L3: 5.7 us x 80 = **-0.46 ms, -5%** | +5% | eng-days | untested |
| L6 | **FP4 KV record** (`nvfp4_ds_mla`) | halves KV bytes per decode step; costs 8% of prefill | `KV_DTYPE=nvfp4_ds_mla INDEXER_KV_DTYPE=mxfp4` | ours, measured: r21 k5 FP4 KV C1 **374 / 369 / 393** vs served 371 / 375 / 376; KV 10.75M vs 4.19M | +4.5% at 120k, -1% at 30k | 40 | measured by us |
| L7 | **Shape the draft distribution (q support / calibration)** | p is top_p=0.95-truncated, q (draft) is not; and q may be over-dispersed relative to p, which costs acceptance at position 1 | `dspark_draft_topk`: the mechanism is in the image and cache-correct (`_sample_sequential_topk`: non-candidates -> -inf, then "the normal dense sampling and rejection paths consume that truncated distribution unchanged"; `_sample_logits` always goes through `gumbel_sample`), **but it is gated**: `speculative.py:1545-1552` admits only `Qwen3DSparkModel`/`Qwen3OmniDSparkModel` and ours is `DSparkV41DraftModel`, so a config arm raises ValueError; the field docstring also says "Requires draft tensor parallel size 1" and we are TP4, so `topk` runs per vocab shard and the effective support is up to 4x k. **Needs the 2-line gate patch, and no route avoids it**: the image's chain is `dspark_draft_topk = self.dspark_draft_topk` -> if None, `getattr(hf_config, "dspark_draft_topk", None)` -> `if dspark_draft_topk is not None:` -> architecture check -> raise. So writing the key into the checkpoint's `text_config` (the flattening in `DeepseekV41Config` makes it visible) still hits the raise; `--hf-overrides` never reaches the draft config (`compose_draft_hf_overrides`: "Dict overrides are target-specific key patches and are not applied to the draft"). The patch is: add `DSparkV41DraftModel` to the allowed set at `speculative.py:1543-1552` (one file, overlay-mounted, no image rebuild, one boot). A q-temperature/logit scale would be the fuller version and needs a patch | vLLM: draft logits cached pre-temperature, scaled in-kernel; top_p/top_k applied to p only (`work/engine/FINDINGS.md` §2.4) | +2 to +8% (sign is empirical: truncation removes q mass where p is small but renormalises q upward inside the support) | 60 + a rebuild/overlay | untested, gated |
| L8 | **Drafter input redesign: sequence-wise target hidden states, more layers** | DLite ablation: sequence-wise beats feature-wise, 7 -> 14 hidden states +5%, then saturates | arXiv 2610.23787 Table 5 | avg tau 3.04 -> 3.30 (+8.5%) sequence-wise; 3.14 -> 3.30 for 7 -> 14 states (Qwen3-8B). Ours: 3 states, feature-wise | +5 to +10% tau in the retrain | 0 (rides L2) | sent to drafter |
| L9 | **On-policy distillation for the drafter (SpecOPD)** | "can push well-trained drafters beyond acceptance-length plateaus" | arXiv 2611.02846 | training recipe; the plateau it targets is exactly ours (well-trained head stuck at a=0.79 on in-domain prose) | rides L2 | reading | sent to drafter |
| L10 | **PCTree: tree drafting on the existing Markov head, no retraining** | converts DSpark's linear block into a tree, scores children per concrete parent with the pretrained Markov head | arXiv 2608.02123 | "+3.1% to 29.5%" speedup over matched DSpark at B=7, but it buys that with nodes: N=32 for +0.93 accepted on 4B GSM8K = 0.037/node vs our 0.24/node bar | +1 to +5% | 2-5 eng-days | dead on row economics |
| L11 | **Hybrid Spark: DSpark + n-gram per request in one engine** | runs a model-based and a model-free drafter on every step, gives each request the better chain, inside adaptive verification, on GPU | arXiv 2611.05611 (paper only, no code yet) | "**1.10-1.62x** the output throughput of DSpark on **code editing** at batch size 4 ... **0.97-1.01x** on chat at temperature 0" | 0 to +10% here (thinking tokens dominate C1) | days | untested |
| L12 | **FlashInfer 0.7.3 Cake SM120 NVFP4 sparse-MLA decode** | allocation-free cake decode kernel for DSV4 on SM120 | `cake_sparse_mla_sm120_dsv4_nvfp4_decode` | FlashInfer 0.7.3 docs + issue #6594; our attention is 7.07+2.1 us x 40 = 0.37 ms | +1 to +2% | 1 eng-day | untested |
| L13 | **Use the confidence head as a diagnostic** | the checkpoint ships `mtp.2.confidence_head.proj.weight`; vLLM loads it and runs it only with adaptive verification | `enable_adaptive_verification: true` for one test-VM run, read `draft_token_confidence_probs` | `vllm/models/deepseek_v41/nvidia/dspark.py:147-152, 356-361`; `speculator.py:106-111`; tensor in `model-00046-of-00048` | 0 (calibration vs capacity) | 30 | untested |
| L14 | **k=3 for prose-only** | shorter step, smaller drafter graph | `SPEC_TOKENS=3` | Moet: "prose unchanged single-stream", +13% at C4/C8, **code -24% single-stream** | 0 on our mixed workload | 40 | measured by Moet |
| L15 | **Serve a lower temperature** | alpha vs T | `--override-generation-config` temperature 0.8 / 0.7 | Fraser thinking-on at T=0 = 3.5 tok/step vs our 3.45 at T=1.0 -> flat on prose | +0 to +10% | 20/pt | tooling ready |
| L16 | **D3D: decoupled drafting so drafting overlaps verification** | would take the drafter's 0.8 ms off the step's critical path and let a bigger drafter pay for itself | arXiv 2611.06425 | abstract only | +9% step if it works | watch | untested |
| L17 | **`dspark_scheduler` confidence-scheduled verification** | trims verify width from calibrated confidence | **NOT in the served image** (all 41 SpeculativeConfig fields listed on the box; the only dspark one is `dspark_draft_topk`) | it lives in the v0.24.0 patch file in the Moet clone, not in our nightly image; Moet measured the idea at -14% code single-stream anyway | -14% | 0 | dead |

L1 (FLy) and L2 (retrain) are the only leads that move acceptance. L1 is one config change and needs no
training, so it should be measured before the training budget is committed: if FLy buys 10-20% of accepted
length, the retrain target drops and the whole plan gets cheaper. L3-L6 are additive, independently
testable step-time shavings worth ~9% combined, and should run in parallel with the drafter work.

---

## 2. Answers to the Lead's open questions (primary sources)

### (a) Can vLLM spec decode combine two draft methods (ngram + dspark)? Does the r21 image support `method: ngram`?

**In the r21 base: no.** `SpeculativeMethod` is a single `Literal`; one `method` per engine
(`vllm/config/speculative.py:73-83`). No mixture-of-drafters plumbing exists in df3bd094.

**But a working design was published on 2026-11-03 and it is instantiated in vLLM**: Hybrid Spark
(arXiv 2611.05611) "runs a model-based and a model-free drafter on every step and gives each request the
better of their chains, within adaptive verification and entirely on the GPU ... Instantiated with DSpark and
an n-gram drafter in vLLM, achieves 1.10-1.62x the output throughput of DSpark on code editing at batch size
4 ... while staying at 0.97-1.01x of it on chat at temperature 0." So combining them is a real design, not a
dead end; it just is not in our image and there is no code release yet. See L16.

**`method: ngram` is present in the base, but it forces the V1 model runner, which kills DSpark and the
Moet V2 decode path.** Chain, in `work/engine/src/vllm` at df3bd094:
- `ngram` / `ngram_gpu` are in `SpeculativeMethod` (`config/speculative.py:75`, `:68`), and a GPU
  speculator exists (`v1/worker/gpu/spec_decode/ngram/speculator.py`: Triton suffix scan, graph-captured).
- `_get_v2_model_runner_unsupported_features()` lists `"ngram/ngram_gpu speculative decoding"`
  (`config/vllm.py:2048-2049`), so `use_v2_model_runner` returns False with a warning (`:734-745`).
- `_get_v1_model_runner_unsupported_features()` lists `"dspark speculative decoding"` (`:3076-3077`):
  DSpark runs only on MRv2.
- So ngram and dspark cannot both be on, and an ngram run is a stock-V1 decode step without the sm_120
  kernels this deployment exists for. The Moet r21 patch does not touch ngram (all 7 mentions are base
  context; it only adds `"dspark"` to the Literal and the runner lists).
- Confirmation run: `--speculative-config '{"method":"ngram","num_speculative_tokens":5}'`, then grep the
  log for "Model Runner V2 does not yet support". Expected value on max-effort thinking prose: low.

### (b) Any trained drafter / MTP head for DS-V4.1-1017 published since 2026-10-25?

**One, and its card says it does not help us.** `Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft`
(2026-10-29, `fp8/mtp-ft.safetensors`, 7.93 GB): "A fine-tune of the three MTP layers (`mtp.0` to
`mtp.2`) ... Everything else in the drafter was trained on the served model's own outputs. The gain is
small: more accepted draft tokens in non-thinking mode, **about flat in thinking mode at temperature 1**."
Our workload is thinking at max effort, T=1.0. Skip.
Everything else is an extraction of the shipped drafter, not new weights: `unsloth/...-GGUF/dspark/`,
`sirfyyn/...-DSpark-FLE-GGUF`, `Baekpica/...-VQ-GGUF`, `fraserprice/DeepSeek-V4-Flash-DSpark` (V4-Flash,
not V4.1), `mlx-community/...-MTP-bf16`. The SubconsciousDev drafter was already measured at 2-6% slower.

### (c) Did 0xSero publish the 675.9 tok/s recipe?

**No reproducible recipe; his own repo contradicts the number for single stream.** Claim: X post
2026-10-22, "Deepseek-v4.1-flash-1017 @ 675.9 tok/s c1 Recipes coming"
(`x.com/0xSero/status/2113067168954982911`). His repo (now
`sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000`, SGLang, DSpark block 5, Engram on NVMe, 4x
RTX PRO 6000 at 275 W) measures in its newest local sweep (8,192 forced output tokens, synthetic
repeated text): **C1 total decode 196-232 tok/s** at 512 to 500k input tokens; C8 599.7-747.7 aggregate.
Nothing reaches 675.9 single stream. Its own 2026-09-19 head-to-head vs Moet had it at 173-288 C1.
Treat 675.9 as a different workload, a burst peak, or marketing until a recipe lands.

### (d) What exactly does `VLLM_MOET_P2P_AR_FUSED_MAX_ROWS` bound, and what happens above it?

Source: `tools/dsv41_sm120/p2p_ar/p2p_allreduce_sm120.py` in the r21 tree (`fused_max_rows()` :226-228,
`push_for_fused()` :231-242, `mode()` :194-206) and step 24 in `tools/dsv41_sm120/README.md:35`.

It bounds **token rows in one boundary all-reduce tensor** - the `M` of the `[M, 5120]` bf16 sublayer
boundary that feeds `mhc_post_norm`, 80 of them per decode step (2 per layer x 40) plus the Engram and
final boundaries. Not bytes, not calls, not a queue depth.

- `M <= FUSED_MAX_ROWS` (also `M <= M_MAX = 64` and `M` in the start-up NCCL-order table): step 24's
  **fused** path. No all-reduce kernel of its own; the push is issued and `mhc_post_norm` waits on the
  three peers' flags, adds the four partials in NCCL's own order (bit-identical), then post-mix.
  Measured 5.7 us push + 7.7 us fused kernel vs NCCL 11.8 + post_norm 3.9.
- Above it: `mode()` falls through to step 23's **two-shot P2P** (reduce-scatter + all-gather, 737 KB
  egress for [48, 5120]) while `M <= VLLM_MOET_P2P_AR_MAX_ROWS` (code default 64), else **NCCL ring**.
- Row counts whose NCCL order was not reproduced bit-for-bit at start-up also fall back to NCCL.
- r21 serves **8** on purpose: `P2P_AR_FUSED_MAX_ROWS` 16 -> 8 is recorded as "C2 -1.3%", and the fused
  path wins at 6 rows (12.5 vs 14.4 us) but loses at 12 (20.0 vs 18.9) and 16 (24.9 vs 21.4). Our k=5
  verify step is 6 rows, k=7 is 8, so 8 is exactly right; nothing to gain at single stream. At C2+ some
  boundary tensors exceed 8 rows and take the two-kernel path, which is the measured-intended behaviour.

### (e) Any sm_120-specific decode kernel work in vLLM after nightly df3bd094 that is NOT prefill?

**Almost none, and none of it is worth >1%.** Diffed `work/engine/src/vllm` (HEAD = df3bd094, 2026-10-20)
against origin/main `34b780fe` (2026-11-08, 1,490 commits). Three commits mention SM120:

| commit | what | effect on our decode |
|---|---|---|
| `2feae41cd` #61821 | bias in the SM120 NVFP4 CUTLASS GEMM epilogue | none (not our expert path) |
| `7ecc83a98` #60762 | declare 64-token kernel blocks for SM120 sparse backends | correctness declaration, no speed |
| `dbcf56882` #58305 | compact FP8 KV rows for **NoPE** MLA models on SM120 | our model has RoPE |

The substantial DSv4.1 decode work in that window is gated to other hardware: `c58e7f6d0` #63921
"latency_specialized decode" (`--additional-config '{"decode_backend":"latency_specialized"}'`)
**requires GB300/SM103, TP4, NVLink multicast** (`vllm/models/deepseek_v41/nvidia/specialized_decode/
AGENTS.md`); `4b01cbad1` #62633 swap-AB decode attention is "for DeepSeek V4.1's small decode steps
(**SM100**)" on tcgen05, absent on sm_120. Portable DSpark work is small: fused Markov by inverse CDF
(#64442), Markov-bias -> Gumbel fusion (#64476), resample vocab skip (#65100), DSpark SWA MLA draft
layers (#63795, Triton MLA, not our FlashInfer path), batch-sharded sequential block (#63886, batch>1).
**Verdict: do not spend VM time on a nightly rebase for decode.**

### (f) What the served image actually supports (verified on the box, not from source)

`sudo docker exec ds41 python3` on the production container, 2026-11-08. The image is the df3bd094 base plus
the Moet patch set, and the two differ exactly where it matters:

- `RejectionSampleMethod = Literal['standard', 'synthetic', 'block', 'fly']` - **FLy is available** (L1).
- All 41 `SpeculativeConfig` fields include `fly_window_size` / `fly_entropy_threshold` (0.3) /
  `fly_entropy_top_k` (3), `dspark_draft_topk`, `num_speculative_tokens_per_batch_size`,
  `synthetic_acceptance_rates/length`, `prompt_lookup_min/max`, `suffix_decoding_*`,
  `enable_adaptive_verification`, `draft_sample_method`, `rejection_sample_method`.
- **No `dspark_scheduler` / `dspark_per_request` / `dspark_pad_to_bucket` / `dspark_confidence_threshold` /
  `dspark_budget_frac`**: those live in `patch/vllm-moet-v0.24.0.patch` in the Moet clone (the v0.24.0 line),
  not in the nightly base we serve. Moet's confidence scheduler is therefore not an option for us, and the
  confidence head can only be exercised through `enable_adaptive_verification`.
- `dspark_draft_topk` is a SpeculativeConfig field and is unset; it is the split-vocabulary top-k of the
  **greedy** draft path, so it does nothing for our probabilistic drafting.

---

## 3. Verified dead ends (do not spend VM time)

| Dead | Number | Source |
|---|---|---|
| Moet r22-r28 for decode | r26b: "decode unchanged (129 steps/s)"; r23: "Fraser bench unchanged (688 tok/s median, 746 best, 129.0 steps/s over 40 reps)" | Moet `CHANGELOG.md` |
| r21-mxfp8 (steps 38-39) | Fraser 679 / 740 at 128.4 steps/s vs r21 680 / 741 at 127.8 | Moet `CHANGELOG.md` |
| k > 5 at today's alpha | k6 346/343/359, k7 329/356/345 vs k5 353/379/380 tok/s | our README |
| Adaptive verification on sm_120 | code 313 -> 273 tok/s (-14%) single stream | Moet `docs/dsv41-sm120-port.md:360` |
| SubconsciousDev drafter | -1..3% reasoning, -4% code, 2-6% slower | our README |
| Avifenesh drafter-ft | "about flat in thinking mode at temperature 1" | its HF card |
| 0xSero 675.9 c1 | his repo: 196-232 tok/s C1 | sybil-solutions README |
| Moet step 35 router+topk fusion | critical path did not move (8,271 -> 8,260 us) | Moet Dockerfile step 35 |
| Moet step 37 MoE mega | "measured slower in the served graph, opt-in build, OFF" | Moet git `d464ddf` |
| ECC off | 114 steps/s unchanged, same KV pool | our README |
| Power / clocks | 285-318 W of 600 W at 2400-2415 of 2430 MHz, `clocks_event_reasons` 0x0 | our README |
| Card memory clock | only supported value 12481 MHz; reference host runs +5000 MHz -> its 127.8 steps/s is unreachable here | `work/engine/FINDINGS.md` §0 L1 |

## 4. Research queue (next)

Done since the first draft: PCTree node economics; DeepSpec (official DSpark training code, config, loss);
the checkpoint's drafter config, the confidence head it ships, and FLy's availability in the image;
DLite / Hybrid Spark / Parallel Reconsideration / SpecOPD / D3D read and costed against our row budget;
the served image's 41 SpeculativeConfig fields listed on the box; MRV2 graph sizing confirmed with the
engine (decode FULL graphs are multiples of k+1, so k=5 is a 6-row graph).

Open, in priority order:

1. **FLy arm results** (L1). Everything else on the drafter axis is gated on whether a lossy sampler can buy
   accepted length for free. Gates: AL, tok/s, GSM8K-100, greedy agreement.
2. Per-request breakdown of the acceptance curve at 30k vs 120k (the drafter's curve is aggregate over two
   codebench-like requests; if the decay is worse at 120k that is a separate lever - local positions).
3. TVP loss form (arXiv 2611.00588) vs DeepSpec's 0.1 CE + 0.9 TV, for the retrain; and SpecOPD's on-policy
   recipe. Both feed task-3 and are the drafter teammate's call.
4. Cost the DeepSpec path for our target: the target cache is the bottleneck ("roughly 38 TB for the default
   Qwen3-4B setting"; our box decodes at ~370 tok/s aggregate). Cheapest source is the model's own
   thinking-mode outputs from production traffic.
5. FlashInfer: does `cake_sparse_mla_sm120_dsv4_nvfp4_decode` accept an FP8 KV record or only NVFP4
   (decides whether L12 needs the FP4 KV recipe).
6. The residual step gap: our step is 20% slower than the Moet reference host at the identical k=7/FP4-KV
   config (102 vs 127.8 steps/s), of which the memory clock explains 14-16%. If the remaining ~5% is PCIe
   topology (our 2 switch pairs + NUMA hop) it is not tunable; if it is VM overhead it might be. One profile
   on the test VM to check the mHC boundary latency against the 13.4 us the Dockerfile measures on the
   reference host.
