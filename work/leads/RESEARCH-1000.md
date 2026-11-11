# 1000 tok/s exact: what the literature says, and the arithmetic that decides it

Written 2026-11-09 for Han. Owner: research-drafter teammate.
Target: **1000 tok/s single stream, same weights, exact rejection (no FLy)**, from the measured exact
baseline 377.7 / 358.3 tok/s at 30k / 120k (accepted length 3.45 / 3.31, step 9.1 ms).
Labels: **[us]** = measured in this repo; **[them]** = measured by the cited source; **[me]** = my
estimate from the two, arithmetic shown so it can be checked or killed.

---

## Verdict

**1000 is not an acceptance problem, it is a row-cost problem.** With our measured marginal verify row
at 0.61 ms, 1000 tok/s requires `AL(k) = 6.05 + 0.61k` (see §2). At k=5 that asks for AL 9.1 when the
maximum is 6. **No realistic k reaches 1000 at today's row cost.** Even an oracle drafter that accepts
every row gets 823 tok/s at k=8 and 1,053 at k=15 — the k=15 case is the only one that clears 1000 and it
needs 15 of 15 rows accepted [me]. So every "get a better drafter" plan, including the
DSpine/DFlash2/JetSpec++ class, tops out near **470 tok/s** on our engine [me] — that is the number that
reframes the project.

Four leads, ranked by expected value. They multiply; no single lead clears 1.7x.

| # | Lead | Mechanism | Multiplier | Eng-days | Risk |
|---:|---|---|---|---:|---|
| **R1** | **Laddered (chunked) verification** | verify a short prefix, extend only when every row in the chunk was accepted. Rows computed per step drop from k+1 to ~AL, which is exact by construction (acceptance is prefix-ordered) and needs no retraining | **1.4-1.55x at k=15**, 1.1x at k=5 [me] | 3-6 | the second chunk is a second target pass in one step: two graph shapes, KV written twice for the tail, `num_accepted` read on the host between chunks. vLLM MRv2 scheduler surgery |
| **R2** | **2-bit routed experts** | halves the row-dependent expert bytes: c 0.61 -> 0.31 ms, T(5) 9.1 -> 7.3 ms [us] | **1.24x at k=5, 1.5-1.7x at wide k** [us]/[me] | 10-20 (re-quantise 14,208 experts + b8x kernel) | quality gate on GSM8K-100 >= 96; it is a numeric change to the served weights, not a decoding change |
| **R3** | **Block-16 parallel drafter + path selector** | DFlash2's recipe: 5-layer parallel drafter trained at block 16, plus a 2M-param adjacent-pair selector over top-16 candidates. AL 3.45 -> 5.0-6.2 at T=1.0 thinking [them]/[me] | **+45-80% AL**; 1.25x tok/s alone at our row cost, 1.6-1.7x with R1+R2 | 4-6 + one training run (~2 days) | the last retrain was a negative result at block 5; this one changes the block size, which is the actual lever |
| **R4** | **Draft/verify pipelining (foresight, JetSpec++ style)** | run the drafter concurrently with the target's verify so its 0.75-0.8 ms leaves the critical path, and spend the freed window on extra refinement passes | **1.10-1.12x** step, plus free AL from extra passes [them]/[me] | 5-10 | hardest item: needs async scheduling in MRv2, and our 4 GPUs are all busy with TP4 so the overlap has to come from idle SMs inside a DRAM-bound step |

**Stack, computed on the step model rather than by multiplying multipliers** [me]: R2 takes B 6.05 -> 5.75
and c 0.61 -> 0.31; R4 and the two already-designed step items (mHC push fusion -0.46, dense-GEMV
short-circuit -0.90 [us]) take B to **3.64 ms**; R1 makes the rows paid equal AL. Then

```
T = 3.64 + 0.31 * AL      =>   1000 tok/s at AL 5.40   (T 5.31 ms)
```

**AL 5.40 is inside DFlash2's published T=1.0 thinking band (5.20-5.97) [them].** That is the whole plan:
the step stack (R2+R4+two known items) plus the ladder (R1) plus a block-16 drafter at the published T=1
frontier (R3). The single highest-EV lead is **R1**: the only one that attacks the row axis without touching
weights, without a training run, and it is the multiplier that makes R3's wide block affordable at all.
Do R1 and R2 in parallel, R3 in the background (it is the long pole), R4 last.

---

## The arithmetic

Measured step model [us]: `T(k) = B + c*k` with `c = 0.61 ms` (marginal verify row, from the k-sweep
pair) and `B = 6.05 ms` (the k=0 intercept implied by T(5)=9.1 ms; extrapolated from two points, not
measured directly). Then

```
tok/s = 1000 * AL(k) / (6.05 + 0.61 k)      =>   1000 tok/s  <=>  AL(k) = 6.05 + 0.61 k
```

**Frontier, exact decoding, today's step cost** (`AL_est` = my estimate for a properly trained block-16
drafter, from DFlash2's measured flat 0.85 conditional acceptance at T=1, i.e. `1 + 5.667(1-0.85^k)` [them]->[me]):

| k | rows | T ms [us] | AL needed for 1000 | AL max = 1+k | AL_est block-16 | tok/s at AL_est | AL today (block 5) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 6 | 9.10 | 9.10 | 6 | 4.15 | 456 | 3.45 [us] |
| 8 | 9 | 10.93 | 10.93 | 9 | 5.12 | **468** | ~3.6 (k=7: 3.55 [us]) |
| 11 | 12 | 12.76 | 12.76 | 12 | 5.72 | 448 | - |
| 15 | 16 | 15.20 | 15.20 | 16 | 6.17 | 406 | - |

**The one 2026 paper that models exactly our regime** is "Why Speculative Decoding Works Better Than
Predicted on Sparse MoE Models" (arXiv 2610.38981, NeurIPS 2026 main track, 10 MoE models 30B-1T, 8-512
experts) [them]: verification cost is "governed not only by activated-expert bytes, but also by temporal
overlap across draft tokens and GPU execution dynamics", L2 absorbs repeated reads of co-routed experts, and
they predict the unique-expert count from routing-overlap statistics with **<4% error** via a
maximum-entropy IPF model. Two uses for us: their IPF is the right estimator for our row cost at any k (we
have the routing traces, so we can fit `c(k)` instead of assuming it constant), and their headline finding is
counter-evidence to our own measurement — at low batch size on H100 they find MoE layers are a *smaller*
share of runtime than of weights, while our MoE is 3.0 of 9.1 ms [us] for a model that is ~99% expert
weights. The likely reason is that our experts are FP4 on 4 PCIe cards with no L2 sharing across ranks, so
the L2 effect they rely on does not exist for us. Their companion bound (arXiv 2609.22156) is the rigorous
form of our row rule: speculation-budget selection as a Stochastic Shortest Path, with the result that
"rejected candidates form a strict linear boundary" and the local necessary condition is marginal cost vs
expected progress [them, abstract].

Three facts fall out:

1. **The optimum is k ~ 7-8 at c=0.61, and its ceiling is ~470 tok/s.** A DFlash2-class drafter on our
   engine is worth +25%, not +160%. This is the same conclusion the local staircase note reached from the
   other direction (`work/leads/notes/moe-step-staircase.md`: k*=8 at a=0.86) — now with a published AL
   curve instead of a geometric fit.
2. **Wide blocks are uneconomic at 0.61 ms/row.** AL grows logarithmically in k, T grows linearly. That is
   why DSpine's headline AL, dropped in unchanged, *regresses* us: 4.82 accepted at 15.2 ms = 317 tok/s
   [them]->[me]. See §3.
3. **1000 needs AL 5.3-8.3 depending on how much of the step stack lands, and only once the rows paid
   equal AL.** With R1 on and R2 alone (`B 5.75, c 0.31`) the requirement is `AL = 5.75/0.69 = 8.3`; adding
   R4 (-0.75 ms) gives 7.3; adding the mHC push fusion (-0.46 ms [us]) 6.6; adding the dense-GEMV
   short-circuit (-0.90 ms [us]) **5.3** [me]. The published T=1 thinking frontier is 5.2-6.0 [them], which
   is why every item in the stack is load-bearing and none is optional.

**Which lead attacks which axis**

| axis | today | R1 ladder | R2 2-bit | R3 block-16 | R4 pipeline | other |
|---|---|---|---|---|---|---|
| c (ms/row) | 0.61 | **rows 16 -> ~8** | 0.61 -> 0.31 | - | - | - |
| B (ms, 1 row) | 6.05 | - | -0.30 (rest of the 1.8 goes to c) | - | -0.75 | mHC fusion -0.46 [us], dense GEMV -0.9 [us] |
| AL | 3.45 | - | - | **3.45 -> 5.0-6.2** | +refinement passes | - |

---

## DSpine and the parallel-drafter line

### DSpine, exactly (arXiv 2609.36173, Tencent YouTu Lab + SJTU + Xiamen; v1 2026-09-28, v2 2026-10-10)
Paper: https://arxiv.org/abs/2609.36173 · project page: https://dspine-project.github.io/

What it is [them]: a **parallel drafter** on DFlash's skeleton — 5 Transformer layers, block-causal
attention, target features from 5 target layers injected as K/V at every layer, target embedding + LM head
shared and frozen. It drafts a block of 1 anchor + **15 candidates in one forward pass**. Two mechanisms:
**adjacent causal injection** (after every layer, each position receives a gated message carrying its
predecessor's *predicted feature*, so the causal chain unfolds over depth while positions update in
parallel; ~33M params, ~3% of the backbone) and **predecessor-conditioned decoding** with a **transition
cache** (top-16 candidates per position, all 16x16 = 256 adjacent-pair scores precomputed in one pass, then
selection is a walk over a lookup table). Correction to the brief: the "256-way transfer cache" is the
16x16 transition table, not 256 candidate paths.

Numbers, Qwen3-8B/4B, all drafters retrained on the same ShareGPT data, same 5-layer/block-16 budget
[them]: 7-benchmark mean acceptance length **3.77 (DFlash) -> 4.82 (DSpine), +27.8% at T=0**; leads the
strongest baseline by 10.0% greedy (DFlash2) and 12.8% under sampling (DSpark). SGLang throughput
+23.3% over DFlash, +11.1% over DFlash2, up to 4.0x over AR. A DSpine round costs 7.14 ms vs DFlash2's
7.19 ms; injection adds <=0.15 ms/round; transition-cached decoding 0.36 ms vs DSpark's 0.88 ms and
Domino's 0.97 ms. Verification is 72-76% of round time for every method.

Qwen3.8-27B support was added Nov 2026; the 27B figures in circulation (**~2B draft params, block 16, 8,192
sliding window, GSM8K 7.20 vs DFlash2 5.69, WorkBuddy-Office 5.63 vs 4.24, SWE-bench Pro 4.99 vs 4.19,
2.7-3.8x end-to-end on agent benchmarks, 33 -> 156 tok/s on an H100**) are **vendor numbers relayed by a
third-party blog**, not in the arXiv v2 (https://www.mindstudio.ai/blog/dspine-speculative-decoding-qwen).
Treat them as abstract-level.

**What transfers to us:** the mechanism, and only the mechanism. It is a *drafter-side* change; the verify
side is untouched, so it costs nothing on the axis that actually binds us. The transition cache is the same
idea as DFlash2's selector (score all adjacent pairs once, then walk), and our image already contains a
graph-captured implementation of that walk (`_selector_walk_kernel`, see R3). The block-causal attention +
"predict the predecessor's feature, not its token" trick is directly usable on a DSpark-architecture head.

**What does not transfer:** the numbers. All headline numbers are **T=0** on dense 4B/8B targets with
thinking disabled, and a dense target has no per-row expert traffic at all — DSpine's own round time is
72-76% verification, same as ours, but its verification cost is flat in rows because the weights are read
once regardless of block width. On our engine the *only* thing that matters is that each extra row is
0.61 ms. **Concretely: DSpine's 4.82 accepted, at our k=15 step of 15.2 ms, is 317 tok/s — a 16% regression
against today's 377.7** [them]->[me]. Plugged into the ladder (rows ~ AL) it is 6.05 + 0.61*4.82 = 9.0 ms
= 536 tok/s [me]. With R2 as well, 5.75 + 0.31*4.82 = 7.2 ms = 665 tok/s [me]. That is the whole story of
this section: **the parallel-drafter line is a 1.4-1.8x on our box, not a 3x, and it is 1.4x only if the row
cost is fixed first.**

### The DFlash / DFlash2 line (the one that actually gives AL per row)
- DFlash (arXiv 2602.06036, Jan 2026, github.com/z-lab/dflash): block-diffusion drafter, whole block in one
  forward pass. Now in SGLang, vLLM, TensorRT-LLM, llama.cpp; NVIDIA claims up to 15x on Blackwell (batched,
  marketing).
- **DFlash2 (Inco AI, 2026-08-18, https://inco.ai/blog/dflash2/)** is the relevant one, because it publishes
  a **matched-comparison table at exactly our sampling config: thinking enabled, T=1.0, top_p 0.95, top_k 20,
  presence penalty 1.5, lossless rejection sampling** [them]. Qwen3.5-4B, per-request mean AL:
  **MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash2 5.97**. Qwen3.8-27B at block 8: MTP 4.28, DSpark 3.62,
  DFlash2 4.80. Muse Glimmer at block 16: DFlash 4.44, DSpark 4.48, DFlash2 5.70. Serving: 2.7-3.4x over AR
  at batch size 1 in SGLang.
- Its two mechanisms are cheap and both are *drafter-side*: a **path selector** over the top-16 candidates at
  each position (2.0M params, **+0.6% cycle latency**, +0.34 AL at T=0 / **+0.47 at T=1** over DFlash, and it
  beats DSpark's sequential correction with 40x fewer params and 16x less latency overhead) and a **two-tap
  dynamic convolution** (16.5M params, +0.7% latency) that recovers most of what 10 extra layers buy, because
  **suffix decay is a local problem**. Together +1.3% cycle latency for +1.05 AL (21%).
- The headroom number worth remembering: on Qwen3-4B GSM8K, DFlash's top-1 is right 85.4% of the time at
  position 0 but the right token is in its **top 16 99.5%** of the time, and still 87.8% at position 6; an
  oracle selector lifts AL from **4.27 to 6.79** [them]. That gap is pure selection, and selection costs no
  extra verify rows. **It is the cheapest accepted token in the literature for a verifier like ours.**
- Conditional acceptance at T=1 on MATH-500 (Qwen3.5-4B): DFlash2 holds **85-86% from position 0 to 14**,
  DSpark decays 87.2 -> 79.9, DFlash 88.4 -> 77.5, MTP 84.6 -> 77.9 [them]. Our curve is
  c = [0.812, 0.759, 0.730, 0.672, 0.725] [us]. We are already at DSpark-like decay; the flat-0.85 shape is
  what a selector+conv buys.

### JetSpec++ (arXiv 2610.19792, UCSD Hao AI Lab, v2 2026-10-15) — the closest thing to a 1000 tok/s existence proof
https://arxiv.org/abs/2610.19792 · code: https://github.com/hao-ai-lab/JetSpec

[them] 5-layer causal parallel drafter, 1.05B params for Qwen3-8B / 1.73B for Qwen3.8-27B, trained on block
16 (anchor + 15). Mean AL **7.50 at J1 and 8.94 at J3 at T=0, budget 16** (DSpark 6.89). At budget 32 it
gains further while DFlash/DSpark/Domino lose. Lossless, proven (point-mass proposal + exact-match
verification). Serving: **1,361 tok/s for a single request on GB200**, 1.54x over DSpark.

The number that matters for us is in the text, not the abstract: **"For an 8B target, graphed verification
takes 6.1 to 6.3 ms... DSpark adds 3.76 and 3.80 ms of drafting per round, whereas JetSpec++ exposes only
0.61 and 0.66 ms, with the realized outcome already pre-drafted in 92.9% and 91.7% of rounds"** [them].
So 1,361 tok/s = AL 9.41 at a ~6.9 ms round on a **dense 8B**, with the drafter on a **second GPU**. On the
27B target verification alone is 27.4-27.8 ms and it is 1.31-1.37x over DSpark. Three transferable pieces:
(a) hide the drafter behind verification (R4), (b) foresight-draft the next block for the top-B likely
outcomes, (c) spend the freed window on Jacobi self-refinement passes, which is *free AL*. What does not
transfer: their verify is 6.1 ms for 16 rows on an 8B; ours is 9.1 ms for 6 rows on a 502B MoE. Their row
cost is ~0.1 ms, ours is 0.61 ms, a 6x difference in the same units.

---

## Leads detail

### R1 — Laddered (chunked) verification: the row-cost lever that needs no weights
**Mechanism.** Acceptance is prefix-ordered, so verifying rows 1..m and then, only if all m were accepted,
rows m+1..k+1 produces a bit-identical result to verifying k+1 rows at once. Rows computed per step fall
from k+1 to approximately the accepted length. At k=15 with AL 6.17 the ladder computes ~7.5-8 rows instead
of 16 [me: E[rows] for a 3-4 stage ladder [4,4,8] / [2,2,4,8] with a flat 0.85 conditional], i.e.
**-5.0 ms of the 15.2 ms step**. It is exact, needs no retraining, and is orthogonal to every other lead.
**Number.** At k=5 it is only 9.1 -> 8.15 ms (+12%, because AL 3.45 is already close to the 6 rows we pay) [me]. Its
value is entirely that it makes wide k affordable: k=15 goes 15.2 -> 10.8 ms, 406 -> 573 tok/s, and
15.2 -> 9.8 ms / 629 tok/s in the limit where the rows paid equal AL exactly [me].
**Primary sources.** FASER (arXiv 2604.20503) breaks "the verification phase into frontiers, or chunks" and
does "early pruning of rejected tokens inside the verification phase", in vLLM, up to +53% throughput
[them, abstract only]. EVICT (arXiv 2605.00342) truncates the draft tree before verification to the
cost-effective prefix, training-free and lossless, 1.21x over EAGLE-3 on MoE backbones [them, abstract].
DScale (arXiv 2609.37532) does dynamic verify-length allocation and packs scored prefixes "into half the
native verification capacity" with captured graphs reused, +22.2% over DSpark on Qwen3-8B [them, abstract].
The premise is stated plainly in arXiv 2611.04055: "Verifying more draft tokens can accept more of them,
but work after the first rejection is wasted" [them, abstract].
**Patch site.** The target pass that runs 1+k rows is `vllm/v1/worker/gpu/model_runner.py`
(`self.num_speculative_steps = vllm_config.num_speculative_tokens`, :279; decode_query_len at :487); the
accept/reject decision is `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`. Add a config field
(`spec_verify_ladder: [4, 12]`), run chunk 1, read `num_accepted` on the host, run chunk 2 only when
`num_accepted == chunk-1`. Both M values must be graph-captured (the engine already captures several).
**Risk.** A second dependent target pass inside one scheduler step; MLA context KV read twice (0.37 ms
each, [us]); the r21 fused mHC all-reduce is tuned to specific row counts (`VLLM_MOET_P2P_AR_FUSED_MAX_ROWS`
= 8 [us]), so a ladder changes M per chunk and may fall off the fused path. Also the biggest chunk that fits
the tuned graph is 6 rows today, which argues for a [6, 10] ladder rather than [2,2,4,8].

### R2 — 2-bit routed experts: halve the row, not the step
**Mechanism.** The marginal row is almost pure routed-expert bytes (one row touches ~5.5 more distinct
experts/layer x 4.7 MB x 40 layers = 1.03 GB [us, consistent with the measured 0.61 ms at 1.45 TB/s]).
Halving the bytes halves c as well as the 3.0 ms MoE floor.
**Number.** [us] T(5) 9.1 -> 7.3 ms (-1.8 ms), i.e. 377.7 -> 471 exact tok/s at AL 3.45; combined with the
ladder and R3 it is the largest single multiplier in the stack (1.5-1.7x at wide k) [me]. Scope: 14,208 experts (384 x 37
layers), 35.5M params each, 502.8B total [us, `work/STATUS.md` round 64].
**Patch site.** Moet `docs/v024-2bit.md` (2-bit planes + 4-bit recovery), the b8x/b12x kernel; the checkpoint
re-quantisation is a separate pipeline from the served FP4 weights.
**Risk.** It changes the served numerics. Han's constraint is "no quality regression" = exact verification;
2-bit experts keep the distribution exact but may move GSM8K. Gate on GSM8K-100 >= 96 and a codebench pass
rate, same as the FLy gate.

### R3 — Block-16 drafter + path selector (the AL lever)
**Mechanism.** Three separable pieces, in order of AL-per-cost: (i) train the existing DSpark drafter at
**block 16** instead of 5 (`dspark_block_size`), which is the single biggest AL jump and the one the last
retrain did not do; (ii) add DFlash2's **adjacent-pair path selector** over the top-16 candidates (2M params,
+0.6% cycle latency) instead of DSpark's sequential Markov correction; (iii) add the **two-tap dynamic
convolution** (16.5M params, +0.7%) to kill suffix decay. All three are drafter-side and cost zero verify rows.
**Number.** Published, matched config, T=1.0 + top_p 0.95 + thinking + lossless: DSpark block 16 = 5.49 vs
our DSpark block 5 = 3.45 [them]/[us]; DFlash2 = 5.97 (+8.7% over DSpark, from the selector + conv alone)
[them]. My estimate for us: **AL 5.0-5.8 at k=15, 4.4-4.9 at k=8**, since our workload is harder than
GSM8K/MT-Bench [me]. That is +45-70% AL. At our row cost it is worth only 1.25x tok/s; with R1+R2 it is
1.6-1.7x.
**Primary sources.** DFlash2 blog (above) for the selector/conv numbers; DSpine (above) for the
predecessor-feature injection alternative; DLite (arXiv 2610.23787) for the input design — sequence-wise
beats feature-wise, 7 -> 14 target hidden states +5% then saturates (our drafter takes 3 states,
feature-wise) [them, in `work/leads/notes/spec-decoding-nov-2026.md`]; SpecOPD (arXiv 2611.02846) for the
training recipe that matches our plateau — **"up to 41.45% higher mean acceptance length"**, block size 16,
spans an 8B, a 27B hybrid and a **125B sparse-linear** target, and four domains including **agentic** [them,
abstract]. SpecOPD is the right loss for the block-16 retrain.
**Patch site.** Training: `work/mtp/run.sh` `BLOCK=9` -> `BLOCK=16`, `K=15`; DeepSpec `block_size` config
field; loss from SpecOPD's E2E-TV surrogate. Serving: `SPEC_TOKENS=15`; and the good news — **the image
already ships DFlash2's machinery**: `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` has
`DFlash2Speculator(DFlashSpeculator)` with `_candidate_top_k_key = "selector_top_k"` and a graph-captured
`_selector_walk_kernel`, and `vllm/model_executor/models/qwen3_dflash2.py` has the grouped-conv Triton
kernel [us, r21 tree]. So (ii) and (iii) are a new `DSparkSpeculator` subclass reusing that walk, not new
kernels. `SpeculativeMethod` currently has `dflash` but not `dflash2`/`dspark+selector`
(`vllm/config/speculative.py:73-83`), so the method literal needs the entry.
**Risk.** The retrain at block 5 was a decisive negative (teacher-forced 0.8676 -> 0.8983, serving 3.25/3.42
vs 3.45/3.31 [us]). The block size is the difference: at block 5 the tail is already saturated, so a better
head cannot show up. Do not repeat the retrain at block 5.

### R4 — Pipelined drafting (foresight), the step lever that also buys AL
**Mechanism.** Draft block t+1 while block t verifies. JetSpec++ does it with a hidden-to-hidden adapter that
predicts the target's hidden states for the block under verification, a decoder that predicts the outcome
distribution P(k,a), and a speculation cache of the top-B outcomes (B=4). Because drafting then runs inside
the verification window, extra refinement passes are free.
**Number.** [them] On Qwen3-8B the drafter goes from 3.76-3.80 ms exposed (DSpark) to **0.61-0.66 ms**, with
a 92.9% cache hit rate; 1.54x over DSpark. On the 27B target: 0.71-0.77 ms exposed vs 4.46-4.52 ms, 1.31-1.37x.
For us the drafter is only 0.75-0.8 ms of 9.1 ms [us], so the direct step win is **1.09-1.10x** [me]; the
real prize is that a 2-3x more expensive drafter becomes free, which is what pays for multi-pass refinement
(+AL) and for a block-16 backbone.
**Primary sources.** JetSpec++ (above); D3D (arXiv 2611.06425, abstract only) for the decoupled-drafting
framing; SSD (Kumar et al., cited in JetSpec++) and SwiftSpec for the speculation-cache idea.
**Patch site.** `vllm/v1/worker/gpu/model_runner.py` async scheduling: the drafter must run in a side stream
between the target's attention and its MoE epilogue, or after the target's last layer but before the
sampler. Our step is DRAM-bound for 3.0 of 9.1 ms with idle SMs [us], so the overlap is physically available.
**Risk.** Highest engineering risk of the four. Graph capture of two concurrent shapes, and the target's
hidden states for the *next* block do not exist yet, which is exactly why JetSpec++ needs a trained adapter.

### R5 — Adaptive verify width per step (small, cheap, exact)
EVICT (arXiv 2605.00342): training-free, hyperparameter-free, lossless; uses fine-grained drafter signals
plus an offline-profiled verification cost to keep only the cost-effective prefix; **2.35x over AR and 1.21x
over EAGLE-3 on MoE backbones** [them, abstract]. The rigorous version of our row rule is arXiv 2609.22156:
speculation-budget selection as a Stochastic Shortest Path over reference sequences, with the finding that
"rejected candidates form a strict linear boundary" and a necessary condition balancing marginal cost
against expected progress (24 pages, [them, abstract]). Local caution: `enable_adaptive_verification` is in
the image and measured **-14% on code single stream on sm_120** [us], so this must be EVICT's variant, not
the shipped confidence scheduler. Expected +5-15% [me]; 2-3 eng-days; the checkpoint's
`mtp.2.confidence_head.proj.weight` is the free signal.

### R6 — Wildcard worth one measurement: routing-consistency-aware drafting
Every row we verify loads its own experts. If the drafter were trained (or sampled) to emit blocks whose
tokens route to the *same* experts as the anchor, the marginal row cost would fall below 0.61 without any
quantisation. There is no literature on this that I found; it is a training-time objective plus a
measurement of the routing overlap of drafted blocks. Cheap to measure first: log the distinct-expert count
per drafted block on the existing traces. [me, speculative]

---

## Dead ends (with the source)

| Dead | Number | Source |
|---|---|---|
| FLy / any lossy acceptance | 463.5 tok/s but changes the distribution ("100%"->"500%", "FLy"->"Flang"); banned | [us], Han's call |
| **DSpine/DFlash2/JetSpec++ AL dropped in at block 16 unchanged** | 4.82 accepted at our 15.2 ms k=15 step = **317 tok/s**, worse than 377.7 | [them] AL x [us] row cost -> [me] |
| Better drafter at block 5 | retrain, 2031 steps, teacher-forced 0.8676 -> 0.8983, serving 3.25/3.42 vs 3.45/3.31 | [us], `work/STATUS.md` rounds 58-60 |
| k > 5 with the block-5 checkpoint | k6 346/343/359, k7 329/356/345 vs k5 353/379/380 | [us], README |
| Trees (PCTree and the whole tree line) | 0.037 accepted per node vs our 0.24/node bar (N=32 for +0.93 AL on 4B GSM8K) | PCTree arXiv 2608.02123 [them] + [us] row cost |
| Multi-draft optimal verification (UniVer, SpecTr, SpecTr-GBV, OT-with-membership-cost) | UniVer: **+4.2-8.5% AL** over standard recursive rejection, exact. For a *linear* block the standard rule is already optimal, so this buys nothing until we build a tree | arXiv 2605.04543, 2604.25925, 2310.15141 [them, abstracts] |
| Mixture-of-drafters / n-gram hybrid (Hybrid Spark) | 1.10-1.62x on code editing, **0.97-1.01x on chat at T=0**; our workload is thinking prose. Paper only, no code | arXiv 2611.05611 [them] |
| Adaptive verification as shipped in vLLM/Moet | -14% single stream on code, sm_120 | [us], Moet `docs/dsv41-sm120-port.md:360` |
| Host RAM / NVMe expert prefetch | 4.2 GB per 9.1 ms step = 462 GB/s per rank vs PCIe Gen5 ~60 GB/s: 8x short | [us], `work/leads/notes/moe-step-staircase.md` |
| A nightly rebase for decode kernels | 3 SM120 commits in 1,490, none on our path; the real DS4.1 decode work is SM100/GB300-gated | [us], LEADS.md §2(e) |
| 0xSero's 675.9 tok/s c1 | his own repo measures C1 196-232 tok/s | [them], sybil-solutions README |
| JetSpec++'s 1,361 tok/s as a target for our box | that is a dense 8B whose 16-row verify costs 6.1-6.3 ms on GB200 with the drafter on a second GPU | arXiv 2610.19792 §5.4 [them] |

## Has anyone shipped 1000 single stream on 4x RTX PRO 6000?
**No.** As of 2026-11-09 the fastest public single-stream number for this model family on this hardware is
**our own 377.7 exact / 463.5 FLy**; the strongest third-party is local-inference-lab's "Karmic Kraken" at
**C1 249.1 tok/s** (4x RTX PRO 6000 Max-Q, adaptive DSpark k7, greedy drafting + standard rejection, 98.58
steps/s) [them, https://github.com/local-inference-lab/rtx6kpro, checked 2026-11-08]. Everything at or above
1000 in public is **aggregate** (2x RTX PRO 6000, DS4-Flash 284B: 1,003 tok/s at 24 streams, peak 1,170 at
32 [them, r/LocalLLM 2026-11-06]; 8x RTX PRO 6000: vLLM+DSpark 946 tok/s aggregate [them, helix.ml]) or
**single-request on different hardware** (JetSpec++ 1,361 tok/s on GB200, dense 8B). Nobody has published a
single-stream 1000 on a 502B MoE over 4 PCIe Blackwell cards.

## What I could not verify
- The DSpine Qwen3.8-27B numbers (2B params, AL 5.63/4.99, 2.7-3.8x) are from a third-party blog summarising
  Tencent's model card; the arXiv v2 covers only Qwen3-4B/8B. Labelled as such above.
- DSpine/DFlash2/JetSpec++ Table 1 cell values (per-benchmark AL at T=1) do not render in the arXiv HTML
  extraction; only the text-stated means are quoted here. FASER, EVICT, DScale, UniVer, SpecOPD,
  2609.22156, 2611.04055 are **abstract-only** reads — no numbers beyond what the abstract states.
- `B = 6.05 ms` is a two-point extrapolation, not a measured k=0 step.
- The chunked-verification row count (~7.5-8 rows at k=15) is an arithmetic estimate from a flat 0.85
  conditional; it should be simulated against our measured c-vector before any code is written.
