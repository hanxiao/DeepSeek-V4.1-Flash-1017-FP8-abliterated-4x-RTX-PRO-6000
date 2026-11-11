# 2-bit codebook and block scale for an MXFP4 weight source

Scope: the routed-expert format `2 bits/weight + per-32 E8M0 scale`, decoded as `E2M1 nibble × s`.
Everything marked **[own]** is this study's derivation; **[pub]** is published and cited in §12.

Artefacts in this directory: `q2_ops.py` (quantiser primitives), `q2_core.py` (FP4 source model),
`q2_derive.py` (continuous optima + ladder), `q2_final.py` (exhaustive codebook sweep),
`q2_layer.py` (imatrix / GPTQ layer experiment), `ladder_on_checkpoint.py` (the same ladder on the
real checkpoint, verified end-to-end on a synthetic MXFP4 checkpoint).

---

## 0. The answer in one table

Block 32, mean per-block relative RMS against the FP4 weights, i.i.d.-Gaussian source calibrated to
your two measured anchors.

| 2-bit scheme | error **[own]** | your measurement | cost |
|---|---:|---:|---|
| uniform `amax/2`, integer grid | 40.0 % | 40.26 % | current baseline |
| **current** `{±0.5, ±1.5}·2^j s0` | 36.3 % | **37.94 %** | current |
| **`{±1, ±4}·2^j s0`** | **34.5 %** | not measured | one constant in the decode |
| best real-valued global codebook, pow2 scale (not representable in the MMA) | 34.2 % | — | 4-entry fp16 LUT |
| `{±1, ±4}` × free per-block scale | 32.7 % | not measured | fp8 scale + new decode |
| `{±0.5, ±1.5}` × free per-block scale | 33.0 % | not measured | fp8 scale only |
| best real-valued global codebook × free scale | ~32.5 % | — | both of the above |
| per-block 4-level codebook × free scale | 30.2 % | 31.47 % | +0.5 bit/w + dynamic LUT |

The codebook ratio and the scale format are **substitutes, not two levers**. Starting from 36.35 %:
the fine scale alone is worth 3.38 pp, the free codebook alone 1.82 pp, and both together 3.64 pp. The
codebook eats most of the scale's value for zero cost, and after it the mantissa has 0.26 pp left. The
per-block codebook is a third, smaller step. And one lever is invisible in this metric and much bigger
on the output error (error feedback, §7).

**Recommendation: change the codebook to `{±1, ±4}` and stop there for the kernel.** It is one
constant, costs zero bits, and buys ~1.8 pp of weight RMS (2–4 pp on your real weights). Keep the
E8M0 scale and block 32. Spend the real effort on the offline rounding (imatrix-fitted scale, then
GPTQ), not on the format.

---

## 1. What the format can actually represent (this bounds every answer below)

The stored weight is an **E2M1 nibble times a per-block scale**. So:

1. The four codebook values must be **four of the 15 E2M1 values** `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
   There are 1365 such 4-subsets. A codebook of arbitrary real values would need the value to live
   outside the nibble, i.e. an fp16 operand or a different MMA — not the b12x `e2m1x8` path.
2. Since the scale is a free power of two per block, a codebook and its double are the same codebook.
   The only shape degree of freedom is the **ratio** `q/p`, and the E2M1 grid only offers
   `{1.33, 1.5, 2, 2.67, 3, 4, 6}`.
3. If the scale is consumed by the tensor core's block-scaling, it is **ue8m0 at block 32** (MXFP4) or
   **ue4m3 at block 16** (NVFP4) on Blackwell — a non-power-of-two scale at block 32 is not a hardware
   option, it needs the scale applied outside the MMA. [pub: Colfax block-scaling tutorial; NVIDIA NVFP4 blog]

Consequence: your current codebook is the ratio-3 point, and the achievable set is a 7-point ladder of
ratios. That is the whole design space for the cheap lever.

---

## 2. The optimal 4-level codebook for a Laplacian / Gaussian source **[own]**

Lloyd–Max on a 400k-point grid, unit variance, verified against Max (1960).

| source | optimal levels (free) | ratio b/a | MSE | rel-RMS | check |
|---|---|---:|---:|---:|---|
| Gaussian | ±0.4528, ±1.5104 | 3.34 | 0.11748 | **34.28 %** | Max 1960 tabulates D = 0.1175, x = ±0.9816, y = ±0.4528/±1.510 → **matches to 4 digits** |
| Laplacian (unit var) | ±0.4198, ±1.8339 | 4.37 | 0.17619 | **41.98 %** | no published table found; own |

Two things follow immediately.

- **The sign-symmetric constraint costs nothing.** For a symmetric pdf the free optimum *is*
  symmetric: my free solution comes out as `[-1.5103, -0.4527, +0.4528, +1.5104]`. On the discrete
  FP4 source the same holds: best symmetric 4-subset 34.53 % vs best asymmetric 4-subset 34.53 %
  (identical), and best real-valued asymmetric codebook 34.22 % vs symmetric 34.80 %. So **the
  31.5 % you measured with "free Lloyd, not sign-symmetric" is not buying asymmetry** — it is buying
  per-block adaptation. A global sign-symmetric codebook is not leaving anything on the table.
- **A Laplacian wants a wider codebook than a Gaussian** (ratio 4.37 vs 3.34). Your real weights sit
  between the two: the measured 37.94 % is 1.6 pp worse than my Gaussian model's 36.3 % for the same
  scheme, which is the signature of heavier tails. That matters below: the codebook fix is worth
  *more* on your weights than on a Gaussian.

Reference point: the high-rate rate–distortion bound at 2 bits for a unit-variance Gaussian is
`σ²·2^(-2R) = 0.0625`, i.e. 25 % RMS. Scalar quantization is stuck at 34.3 % (1.3 dB above the bound)
no matter how the codebook is chosen. Closing that gap needs *vector* codebooks — QuIP#'s 2-bit E8P is
a 2^16-entry codebook over 8 weights, looked up from a 2^8 table, 1 KiB total [pub]. A scalar 4-level
table cannot get there; do not expect the codebook lever to be worth more than ~2 pp.

---

## 3. The ladder on your exact format, and why ratio 4 wins **[own]**

Full sweep of the 21 sign-symmetric pairs (= every reachable ratio), pow2 scale, block 32:

| codebook | ratio | Gaussian | Laplacian | t4 |
|---|---:|---:|---:|---:|
| **{±1, ±4}** (or {±0.5, ±2}) | **4.00** | **34.53 %** | **38.08 %** | **38.98 %** |
| {±1.5, ±6} | 4.00 | 35.40 % | — | — |
| **{±0.5, ±1.5}** (current) | 3.00 | 36.35 % | 43.53 % | 42.98 % |
| {±1.5, ±4} | 2.67 | 37.09 % | — | — |
| {±0.5, ±3} | 6.00 | 37.23 % | — | — |
| {±0.5, ±4} | 8.00 | 39.02 % | — | — |
| {±1, ±2} | 2.00 | 42.33 % | — | — |
| {±1, ±1.5} | 1.50 | 49.88 % | — | — |

**Why 4 and not the continuous optimum 3.34:** a power-of-two scale grid is a 2× grid. The only
codebooks that are invariant to a 2× rescale of the *level spacing* are those whose ratio is a power
of two. With ratio 4 the codebook is exactly self-similar under the scale grid, so the scale
quantization error is absorbed; with ratio 3 it is not, and the ±41 %/−29 % scale miss is paid in full.
Ratio 4 is therefore the joint optimum of (shape, scale grid), not a compromise. Note also the *phase*
matters: {±1.5, ±6} has ratio 4 but sits 1.585 octaves off the grid, and loses 0.9 pp.

Corroboration from the other direction: with a *continuous* scale the optimum ratio moves back toward
the continuous value (my fitted global codebooks land at 3.1–3.8), and ratio 4 still wins because the
real distribution is heavier-tailed than Gaussian. Both arguments point at 4.

**The exhaustive 1365-subset sweep never beats `{±1, ±4}`** on any of the four distributions I tried.
The best real-valued (non-representable) codebook under the pow2 scale is 34.22 % vs 34.53 % — 0.3 pp,
which is the entire value of a 4-entry fp16 LUT. Not worth an extra load.

---

## 4. Q1, answered

For weights that are already E2M1 with per-block scales, the best 2-bit quantizer is:

> a **global, sign-symmetric 4-level codebook with ratio 4** — `{±1, ±4}` in E2M1 units — plus a
> **per-block power-of-two scale searched over a wide exponent range**, and nothing else.

- Expected error **34.5 %** (Gaussian-calibrated model), **38.1 %** on a Laplacian.
- Your measured 37.94 % sits between those; scaling the model's *relative* improvement onto your
  measurement gives **34–36 %**, i.e. **2–4 pp better**, for one changed constant.
- Against the "theoretical best": the free-Lloyd global codebook with a free scale reaches ~32.5 %,
  so `{±1,±4}`+pow2 captures 34.5 of the ~8 pp of total headroom from 40.3 → 32.5, i.e. **~25 % of
  everything that is reachable with a global codebook**, at 1 % of the effort.
- The 31.5 % "free Lloyd per-block" number is **not** a codebook-shape result; it is a per-block
  adaptation result. Do not read it as evidence that the codebook has 6 pp of headroom.

---

## 5. Q2: non-power-of-two scale

Apples to apples, same codebook, only the scale rule changes **[own]**:

| codebook | pow2 scale | continuous (LS) scale | gain from the fine scale |
|---|---:|---:|---:|
| `{±0.5, ±1.5}` (current) | 36.35 % | **32.97 %** | **3.38 pp** |
| `{±1, ±4}` | 34.53 % | **32.71 %** | **1.82 pp** |
| gain from the codebook | 1.81 pp | **0.26 pp** | |

**So the scale is the bigger single lever, and the codebook is the free one that eats most of it.**
From 36.35 %: the mantissa alone gets 3.38 pp, the codebook alone 1.82 pp (55 % of it), both 3.64 pp.
After the codebook is at ratio 4, the mantissa has only 0.26 pp of new error left to remove. The ratio-4
codebook is a poor man's mantissa — it makes the level spacing self-similar under the 2× scale grid,
which is precisely what a mantissa would have bought.

Two independent implementations agree on the 32.7 % figure (`q2_layer.py` row E and
`ladder_on_checkpoint.py`), so I trust it.

Cost side: at block 32 the hardware scale path is ue8m0, so a mantissa needs a different decode (or an
epilogue multiply). Verdict: **take the free one.** The codebook buys the first 1.8 pp at zero cost; the
mantissa buys the remaining 0.26 pp for a new decode. If you are going to touch the scale format at all,
go to NVFP4 (block 16 + ue4m3) rather than inventing an fp8 scale at block 32 — that is one hardware
switch instead of a custom decode, and it also shrinks the block.

Published corroboration at 4 bits, same mechanism: NVIDIA's own comparison of E8M0 vs E4M3 scaling
reports an average MSE of 0.08 for E4M3 and states that E8M0 "snaps the scale factor to nearest 2^n,
which can create a large quantization error for the block maximum (amax)" [pub].

---

## 6. Q3: block size

**[own]**, Gaussian-calibrated, mean per-block rel-RMS, `{±1,±4}` codebook:

| block | scale bits/weight | pow2 scale | continuous scale | per-block codebook |
|---:|---:|---:|---:|---:|
| 8 | 1.000 | 30.8 % | 26.9 % | 18.4 % |
| **16** | 0.500 | **32.7 %** | **30.1 %** | 25.6 % |
| **32** | 0.250 | **34.5 %** | **32.7 %** | 30.2 % |
| 64 | 0.125 | 35.5 % | 34.9 % | 32.9 % |
| 128 | 0.062 | 35.6 % | 34.8 % | 34.3 % |

- **16 vs 32: −1.8 pp (pow2 scale) or ≈−2.7 pp (continuous scale), for +0.25 bits/weight.** Worth it
  only if you were going to NVFP4 anyway.
- **64 vs 32: +1.0 pp worse, saves 0.125 bits/weight.** A bad trade at 2 bits; the scale is not what
  costs you.
- 32 is the knee. Below it the pow2 penalty *grows* (2.6 pp at block 16 vs 1.0 pp at block 64), because
  the block's own fitting error shrinks while the 2× scale grid error does not. **Block size and scale
  format interact**: shrinking the block without fixing the scale format gets you half the win.
- Published anchor, 2-bit, same direction: GPTQ on OPT-175B, WikiText2 — FP16 8.34, group 128 9.58
  (+1.24), group 64 9.18, group 32 8.94 (+0.60) [pub]. Going 128→32 at 2 bits buys 0.64 ppl there;
  going below 32 was not worth publishing.

---

## 7. Q4 and Q5: activation weighting and error feedback **[own measurements]**

Setup: synthetic linear layer, out=256, in=1024, FP4(MX, block 32) source, X with per-channel gains
(lognormal σ=0.6, 8 channels ×20) and a Toeplitz correlation ρ=0.35, so H is not diagonal.
`H p99/p50 = 21`, `max/med = 2133`. Two facts make this the right instrument:

- For a linear layer `‖XΔWᵀ‖²_F = n·tr(ΔW H ΔWᵀ)`, so the **imatrix objective *is* the output error**
  of the calibration set, not a proxy. The "H-weighted" and "output" columns below are the same number
  by construction.
- 37.9 % output error for the current codebook, matching the 37.94 % weight RMS. The instrument agrees
  with your checkpoint measurement.

| scheme | weight rel-RMS | **output error** | cost |
|---|---:|---:|---|
| A current `{±0.5,±1.5}`, pow2, plain | 36.52 % | **37.89 %** | — |
| C `{±1,±4}`, pow2, plain | 34.72 % | **34.94 %** | one constant |
| B current codebook, scale fitted to H | 42.08 % | **22.11 %** | offline only |
| D `{±1,±4}`, scale fitted to H | 37.39 % | 27.90 % | offline only |
| G codebook fitted to H, scale fitted to H | 42.81 % | 22.42 % | offline only |
| O current codebook, continuous scale fitted to H | 36.91 % | 17.72 % | fp8 scale |
| F `{±1,±4}`, continuous scale fitted to H | 36.40 % | **17.21 %** | fp8 scale |
| I per-block codebook, fitted to H | 33.63 % | **13.49 %** | +0.5 bit/w |
| J GPTQ, current codebook | 53.21 % | **5.64 %** | offline only |
| K GPTQ, `{±1,±4}` | 62.30 % | 7.37 % | offline only |
| N GPTQ + per-block codebook | 39.35 % | 3.55 % | +0.5 bit/w |
| M GPTQ + act-order | 49.94 % | 2.00 % | offline only |

**Q4 answers.**

1. Fitting the *existing* per-block scale to the H objective instead of plain MSE takes the output
   error from 37.9 % to 22.1 %, **−42 %, for zero kernel change and zero extra bits**. This is the
   single best-value lever in the whole study, and it *worsens* the weight RMS (36.5 → 42.1 %), so your
   current quality gate would reject it. That is the most important thing in this report after the
   codebook.
2. **A codebook choice alone captures about a tenth of the H gain**: 37.9 → 34.9 % (−8 % of the output
   error) versus −42 % for the scale and −85 % for GPTQ. Fitting the codebook *to* the H objective
   instead of to plain MSE changes nothing (22.42 vs 22.11 %).
3. The codebook ranking **flips** under H weighting with a pow2 scale: ratio 3 (22.11) beats ratio 4
   (27.90), even though ratio 4 is better in plain MSE. With a continuous scale it flips back
   (17.21 vs 17.72). If you adopt imatrix, **re-tune the codebook under the H objective**, do not reuse
   the MSE-optimal one.
4. Per-block codebooks capture much more of the H gain than any global codebook (13.49 %), because the
   3 extra degrees of freedom can be spent on the important columns inside the block. But they cost
   +0.5 bit/weight and a dynamic LUT, and GPTQ beats them for free.

Caveat on the −42 %: its size is a property of your H, not a constant. It is large here because my
within-block importance spread is 20× (8 injected channels out of 1024). With a flatter H the gain
shrinks toward the plain-MSE number. Measure it on your imatrix before budgeting for it — the direction
is certain, the magnitude is not.

Published support for the mechanism: SqueezeLLM formulates exactly this — a sensitivity-weighted
(Fisher) k-means codebook — and reports that non-uniform quantization alone is *not* enough at 2 bits
(LLaMA2-7B 2-bit: 35.49 ppl vs 5.47 FP16; 13.64 with 0.1 % outliers kept) [pub]. AWQ's activation-aware
scaling and llama.cpp's imatrix are the same idea at the granularity of a scale, not a codebook.

**Q5 answers.**

- Error feedback is worth **−85 %** on the output error (37.9 → 5.6 %) with the *same* codebook, same
  block scale, same kernel. It is by far the biggest lever, and it costs nothing at inference.
- It **makes the weight RMS much worse** (36.5 → 53.2 %). If your acceptance test is weight RMS you
  will conclude GPTQ is a regression. It is not: it moves error out of the directions H cares about.
- **It interacts with the codebook, and the interaction flips the ranking**: with GPTQ, ratio 3 beats
  ratio 4 (5.64 vs 7.37 %, a 30 % relative difference). So the codebook must be chosen *inside* the
  rounding loop, not before it. Concretely: run GPTQ twice with the two candidate tables and pick the
  better.
- act-order gives another 3.5× in my setting (5.64 → 2.00 %), but treat that as an upper bound: my H has
  a static importance ordering and `max/med = 2133`, the best possible case for it. Published evidence
  at very low bits is mixed — **QMoE found activation reordering actively harmful** for ternary
  quantization of switch-base-128 (loss 1.99 → 2.23) and true-sequential neutral [pub]. Test it, do not
  assume it.
- Published scale of the effect at 2 bits: GPTQ OPT-175B WikiText2 8.34 → 9.58 (g128, +15 %) / 8.94
  (g32, +7 %) [pub]; RTN at 3 bits collapses (7.3e3 ppl) where GPTQ holds 8.68 [pub]. At 2 bits,
  data-dependent rounding is the difference between usable and broken.

---

## 8. Q6: MoE routed experts

**Published evidence (this is the best I found, with numbers).**

GEMQ (2026) evaluates uniform-bit vs expert-level mixed-precision allocation at the *same average
bit budget*, GPTQ as the base quantizer, attention fixed at 4 bits [pub]:

| model, avg bits/expert | uniform | PMQ (expert-level allocation) | GEMQ |
|---|---:|---:|---:|
| Mixtral-8×7B, 2.5 | **65.49** | 64.34 | 65.13 |
| Mixtral-8×7B, 2.0 | 55.72 | **60.38** | 59.98 |
| Mixtral-8×7B, 1.5 | 47.45 | 51.78 | 52.00 |
| DeepSeekV2-Lite, 2.0 | 50.06 | 52.52 | **53.80** |
| Qwen1.5-MoE, 2.0 | 41.06 | 52.41 | **54.87** |

(avg zero-shot accuracy over 7 tasks)

Reading: **at 2 bits, quantizing all experts equally costs 4–5 accuracy points (up to 14 on
Qwen1.5-MoE); at 2.5 bits it costs nothing** (uniform even wins on Mixtral). The payoff of
differentiated allocation appears only in the aggressive regime. That is consistent with your own
result: 2-bit routed experts at 37.94 % weight RMS cost −0.38 pp on 1,319 GSM8K questions, i.e. your
checkpoint's experts are *not* the sensitive part at 2 bits, so per-expert allocation would buy little
here. Do not spend the effort unless the eval says otherwise.

Other published points that matter for the mechanism:

- **The router is the fragile part, not the expert weights.** GEMQ measures that "over 40 % of tokens
  are routed to different experts after 1.5-bit quantization" [pub], and gets a large part of its gain
  from fine-tuning only the routers (0.04 % of parameters). If you ever go below 2 bits, protect or
  re-fit the gate before touching expert precision.
- **Attention is more sensitive than experts** — GEMQ's ablation: dropping attention to 2 bits degrades
  significantly, raising it to 8 bits is cheap and helps [pub]. Your format change only touches routed
  experts, which is the right place to spend bits.
- **Large MoEs are unusually robust to weight quantization.** QMoE: RTN alone at 2 bits on
  switch-base-128 gives 2.27 vs 1.73 BF16, but on the 2048-expert model 1.33 vs 1.18, and "vanilla
  rounding does not lead to a complete model collapse even at ternary precision" [pub]. A 2-bit MoE is
  not the same risk as a 2-bit dense model.
- **Unsloth Dynamic 2.0** is per-layer/per-tensor bit selection driven by KLD against the base model,
  not a per-expert frequency rule; the published deltas are KLD improvements of ~4–7 % at the same size
  (e.g. Q2_K_XL 0.2297 → 0.2209) [pub]. It is a layer-allocation method, not evidence about rare
  experts.

**My derivation for the rare-expert question [own].** For a routed expert, the H-weighted objective is
`Σ_tokens (xᵀΔw)²`, so an expert's contribution to the loss scales with its **token count**. A single
global codebook plus a per-block MSE-optimal scale treats every expert identically in *relative* terms,
which over-spends on rare experts and under-spends on hot ones. Two consequences:

1. The right cheap fix is the same as §7: fit each block's scale to a **frequency-weighted** objective
   (weight = the expert's activation count, or its imatrix). That is per-expert dynamic quantization
   done at the only granularity the format has, at zero kernel cost.
2. A per-expert *bit width* would be better, but it breaks a fixed 2-bit kernel; the format decision
   dominates. If you ever want it, the published recipe is an LP over per-expert error curves [pub:
   PMQ, GEMQ], not a frequency heuristic alone.

Caveat in the other direction: rare experts see few calibration tokens, so their imatrix/importance
estimate is noisy — EAQuant calls this out as "sparse expert calibration" [pub]. With 128 calibration
sequences GEMQ reports expert-importance estimates stable at Pearson 0.99 across resamples, but that is
for a 8-expert-per-layer model; DeepSeek-V4-Flash has far more experts per layer, so expect noisier
per-expert weights and prefer a shrunk (frequency-only) weighting.

---

## 9. Ranked levers: error reduction per unit of implementation effort

Δ weight RMS is relative to the current 36.35 % (calibrated simulator, mean per-block rel-RMS).
Δ output error is relative to scheme A's 37.89 % (layer experiment, H-weighted = output error).
Your real numbers will be 1–2 pp worse in absolute terms (my model is 1.4 pp optimistic on the current
scheme) but the ordering is stable across four source distributions.

| # | lever | Δ weight RMS | Δ output error | bits | effort | verdict |
|---|---|---:|---:|---:|---|---|
| 1 | **codebook `{±0.5,±1.5}` → `{±1,±4}`** | **−1.8 pp** (−5.4 on Laplacian) | −8 % | 0 | one constant, `nib = 2 + 4*(b&1) + 8*(b>>1)` | **do it** |
| 2 | **fit the per-block scale to diag(H) instead of plain MSE** | +5.6 pp (worse) | **−42 %** | 0 | offline only, needs imatrix | **do it** |
| 3 | **GPTQ error feedback on top of the fixed codebook** | +17 pp (worse) | **−85 %** | 0 | offline pass, needs calibration set | **do it if you have the pipeline** |
| 4 | widen the exponent search `j` to `{-8..8}` | 0.00 pp | 0 | 0 | free | already sufficient, verify only |
| 5 | per-block 4-level codebook | −2.5 pp | −64 % | +0.5 | dynamic LUT in the inner loop | not worth it vs 2+3 |
| 6 | block 32 → 16 (+ NVFP4 ue4m3) | −3.3 pp | — | +0.25 | different MMA path, new checkpoint layout | only if you go NVFP4 wholesale |
| 7 | fp8 mantissa scale at block 32 | −1.8 pp alone, **−0.26 pp after #1** | −20 % (with #2) | 0 | new decode, breaks ue8m0 | skip unless you go NVFP4 |
| 8 | 4-entry fp16 LUT (arbitrary real codebook) | −0.3 pp | — | 0 | extra load per block | skip |
| 9 | asymmetric global codebook | −0.0 pp | — | 0 | 4-entry LUT / PRMT | skip |

Two traps worth naming:

- **Levers 2 and 3 make the weight RMS worse.** 42 % and 53 % vs 36.5 %. If the acceptance test stays
  "mean per-block relative RMS against the FP4 weights", the two best levers will look like
  regressions. Change the metric to the H-weighted (imatrix) error, or to a task eval.
- **Lever 1 and levers 2/3 are not independent.** The codebook that is best for plain MSE (ratio 4)
  is not the best under H weighting (ratio 3 wins there, 22.1 vs 27.9 %), and not the best under GPTQ
  (ratio 3 again). Tune the table last, inside the final rounding objective.
- **Levers 1 and 7 are the same lever.** Ratio 4 under a pow2 scale and ratio 3 under a continuous
  scale both land near 33 %. Do not budget for both and expect them to add.

---

## 10. Concrete recipe for this format

**Format: keep it.** 2 bits + E8M0 per 32 + E2M1 nibble. Nothing in this study justifies changing the
container.

1. **Codebook.** Replace the four E2M1 codes `{1, 3, 9, 11}` with `{2, 6, 10, 14}`:
   ```
   nib = 2 + 4*(b & 1) + 8*(b >> 1)        // {+1, +4, -1, -4} * s, E2M1 codes {2,6,10,14}
   ```
   Branch-free, same instruction count as today (one MAD plus one shift-OR), same 4-bit nibble, no
   extra memory. Equivalent alternative with identical error: `{1, 4, 9, 12}` = `{±0.5, ±2}`,
   `nib = 1 + 3*(b&1) + 8*(b>>1)`. Do **not** use `{±1.5, ±6}` even though its ratio is also 4 — it is
   off-phase with the power-of-two scale grid and loses 0.9 pp.
   Expected: 34.5 % in my model, **34–36 % on your weights** (from 37.94 %).
2. **Scale.** Keep E8M0. Search `j` over at least `{-4..+4}` (the ratio-4 codebook shifts the optimal
   `j` down by 1 vs today, so the current `{0..3}` window is no longer centred — widening is free and
   the check costs one line).
3. **Choose `j` with the imatrix, not with plain MSE.** `j_b = argmin_j Σ_{k∈block} h_k (w_k − ŝ_k)²`.
   Offline only, no format change, −42 % output error in my simulation. If you have no imatrix, skip.
4. **Then GPTQ.** Quantize column by column with error feedback, re-deriving the block scale from the
   error-fed-back weights, and re-check the codebook inside the loop (ratio 3 vs 4 flips here).
   −85 % output error. Skip act-order unless you measure it helping.
5. **Do not** go to block 16, an fp8 scale, or per-block codebooks for the current 2-bit target. Each
   is ~2 pp for a format change; the three offline levers above are ~2 pp, −42 % and −85 % for none.
6. **Change the acceptance metric** from weight RMS to the H-weighted error before you judge any of
   this, otherwise steps 3 and 4 will read as regressions.

Expected end state: 34–36 % weight RMS (from 37.94 %), and an output error ~5–6× lower than today at
the same 2 bits, with the kernel's data path untouched.

**Verify on the real checkpoint** (14 s per tensor, prints the whole ladder including the H-weighted
column when an imatrix is supplied):

```
python3 ladder_on_checkpoint.py --model-dir /data/models/ds41-abl --layer 6 --experts 0
python3 ladder_on_checkpoint.py --model-dir /data/models/ds41-abl --layer 6 --experts 0 --imatrix /data/imatrix
```
It reproduces the anchors on a synthetic MXFP4 checkpoint: u2 40.65 (yours 40.26), current 36.50
(yours 37.94), ratio-4 34.72, per-block Lloyd 30.61 (yours 31.47).

---

## 11. What is own vs published, and what I could not check

**Own (derived here, no external source):** every number in §2–§7 tables except the two Max-1960 rows
and the GPTQ/MoE rows explicitly marked [pub]; the ratio-4 argument; the block-size table; the whole
layer experiment; the rank order; the rare-expert argument in §8.

**Published (cited, with the exact claim):**
- Max 4-level Gaussian optimum D = 0.1175, levels ±0.4528/±1.510, SNR 9.30 dB — Max (1960), and the
  table reproduced in Gibson's *Digital Compression*, Table 4.1. My Lloyd reproduces it to 4 digits.
- 2-bit GPTQ group-size ladder on OPT-175B/BLOOM-176B (8.34 → 9.58 g128 → 8.94 g32), RTN collapse at
  3 bits — Frantar et al., GPTQ.
- act-order harmful at ternary, large MoEs robust to RTN at 2 bits — Frantar & Alistarh, QMoE.
- SqueezeLLM sensitivity-weighted k-means, 2-bit LLaMA2-7B 35.49 ppl vs 5.47 FP16 — Kim et al.
- Expert-level mixed precision at equal bits, +4–5 points at 2 bits, 0 at 2.5 bits; >40 % of tokens
  re-routed at 1.5 bits; attention more sensitive than experts — GEMQ (2026).
- NVFP4 = block 16 + E4M3 vs MXFP4 = block 32 + E8M0, and NVIDIA's E8M0-vs-E4M3 MSE comparison —
  NVIDIA developer blog; hardware block-scaling scale formats — Colfax CUTLASS tutorial.
- Unsloth Dynamic 2.0 = per-layer/tensor selection by KLD, KLD deltas 4–7 % at equal size — Unsloth blog.
- QuIP# 2-bit E8P: 2^16-entry 8-dim lattice codebook via a 2^8 lookup, 1 KiB — Tseng et al.

**Not checked, and it matters:** I do not have the checkpoint (it lives on the VM at
`/data/models/ds41-abl`), so every number above is from a simulator calibrated to your two anchors.
The simulator reproduces a synthetic MXFP4 tensor *exactly* (u2 40.65, current 36.50) but your real
tensor measures 40.26 and 37.94 — i.e. the real weights are 1.4 pp harder than i.i.d. Gaussian, which
is why I quote a 2–4 pp range for the codebook fix rather than the model's 1.8 pp. Run
`ladder_on_checkpoint.py` for the exact figures.

---

## 12. Sources

1. Max, J., *Quantizing for Minimum Distortion*, IRE Trans. Inf. Theory, 1960.
   https://web.stanford.edu/class/ee398a/handouts/papers/Max%20-%20Quantizing%20for%20Min%20Distortion.pdf
   (table values D = 0.1175, x = ±0.9816, y = ±0.4528/±1.510, SNR 9.30 dB reproduced in
   https://web.ece.ucsb.edu/~gibson/books/digitalcompression/Chapter4Table.pdf)
2. Frantar, Ashkboos, Hoefler, Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative
   Pre-trained Transformers*, ICLR 2023. https://arxiv.org/abs/2210.17323
3. Frantar, Alistarh, *QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models*, MLSys 2024.
   https://arxiv.org/abs/2310.16795
4. Kim et al., *SqueezeLLM: Dense-and-Sparse Quantization*, ICML 2024. https://arxiv.org/abs/2306.07629
5. Wang et al., *GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs*, 2026.
   https://arxiv.org/html/2605.23078v1
6. Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*,
   ICML 2024. https://arxiv.org/abs/2402.04396
7. NVIDIA, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*, 2025.
   https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference
8. Colfax Research, *Hardware-supported Block-scaling with NVIDIA Blackwell* (scale factors are 8-bit
   unsigned; ue4m3 for nvf4, ue8m0 for mxf4).
   https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus
9. OCP, *Microscaling Formats (MX) Specification v1.0* (E2M1 element, E8M0 scale, block 32).
   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
10. Unsloth, *Dynamic v2.0 GGUFs*. https://unsloth.ai/blog/dynamic-v2
11. Fu et al., *EAQuant: Enhancing Post-Training Quantization for MoE* (sparse expert calibration,
    router instability). https://arxiv.org/html/2506.13329v3
12. In-repo: `work/quant/re-quant2.py`, `work/quant/quantize_2bit.py`, `work/STATUS.md` (rounds 66, 70)
    for the measured 40.26 / 37.94 / 31.47 anchors and the `{1,3,9,11}` decode.
