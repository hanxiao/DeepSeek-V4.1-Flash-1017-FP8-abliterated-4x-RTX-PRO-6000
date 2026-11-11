# Training-free 2-bit weight quantization for a DeepSeek-V4-class MoE

Research report, 2026-11-11. Scope: methods that need no gradient updates and no end-to-end learned codebooks, judged against one hard constraint: **a global 4-entry codebook plus a per-32-element block scale**, decoded to an E2M1 nibble in 2 ALU ops.

All numbers below are either quoted from a cited source or measured in this repo (`work/STATUS.md` rounds 65-77). Anything I could not verify is marked **[unverified]**.

---

## 1. Bottom line

1. **Our format is already the best training-free 2-bit format that fits our kernel.** Every published method that beats plain 4-level scalar quantization at 2 bits (QuIP#, AQLM, QTIP, VPTQ, IQ2, PagedMoE) does it by changing the *format*: vector codebooks, trellis state, rotations, or per-block codebooks. None of them fits a 4-entry global LUT without a new kernel.
2. **The dominant term in our 37.94% is not the codebook shape, it is the scale grid.** In a calibrated model of our own blocks, refining the per-block scale from power-of-two (E8M0) to 3-mantissa-bit (E4M3) takes the error from 33.02% to 28.99% (-12.2% relative), while changing the codebook ratio from 3 to 4 buys only -2.9%. With a continuous scale the sign-symmetric codebook is competitive with a free per-block Lloyd codebook, i.e. the sign-symmetry constraint is nearly free; the scale granularity is not.
3. **The cheapest real change is to stop treating the scale byte as E8M0.** It is already one byte per 32 weights; reinterpreting it as E4M3 (or any float with 3 mantissa bits) costs zero bytes, needs no new tensor, and captures about 90% of the gain available from scale refinement.
4. **Codebook dimensionality is worth another ~15% of MSE but costs the kernel.** On i.i.d. Gaussian at 2 bits, QTIP's own table gives 0.118 MSE for scalar Lloyd-Max, 0.089 for 8-D E8P, 0.069 for 256-D trellis, against a rate-distortion bound of 0.063. That is the whole value of VQ/trellis, and all of it is inaccessible to a 4-entry LUT.
5. **Our verbosity effect is a known, named effect and our magnitude is normal.** "Quantization Inflates Reasoning" (arXiv 2606.25519) reports CoT token inflation of +3.7% to +14.4% at INT4 and +19% to +60% at INT3 on a 30B MoE; we see +27% to +36% at 2 bits on 502B. Their INT3 numbers bracket ours.

---

## 2. Summary table

Format column = what the method must change relative to "one global 4-level codebook + one scale per 32 weights".

| Method | Format | Training-free? | 2-bit accuracy evidence | Fits our kernel? |
|---|---|---|---|---|
| RTN, uniform int2, group 128 | per-group affine (scale + zero point) | yes | LLaMA3-8B WT2 PPL **1900** (F16 6.1); LLaMA3-70B **4.6e5** (F16 2.9) [2404.14047] | partial: zero point breaks it, scale granularity does not |
| GPTQ @ 2 bit | per-group affine, Hessian-weighted rounding | yes (no gradients) | LLaMA3-8B WT2 **210**, LLaMA3-70B **11.9** vs 2.9 [2404.14047] | partial, same as above |
| AWQ @ 2 bit | per-group affine + per-channel activation-aware scaling | yes | LLaMA3-8B WT2 **1.7e6**, LLaMA3-70B **1.7e6** (collapse) [2404.14047] | no (per-channel scaling needs an activation-side multiply) |
| HQQ @ 2 bit | per-group affine, zero point, fp16 scale | yes (calibration-free) | 2-bit Llama-2-70B beats fp16 Llama-2-13B PPL [dropbox blog] | no (fp16 scale + zero point) |
| QuIP# @ 2 bit | RHT rotation + 8-D E8P lattice VQ + BlockLDLQ | no-FT variant exists; headline numbers use FT | L2-70B WT2 **3.91**, C4 **5.71** (F16 3.12/4.97) [2406.11235 Table 4]; LLaMA3-70B WT2 13.0 [2404.14047] | no (needs a Hadamard around the GEMM + 8-D LUT) |
| AQLM @ 2 bit | additive 8-D VQ, 2 learned codebooks | no (codebooks learned per layer) | L2-70B WT2 **3.94**, C4 **5.72** [2406.11235 Table 4] | no (1 MiB codebook) |
| QTIP @ 2 bit | trellis-coded quantization, L=16 bitshift trellis, computed codes | yes (computed-code variants need no FT) | L2-70B WT2 **3.78**, C4 **5.56**, better than QuIP#/AQLM at every size [2406.11235 Table 4] | no (stateful decode, L-bit window) |
| VPTQ @ 2 bit | vector quantization, data-dependent centroid codebooks | no retraining, but k-means codebooks | vendor claim: 2-bit Llama-3.3-70B keeps >95% of accuracy [2409.17066, TDS write-up] **[unverified independently]** | no (codebook LUT) |
| llama.cpp IQ2_XXS / IQ2_M / Q2_K + imatrix | 2-2.6 bpw, per-block scale + small grid LUT, importance-matrix least squares | yes | Gemma-3-27B, from Unsloth's Dynamic 3.0 page: MMLU IQ2_XXS **59.20**, IQ2_M **66.47**, Q2_K_XL **68.70**; KLD 0.536 / 0.266 / 0.230 against 0.0249 for Q4_K_XL [unsloth docs] | closest cousin: grid LUT + block scale, but the grid is 256 entries |
| Unsloth UD-Q2_K_XL, Dynamic 2.0/3.0 | mixed precision, 2-6 bpw per tensor, calibrated (Q2_K_XL is ~2.5-2.7 bpw, not uniform 2-bit) | yes | same table as above; DeepSeek-V3.1 1-bit UD: 671 GB -> 192 GB [unsloth docs] | it is a bit-allocation recipe, not a format; the idea is adoptable, the format is not |
| BitNet b1.58 (ternary) | 1.58 bit, per-tensor scale, QAT from scratch | no | needs training from scratch [2402.17764] | no |
| OCP MX (MXFP8/6/4, MXINT8) | E8M0 per-32 scale + element format | n/a | **no MXFP2 exists in MX v1.0**; smallest element is MXFP4 E2M1 [OCP MX spec] | our format *is* the natural MXFP2 analogue |
| PagedMoE tree codebook | prefix-compatible nested bitplanes, per-block | yes | uniform 2-bit gives **54.7%** relative routed-output error, 3-bit 29.6%, 4-bit 15.3% on Qwen3.5-397B-A17B; their 4-bit weight error 10.02% vs GPTQ-INT4 14.8% [2611.03173] | no (nested bitplanes + tree codebook) |
| Ours: sign-symmetric {±0.5s, ±1.5s} + 2^j scale search | global 4 levels + E8M0 per-32 | yes | 37.94% block RMS; GSM8K-1319 96.06% vs 96.97/96.59% base [STATUS round 71] | it is the baseline |

---

## 3. Q1: what the 2025-2026 landscape actually contains

**The 2-bit scalar family is broken, and everyone knows it.** With group size 128, plain GPTQ and AWQ at 2 bits collapse on LLaMA3: WikiText2 210 and 1.7e6 on the 8B, 11.9 and 1.7e6 on the 70B, against 6.1 and 2.9 in fp16 ([2404.14047](https://arxiv.org/abs/2404.14047)). QuIP's incoherence processing does better (85.1 and 13.0) but is still far off. This is why all credible 2-bit results since 2024 come from either vector quantization or from mixing bit widths.

**The VQ / trellis frontier, with numbers.** QTIP's Table 4 is the cleanest apples-to-apples comparison at 2 bits on Llama-2-70B (WikiText2 / C4, ctx 4096): fp16 3.12 / 4.97, AQLM 3.94 / 5.72, QuIP# 3.91 / 5.71, QTIP 3.78 / 5.56 ([2406.11235](https://arxiv.org/abs/2406.11235)). So the *entire* published 2-bit frontier on a 70B model spans 3.78 to 3.94 perplexity, i.e. +0.66 to +0.82 over fp16. There is no method that makes 2 bits free on a dense 70B.

**Why dimension is the lever, quantitatively.** QTIP Table 1 gives distortion for an i.i.d. Gaussian source at 2 bits: scalar Lloyd-Max 0.118 MSE, QuIP#'s 8-D E8P 0.089, QTIP's 256-D trellis 0.069, rate-distortion bound D_R = 0.063. In RMS terms that is 34.3% / 29.8% / 26.3% / 25.1%. Two conclusions follow. First, the gap between a perfect scalar quantizer and a perfect 256-D quantizer is only 1.30x in RMS. Second, the distance from our 37.94% to that 34.3% scalar bound is about the same size as the whole benefit of going to trellis codes. The two numbers are not on the same source (ours are FP4-grid blocks, the bound is Gaussian), so read it as an order-of-magnitude comparison: our headroom against scalar quantization is comparable to the headroom everyone else buys with vector quantization.

**MoE-specific 2026 work.** PagedMoE ([2611.03173](https://arxiv.org/abs/2611.03173)) measures relative routed-output error on Qwen3.5-397B-A17B layer 30: uniform 4-bit 15.3%, uniform 3-bit 29.6%, uniform 2-bit 54.7%, and a routing-weight-driven allocator at a 2.5-bit average brings it to 36.8%. Its own 4-bit codec reaches 10.02% relative weight error against ~14.8% for GPTQ-INT4. RotaryQuant ([2608.08081](https://arxiv.org/pdf/2608.08081)) ships 4-bit dense, 2-bit routed experts and an 8-bit shared expert on a 26-120B MoE, on the grounds that the shared expert's activation kurtosis resists compression. Both are precedents for our exact operating point (2-bit routed experts, denser elsewhere), and both report that the cost is "model- and task-dependent" rather than a fixed number.

**imatrix.** llama.cpp's importance matrix is a per-input-channel weighting used inside the block least-squares fit ([tools/imatrix/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), introduced in [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)). It is training-free (a weighted least-squares solve, no gradients) and it is what the IQ2 family is built on. **I could not find a published ablation that isolates imatrix's contribution at 2 bits with numbers.** The best proxies are: GPTQ vs RTN at 3 bits on LLaMA3-8B, 8.2 vs 27.9 PPL (3.4x) and at 2 bits 210 vs 1900 (9x) ([2404.14047](https://arxiv.org/abs/2404.14047)); and the calibration-corpus effect at INT3 in [2606.25519](https://arxiv.org/abs/2606.25519), where switching from WikiText to math-CoT calibration moves Qwen3-4B INT3-GPTQ accuracy from 28% to 65% while cutting CoT length from 30k to 12k tokens. Activation-aware weighting matters more than codebook shape at low bits, but see §5.3 for why our format caps how much of it we can use.

**MX / OCP.** MX v1.0 defines MXFP8, MXFP6, MXFP4 and MXINT8 only; **there is no MXFP2** ([spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [2310.10537](https://arxiv.org/abs/2310.10537)). The smallest standard MX element is E2M1, which is exactly what our checkpoint stores. Recent MX work goes sideways rather than down: M²XFP ([2601.19213](https://arxiv.org/html/2601.19213v1)) keeps 4-bit elements and adds 8 bits of per-block metadata to the 8-bit E8M0 scale, and "Is Finer Better? The Limits of Microscaling" ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/97ca7168c2c333df5ea61ece3b3276e1-Paper-Conference.pdf)) studies block size rather than 2-bit elements. So: our scheme is the natural "MXFP2-like" format, and the literature on that specific format is thin. That is an argument for measuring rather than importing.

**One more 2026 entry.** D²Quant ([2602.02546](https://arxiv.org/html/2602.02546v1)) reports sub-4-bit weight-only PTQ combining "DSQ and DAC". I did not verify its 2-bit numbers; treat it as a pointer, not evidence.

---

## 4. Q2: which methods fit "one global 4-level codebook + a per-block scale"

Three groups.

**Group A, fits with zero or near-zero kernel change (only the constants change).**
- A fixed global 4-level codebook with any ratio and any assignment of the 4 codes to E2M1 nibbles. Our decode `nibble = 1 + 2*(b&1) + 8*(b>>1)` is the ratio-3 codebook {0.5, 1.5}; ratio 4 is `nibble = 1 + 3*(b&1) + 8*(b>>1)` ({0.5, 2}) or `nibble = 2 + 4*(b&1) + 8*(b>>1)` ({1, 4}, which is the upstream v0.24 codebook {-4,-1,1,4} in E2M1 codes {2, 6}). Same op count, one constant.
- A finer per-block scale, if the kernel converts the scale byte to float rather than feeding it to a hardware MX scale-factor operand. The byte already exists.
- imatrix / Hessian-weighted *selection* of which of the 4 codes and which scale exponent each block uses. Offline only, format unchanged.

**Group B, needs a second per-block byte but keeps the 2-op decode.**
- A per-block codebook pair (a, b) drawn from the E2M1 magnitude grid, 1 byte per block (+0.25 bits/weight). Decode becomes `nibble = a + (b-a)*(c&1) + 8*(c>>1)`: one multiply-add more than now. This is the "tensor-sym" idea upstream already ships, generalized per block.

**Group C, needs a new kernel.** Everything with a table larger than 4 entries or with state: IQ2-style 256-entry grids, QuIP#/AQLM/VPTQ vector codebooks, QTIP trellis state, PagedMoE nested bitplanes, any rotation (QuIP#, QuaRot, RotaryQuant). Note that our kernel already measured the cost of table lookups: a `__constant__` LUT with divergent indices cost **+8 ms per step**, a shared-memory LUT **+0.15 ms**, arithmetic expansion **+0.6 ms** (`work/STATUS.md` round 75). Any per-block LUT is therefore a memory-hierarchy decision, not a free change.

---

## 5. Q3: how much does the codebook shape matter, and how much does activation-awareness matter

### 5.1 Published anchors

- **Scalar vs vector vs trellis, i.i.d. Gaussian, 2 bits** (QTIP Table 1): 0.118 / 0.089 / 0.069 MSE, bound 0.063. In RMS: 34.3% / 29.8% / 26.3% / 25.1%.
- **Codebook *shape* at fixed dimension**: QuIP#'s appendix C.3 reports that the E8 lattice codebook beats a k-means Gaussian codebook of the same dimension, and that a uniform codebook beats a Gaussian-shaped one on post-rotation weights ("allocating more information to the edge of the distribution ... is useful") ([2402.04396](https://arxiv.org/abs/2402.04396)).
- **Our own weights, 2 bits, block 32**: uniform 40.26%, sign-symmetric {0.5,1.5} with per-block 2^j scale search 37.94%, free per-block Lloyd 31.47%, 3-bit Lloyd 16.61% (`work/STATUS.md` rounds 65, 66, 70; script `work/quant/re-quant2.py`).
- **Our own model, end to end**: at 37.94% weight RMS, GSM8K-1319 goes 96.97% / 96.59% (two base runs) -> 96.06%, i.e. -0.7 to -0.9 pp, about twice the harness noise, with 22 truncations instead of 4-7 (`work/STATUS.md` round 71).

### 5.2 A calibrated decomposition of our 37.94%

To rank the options I built a model of our blocks: 32 i.i.d. Gaussian weights, quantized to MXFP4 with the OCP rule (s0 = 2^(floor(log2 amax) - 2), E2M1 round-to-nearest, clip at 6), then requantized to 2 bits on the same block grid. It reproduces the measured anchors as follows (measured -> model): uniform 40.26 -> 39.4, per-block Lloyd 31.47 -> 30.2, sign-symmetric+j 37.94 -> 33.0. The first two match within 1 pp; the model is 5 pp optimistic on the j-search variant, and it under-weights per-block adaptivity (it predicts -8.6% for per-block Lloyd where the real weights give -17.1%). Read the **relative** deltas as a lower bound for adaptivity-type changes and as a fair estimate for shape-only changes. One script, one framing: `work/quant/sim/q2final.py`.

| Scheme | Model rel-RMS | Delta vs current | What changes in the kernel |
|---|---:|---:|---|
| {0.5, 1.5} + j in 0..3 (current) | 33.02% | - | - |
| {0.5, 2.0} ratio 4, + j | 32.07% | -2.9% | 2 constants |
| {1.0, 4.0} ratio 4 (= upstream {-4,-1,1,4}) | 32.18% | -2.6% | 2 constants |
| {0.5, 1.0} ratio 2 | 38.80% | +17.5% | 2 constants |
| {0.5, 3.0} ratio 6 | 36.41% | +10.3% | 2 constants |
| per-block (a,b) pair from the E2M1 grid, + j | 30.73% | -6.9% | +1 byte/block, 1 extra MAD |
| {0.5, 1.5}, scale on a 2^(1/2) grid | 31.02% | -6.1% | scale byte reinterpreted |
| {0.5, 1.5}, scale on a 2^(1/4) grid (E4M3-like) | **28.99%** | **-12.2%** | scale byte reinterpreted |
| {0.5, 1.5}, scale on a 2^(1/8) grid | 28.76% | -12.9% | same, no longer 1 byte |
| per-block free Lloyd, 4 levels | 30.18% (measured 31.47%) | -8.6% (measured -17.1%) | 8 B/block + divergent LUT |
| per-block free Lloyd, 8 levels (3-bit) | 15.22% (measured 16.61%) | -53.9% (measured -56.2%) | 2x the bytes |

Three findings that answer the question directly.

1. **Sign symmetry looks nearly free, and this is the least certain claim in the report.** In the model, the sign-symmetric codebook with a near-continuous scale (28.76% at 2^(1/8)) beats a free 4-level per-block Lloyd codebook (30.18%), because 2 parameters fitted on 32 samples overfit less than 4 do. On the real weights per-block Lloyd does much better than the model predicts (31.47% measured, a -17.1% gain against the model's -8.6%), so the model is not trustworthy on per-block adaptivity. Read this as: the symmetry constraint is probably cheap, not proven free.
2. **The scale grid is what costs us.** Refining the scale from 2^j to a 3-mantissa-bit float recovers 12% of the error in the model, more than the ratio change and more than the per-block-pair change, at zero extra bytes. Because the model under-weights per-block adaptivity, the true gain on real weights is likely larger than 12%; it is also the only option here whose whole cost is one byte reinterpretation.
3. **The codebook ratio interacts with the scale grid.** With a coarse power-of-two scale, ratio 4 wins because a wider codebook partially compensates for a scale that can only move in factors of 2. With a fine scale, ratio 3 wins, because the optimum shape is the Lloyd-Max ratio of about 3.3. Do not change both at once without re-measuring.

### 5.3 Activation-aware weighting vs plain weight error

No clean public ablation at 2 bits exists (see §3). What the evidence supports:

- Hessian-weighted rounding (GPTQ) vs plain RTN is worth 3.4x in perplexity at 3 bits and 9x at 2 bits on LLaMA3-8B, but both are still broken at 2 bits, so the ratio is not a useful guide to how much it buys once a good codebook is in place.
- Calibration *data* choice is worth 28% -> 65% accuracy at INT3 ([2606.25519](https://arxiv.org/abs/2606.25519)), which is far larger than any codebook tweak in this report. That is a warning: if we ever add a calibration set, the corpus matters more than the algorithm.
- For our format the actionable part is narrow but free: choose the per-block scale exponent j and the per-element code to minimize a Hessian-weighted error (sum_k H_kk (w_k - ŵ_k)^2) instead of the plain L2 error. Same 4 levels, same byte layout, offline only. Magnitude unknown; measure it.

---

## 6. Q4: speculative decoding acceptance and output verbosity

**Verbosity: yes, this is a published, named effect, and our magnitude is normal.** "Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models" ([2606.25519](https://arxiv.org/abs/2606.25519)) defines the CoT Token Inflation Ratio (CTIR) and reports, on Qwen3-30B-A3B (a 30B MoE):

| Setting | CTIR (avg over 6 reasoning benchmarks) |
|---|---:|
| INT4 GPTQ | +3.7% |
| INT4 HQQ | +3.9% |
| INT4 RTN | +43.0% |
| INT3 GPTQ | +19.0% |
| INT3 AWQ | +32.8% |
| INT3 HQQ | +60.1% |

and on Qwen3-4B (dense), INT3 AWQ reaches +276% and INT3 HQQ +292%. Their mechanism analysis is the useful part: inflation comes with *more intermediate steps and more semantic repetition*, it is stronger on instances the fp16 model answers correctly, and it is not merely longer output, it is a changed reasoning trajectory. They also show the serving cost: a 1.1x CTIR raises p90 TTFT from 1050 s to 1400 s and drops throughput from 0.055 to 0.047 req/s.

Our +27% to +36% (849,225 output tokens against 622,883 / 671,210, `work/STATUS.md` round 71) sits between their INT3 and INT4 numbers for a 30B MoE, on a model 17x larger, at 2 bits. So: expected, consistent with the literature, and not a sign of a bug in our requant.

Note also that our own repo already treats completion-token inflation as a quality signal: `work/quant/PLAN-1000.md` sets a "+2% completion-token inflation" kill criterion and calls +8-11% "a real quality bug that accuracy alone did not catch". By that standard our +27-36% is a large behavioural change, even though GSM8K moved only -0.7 pp.

**Acceptance rate: sensitive to the codebook, and sign bias can zero it.** Two pieces of evidence.

1. Upstream's own table for the v0.24 2-bit line on a 159B DeepSeek-class model (`work/quant/PLAN-1000.md`, from the repo's `docs/quality.md`): FP4 experts give MTP accepted length 2.68 and draft acceptance 84.1%; a 3-bit K=8 codebook 2.60 / 79.8%; a naive 2-bit *asymmetric* codebook collapses to accepted length 1.00 / acceptance 0.0% and breaks 12 of 12 coherence prompts; a 2-bit sign-symmetric {-4,-1,1,4} codebook gives 2.73 / 86.3% and 12 of 12 coherent. Their conclusion: "symmetry, not granularity, is the lever". This is the single most relevant published-or-internal datapoint for our kernel, because it is the same model family and the same 4-level format.
2. Our own A/B (`work/STATUS.md` round 76): requantizing both the target and the DSpark drafter to the 2-bit codebook gave step 8.84 -> 8.47 ms (-4.2%) but accepted length 2.80 -> 2.68 (-4.3%), i.e. net zero tok/s. The acceptance loss came from the drafter, and keeping the drafter at 4 bits is the obvious next move.

The general literature agrees that acceptance is the fragile part: "GPTQ quantization to the draft model leads to substantial degradation of the acceptance rate" ([2505.22179](https://arxiv.org/abs/2505.22179)). I could not extract that paper's acceptance-rate tables (they did not survive text extraction), so I am quoting the sentence, not a number.

**What is not established:** whether acceptance and verbosity move monotonically with weight error. Nobody has published a CTIR-vs-weight-RMS curve. If reducing 37.94% to ~33% also cuts inflation, that would be worth more than the accuracy gain; it is a cheap experiment on the harness we already have.

---

## 7. Q5: ranked recommendations

Ranked by (error reduction) / (implementation effort), with the offline requant cost held constant across all of them: any of these needs the same 4-hour pass over the 92,160 routed-expert tensors, so only the kernel delta counts.

| Rank | Change | Est. error reduction | Effort | Notes |
|---|---|---:|---|---|
| 1 | **Reinterpret the scale byte as E4M3 instead of E8M0** (3 mantissa bits) | **-12% relative in the model**, so 37.94% -> roughly 33-34% on real weights | 1-2 days: requant writes a float scale, kernel converts e4m3 -> float instead of 2^e | Zero extra bytes. That is about half to two thirds of the per-block-Lloyd gain (31.47%) at a tenth of the kernel cost. Verify the scale is converted in-kernel and not fed to a hardware MX scale-factor operand; if it is, fold it into the epilogue as a per-block fp32 multiply |
| 2 | **Sweep the codebook ratio on real weights and pick the best 4 E2M1 codes** | -2.9% relative if the scale stays power-of-two; ~0 if rank 1 is adopted | 4 hours offline + 2 constants | Ratio 4 ({0.5,2} or {1,4}) wins with a coarse scale; ratio 3 wins with a fine scale. Upstream ships {-4,-1,1,4}, independent evidence for ratio 4 at coarse scale |
| 3 | **Hessian/imatrix-weighted choice of scale exponent and codes** | unknown, plausibly a few % of *output* error, which is what benchmarks see | 1-2 days offline, 0 kernel | Same format. Needs a calibration set and an H diagonal. Highest information per hour of the cheap options |
| 4 | **Per-block (a,b) codebook pair from the E2M1 grid** | -6.9% relative | 3-5 days: +1 byte/block, expansion becomes 1 MAD more, requant searches 84 (a,b,j) candidates per block | The "tensor-sym per block" upgrade. Dominated by rank 1 on error-per-effort, but stackable with it |
| 5 | **Per-block free Lloyd codebook** (4 fp16/block) | -17.1% relative (measured 31.47%) | 5-10 days: 8 B/block, and the LUT must live in shared memory (a `__constant__` LUT cost +8 ms/step) | The best 4-level result we have measured, but the most expensive per point of error |
| 6 | **3-bit** (2-bit plane + 1 refinement bit) | -56.2% relative (16.61% measured) | 6 days | 1.5x the bytes, so it forfeits the reason for 2 bits. Note the 2-bit FC1 path only netted -4.2% step before the AL loss, so bytes are not currently the binding constraint |
| 7 | **Keep an FP4 delta/recovery pool for hot experts** | does not reduce weight RMS | already exists upstream | Fixes specific failures (the 17x23-100 probe) rather than the average. Orthogonal to ranks 1-6 |

Two things I would do before any kernel work, because they are cheap and they bound the value of everything above:

- **Run the offline sweep first.** `work/quant/re-quant2.py` already computes relative RMS on real blocks. Extend it to score (codebook ratio) x (scale granularity) x (weighted vs unweighted) in one pass. That converts every estimate in this report into a measurement for the cost of one VM boot.
- **Measure whether error reduction moves token inflation.** Rerun the 1,319-question GSM8K with the best rank-1 config and compare output tokens (849 k baseline). If inflation drops with the error, the ranking above is undervaluing it.

**What I would not do.** Do not adopt QuIP#, AQLM, QTIP, VPTQ or IQ2 for this kernel. On Llama-2-70B at 2 bits they land within 0.16 perplexity of each other (3.78 to 3.94) and all remain +0.66 to +0.82 over fp16, and each requires either a rotation around the GEMM or a codebook larger than L1. Our kernel's measured sensitivity to lookup cost (+8 ms for a divergent constant-memory LUT) makes that trade worse for us than for anyone else.

---

## 8. Sources

- QTIP, trellis-coded 2-bit, distortion table and Llama-2 perplexities: https://arxiv.org/abs/2406.11235
- QuIP#, E8P lattice codebooks, k-means comparison (appendix C.3): https://arxiv.org/abs/2402.04396
- AQLM, additive quantization with learned codebooks: https://arxiv.org/abs/2401.06118
- AWQ: https://arxiv.org/abs/2306.00978
- GPTQ: https://arxiv.org/abs/2210.17323
- HQQ, 2-bit Llama-2-70B vs fp16 13B: https://dropbox.tech/machine-learning/halfquadratic-quantization-of-large-machine-learning-models
- VPTQ, extreme low-bit vector PTQ: https://arxiv.org/abs/2409.17066
- "How Good Are Low-bit Quantized LLaMA3 Models?", 2-3 bit GPTQ/AWQ/QuIP tables: https://arxiv.org/abs/2404.14047
- BitNet b1.58, ternary QAT: https://arxiv.org/abs/2402.17764
- Unsloth Dynamic 3.0 GGUFs, 2-bit MMLU and KLD tables: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
- Unsloth Dynamic v2.0: https://unsloth.ai/blog/dynamic-v2
- llama.cpp imatrix README: https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
- llama.cpp imatrix discussion: https://github.com/ggml-org/llama.cpp/discussions/5263
- llama.cpp quant sweep on Llama-3.1-8B (3-bit and up): https://arxiv.org/abs/2601.14277
- OCP Microscaling Formats v1.0 (MXFP8/6/4, MXINT8; no MXFP2): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microscaling Data Formats for Deep Learning: https://arxiv.org/abs/2310.10537
- M²XFP, metadata-augmented microscaling: https://arxiv.org/abs/2601.19213
- "Is Finer Better? The Limits of Microscaling", ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/file/97ca7168c2c333df5ea61ece3b3276e1-Paper-Conference.pdf
- PagedMoE, routed-output error at 2/3/4 bits on a 397B MoE: https://arxiv.org/abs/2611.03173
- RotaryQuant, 2-bit routed experts on a 26-120B MoE: https://arxiv.org/pdf/2608.08081
- "Quantization Inflates Reasoning", CoT token inflation ratios: https://arxiv.org/abs/2606.25519
- "Speculative Decoding Meets Quantization": https://arxiv.org/abs/2505.22179
- D²Quant, sub-4-bit PTQ (numbers not verified): https://arxiv.org/html/2602.02546v1

Repo evidence used: `work/STATUS.md` rounds 65, 66, 70, 71, 75, 76 (weight RMS, GSM8K-1319, output tokens, step and AL measurements); `work/quant/re-quant2.py` (uniform and per-block Lloyd error on real experts); `work/quant/PLAN-1000.md` (upstream v0.24 acceptance table, PagedMoE and RotaryQuant summaries, the +2% token-inflation kill criterion); `work/quant/sim/` (this report's calibrated model).
