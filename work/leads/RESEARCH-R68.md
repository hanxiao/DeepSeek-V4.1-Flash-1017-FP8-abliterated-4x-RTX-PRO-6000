# Round 68 research pack — two passes, both read-only, 2026-11-10

Both passes were run against the question "what training-free, exact change raises single-stream
decode tok/s on this stack". Their conclusions are used in `work/PLAN-600.md` and `work/STATUS.md`
round 68. Kept here because the negative results are as load-bearing as the positive ones.

---

## Pass A — TensorFold / MiaAI-Lab patch-stack mining (112 engine patches, 2 recipes)

Sources: github.com/ashhart/TensorFold (PR #2257, #2162, #1524, #1623, #1752), the two MiaAI-Lab
recipes and their `docs/speed/RESULTS.md` + `docs/handoff.md` + `docs/glm.md`, patch files
`patches/engine/0001-0098`.

**Headline: our window is already optimal.** Fitting our measured points gives a per-draft
acceptance p ~ 0.752; maximising tokens/ms over depth gives r=3 -> 0.348, **r=5 -> 0.364**, r=8 ->
0.340. We ship r=5. TensorFold's +20 % from 32-row windows was bought at a fixed/marginal cost ratio
b/a = 0.052; ours is 0.101, i.e. we are row-cost-dominated and they are fixed-cost-dominated.

Ranked transferable items (all bit-exact):

| # | item | measured delta | source | port difficulty |
|---|---|---|---|---|
| R1 | one-shot IPC peer-store all-gather replaces NCCL all-gather, 90 sites | 3.2-3.6 us vs ncclAllGather 10.3 us at 16 KiB/rank; 0.25-0.77 ms/step at 90 sites | patch 0034, docs/glm.md:40,53 | TP2-only upstream; TP>=3 uses RoCE (#1524 +2.57 % at TP3); generalise to world-4 IPC |
| R2 | PDL chain along the whole decode | 0069-0071 +3.1 %, 0082 +1.0 %, 0083 +0.3 %, 0084 +0.4 % cumulative on sm_120 = ~0.30-0.35 ms | docs/handoff.md:122-127 | hard: needs griddepcontrol inside CUTLASS/FlashInfer kernels |
| R3 | DSA decode select on many CTAs | +3.3 % step at 200K (~+2-3 % at 120k); TRAP -11.7 % at ~690K single row | patch 0067/0091 | medium |
| R4 | decode glue fusions (greedy rows one launch, two-problem projections, router select+group, shared-window segments) | +1.6-2.2 % plain step = 0.11-0.16 ms | 0072-0075 | medium; microbench and end-to-end disagree 3x |
| R5 | DeepSeek-specific: back up only the ratio-2 recurrent carries before a window (89 device copies, 6.4 MB/window on Flash-1017) instead of every layer's ring | ~0.2-0.27 ms = +2-3 % | TensorFold CHANGELOG #1752 | pure host-side loop; check whether our vLLM tree has the same redundancy |
| R6 | shared-expert side stream | 1.4 % at 1 stream = 0.13 ms (GB10 falsification; microbench says 70.6 -> 60.6 us/layer) | patch 0006/0010 | easy in vLLM (second stream + event) but the launch lives inside b12x |
| R7 | L2 prefetch of the next layer's weights on a side stream | +3.02 % TP3 with drafts, +0.76 % TP2 | PR #2162 | 0-0.3 ms for us; discrete 96 GB cards have more L2 |
| R8 | load-timed draft-window cost table instead of frozen seeds | the seeds 9.4 + 1.31 were stale; the rule was already within noise of the best width | 0036/0038/0039/2257 | transferable half only |

Not applicable: EXL3 trellis items (we are FP4), all KDA items (V4.1 has no gated-delta layers),
multi-stream items, RoCE/Thunderbolt, DENSE=q8/HEAD=q8 (re-quant), power-cap skew (+10-15 % on
GB10 with a desktop GPU; our four cards are identical and no clock skew was found).

Unverified: 0034 is never A/B'd end-to-end; 0074's 13.8 us -> <1 us microbench contradicts glue5's
+1.6-2.2 % total by 3x; #2257 is a draft PR and its TP4 numbers are pre-rebase.

---

## Pass B — exact multi-draft / tree verification theory

Read: SpecTr (2310.15141), SpecHub (2411.05289), Hu et al. (2502.18779), Global Resolution
(2511.15898), SpecInfer (2305.09781), Sequoia (2402.12374), EAGLE-2 (2406.16858), Medusa
(2401.10774), DDTree, block verification (2403.10444), JetSpec++ (2606.18394), xPress (2608.02438),
LiLiCorr (2608.20530), DSpine (2609.36173).

**The decisive arithmetic.** Chain accepted length is exactly

    E[L] = 1 + sum_{i=1..k} prod_{j<=i} c_j

with our c = [0.81, 0.76, 0.73, 0.67, 0.72] this gives **3.393** at k=5 (we measure 3.31-3.44; the
2.5 % gap is EOS/averaging). A verified row must yield more than `row_ms * tok/ms = 0.585 * 0.345 =
0.202` tokens to pay for itself. Marginal yields by position: 0.810 / 0.616 / 0.449 / 0.301 / 0.217
then 0.142 — so **k=5 is the exact optimum** (k=4 = 0.3435 tok/ms, k=6 = 0.3158, k=5 = 0.3452).

**Multi-draft cannot pay at our row cost.** m i.i.d. chains buy breadth at depth 1 only; the
rigorous bound is `alpha_m - alpha_1 <= dTV(p,q) - eps`, and with c=0.81 (dTV 0.19, eps ~0.10-0.15)
the whole headroom is +4..9 pp, >=90 % of it captured at m=2. Cost: 5 extra rows (2.93 ms) + a second
draft pass (0.78 ms) for +0.25 tokens. Break-even would need FIX > 18.1 ms (ours is 5.54).

**Trees**: `E[L] = sum_v prod_{i in Path(v)} p_i` (Sequoia Prop 3.4), so a uniform B-ary tree gives
E = sum_d A^d with A = sum_k p_k. Breadth does buy acceptance, but the node count explodes: B=2,D=5
= 63 nodes = 52.5 ms = 0.082 tok/ms (4.2x worse than the chain). Trees pay only in the
fixed-cost-dominated regime (Sequoia's single-GPU dense setting), not ours.

**Accepted path is always an exact sample from p**, never "q restricted to the tree" — so the naive
"test candidates in order and accept the first that passes" IS valid RRS (SpecHub Alg.1), it does not
degenerate; what is invalid is applying the maximal-coupling test per candidate without the residual
update (SpecTr §4 counterexample p=Ber(1), q=Ber(1/2)).

**The one AL lever left is c_1**: dE/dc_1 = (E-1)/c_1 = 2.954 exactly, so +0.05 on c_1 is +4.3 % AL.
That is a proposal-calibration problem (q closer to p), which is what the draft-side logit scale and
the draft top-k truncation attack. **Measured this round: both are neutral (+-2 %, inside noise)** —
the shipped drafter is already well calibrated. See `work/STATUS.md` round 68.

Also from pass B: `synthetic_acceptance_length` is a non-exact but cheap way to price the whole 600
question (set it to 5.0 at k=7 and read the step); `dspark_draft_topk` needs `draft TP=1` in this
vLLM and the v41 draft model lacks `apply_markov_bias_gathered` (only Qwen3DSparkModel has it), so
E5 costs a model-side implementation; `Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft` is a
drop-in published fine-tune of mtp.0-2 worth +1.0 % thinking / +3.3 % all (its own numbers) — and its
card's real finding is that `draft_sample_method: probabilistic` (which we already serve) was worth
+13.7 % over `greedy` on their box.
