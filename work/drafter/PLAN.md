# PLAN: raise accepted length toward 5.5 and take single-stream decode to 600 tok/s

Owner: drafter (task-3). Date: 2026-11-08. Status: v3. Incorporates the leads' note
(`work/leads/notes/dspark-drafter-training.md`), the preliminary k-sweep pairs (k5 = 110 steps/s
at AL 3.45; k7 = 97 at 3.55) and the Lead's baseline per-position vector
(`work/bench/x-base/think.jsonl`, analysed in `work/drafter/analysis/`). Still pending the full
k-sweep per-position JSONL; section 6 says how to fold it in.

Number tags: **[M]** measured by us, **[R]** workspace README/Moet envelope, **[P]** published with
a URL in section 7, **[A]** my model with the procedure to replace it by a measurement.

**Bottom line.** At today's step time the goal needs accepted length 5.45, and accepted length is
`1 + sum s_i` with `s_i = prod_j a*r^(j-1)`. The Lead's baseline measurement (section 2.2) gives
**a = 0.79-0.81, r = 0.965** on the C1 workload: base rate and decay, not one flat alpha. That
shape caps the k=5 ceiling at ~380 tok/s today and makes every row past k=5 a loss. The row budget
allows only **a = 0.88-0.90 with r = 0.99 and a block trained at 7-9**: 492-524 tok/s at today's
step time, and **610-700 with FLy at W=1-2** (the multiplier is shape-invariant, so it multiplies
with the retrain). The binding constraint becomes FLy's quality, not the arithmetic. Position 1 is
the backbone's clean
measurement and the weak link is the parallel backbone at every position (section 2.5), so the
retrain objective is backbone accuracy on thinking traces. That is one training run plus one
config experiment. Cheap parallel win: **FLy** (config-only, +15-45% at the right window W=1-2,
+4-7% at vLLM's default W=4; lossy) and draft-logit shaping (top_p/top_k/scale, section 3).

---

## 1. The arithmetic of 600 tok/s

### 1.1 Definitions

With draft width `k` and accepted length `AL` (accepted draft tokens plus the bonus token, the
quantity `fraser_bench.py` reports as `mean_accepted_length = 1 + accepted/drafts`):

```
tok/s   = AL / T_wall                       T_wall = 1000 * AL / tok_s   (ms)
AL(k)   = 1 + sum_{i=1..k} s_i               s_i = alpha^i  for a flat conditional rate alpha
s_i     = pos_i / drafts                     from vLLM's per-position counter (section 2.1)
```

`T_wall` is the first-token-to-last-token wall clock per step, i.e. what the customer feels. It
depends on context: 9.13 ms at 30k/120k on C1, 8.84 at 60k, 8.77 mean on short prompts. The k-to-k
**difference** `dT/dk` is immune to that constant.

### 1.2 Today's operating point and its provenance

| quantity | value | tag | source |
| --- | ---: | --- | --- |
| single-stream decode, codebench C1, 30k / 60k / 120k | 371 / 375 / 376 tok/s | [M] | README §Engines, served r21 k5 FP8 |
| accepted length on that workload | 3.40-3.45 | [M] | Lead's C1 (3.416/3.468/3.522) and thinkbench C1-like (3.40) |
| **T_wall at k=5, C1 30k / 60k / 120k** | **9.13 / 8.84 / 9.13 ms** (109.5 / 113.2 / 109.5 steps/s) | [M] | Lead's measured C1: 374.7/392.5/385.6 tok/s at AL 3.416/3.468/3.522 |
| T_wall, short prompts (thinkbench, 13 requests) | 8.61-9.00 ms, mean 8.77 | [M] | `work/bench/x-base/think.jsonl` |
| k=7, same workload | 97 steps/s, AL 3.55 -> 344 tok/s | [M] | preliminary sweep |
| **effective row cost `dT/dk`** | **0.61 ms/row** k-marginal; 0.69 upper bound | [M] | k5/k7 sweep pair; Lead's C1 vs C2 (6 rows 9.13 ms, 12 rows 14.0 ms) gives the upper bound |
| C2 / C3, 30k | 243.9 / 200.2 per-stream tok/s at AL 3.41 / 3.47 -> 14.0 / 17.3 ms per step; aggregate 488 / 601 | [M] | Lead's measured C2/C3 |
| Fraser prompt, k=5 FP8 (served) | 534/521 tok/s, 111 steps/s, AL 4.8 | [M] | README §Fraser |
| Fraser prompt, k=5 FP4 | 543/537 tok/s, 113 steps/s, AL 4.7 | [M] | README §Fraser |
| Fraser prompt, k=7 FP4 + 64 GiB offload | 555 tok/s, 102 steps/s, AL 5.45 | [M] | README §Fraser |
| Moet's own host, k=7, 400 W + GDDR7 OC | 680 tok/s median, 127.8 steps/s, AL 5.34 | [P] | Moet README §Fraser |
| C1 step anatomy, -r21, k=7 (profiler) | 7.7 ms = MoE 2.7 + dense GEMV 0.9 + mHC 1.1 + drafter 0.75 + sparse attn/indexer 0.5 + 1-2% | [M] | `work/engine/src/moet/docs/dsv41-sm120-port.md:1930` |

**Row cost, the second-most important number after the acceptance vector.** Measured three ways:
- The k5/k7 sweep pair: **0.61 ms/row** [M] - the k-marginal, which is the right number for
  planning.
- Lead's C1 vs C2 decomposition: 0.69 ms/row [M] - an upper bound that also contains the second
  request's sampler/prep rows and the second draft chain's tail.
- Mechanism, split by the engine from the source + the repo's recorded-routing MoE measurement:
  **85-100% of a row is expert bytes.** The fused b12x kernel costs 2.2-2.6 us per extra distinct
  expert per layer (65.9 us/layer at 22.5 distinct, 95.4 at 36), one row adds ~5.5-6 distinct
  experts/layer (union saturation) -> +14 us/layer = **+0.57 ms/row**; the non-MoE row costs
  (mHC push payload, router+top-6, sparse MLA+merge, indexer top-k, sampler) sum to only
  0.05-0.15 ms; weight-bound parts (dense GEMVs, lm_head, Engram wkv, shared expert) are zero per
  row. So the row floor *without changing the expert format* is **0.47-0.55 ms**, not 0.45:
  driving 0.61 -> 0.45 is ~25% off the expert bytes, i.e. it is the 2-bit/4-bit expert-format work
  (engine L2), not row-overhead cleanup. `work/engine/FINDINGS.md` §5.
Use **0.61 ms/row** for planning, 0.69 as the upper bound (includes per-request extras),
0.47-0.55 as the format-unchanged floor, 0.45 only with the 2-bit experts.
Graph sizing (leads, verified): MRV2 captures FULL decode graphs in multiples of (k+1) - k=5 is a
6-row graph, k=7 an 8-row graph, no free padding, so every +1 k is a real row
(`vllm/v1/worker/gpu/cudagraph_utils.py:321-345`).

### 1.3 What 600 requires

| T_wall (ms/step) | steps/s | AL needed for 600 |
| ---: | ---: | ---: |
| 6.0 | 167 | 3.60 |
| 7.0 | 143 | 4.20 |
| 8.0 | 125 | 4.80 |
| 9.0 | 111 | 5.40 |
| **9.1 (today)** | **110** | **5.45** |
| 10.0 | 100 | 6.00 |
| 10.57 | 95 | 6.34 |

At k=5 the maximum possible AL is 6.0, so 600 at today's step time requires 92% of all drafts
accepted. **600 is a two-axis goal**, and the acceptance axis has to reach AL 5.4-6.0, which means
k>5 and a trained longer block.

### 1.4 Step time vs draft width

`T(k) ~= 9.1 + 0.61*(k-5)` ms (measured k-marginal; `9.1 + 0.69*(k-5)` on the C1-vs-C2
upper bound, `9.1 + 0.45*(k-5)` on the mechanism floor).

Fit procedure for the full sweep (10 lines of python):
1. per request: `T_wall = 1000*AL/tok_s`, `AL = 1 + accepted/drafts` from counter deltas.
2. median T_wall per k at 30k and 120k separately (context changes sparse attention cost).
3. least-squares `T(k) = a + b*k` over k=3..7; report b with its CI. If b is outside 0.35-0.65
   ms/row, re-derive sections 1.4-1.6.
4. replace `T5=9.1, b=0.61` everywhere below.

### 1.5 The acceptance model: base rate `a` and decay `r`

With a flat conditional rate `alpha`, `AL = 1 + sum alpha^i` saturates geometrically. The measured
vector (section 2.2) is better described by `c_i = a * r^(i-1)`, i.e.

```
AL(k) = 1 + sum_{i=1..k} prod_{j=1..i} a * r^(j-1)
```

| a | r | AL k=5 | AL k=7 | AL k=9 | AL k=11 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.812 | 0.972 (fit variant of measured C1) | 3.58 | 3.89 | 4.03 | 4.09 |
| 0.85 | 0.98 | 3.80 | 4.45 | 4.74 | 4.83 |
| 0.88 | 0.99 | 4.35 | 5.08 | 5.57 | 5.89 |
| 0.90 | 0.99 | 4.56 | 5.41 | 6.00 | 6.40 |
| 0.90 | 1.00 (flat) | 4.69 | 5.70 | 6.51 | 7.18 |

Two consequences that the flat model missed:
- **The base rate `a` is the dominant term.** Raising `a` from 0.81 to 0.90 at r=0.99 lifts AL(9)
  from 4.03 to 6.00 (+49%).
- **The decay `r` is worth ~8% at k=9**: at a=0.90, going from r=0.99 to r=1.00 lifts AL(9) from
  6.00 to 6.51.
- The model and the measured k=7 result agree only if the tail is worse than the in-block rate.
  Extrapolating the measured shape with the in-block average conditional (0.721) gives
  AL(7)=3.67; a geometric fit through the measured points gives 3.53-3.89 depending on the fit
  (a=0.793/r=0.965 -> 3.63; a=0.812/r=0.972 -> 3.89; the measured 3.55 implies a constant tail
  conditional of ~0.47 against 0.72 in-block). So the untrained-block deficit is real but modest:
  **+0.12 to +0.15 AL at k=7**, not the +0.3-0.4 a naive geometric extension suggests.

### 1.6 The frontier (decay-corrected)

tok/s with `T(k) = 9.1 + dT*(k-5)`. Left: dT=0.61 (measured k-marginal). Right: dT=0.45, which
needs the 2-bit/4-bit expert-format work (the format-unchanged floor is 0.47-0.55, engine §5).

| (a, r) | dT=0.61 best | dT=0.45 best |
| --- | ---: | ---: |
| 0.812, 0.972 (measured today) | 374 at k=5 (measured 375) | 395 at k=6 |
| 0.85, 0.98 | 437 at k=6 | 445 at k=7 |
| 0.88, 0.99 | 493 at k=7 | 513 at k=8 |
| 0.90, 0.99 | 525 at k=8 | 551 at k=9 |
| 0.90, 1.00 (flat) | 565 at k=10 | 609 at k=12 |

**FLy scenario (lossy, config-only): the multiplier is a function of the window W, and the
default is the weakest setting.** `can_defer` in `vllm/v1/spec_decode/fly.py` requires
`pos + W < num_draft_tokens` *and* all of the next W draft tokens natively accepted, so at k=5 the
number of deferrable positions is k-W: W=4 lets only position 0 defer, W=1 lets positions 0-3.
vLLM's default `fly_window_size = min(6, k-1) = 4` at k=5 is therefore the worst case. Simulated on
the measured c-vector (leads' simulation; my independent-position model reproduces the ordering and
lands 2-5 pp lower):

| W | deferrable positions | AL gain, g=0.5 / 0.8 / 0.9 / 1.0 |
| ---: | ---: | --- |
| 1 | 4 | +20-23% / +35-38% / +40-44% / +45-49% |
| 2 | 3 | +11-15% / +19-24% / +21-27% / +24-30% |
| 3 | 2 | +6-9% / +9-14% / +11-16% / +12-18% |
| 4 (default) | 1 | +2-4% / +4-6% / +4-7% / +4-7% |

Caveats: positions are treated as independent (they are not) and the gate probability g is
correlated with rejection, so the true column is nearer g=1; read the table as an ordering, not a
forecast. The W=1-2 acceptance rule is loose (it repairs any mismatch when the next token was
accepted and entropy is above threshold), so GSM8K-100 is the quality gate.

| scenario | AL | T (ms) | tok/s | 600? |
| --- | ---: | ---: | ---: | --- |
| today + FLy W=4 (default, 1.06x) | 3.55 | 9.1 | 390 | no |
| today + FLy W=2 (1.24-1.27x) | 4.30 | 9.1 | 473 | no |
| today + FLy W=1 (1.43x) | 4.95 | 9.1 | 544 | no |
| a=0.88, r=0.99, k=7 + FLy W=2 (1.24-1.32x) | 5.08 -> 6.3-6.7 | 10.32 | 610-650 | **yes** |
| a=0.90, r=0.99, k=8 + FLy W=2 (1.24-1.33x) | 5.73 -> 7.1-7.6 | 10.93 | 650-700 | **yes** |
| a=0.90, r=1.00, k=10 + FLy W=2 (1.24x) | 6.86 -> 8.5 | 12.15 | 700 | yes |

So with a good window FLy is worth +15-45%, not +8-14%: **at W=1-2 it carries 600 with a=0.88 and
r=0.99 alone** (610-660), i.e. the retrain target comes down and the arithmetic stops binding. The
multiplier is roughly shape-invariant (leads' model: 1.43/1.27 at W=1/2 on today's shape,
1.40/1.32 on a=0.88/r=0.99/k=7; my model 2-5 pp lower), so **FLy and the retrain multiply rather
than substitute** - a good E1 result should be applied to the retrained shapes at the same
multiplier, not a discounted one. At the default W=4 it is nearly worthless (1.06x). That is why
the arm order matters: W in {2, 1, 4-control} at threshold 0.3, then a threshold sweep at W=1.

Consequences:
1. At today's shape, k=5 is optimal and every extra row is a loss (measured k=7: 344 tok/s, -9%).
2. **The base rate alone, inside block 5, caps at ~450-510 tok/s** (a=0.90, r=0.99-1.0, k=5) - a
   better drafter inside the same block does not reach 600.
3. **a=0.88-0.90 with r=0.99 and a block trained at 7-8 gives 492-524 tok/s at today's step
   time; adding FLy at W=2 gives 610-700.** So 600 is reachable on the retrain + FLy alone; the
   remaining risk is quality (FLy is lossy) and the retrain landing a=0.88, not the arithmetic.
   r=1.0, dT=0.45 and the 2-bit experts are now upside, not prerequisites.
4. Row budget: a row pays iff `s_k > AL*dT/T` (~0.23 tokens/row at dT=0.61). At today's shape rows
   1-5 pay (hence k=5); at a=0.88/r=0.99 rows 1-8; at a=0.90/r=0.99 rows 1-9; at a=0.90/r=1.0 rows
   1-12. **Train block_size = 9, not 16.**
5. **k and the expert format are one trade, not two.** 85-100% of a row is expert bytes and the
   expert bytes are also the engine's L2 target. With a 0.47-0.55 ms row (format unchanged) the
   a=0.90 frontier is 524-535 tok/s; only the 2-bit experts (0.3-0.45 ms/row) open the 550-600
   region. The drafter retrain and the engine's expert work are coupled: neither reaches 600 alone,
   and the retrain's block-size part is worthless without the row cost coming down.

---

## 2. Per-position acceptance

### 2.1 Reading the counter

`vllm:spec_decode_num_accepted_tokens_per_pos_total{position=i}` counts accepted tokens at draft
position `i` (0-based). With `drafts` = `vllm:spec_decode_num_drafts_total`:

```
s_i  = pos_i / drafts          unconditional survival (alpha^i for a flat rate)
c_i  = s_i / s_{i-1}           conditional acceptance, s_{-1}=1
AL   = 1 + sum s_i = 1 + accepted/drafts
```

`bench/thinkbench.py:51-53` already scrapes the full vector (8 slots) and splits it into
`spec_think` / `spec_answer` at the first answer token (`thinkbench.py:143-150`). That split is the
most valuable cut we have, because **acceptance is a property of the generated text, not of
context depth**: the GLM-5.2 probe in the Moet repo shows acceptance flat from 8K to 891K context
([P] `work/engine/src/moet/docs/benchmarks/mtp-acceptance-depth-2026-07-13.md`). Our flat
371/375/376 across 30k/60k/120k is the same phenomenon: the workload, not the depth, sets alpha.

### 2.2 The measured vector (Lead's baseline, 2026-11-08)

`work/bench/x-base/think.jsonl` (13 requests, k=5, temp 1.0, top_p 0.95) gives the first direct
per-position measurement on this stack. Aggregate over the two codebench-like requests
(`cb30` 30k-context code task + `code1`), 1,962 draft steps, which is the C1 workload:

| i | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_i` measured | 0.8124 | 0.6162 | 0.4495 | 0.3022 | 0.2192 |
| `c_i` measured | 0.8124 | 0.7585 | 0.7295 | 0.6723 | 0.7251 |
| `s_i` geometric a=0.793 | 0.793 | 0.629 | 0.499 | 0.396 | 0.314 |

`AL = 1 + sum s_i = 3.40`, which matches the C1 medians in `work/bench/x-base/summary.json`
(3.416 / 3.468 / 3.522 at 30k / 60k / 120k). Full analysis in
`work/drafter/analysis/pos-vector.txt`.

**The conditional rate is not flat: it decays.** A fit `c_i = a * r^(i-1)` gives **a = 0.793,
r = 0.965** (SSE 0.0035, reproduces AL(5)=3.39 vs 3.40 measured). Position 1 is the *best*
position, not the worst; each later position loses ~3.5% conditional acceptance. Across all 13
requests the fit is a = 0.70, r = 0.95 and the per-prompt spread is large:

| prompt | AL | `c_1` | `c_5` | fit r |
| --- | ---: | ---: | ---: | ---: |
| cb30 (30k code) | 3.44 | 0.820 | 0.800 | 0.99 |
| code1 | 3.40 | 0.812 | 0.717 | 0.97 |
| math1 | 3.35 | 0.787 | 0.668 | 0.96 |
| logic1 | 3.29 | 0.800 | 0.622 | 0.94 |
| zh1 | 3.09 | 0.755 | 0.691 | 0.97 |
| plan1 | 2.38 | 0.630 | 0.545 | 0.96 |
| prose1 | 2.36 | 0.637 | 0.509 | 0.93 |

So the earlier flat-alpha model (from the README k-sweep) was an approximation that only holds
for the first two positions. The correct model has **two parameters: the base rate `a` and the
decay ratio `r`**.

**Structural reading (Lead's, and the data supports it).** In DSpark, position 1 is sampled from
the parallel backbone's anchor query alone; positions 2-5 are sampled from the backbone logits plus
the Markov head's first-order bias. Measured: `c_1` = 0.62-0.82 and varies by prompt type more than
the tail does (`c_5` = 0.51-0.80, and on C1 `c_5` is actually higher than `c_4`). The tail decay
is mild; the weak link is the **parallel backbone at every position**, and position 1 is its
cleanest measurement. The retrain objective is therefore *backbone accuracy*, not Markov-head
sharpening. Positions 6-9 are still unmeasured (the counter only reports what was drafted), and
are expected to be worse than the in-block shape because they are untrained (`dspark_block_size:
5`).

### 2.3 What it implies for k

Marginal value of the k-th row = `s_k`; marginal cost = `dT = 0.61 ms`. The average today is
2.68 ms/token (9.1/3.40), so a row pays iff `s_k > 0.23`.

| k -> k+1 | s_k (measured shape extended) | ms per extra token | verdict |
| ---: | ---: | ---: | --- |
| 5 -> 6 | 0.19 (if trained; ~0.03 measured today) | 3.2 (today: 20) | lose 5-8% today |
| 6 -> 7 | 0.13 (if trained) | 4.8 | lose 9% today (measured) |
| 5 -> 7 on Fraser | 0.65 over 2 | 1.9 | win 4% (measured) |

In cost form (engine's inequality): a row pays iff `s_k * step > AL * dstep`, i.e. at s_5=0.219,
step 9.1 ms, AL 3.40, the k=6 row needs `dstep < 0.42 ms` if the tail conditional is the in-block
0.72, and `< 0.28 ms` at the measured tail 0.47. The measured row cost is 0.55-0.69 ms, so k=6
loses ~3-8%. **A block-9 retrain that only fixes the tail (alpha unchanged at 0.81) gives
AL(9)=3.81 for +4 rows, i.e. ~331 tok/s vs 374 - a 12% loss.** The block-size work therefore only
pays *together with* the alpha lift: at a=0.90/r=0.99 the row marginals are 0.27-0.46 and the rows
pay (section 1.6). Keep k=5 until alpha moves; the block retrain is not an independent win.

### 2.4 Confidence-gated draft length: cannot pay on this path

DSpark's answer to tail waste is confidence-scheduled verification: a confidence head predicts
`c_i = 1 - TV(p_target, p_draft)` and a hardware-aware scheduler trims each request's verify
prefix ([P] arXiv 2607.05147 §3.2). On our box it is dead for three independent reasons:

1. **Not supported.** `enable_adaptive_verification=true` fails at boot because
   `DeepseekV41IndexerBackend` does not support device-side trimming (Moet port doc gap #6). The
   served config sets it false.
2. **Measured negative.** Confidence-scheduled verification costs 14% of single-stream code
   decode on SM120 ([M] README §Engines). Its win requires verification to be cheap per row; here
   it is 0.61 ms/row.
3. **No batch to protect.** The scheduler optimises aggregate throughput under load; at batch 1,
   trimming rows just changes k, which is a static choice.

Do not spend VM time on adaptive verification. (The confidence head still has diagnostic value:
its per-position output is a cheap proxy for `c_i` if we ever want it offline.) The served image's
`SpeculativeConfig` has 41 fields; the only dspark-specific one is `dspark_draft_topk` - a
base-logit top-k for the *probabilistic* path too (`_sample_sequential_topk` feeds the truncated
distribution into the sampler and the rejection cache unchanged), but gated to Qwen3 architectures
and draft TP1 (section 3, cheap lever 2). There are no scheduler knobs (`dspark_scheduler`,
`dspark_per_request`, `dspark_confidence_threshold`, `dspark_budget_frac` are in Moet's v0.24.0
patch, not our nightly image). `num_speculative_tokens_per_batch_size` exists but batch=1 makes it
a constant.

### 2.5 What the position-1 dominance prices (Lead's questions, corrected arithmetic)

Let `B = (AL-1)/c_1` be the "tail bracket": `AL = 1 + c_1 * B`. On C1, `B = 2.954`; on math1,
`B = 3.035`.

1. **Tail-work ceiling.** If the Markov head made positions 2-5 perfect (`c_2..c_5 = 1`), then
   every `s_i = c_1` and `AL = 1 + 5*c_1`: C1 5.06 (+49%), math1 4.90 (+47%). The Lead's estimate
   of +24% used four terms instead of five; the tail ceiling is ~2x larger than stated. It is also
   not reachable by the Markov head alone: the head is a rank-256 first-order bias, it cannot
   recover the backbone's base-logit error for a specific prefix.
2. **Position-1 sensitivity.** `dAL/dc_1 = B = 3.0`. So +0.05 on `c_1` = +0.15 AL (+4.5%), and
   `c_1` 0.78 -> 0.88 = +0.30 AL (+9%), not +12-15%. Position-1-only ceiling (`c_1` -> 1) is
   +16% (C1) / +20% (math1).
3. **Retrain objective.** Raise the backbone's per-token accuracy (capacity, features, thinking
   data) so that `c_1` and the tail conditionals move together; keep the tail. A drafter that only
   sharpens the Markov head is worth at most the ~+47% tail ceiling, and in practice much less
   because the backbone's error dominates there too.
4. **k>5 is dead today and only revives with a better backbone.** At the measured shape the row-6
   marginal is `s_6 = s_5 * c_6 ~= 0.14-0.19` (with `c_6` ~0.65-0.85), against the 0.24 bar at
   dT=0.61: 3.6-4.4 ms/token versus 2.64 ms/token average. Even a trained block does not pay until
   `a >= 0.88` (section 1.6, rule 4).

---

## 3. Ranked options

Ranked by expected tok/s per unit of engineering + VM time, at today's `T5=9.1 ms`. `AL gain` is
for the agent workload (thinking at max effort, temp 1.0, top_p 0.95), not Fraser.

| # | option | mechanism | AL gain | tok/s | cost | lossless | risk |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | **FLy verification** | defer a rejection at a high-entropy position when the next W drafts are accepted natively | +10-25% | 414-470 | 1 config line + 1 boot + 1.5 h bench | **no** (min retention 96.3%) | low, config-only |
| 2 | **Retrain the MTP head at block 9, 5 layers** | raise a 0.79->0.88-0.90, flatten r 0.965->0.99, train positions 6-9 | +37-76% AL | 492-565 | 3-6 eng-days + 10-15 box-h per run | **yes** | high: two negative fine-tunes on record |
| 3 | top_p-matched drafting | drafter samples untruncated, target is top_p 0.95 -> recover up to 5 pp | +2-4% | 384-391 | 20-line patch + 1 boot | yes | low |
| 4 | ngram/prompt-lookup hybrid | copy runs from the 30k-120k context | +2-6% | 384-398 | 1-2 eng-days, bound offline first | yes | low value on thinking text |
| 5 | tree / branch drafting | verify m candidates per position | published +23-32% AL, but rows 6 -> 11 | 384-417 | 3-5 eng-days, patches speculator + verify + indexer | yes | PCTree measured 0.037 accepted/node vs 0.23 bar |
| 6 | k=7 with the current head | 2 more rows, `s_6+s_7 = 0.075` | +3% | 344 | free (already measured) | yes | loses 9% |
| 7 | drafter on 1 GPU | removes ~4-5 collectives from a 0.75 ms draft graph | 0 | +1% | 1 boot, +7.2 GiB on rank 0 | yes | memory |
| 8 | adaptive verification | section 2.4 | 0 | -14% | - | no | dead |

### (a) Retrain the MTP head (the only lossless route to 600)

**Why the current head is stale.** The served head is `mtp.0/1/2` inside the pretrained checkpoint:
3 layers, `dspark_block_size: 5`, aux hidden states from target layers 37/38/39, Markov rank 256,
its own MoE at 128 experts top-3, `dspark_noise_token_id: 128799` ([P] checkpoint config). It was
trained jointly with the base model, at block 5, on pretraining data. The served model is
post-trained (thinking at max effort, abliterated), and **positions 6+ were never trained at all**,
which is exactly what the k=7 shortfall shows (3.67 predicted at the in-block rate, 3.55
measured). The same head
reaches alpha 0.90-0.93 on thinking-off code, so capacity exists; the mismatch is text regime and
block length.

**The target is two numbers, not one: a = 0.88-0.90 and r = 0.99, with the block trained at 9.**
The measured vector (section 2.2) decomposes into a base rate (position 1: 0.81 on C1, 0.64 on
prose) and a per-position decay (r = 0.965 on C1, 0.93 on prose). Per section 2.5, position 1 is
the backbone's clean measurement and the tail conditionals are backbone-dominated too, so the run's
real objective is **backbone per-token accuracy on thinking traces**, with the Markov head's
decay as a secondary target. Both ride one training run:
`block_size` is a DeepSpec hyperparameter (its released checkpoints are block 7; the paper trains
block 16 for its scaling study) [P].
- For the base, in this order: (1) add the post-norm LM-head input as a 4th aux feature (the
  representation the target's own next-token distribution is computed from - targets position 1 by
  construction); (2) switch the aux composition from feature-wise to sequence-wise (DLite,
  arXiv 2610.23787 Table 5: 3.04 -> 3.30, +8.5%); (3) 7-14 target layers instead of 3 (+5%,
  saturating past 14); (4) 5 draft layers instead of 3 (DeepSpec's default). Do the Markov rank
  change last: it is a first-order bias and the smallest of the four.
- For the decay: the sequential/Markov head is the only thing that can condition position i on
  positions <i. DLite's ablation has rank 256 -> 512 helping (512 -> 4096 marginal) [P], and
  SpecOPD's on-policy distillation is the published recipe for pushing a plateaued drafter beyond
  its acceptance-length plateau (up to +41.45% mean AL on DSpark/DFlash) [P].
- Do not chase DLite's headline "+70% over DSpark on agentic": the leads' read is that their DSpark
  baseline degrades, and their own SWA-128 vs SWA-512 result says a 128-token window costs nothing
  in acceptance once trained - which we already have.

**What must be different from the fine-tunes that failed.** The closest precedent is
`Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft` (2026-10-29) [P]: it fine-tuned exactly our three
layers (`mtp.0`-`mtp.2`, routed experts and embeddings frozen) on agent sessions and
Open-PerfectBlend prompts regenerated by the served model, target features from layers 37-39,
loss 0.1 CE + 0.9 L1 with gamma 4, Markov rank 256, confidence head, 1,140 steps, lr 6e-5, seq len
640, 256 anchors, 2 GPUs, **block size 5**. Result: +3.3% accepted length overall, **+1.0% in
thinking mode at temperature 1** (held-out), tok/s flat. SubconsciousDev's retrain was 1-3% worse
on reasoning, 4% on code [M].
- So the run must change what they did not: **(i) block size 9-10** - they kept 5, so positions 6+
  stayed dead and the extra rows could never pay, which is the single largest difference between
  their result and the 600 target; **(ii) capacity: 5 draft layers instead of 3** (DeepSpec's
  default is 5; two extra layers cost ~0.5 ms of draft graph, +5% step, cheap if alpha moves);
  **(iii) the loss is already TV-heavy** in the official recipe (`ce_loss_alpha=0.1,
  l1_loss_alpha=0.9`), so do not expect the "TV instead of CE" gain; TVP's fair comparison is
  +6.2-12.9% over the public DFlash2 recipes [P].
- Their card also confirms the proposal-distribution lever from the other direction: switching
  `draft_sample_method` from greedy to probabilistic took their thinking-mode decode from 59.7 to
  67.9 tok/s at 1 user (+13.7%) with either drafter [P]. We already serve probabilistic, so that
  win is banked - and it is the same mechanism as E3 (top_p matching), which is why E3 is worth an
  hour even though its expected gain is single-digit.

**Recipe (DeepSpec's own, `config/dspark/dspark_qwen3_4b.py`, [P]):**
`block_size=9, num_draft_layers=5, target_layer_ids=[37,38,39], num_anchors=512, markov_rank=256,
markov_head_type='vanilla', confidence_head_alpha=1.0, confidence_head_with_markov=True,
loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9, lr=6e-4, warmup_ratio=0.04, bf16,
global_batch_size=512, num_train_epochs=10, max_length=4096, DDP`. DeepSpec ships DSpark
draft-model code for Qwen3 and Gemma4 only, so a DeepSeek-V4.1 target needs a new
`deepspec/modeling/dspark/deepseek_v41/` (the head is the same: backbone + Markov head +
confidence head; the backbone class is V4.1-specific with mHC + MoE + MLA). That port is the
engineering cost; alternatively wrap the checkpoint's own `inference/model.py`, which already
implements the DSpark forward in plain PyTorch and loads the native FP8/FP4 layout [P].

**Self-fed inputs (the part that makes the Markov chain right).** At inference the drafter's
sequential stage samples left to right and feeds its *own* sampled token into the Markov head at
the next position (`dspark/speculator.py:156-207`); the backbone, however, sees the target's
hidden states and the anchor. Training must match that: for depth i, the previous-token input is
the token the chain actually produced at depth i-1 (the target's own sampled token on the
on-policy trace, which is also what the drafter would have had to match), not a ground-truth
continuation token from another model, and the supervision at each depth is the target's
distribution at t+i (the `L_tv` term). If the trainer instead teacher-forces the ground-truth
tokens at every depth, the Markov head learns a transition function for a token sequence it will
never see at inference - that is the exposure-bias failure mode that makes "plain fine-tunes"
look flat.

**Data pipeline, exact:**
1. **Source: production traffic, not regeneration.** The served distribution is exactly what we
   want, it is free, and it avoids the target-cache problem. DeepSpec warns its target cache is
   "roughly 38 TB for the default Qwen3-4B setting" [P]; regenerating 1B tokens at our measured
   aggregate ~700-1,000 tok/s (8 streams, [M] README §Engines) is 12-17 days of box time.
   Capture thinking-mode agent requests/responses (the engine can log request/response pairs, or
   put a tee in front of it on the test VM) until 5-20M output tokens. At a few sessions a day
   this may take days of production traffic; if that is too slow, a dedicated 4-GPU generation run
   at 8 streams produces 5-20M tokens in 1.5-8 box-hours.
2. Token ids + thinking/answer boundary only (~80 MB). Prompt set for the synthetic fallback:
   `bench/codebench.py`'s three tasks and its 30k/60k/120k contexts plus public coding-agent traces.
3. Target cache: one frozen-target forward over the traces storing (i) aux hidden states at layers
   37/38/39 and (ii) top-64 target logprobs per position. Hidden states are 3 x 5120 x 2 B =
   30 KB/token -> 150 GB for 5M tokens, 600 GB for 20M; top-64 logprobs add ~0.5 KB/token. Fits
   the 16-disk RAID0, not the 400 GB boot disk. Extraction is ~0.3-0.5 box-hours at the measured
   19K tok/s prefill [M].
4. Train: one frozen-target forward per packed batch + forward/backward of the 5 draft layers.
   Estimated 60-90K tokens/s on 4 GPUs -> 20M tokens x 3 epochs ~= 1-2 box-hours. Use the
   anchor-bounded packing from the paper (multiple anchors per sequence, token-level attention
   indices).
5. Evaluate offline first: on 200 held-out prompts, per-position `TV_i` and implied
   `AL = 1 + sum(1-TV_i)`. **Gate: do not serve unless the offline AL at k=9 is >= 5.5.** Then
   serve the new `mtp.*` weights as a checkpoint overlay (set `num_nextn_predict_layers` in the config to the new
   count; the engine reads it at `vllm/models/deepseek_v41/nvidia/dspark.py:85-90`)
   and run codebench C1 + GSM8K-100. Because rejection sampling is exact, the target output
   distribution cannot change; any quality delta is a bug, so this is a clean A/B.

**Budget: 10-15 box-hours per run = $90-135 at Flex-start $9/h ([M] README §Capacity), 2-3 runs =
$200-400, plus 3-6 engineering days** (DeepSpec V4.1 port) or 3-4 days around `inference/model.py`.
The training compute itself is small - Avifenesh's run was ~12M supervised tokens in 1,140 steps on
2 GPUs - so the cost is the engineering and the target cache, not the GPUs.

**Calibration against the published numbers.** The DSpark paper's Table 1 (all at T=1.0) reports
accepted length 3.29-3.72 on open-ended chat, 4.86-5.52 on code and 4.89-6.21 on math, on Qwen3
targets with DeepSeek's own drafters [P]. Our measured 3.40-3.52 on thinking prose is exactly the
published *chat* level; asking for 5.5 is asking for **code-level acceptance on prose**. That is
the honest size of the bet.

**Expected gain and its risk.** Published aligned-training gains are 6-13% (TVP over public
recipes) to 41% (SpecOPD on-policy distillation on DSpark/DFlash) [P], but the two fine-tunes we
know of were flat or worse. My estimate: reaching a=0.88-0.90 with r=0.99 has probability ~40%;
a=0.85-0.88 with r=0.97 ~35%; flat or worse ~25%. At a=0.88-0.90, r=0.99, block 9:
**492-524 tok/s at today's step time** (565 if the decay also flattens), and 600 with FLy plus one
of (r=1.0, dT 0.45) on top (section 1.6, 5). **Falsifier: if the retrain moves a but not r (or vice versa), the model above says which half
of the frontier is missing, and the second iteration should target the other parameter.**

### (b) Tree / branch drafting

Mechanism: verify `m` candidates per position, so a position survives if the target's sample is
among the `m`. Published: TreeSpark +23.4-31.6% mean AL and +15.1-25.0% single-request throughput
over DSpark on Qwen3-4B/8B greedy [P] arXiv 2610.20175.

Why it is weak **on this box**: the gain needs top-m mass at each position, but the cost is verify
rows: width 2 over 5 positions = 11 rows, +5 x 0.61 = +3.05 ms (+34% step). Even at top-2 mass 0.86,
AL(k=5) = 4.3 and tok/s = 4.3/12.1 = 355 (-6%). And PCTree (training-free tree over the Markov
head) measured **0.037 accepted tokens per verify node** on Qwen3-4B GSM8K [P] arXiv 2608.02123,
against our 0.23/node bar: dead on arrival. Implementation is not small either: the DSpark path is
hard-wired to a uniform `num_query_per_req` and a uniform verify block
(`dspark/speculator.py:49-55`; `dflash/speculator.py:555-560` "Every DFlash step has exactly
num_query_per_req tokens, so we can use FULL CGs"), and the indexer backend has no device-side
trimming. Only revisit after alpha is high and the row cost has been driven down by the engine axis.

### (c) ngram / prompt-lookup hybrid

Mechanism: coding-agent output copies from the 30k-120k context (file contents, tool results, the
task statement). A 5-gram lookup over (context + prefix) proposes those runs at `c ~= 0.95` where
the MTP head has 0.77.

Value bound: if a fraction `f` of blocks sits inside an exact copy run, AL goes from 3.44
(alpha 0.77) toward 5.30 (alpha 0.95), so the gain is at most `f * 1.86` tokens: +0.37 (+11%) at
`f=0.2`, +0.09 (+3%) at `f=0.05`. Partial runs discount that roughly linearly. The thinking block
at max effort is where the tokens are and it is not copyable, so expect small `f`. **Measure `f`
offline first** (E0).

If it pays: `method` is a single Literal (`vllm/config/speculative.py:78-83`) and the V2 runner
builds exactly one speculator (`vllm/v1/worker/gpu/spec_decode/__init__.py:12-95`), so
`method: "ngram"` (exposed, `NgramGPUSpeculator`, a Triton scan,
`ngram/speculator.py:21-140,151+`) replaces DSpark rather than combining with it. The hybrid is a
patch: keep DSpark, and for positions with an ngram hit overwrite the draft token and its cached
draft-logits row to one-hot before verification. Probabilistic rejection reads the cached draft
logits, so a one-hot row makes that position exactly the greedy-draft accept test, still lossless.
~1-2 days.

### (d) Adaptive / gated verification

Section 2.4. Not supported, measured -14%, structurally useless at batch 1. Zero.

### (e) Drafter on one GPU (no TP all-reduce in the draft loop)

`draft_tensor_parallel_size` defaults to the target TP (`vllm/config/speculative.py:1724-1745`;
only 1 or target TP are legal) and the served config does not set it, so the drafter runs TP4. The
draft graph is 0.75 ms of a 7.7 ms step [M] and contains ~4-5 collectives (3 sublayer boundaries +
the sequence-parallel all-gather at `vllm/models/deepseek_v41/nvidia/dspark.py:210-246` + the
drafter lm_head vocab all-gather, Moet steps 23-25), i.e. ~30-70 us = ~1% of the step. The drafter's
9.6 GiB would land on rank 0, which has ~4.5 GiB free at util 0.94. Net <=1% for a memory/KV
sacrifice. Skip.

### (f) Distillation from the target's own outputs

This is (a). "Distillation" without on-policy thinking-mode data is just CE on the target's tokens,
which is what the flat Avifenesh fine-tune did. Do not run a separate CE-only variant.

### Cheap ways to raise `c_1` without a retrain (Lead's question)

Ranked by test-VM time per point of `c_1`:

1. **Proposal shaping: top_p/top_k truncation + a logit scale on the draft.** This is the only
   cheap lever that acts on `c_1` directly, because position 1's proposal is the backbone's raw
   logits with no Markov correction. The target verifies top_p-0.95-truncated logits
   (`rejection_sampler.py:264`), while the draft samples untruncated
   (`speculator.py:531-537`); matching the truncation removes up to 5 pp of always-rejected
   proposal mass. A logit scale `s` (draft temperature) then fixes the drafter's
   over/under-confidence: the optimal `q` is the target's truncated distribution, and `s` is the
   one free parameter that can move TV without touching weights. The Lead's staged overlay is
   exactly this class and is the right first sweep. **Two caveats:** (i) the *scaled and masked*
   logits must be what goes into the draft-logits cache that the rejection ratio test reads,
   otherwise the test is biased and the method is not lossless; (ii) measure `pos[0]/drafts`
   (i.e. `c_1`) as the primary metric, not overall AL - the effect concentrates at position 1, and
   overall AL dilutes it by ~3x. A 3-point sweep of `s` in {0.9, 1.0, 1.1} plus top_p matching is
   ~4 boots, ~10 min each.
2. **Ready-made support shaping: `dspark_draft_topk`.** The field exists in the served image
   (`speculative.py:615`, validated and written into the draft hf_config at `:1522-1553`, read by
   the speculator at `dspark/speculator.py:78-79`), and `_sample_sequential_topk` is exactly the
   cache-correct implementation wanted above: "the selected logits are corrected in place and every
   other entry is set to -inf. The normal dense sampling and rejection paths then consume that
   truncated distribution unchanged." Two caveats before it can be used on our stack: (i) the
   config gate at `speculative.py:1545-1552` admits only `Qwen3DSparkModel`/the Omni variant, while
   our V4.1 draft architecture is `DSparkV41DraftModel` (`speculative.py:1468`), so the field raises
   ValueError unless that one-line gate is patched; (ii) the field docstring requires draft TP1 and
   our drafter is TP4, where `topk` runs on the vocab-sharded logits, so the effective support is up
   to 4x the configured k. It also truncates the *base* logits before the Markov bias, so the
   candidate set is the backbone's top-k, not the final q's. Mechanism: support truncation removes
   q mass where p is small (good) but renormalises q upward inside the support (bad), so it only
   wins if q is over-dispersed. Sweep **k in {8, 16, 32}**, not 64/256: at TP4 the effective
   support is up to 4x k, so 4x32=128 is already a wide support and 4x64=256 is barely a truncation.
   Enabling route: the gate at `:1543-1552` fires whether the value comes from the SpeculativeConfig
   field *or* from `getattr(hf_config, ...)`: the chain is :1528 `self.dspark_draft_topk` ->
   :1529-1532 fallback to the hf_config attribute -> :1533 `if dspark_draft_topk is not None:` ->
   :1543-1552 the Qwen3-only check. The V4.1 config class flattens `text_config` onto the top
   level, so a key in config.json *is* found - and then trips the gate, because the draft
   architecture was already set to `DSparkV41DraftModel` at :1468 (and `update_arch_()` at :1474
   refreshed `model_arch_config.architectures` before the gate). So config.json alone is **not**
   zero-patch: the minimal change is to add `"DSparkV41DraftModel"` to the allowed set in
   `speculative.py:1543-1552` as an overlay. `--hf-overrides` does not propagate to the draft (dict
   overrides are target-only). The patch is **safe**: the gate is the only architecture-specific
   thing in the path; `_sample_sequential_topk` and its `-inf` fill are architecture-agnostic, so
   relaxing the gate changes no behaviour for Qwen3. The block writes the value back to
   `hf_config.dspark_draft_topk` at :1553, so after the patch the SpeculativeConfig field and the
   model-config route are consistent - use one, not both.
3. **FLy** (section 1.6): does not raise `c_1`, but converts a fraction of rejections into accepts,
   which is equivalent to +8-14% AL. Complementary, lossy, config-only.
4. **Feature, not weights: the LM-head input as a 4th aux feature.** The checkpoint's
   `dspark_target_layer_ids = [37,38,39]` are decoder-layer outputs; the post-norm final hidden
   state (the LM-head input) is the representation the target's own next-token distribution is
   computed from. Adding it to `main_proj` is a small architecture change inside the retrain and
   targets position 1 by construction (the Avifenesh fine-tune card lists "layers 37 to 39 and the
   LM-head input" as its target features). Cost: a few hours of engineering, zero extra GPU time
   beyond the training run already planned.
5. **Bigger anchor query capacity** (more draft layers) is the retrain, not a cheap lever.

### Adjacent, not drafter work: FLy and top_p matching

- **FLy** (`rejection_sample_method: "fly"`, `fly_window_size`, `fly_entropy_threshold`,
  `fly_entropy_top_k`; `vllm/config/speculative.py:541-542,619-627`,
  `vllm/docs/features/speculative_decoding/fly.md`): keeps a draft token the target would have
  rejected when the next W draft tokens are accepted natively. Published on **MRV2 + DSpark k=7**:
  1.099x-1.253x decode vs standard SD on DeepSeek-V4-Flash-0731, min quality retention 96.3% [P].
  Config-only, no patch; the V2 runner is the default in this build (`vllm/config/vllm.py:707-745`).
  It is lossy. Mechanism on our workload: with alpha 0.77, 23% of positions reject; of those,
  ~alpha^W are followed by W natively accepted drafts, so a W=2 window can recover roughly
  `0.23 * 0.77^2 ~= 14%` of the lost tokens, i.e. +14% AL, matching the published range. **This is
  the cheapest +10-25% on the table and it is on our exact proposer family.** At alpha 0.90 the
  recoverable share falls (fewer rejections), so FLy and the retrain overlap: expect the combined
  gain to be less than the product.
- **top_p matching (verified by the engine, `work/engine/FINDINGS.md`).** The verifier's `p` is
  top_p-truncated and renormalised: `RejectionSampler._verify` calls
  `self.sampler.apply_sampling_params(...)` (`rejection_sampler.py:264`), which applies
  temperature then top_k/top_p (`vllm/v1/worker/gpu/sample/sampler.py:262-274`). The proposal `q`
  is never truncated: the draft logits are cached **pre-temperature** and only scaled by `temp`
  inside the rejection kernel (`rejection_sampler_utils.py:169`), and the draft is sampled in
  `DSparkSpeculator._sample_logits` (`dspark/speculator.py:116-154`). Acceptance is
  `sum_x min(p(x), q(x))`, so all of q's mass outside p's top-95% set is deterministically
  rejected - at T=1.0, exactly our regime. Two patches, both quality-neutral by construction (the
  rejection step keeps the output distribution exactly p):
  - (a) truncate the draft logits to the same top_p before sampling, and make the **cached**
    `self.draft_logits` hold the truncated values (the rejection kernel reads that cache as q; q
    and the sampled token must come from the same distribution). Cost: 5 top-p passes over
    129,280 logits inside the draft graph, ~10-20 us each (est.), i.e. ~1% of the step.
  - (b) one line, same place: `logits = logits * scale`, sweep scale {0.8, 0.9, 1.0, 1.1}. Draft-side
    temperature; the exactness guarantee is the same, zero risk.
  Primary metric: `pos[0]/drafts` (c_1), not overall AL - the effect concentrates at position 1
  where the proposal is raw backbone logits. Quality gate: GSM8K-100 + needle; greedy-output
  equality is not the right test (any q gives the same output distribution; only the RNG stream
  and which tokens are accepted change).

---

## 4. The first experiments (each under 2 hours of test-VM time)

Boot is 6-8 min with the kernel cache ([M] README §Cold start), so each arm is one restart plus
~40-60 min of requests. Every arm records `tok/s`, `AL`, `steps/s` and the full per-position vector
per request (thinkbench does), and runs GSM8K-100 as the quality gate (baseline 96% on r21 k5 FP8
[M] README §Quality).

### E1 (run this first): FLy on the served config

One-line recipe: in `vm/ds41-start-r21.sh`, replace the `SPEC` JSON with

```json
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic",
 "rejection_sample_method":"fly","fly_window_size":2,"fly_entropy_threshold":0.3,
 "fly_entropy_top_k":3,"enable_adaptive_verification":false,"moe_backend":"b12x"}
```

then `docker stop -t 60 ds41` and re-run the start script on the test VM.

Measure: codebench C1 30k and 120k, 3 runs each; `bench/thinkbench.py` on `cb30,math1` at temp
1.0, 3 reps, 4096 tokens (thinking/answer split + per-position vector); GSM8K-100; Fraser k=5 as
the high-acceptance control.

- Arm order (the window is the lever, not the threshold): `fly_window_size` **2, then 1, then 4 as
  the default control**, all at threshold 0.3; then one threshold sweep (0.2 / 0.5) at the best W.
  At k=5 the number of deferrable positions is k-W, so W=4 (vLLM's default `min(6,k-1)`) lets only
  position 0 defer - it is the weakest arm, not the reference.
- Success: AL >= 3.75-3.80 (+10% over the measured 3.40-3.45) with GSM8K >= 95/100. Expected
  430-540 tok/s at W=1-2 (390 at the default W=4).
- Kill: AL < 3.55 or GSM8K <= 94. If the boot rejects probabilistic+fly (only greedy was validated
  on MRV2 in the doc), retry once with `"draft_sample_method":"greedy"` as a compatibility check
  only; do not judge speed on it, greedy drafting at temp 1.0 loses acceptance.
- Why first: config-only, published 1.10-1.25x on the same proposer, and it settles whether the
  verification side can carry a large share of the gap before anyone spends engineering days.

### E2: FLy at k=7 (second arm of the same session)

Same config with `num_speculative_tokens: 7` and **`fly_window_size: 2`** (not the default 6: at
k=7, W=6 allows only position 0 to defer, W=2 allows four positions). The published DSpark+FLy
result is at k=7. If E1 wins and E2 wins more, the serving config becomes k=7+FLy and the
fine-tune target stays block 9 (positions 6-7 are still untrained today, so E2 mostly tests FLy's
window, not the tail).

### E3a (30 min, run before E3): bound the truncation mismatch with top_p=1.0

One boot with `--override-generation-config '{"temperature":1.0,"top_p":1.0}'`, everything else
frozen, same bench: `dAL = AL(1.0) - AL(0.95)` bounds what making q match p's truncation can
recover, and the per-position `dc_i` tests whether the effect really concentrates at position 1.
**By-construction ceiling:** q-shaping can recover at most the top_p-removed mass of q (<=5% of
q), i.e. <=5 pp of `c_1` and <=~+4-5% AL - so levers 1-2 cannot change the 600 verdict even in the
best case. The boot tells us whether we actually collect those points.
**Confound to state:** top_p=1.0 removes the q/p support mismatch (raises acceptance) *and*
raises the target's entropy (lowers acceptance), so this is a one-sided test - a positive dAL
> 0.05 is strong evidence for E3, a near-zero or negative dAL is ambiguous, not a kill (the leads'
"dAL<=0.05 kills levers 1-2 by construction" is only true if the entropy shift is small). It is
also a bound on the test VM only: top_p=1.0 changes the served output distribution.

### E3: top_p-matched drafting (the lossless cheap one)

Bind-mount a patched `speculator.py` over
`/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`
(the overlay trick the repo already uses for the tokenizer), restart, re-run the E1 bench.

- Patch (a): in `_sample_logits` (`dspark/speculator.py:116-154`), before the sampler call, apply
  top_p=0.95 to the [5, 129280] draft logits and make the **cached** `self.draft_logits` hold the
  truncated values (the kernel reads the cache as q). Patch (b): `logits = logits * scale`.
- Sweep, one boot per arm: (b) with scale {0.8, 0.9, 1.1} (10 min/boot), then (a) if (b) shows the
  drafter is miscalibrated. Measure `pos[0]/drafts` (c_1) as the primary metric, overall AL second.
- Success: c_1 +2% relative or more with GSM8K >= 95/100. Kill: < 1% (run-to-run spread is several
  percent; 3 runs, medians).

### E4: rejection_sample_method block vs standard (one-line config A/B)

The block verifier (Sun et al., `rejection_sampler_utils.py:503-700`) is expected to dominate the
standard one-at-a-time sampler, but nobody has measured it on r21. One boot, same bench. Keep block
unless standard wins by > 2%.

### E5 (engine's decisive row-cost trace): diff one k=5 step against one k=7 step

`--profiler-config.profiler torch --profiler-config.torch_profiler_dir /profiles
--profiler-config.torch_profiler_with_stack false`, `POST /start_profile {"max_iterations":1}`
while a 120k request decodes, once at k=5 and once at k=7, then diff per-kernel totals with
`work/engine/src/moet/tools/sm120_perf/trace_agg.py`. Kernels whose total does not move are
weight-bound; the ones that do are the row cost, split by kernel. This is the only measurement
that can confirm the engine's 85-100%-expert-bytes model on the served graph, and it decides
whether the 0.47-0.55 floor is real.

### E0 (no VM, 20 minutes): bound the ngram hybrid

From saved C1 outputs (`bench/codebench.py --save-output`), per output position check whether a
5-gram match in (context + prefix) predicts the next 5 tokens; report coverage `f`, run lengths,
and the thinking/answer split. If `f < 0.1`, kill option (c).

### Decision table after the sweep + E1-E4

| observation | action |
| --- | --- |
| E1 AL >= 3.8, GSM8K >= 95 | adopt FLy, bank +10-25% |
| E2 (k=7+fly) beats E1 | serve k=7+fly; fine-tune still targets block 9 |
| E3 c_1 +2% | keep the logit-shaping patch; it is free and lossless |
| E4 standard beats block by >2% | switch rejection_sample_method (unexpected; re-check the block path) |
| sweep `s_6+s_7 < 0.46` at k=7 | tail dead at current block: only a retrained block moves it |
| sweep thinking `c_i` < 0.80 | distribution mismatch is real -> fund the full retrain (block 9, 5 layers) |
| sweep thinking `c_i` >= 0.85 | headroom is block length/capacity only -> size the run down (block 9, 3 layers) |
| row cost `dT/dk` <= 0.45 | 600 is reachable on the retrain alone; no FLy needed |
| row cost `dT/dk` >= 0.60 | 600 needs the retrain + FLy or ~5-10% of step work |

---

## 5. The 600 tok/s path, assembled

Decay-corrected, `T5=9.1 ms`, `dT=0.61 ms/row` unless stated. The FLy multiplier is W-dependent
(1.6) and **roughly shape-invariant**: at W=2 it is 1.24-1.33 across today's shape and the
retrained shapes (the "fewer rejections" effect is offset by "each repair extends a longer chain"),
so FLy and the retrain multiply rather than substitute. Rows use 1.15 (g=0.5 floor) to 1.33
(leads' g=0.9 model); my model runs 2-5 pp lower.

| step | a | r | k | AL | T (ms) | tok/s | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| today (r21 k5 FP8) | 0.812 | 0.972 | 5 | 3.40 | 9.10 | 374 | [M] 2.2 |
| today + FLy W=4 (default) | 0.812 | 0.972 | 5 | ~3.55 | 9.10 | 390 | [A] 1.6 |
| today + FLy W=2 | 0.812 | 0.972 | 5 | ~4.30 | 9.10 | 473 | [A] 1.6 |
| today + FLy W=1 | 0.812 | 0.972 | 5 | ~4.95 | 9.10 | 544 | [A] 1.6 |
| retrain: a 0.88, r 0.99 | 0.88 | 0.99 | 7 | 5.08 | 10.32 | 492 | [A] 1.6 |
| retrain a 0.88 + FLy W=2 (1.24-1.33x) | 0.88 | 0.99 | 7 | 6.3-6.8 | 10.32 | 610-660 | [A] 1.6 |
| retrain: a 0.90, r 0.99 | 0.90 | 0.99 | 8 | 5.73 | 10.93 | 524 | [A] 1.6 |
| retrain a 0.90 + FLy W=2 (1.24-1.33x) | 0.90 | 0.99 | 8 | 7.1-7.6 | 10.93 | 650-700 | [A] 1.6 |
| retrain + dT 0.45 (2-bit experts) | 0.90 | 0.99 | 9 | 6.00 | 10.90 | 550 | [A] 1.6 |
| retrain + flat decay | 0.90 | 1.00 | 10 | 6.86 | 12.15 | 565 | [A] 1.6 |

So the lever ordering has changed twice under the window analysis. **Today + FLy at W=1 is 544
for a config line; the retrain at a=0.88/r=0.99 plus FLy W=2 is 610-660** (W=1: 690-700). The
retrain target comes down to **a=0.88, r=0.99, block 7** and 600 arrives with margin; the 2-bit
expert/row-cost work is no longer needed for 600 (it would take the result past 700), and the
binding constraint is now **quality**: FLy is lossy (GSM8K gate) and the retrain has to land a=0.88.
What has NOT changed: at the default W=4 FLy is worth only 4-7%, and the retrain without FLy tops
out at 524.

---

## 6. How to fold in the rest of the Lead's sweep

The k=5 vector is already measured (section 2.2, `work/bench/x-base/think.jsonl`). What remains:

1. Fit `T(k) = a + b*k` from the k=3..7 per-request `AL` and `tok/s`. If `b` is outside 0.35-0.65
   ms/row, re-derive 1.4-1.6. This number decides whether the retrain alone reaches 600.
2. Take the k=7 per-position vector and compare positions 1-5 against the k=5 vector: they should
   match within noise; if they do not, block width changes the drafter's behaviour and the k
   comparison is not apples-to-apples.
3. Positions 6-7 at k=7 are the only direct measurement of the untrained tail. Their `s_6, s_7`
   are the "untrained block prize": if `s_6+s_7` is already >= 0.46, the block-size training is
   less urgent than the base-rate training.
4. Split the k=5 vector into thinking vs answer with `spec_think`/`spec_answer`. The answer vector
   should run near 0.9 (Fraser-like); if it does not, the workload is harder than Fraser and the
   a=0.90 target is a stretch.
5. Check the measured `c_1` against the other positions. Today `c_1` is the *highest* (0.81 on C1),
   so the budget split is: base rate -> capacity/data (5 layers, more target features), decay ->
   sequential head (rank 512, SpecOPD-style on-policy distillation). If the sweep's larger sample
   shows `c_1` below the rest instead, swap the split.
6. Replace the numbers in 1.5-1.6 and 5, then re-rank. The ranking is robust: FLy and the retrain
   stay 1-2 for any vector consistent with the measurement.

---

## 7. Sources

Local (workspace, read-only for me):
- `README.md` §Engines, §Speculation and batching, §Fraser bench, §Quality, §Capacity, §Cold start.
- `vm/ds41-start-r21.sh:44-46` (served spec config), `vm/r21.env`.
- `work/engine/FINDINGS.md` (step anatomy; verified top_p/q mismatch and patch locations; 7
  all-reduces per draft step ~90 us; block verifier location).
- `work/leads/notes/dspark-drafter-training.md` (DeepSpec config, Avifenesh, PCTree, row economics),
  `work/leads/notes/spec-decoding-nov-2026.md` (DLite, SpecOPD, DSpark Table 1 at T=1.0).
- `work/engine/src/vllm/vllm/config/speculative.py:78-83,406,541-542,560,619-627,1724-1745,1889-1909`.
- `work/engine/src/vllm/vllm/v1/worker/gpu/spec_decode/__init__.py:12-95` (one speculator).
- `work/engine/src/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:49-55,120-155,156-207,258-276`.
- `work/engine/src/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:315-339,555-560`.
- `work/engine/src/vllm/vllm/v1/worker/gpu/spec_decode/speculator.py:531-537` (draft ignores top_p/top_k).
- `work/engine/src/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:177-179,264`.
- `work/engine/src/vllm/vllm/v1/worker/gpu/spec_decode/ngram/speculator.py:21-140,151+`.
- `work/engine/src/vllm/vllm/models/deepseek_v41/nvidia/dspark.py:85-90,210-246`.
- `work/engine/src/vllm/vllm/config/vllm.py:707-745` (V2 runner is the default).
- `work/engine/src/vllm/vllm/docs/features/speculative_decoding/fly.md`.
- `work/engine/src/moet/README.md:45` (Markov sampling fused), `docs/dsv41-sm120-port.md:1897-1932`
  (r21 step anatomy, gap #6), `docs/benchmarks/mtp-acceptance-depth-2026-07-13.md` (acceptance is a
  property of the text, flat vs context depth).
- `bench/thinkbench.py:51-53,143-150` (per-position vector, thinking/answer split).
- `work/bench/x-base/summary.json` (C1 medians: 374.7/392.5/385.6 tok/s at 30k/60k/120k, AL
  3.416/3.468/3.522), `work/bench/x-base/think.jsonl` (13 requests with the per-position vector).
- `work/drafter/analysis/pos-vector.txt`, `work/drafter/analysis/frontier-decay.txt` (this plan's
  analysis of the above: a=0.793, r=0.965, decay-corrected frontier).
- Checkpoint config: `num_nextn_predict_layers: 3`, `dspark_block_size: 5`,
  `dspark_target_layer_ids: [37,38,39]`, `dspark_markov_rank: 256`, `dspark_n_routed_experts: 128`,
  `dspark_num_experts_per_tok: 3`, `dspark_noise_token_id: 128799`
  (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json).

External:
- DSpark paper (architecture, training objective `L_ce + L_tv + L_conf`, position weighting,
  confidence head = 1-TV, production numbers): https://arxiv.org/abs/2607.05147
- DeepSpec (open-source DSpark/DFlash/Eagle3 training stack, data pipeline, hardware, "fine-tune
  again for thinking mode"): https://github.com/deepseek-ai/DeepSpec
- DeepSpec DSpark config (`block_size=7, num_draft_layers=5, ce 0.1 / l1 0.9, gamma 4, lr 6e-4,
  batch 512, 10 epochs`): https://github.com/deepseek-ai/DeepSpec/blob/main/config/dspark/dspark_qwen3_4b.py
- TVP, TV loss for drafter fine-tuning: https://arxiv.org/abs/2611.00588
- SpecOPD, on-policy distillation, up to +41.45% mean acceptance length on DSpark-class drafters:
  https://arxiv.org/abs/2611.02846
- TreeSpark, learned tree drafting, +23.4-31.6% AL over DSpark: https://arxiv.org/abs/2610.20175
- DLite, drafter input design (sequence-wise vs feature-wise target features +8.5%, 3->14 target
  layers +5%, sequential-head rank 256->512 helps): https://arxiv.org/abs/2610.23787
- PCTree, training-free tree over the Markov head, 0.037 accepted/node: https://arxiv.org/abs/2608.02123
- FLy feature doc + measured MRV2+DSpark speedups: `vllm/docs/features/speculative_decoding/fly.md`
  (upstream), https://arxiv.org/abs/2511.22972
- DSpark-class training recipe with hardware notes: https://forums.developer.nvidia.com/t/open-source-recipe-scaffold-training-a-dspark-class-speculative-decoding-draft-for-nemotron/374876
- A released DSpark drafter with full training hyperparameters (5 layers, 8K seq len, 1024 anchors,
  rank-256 Markov head, 8x H200 = 4 extraction + 4 training):
  https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark
- Avifenesh fine-tune (block 5, 1,140 steps, +3.3% overall / +1.0% thinking T=1; the
  greedy->probabilistic +13.7% at T=1): https://huggingface.co/Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft
- Checkpoint inference reference implementation (the cheapest training codebase to adapt):
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main/inference
