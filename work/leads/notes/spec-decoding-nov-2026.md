# Speculative decoding literature, Oct-Nov 2026: what applies to our stack

Researched 2026-11-08. Every quote is from the paper's abstract/body; the "applies?" column is my judgement
against our measured facts (alpha 0.77 on thinking prose at T=1.0, block 5, ~0.45 ms per verify row, 4x RTX
PRO 6000 PCIe TP4, vLLM-Moet r21).

## 1. DLite: history-free speculative decoding for agentic AI (arXiv 2610.23787, 2026-10-16)

Claim: "agentic workloads introduce long and continuously evolving contexts ... existing draft models suffer
from **up to an 80% degradation in acceptance length** ... DLite achieves **2-6 acceptance length and
outperforms the state-of-the-art speculative decoding method, DSpark, up to 70%**."

Mechanism: reuse target hidden states, predict locally around the current position, no draft KV cache; a
low-rank sequential head restores intra-block causality; two-stage training (isomorphic history-aware teacher
by forward KL, then direct CE+TV alignment).

What matters for us:

- **The killer ablation is in their appendix**: "we additionally train two variants with **sliding-window KV
  caches of 128 and 512 tokens** ... after sufficient training, the three models achieve **comparable
  acceptance lengths**". Our DSpark drafter is already SWA-128 (DSpark paper: "a sliding window attention of
  128"), and our accepted length is flat from 30k to 120k context (371/375/376 tok/s). So DLite's headline
  advantage over a *properly trained SWA* DSpark is not available to us; their DSpark baseline degrades
  because of its training data, not because it has a KV cache.
- **Transferable design choices for a retrained drafter** (their Table 5, Qwen3-8B, avg tau):
  sequence-wise concatenation of target hidden states beats feature-wise (3.04 -> 3.30, +8.5%); 7 -> 14
  hidden states 3.14 -> 3.30 (+5%); beyond 14 saturates; sequential-head rank 256 -> 512 gains, 512 -> 4096
  barely. Our drafter takes 3 hidden states (layers 37/38/39) and concatenates them **feature-wise** into
  `main_proj` (3 x 5120 -> 5120). Sequence-wise plus more layers is a cheap experiment in the same training
  run as a block-size change.
- Their general-workload gains are small: Alpaca tau 3.49 (DSpark) -> 3.63, MT-Bench 3.79 -> 3.88,
  LCB 5.28 -> 5.98. So do not expect +70% on our workload; the +70% is SWE-Bench-like sessions where DSpark
  collapses to 1-2 accepted tokens.

## 2. Hybrid Spark: DSpark + n-gram in one engine, per-request, on GPU (arXiv 2611.05611, 2026-11-03)

Claim: "a framework that runs a model-based and a model-free drafter on every step and gives each request
the better of their chains, within adaptive verification and entirely on the GPU, which keeps asynchronous
scheduling intact ... Instantiated with **DSpark and an n-gram drafter in vLLM**, achieves **1.10-1.62x** the
output throughput of DSpark on **code editing** at batch size 4 and **1.04-1.27x** at batch size 128, while
staying at **0.97-1.01x** of it on chat at temperature 0."

This settles open question (a): a two-drafter engine does exist as a working design, in vLLM, and it is the
only published way to combine n-gram with DSpark. The key trick is a "price per verified draft token" that
puts both drafters' chains on one scale once the n-gram tokens carry calibrated acceptance estimates.

Applies? Our workload is a coding agent, so some output is copied/edited text where n-gram wins; but the
C1 metric is dominated by thinking tokens, and the paper's own chat number at T=0 is 0.97-1.01x (neutral).
Expected here: neutral to +10%, and the implementation does not exist publicly (paper only, Nov 3). Not a
path to 600.

## 3. Parallel Reconsideration (arXiv 2611.05724, 2026-11-03)

Claim: "guided by its own uncertainty, the drafter reconsiders its hesitant guesses with extra passes that
run in parallel with the one-pass draft, adding negligible latency ... improves over the baselines by
**5.3% to 10.2% on acceptance length and by 3.5% to 9.2% on speedup**." Needs "a co-designed training recipe
and serving-time drafting algorithm" that appends refined candidates to the draft tree.

Applies? It is a tree again, so the row cost bites: the extra candidates are verified rows. At +5-10%
accepted for +1-2 rows, it is under our 0.2-accepted-per-row bar unless the candidates replace existing rows
rather than adding them.

## 4. PCTree (arXiv 2608.02123) - already in LEADS.md

Training-free tree over DSpark's Markov head. "+3.1% to 29.5%" speedup over matched DSpark at B=7, but the
phase table shows it buys acceptance with nodes: at k=4, N=32 nodes, +0.93 accepted on Qwen3-4B GSM8K =
0.037 accepted per node, against our 0.2/node bar. Dead here.

## 5. TVP (arXiv 2611.00588, 2026-10-31) and SpecOPD (arXiv 2611.02846, 2026-11-03) - training side

- TVP: loss family on expected acceptance length, power annealed 0 -> 1; "improves acceptance length over CE
  by 9.8% when fine-tuning, and over the public DFlash2 training recipes by 6.2 to 12.9%". Note DeepSpec's
  own DSpark recipe already uses a TV loss at 0.9 weight, so the CE comparison overstates the gain for us.
- SpecOPD: "on-policy distillation (OPD) can push well-trained drafters beyond acceptance-length plateaus".
  This is the training recipe that matches our situation (drafter plateaus at 3.45 on in-domain text); worth
  reading before choosing a loss for the retrain.

## 6. D3D (arXiv 2611.06425, 2026-11-04) - step-time idea, watch

"Larger drafters can improve acceptance, but typically require more time to generate each candidate block.
When drafting and verification execute sequentially..." - decoupled/asynchronous diffusion drafting. If it
works, it removes the drafter's 0.8 ms from the step's critical path, which is a 9% step win and would let us
afford a bigger (higher-acceptance) drafter. No numbers extracted yet; the code is not public.
