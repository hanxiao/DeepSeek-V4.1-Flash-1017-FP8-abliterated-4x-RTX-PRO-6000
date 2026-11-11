# Round 68 (2026-11-10) — 600 tok/s exact, training-free: the plan

## Target, restated by the owner this round

- **600 tok/s single stream** on the daily coding-agent workload (codebench: thinking max, 30k-120k ctx,
  4096 output), **exact decoding** (FLy banned: `100%`->`500%`, `Mountain View`->`Mile V`).
- Aggregate multistream as high as possible.
- **Training-free only** (no drafter/speculator/MTP training; ready-made published checkpoints are OK).
- **No quality regression**; same abliterated weights; no swap to a lower-quant model.

## Where we stand (measured, rounds 1-67)

`tok/s = 1000 * AL / step_ms`. Measured on codebench C1, T=1.0, exact block rejection, k=5:

| | value | note |
|---|---|---|
| step | 9.05 ms at 6 rows (k=5), 10.17 at 8 rows (k=7) | fit `5.69-6.05 + 0.56-0.61*rows` |
| AL | 3.31 (120k) / 3.40 (30k) | per-position c = [.81 .76 .73 .67 .72] |
| tok/s | 366 (120k) / 377 (30k) | FLy reference: 463 (lossy, banned) |
| aggregate | 458 / 580 / 704 / 1082 at C1/C2/C3/C8 | FLy |

The two axes: step 9.05 -> 5.5 ms, or AL 3.31 -> 5.4, or a mix. AL ceiling at any k is ~3.76 with the
shipped block-5 drafter (measured, k=7/k=15).

## This round's lever list, ranked by (expected tok/s) / (hours)

| # | lever | axis | expected | source | status |
|---|---|---|---|---|---|
| 1 | **Exact multi-draft trie**: 2 independent DSpark draft passes -> trie -> one ancestor-masked verify. AL 3.31 -> ~4.9 at 8-9 rows (RRS exact, q can be any distribution) | AL | **+16-22%** | SpecInfer 2305.09781, Sequoia 2402.12374, DDTree; subagent report 2026-11-10 | todo |
| 2 | **`Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft`** drop-in drafter (published fine-tune of mtp.0-2, MIT) | AL | +3-6% | HF, subagent report | todo |
| 3 | **`VLLM_MOET_DRAFT_LOGIT_SCALE`** (draft-side sharpening; q stays consistent so output is still exactly p) | AL | 0..+10% | engine FINDINGS E2 | todo (check the image has it) |
| 4 | **E1: truncate q the same way p is truncated** (top_p) | AL | +3-8% | engine FINDINGS E1 | moot at top_p 1.0 |
| 5 | **two-shot vs fused mHC all-reduce** `P2P_AR_FUSED_MAX_ROWS=0` | step | 0.5-1.5% | engine PATCHES.md | todo (zero code) |
| 6 | **`P2P_AG=1` + `P2P_AR_ONESHOT_MAX_ROWS=8` + `DET_TOPK=1`** | step | 0..3% | engine FINDINGS E4 | todo (zero code) |
| 7 | **`rejection_sample_method: standard` vs `block`** | AL | +-3% | engine FINDINGS E6 | todo (zero code) |
| 8 | **k sweep k=4,5,6 at T=1.0** | both | 0..+4% | engine FINDINGS E3 | todo (zero code) |
| 9 | **E8: TP-shard the drafter's `main_proj`** (78-157 MB replicated/rank/step) | step | +0.5-1% | engine FINDINGS E8 | code |
| 10 | **E10: target lm_head + logits all-gather into the FULL graph** | step | +1% | engine FINDINGS E10 | code |
| 11 | TensorFold R1: one-shot IPC peer-store all-gather instead of NCCL all-gather (90 sites) | step | +3-6% | TensorFold patch 0034 | code |
| 12 | TensorFold R5: DeepSeek-1017 rollback-copy reduction (89 copies/window) | step | +2-3% | TensorFold CHANGELOG #1752 | check tree |
| 13 | TensorFold R6: shared-expert side stream | step | +1.4% | TensorFold patch 0006 | code |
| 14 | TensorFold R2: PDL chain along the decode | step | +3.5-4% | TensorFold patches 0069-0090 | hard port |
| 15 | TensorFold R3: DSA decode select on many CTAs | step | +2-3% at 120k | patch 0067/0091 | code |
| 16 | `VLLM_MOET_MHC_*`, `BF16_GEMM_SITES=+compressor`, `BF16_CHAIN=v2` re-calibration on our cards | step | 0..2% | engine FINDINGS 4.1 | zero code |

Dead (measured, do not re-run): k=7/k=15 at the shipped drafter; FLy (banned); adaptive verification at
bs1 (~0); MoE megakernel; router+topk fusion; ECC off; power/clocks; EP-vs-TP; host/NVMe prefetch;
L2 cross-step expert caching; SGLang on this box; the 2031-step block-5 retrain (AL flat).
2-bit experts = -1.5 ms but it is a re-quant (owner's constraint this round).

## The arithmetic for 600 (with the two best levers)

```
today                 AL 3.31, step 9.05  ->  366
+ trie (AL 4.9)                            ->  541  (rows 8-9, step ~9.9)
+ trie + step -1.5 ms (items 5,6,9,10)     ->  ~640
+ drafter ft (AL 5.1)                      ->  ~660
```

## Infra

- New test VM **`ds41-test-a`**, zone **asia-south2-a**, IP **<test-ip>** (asia-southeast2-c
  flex-start instances do not boot this round: two fresh ones, empty serial console, no TCP at all;
  prod in that zone is fine).
- `bash /tmp/ds41/ssh.sh '<cmd>'` runs root commands on it. Boot script
  `/usr/local/bin/ds41-boot-moet.sh` (unit ds41-boot.service) rebuilds the /data RAID, downloads the
  510 GB checkpoint, restores the r21 image from `/var/lib/ds41/r21-image.tar.zst`, starts the engine.
- Production `ds41-prod` (<prod-ip>) is untouched and serves this agent.
- After every boot: `bash /var/lib/ds41/ecc/mkctx.sh` (rebuilds /tmp/ds41cb contexts) then the arms.

## Round 68 outcome (2026-11-10, measured)

- New test VM `ds41-test-a` (asia-south2-a, <test-ip>), production untouched.
- Exact baseline on it: **371.4 tok/s at 120k, AL 3.44, step 9.27 ms**; 399.0 at 30k.
- **Landed: `VLLM_MOET_DET_TOPK=1` (default 1 in the launcher now) -> step 9.130 -> 8.838 ms (-3.2 %,
  interleaved A/B/A/B with the base reproducing to 0.000 ms), +3.3 % tok/s at matched AL
  (371.4 -> 383.6 at 120k), neutral at 30k, GSM8K-100 97/100.**
- **Dead this round**: every other config switch (one-shot push, P2P_AG, MOE_PREQUANT, the bundle) is
  neutral on the step; the two-shot mHC consumer is 6.6 % worse; proposal shaping (draft logit scale
  0.7-2.0 and draft top-k) is neutral, so the AL axis is closed for training-free exact decoding.
- The decode profile (first ever on our cards, `work/engine/DECODE-PROFILE-R68.md`) prices the step:
  MoE 3.12, dense GEMV **2.54** (not 1.0-1.4), mHC 3.05 total / 1.00 critical, sparse MLA 0.69,
  indexer top-k 0.36 (now removed), quantizers 0.57, router 0.36.
- Next, in order: fuse `act_and_mul` into the shared-expert GEMV (-0.20), extend the prequant pattern to
  the dense GEMV chain (-0.30), mHC push from the producer (-0.40), sparse MLA to its floor (-0.30),
  2-bit routed experts (-1.5). Bundle = 7.12 ms = 483 tok/s; + 2-bit = 5.62 ms = **612 tok/s**.
