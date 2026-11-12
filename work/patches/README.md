# The VM-side artifacts, rescued before the experiment VM was stopped

`/data` on the experiment VM (`ds41-test-a`) is a **RAID0 of the 16 local NVMe SSDs**, not a
persistent disk, so stopping the instance destroys it. This directory is the copy that matters: the
overlay kernels/patches, the calibration scripts, the measurement tools and the launcher.

The checkpoints themselves are rebuildable and are **not** here:

- `/data/models/ds41-abl` — the FP4 base, re-pullable from `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`
  @ `517efab9ccfef7f714dbd7d679f181a86f43ccf9` (510.3 GB).
- `/data/models/ds41-abl-q2` — every routed expert tensor (target **and** drafter) requantised to the
  sign-symmetric 2-bit codebook, plain MSE objective: `python3 requant.py <src> <dst>` (~6 min on 4 GPUs).
- `/data/models/ds41-abl-q2b` — same, with the drafter included (the variant that keeps the drafter at
  4 bits is `-q2`).
- `/data/models/ds41-abl-q2h` — the **imatrix** variant: the per-block scale/code search is weighted by
  `diag(H)` collected by `cal3.py` (8 prompts x 4 TP ranks, 200k tokens). **This is the one to use.**

Files:

| file | what it is |
| --- | --- |
| `b12x_moe_fused_sm120.cu` | the fused decode MoE with the 2-bit FC1 + FC2 weight paths (ratio-3 codebook, shared-memory LUT, pairwise 16-byte staging), plus the 4-bit path it replaced |
| `b12x_moe_fused_sm120.py` | its Python wrapper (the `q2` flag, the pair-count gate) |
| `w4a8_compact_micro_q2.py` | the launcher patch: the in-place 2-bit repack, the 2-bit w2 parked in the w13 tensor's freed second half, the optional imatrix calibration hook |
| `w4a8_compact_projection_q2.py` | the CuTe compact projection with the 2-bit weight path (needed whenever the drafter is also requantised) |
| `w4a8_compact_projection_4b.py` | the stock 4-bit projection (use this when the drafter stays at 4 bits) |
| `w4a8_phase1_q2.py` | b12x's prefill FC1 with the 2-bit path |
| `requant.py` | the requantiser: per-block `2^j` scale search, sign-symmetric 4-level codebook, optional `diag(H)` weighting for w1/w3 |
| `cal3.py` | the offline imatrix calibration (vLLM offline `LLM`, `enforce_eager`, `MoERunner.forward` hook, per-rank dump) |
| `alprobe.py` | the step measurement (`step_metrics` is the prefill-free one) |
| `exp.sh`, `ds41-start-r21.sh`, `doprof_q2.sh` | the arm runner, the launcher, the 2-bit profiler |
| `sweep2bit.py`, `test_q2fc1.py`, `test_q2_t1.py` | the codebook/scale sweep on real weights and the fused-kernel correctness tests |

The **served configuration** that is fastest with no quality change is *not* in here: it is
`reasoning_effort` in `--default-chat-template-kwargs`, and production now runs `high`
(`/etc/ds41/r21.env` on the serving VM, backed up as `r21.env.bak-efforthigh-*`).
