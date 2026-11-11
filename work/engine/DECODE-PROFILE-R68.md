# The decode-step profile on our cards, round 68 (first one ever captured)

Trace: `/var/lib/ds41/x/prof2/prof/dec_dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz` on
`ds41-test-a`, captured with `PROF=1` (the launcher only adds `--profiler-config` when PROF is
exactly 1; `PROF=2` skips it, which is why round 67's attempt produced an empty directory) then
`profdecode.py OUTDIR 3000 3 120`. Served config: exact block rejection, k=5, top_p 1.0, base5 env.
4547 kernel events, 39.5 ms of GPU kernel time over a 35.0 ms wall span (1.13x stream overlap) =
**3 decode steps** (120 = 3 x 40 MoE launches). Profile overhead inflates the wall, so read the
**relative** split, not the absolute sum.

## Per-step GPU time by kernel (divide the trace total by 3)

| kernel | us/launch | launches/step | **ms/step** | share |
| --- | ---: | ---: | ---: | ---: |
| `b12x_moe_fused::moe_fused_kernel` (40 layers, 22.5 distinct experts) | 78.09 | 40 | **3.12** | 24 % |
| `mxfp8_mma_gemv_v3_kernel<1,1,1>` (dense MXFP8 GEMVs) | 10.59 | 188 | **1.99** | 15 % |
| `mhc_proj_kernel<4>` (side stream, per boundary) | 13.49 | 82 | 1.11 | 8 % |
| `mxfp8_mma_gemv_kernel<1,2,1,1>` | 8.41 | 43 | 0.36 | |
| `mxfp8_mma_gemv_kernel<1,0,0,1>` | 4.45 | 43 | 0.19 | |
| **dense GEMV total** | | **274** | **2.54** | 19 % |
| `mhc_pre_big_fuse_with_norm_tilelang` (side stream, Sinkhorn) | 10.44 | 90 | 0.94 | 7 % |
| `push_phased_kernel` (mHC TP4 all-reduce push) | 6.28 | 82 | 0.52 | 4 % |
| `mhc_post_norm_kernel<1,true,true>` (AR consumer + collapse + norm) | 5.95 | 82 | 0.49 | 4 % |
| **mHC total (1.00 ms of it on the critical path)** | | **336** | **3.05** | 23 % |
| `sparse_mla_decode_dsv4_kernel` | 12.80 | 43 | 0.55 | 4 % |
| `sparse_mla_decode_dsv4_merge_kernel` | 3.24 | 43 | 0.14 | |
| `persistent_topk_kernel<512,4u>` (indexer top-k, 8 index layers) | **45.56** | 8 | **0.36** | 3 % |
| `spread_kernel<20,false,true,false,true>` (router BF16 GEMM) | 9.02 | 40 | 0.36 | 3 % |
| `_q_kv_norm_quant_kernel` | 6.84 | 43 | 0.29 | 2 % |
| `act_and_mul_kernel` (shared expert SwiGLU) | 5.72 | 43 | 0.25 | 2 % |
| `mxfp8_quantize` x3 variants (531 launches) | 2.0-4.2 | 177 | 0.57 | 4 % |
| `dsv4_topk_kernel<384>` (routing top-6) | 4.45 | 40 | 0.18 | |
| `ncclDevKernel_AllGather_RING_LL` (lm_head vocab AG + drafter) | 24.62 | 6 | 0.15 | |
| `chain_kernel<16,320>` (indexer weights_proj) | 15.71 | 8 | 0.13 | |
| `_markov_step_kernel` (drafter, 5 sequential steps) | 21.72 | 5 | 0.11 | |
| `deep_gemm::sm120_fp8_paged_mqa_logits` (indexer QK) | 17.72 | 5 | 0.12 | |
| everything else (35 kernels) | | | ~0.35 | |

Sum 13.2 ms of GPU work per step against a ~8.9 ms wall: **4.3 ms of the step is overlapped** (the
mHC side stream and the quantizers), which is why the "mHC total" is 3.05 ms but only ~1.0 ms of it
is on the critical path.

## What this changes in the plan

1. **The dense MXFP8 GEMV is 2.54 ms/step, not the 1.0-1.4 ms the earlier tables carried** (274
   launches x 9.3 us). Its byte floor is 1.24 ms (1.8 GB/rank at 1.45 TB/s), so there is ~1.3 ms of
   theoretical headroom, but the engine's own GEMV note says short bursts on this card run at
   0.6-1.2 TB/s and the realistic remaining upside is ~0.3 ms. It is the second largest item and it
   is latency-bound, not bandwidth-bound.
2. **The indexer top-k is 45.6 us x 8 = 0.36 ms/step**, three times the 15.1 us the reference host
   measured. `VLLM_MOET_DET_TOPK=1` is the switch that replaces `persistent_topk` with the
   `det_topk` kernel (9.4 us at 131K columns on the reference host) and it is **off by default**. This
   is the single cheapest concrete step item found this round: worth up to ~0.28 ms (3 %).
3. **`act_and_mul_kernel` 0.25 ms/step** is the shared expert's SwiGLU running as its own launch after
   its GEMV, 40 times. Fusing it into the shared expert's GEMV epilogue removes it.
4. **531 `mxfp8_quantize` launches/step (0.57 ms)** are the activation quantizers; the engine already
   has a prequant patch for the MoE's rows (`VLLM_MOET_MOE_PREQUANT=1`, measured -82 us on the
   reference host) and the rest are the "launch-gap/quant fusion" item.
5. **The mHC path is 3.05 ms of GPU work per step, 1.00 ms of it critical** (push 0.52 + post_norm
   0.49). The push is 82 launches x 6.28 us: this is the E7 target, and the profile confirms it is
   still 0.52 ms with the served fused consumer, so the config switch (`P2P_AR_ONESHOT_MAX_ROWS`)
   does not remove it.
6. **The MoE is 3.12 ms at 78.09 us/layer** — the DRAM floor of its access pattern per the engine's
   own b12x note. Only a smaller expert format moves it.

## The honest 600 arithmetic with these numbers

step 8.90 ms today. Training-free kernel items identified and priced: DET_TOPK -0.28, act_and_mul
fusion -0.20, quant fusion/PDL -0.30, mHC push from producer -0.40, sparse MLA to floor -0.30,
dense GEMV -0.30 = **-1.78 ms -> 7.12 ms**. At AL 3.44 that is **483 tok/s**. Adding the 2-bit routed
experts (-1.5 ms) gives 5.62 ms -> **612 tok/s**. So 600 is: **the kernel bundle + a smaller expert
format**, and nothing else. The drafter cannot contribute (round-68 research: the AL axis is
exhausted at 3.39-3.44 for any training-free exact scheme at k=5).
