# r21 decode step: anatomy, cost model, and ranked levers

Owner: engine teammate (task-1). Sources: the clone at `work/engine/src/moet` (b0dc737) and
`work/engine/src/vllm` (df3bd094), the served configuration (`vm/ds41-start-r21.sh`, `vm/r21.env`),
the production container read over ssh (read-only), and the repo's own measurements in
`docs/dsv41-sm120-port.md` + `tools/dsv41_sm120/README.md`. Every number below is either quoted from
one of those or marked **estimate** with the arithmetic shown.

Baseline to beat: **376 tok/s** at 120k ctx = 114 steps/s (8.77 ms/step) x accepted length 3.45.
Target 600 tok/s = either 174 steps/s at 3.45 accepted (5.75 ms/step, -34 %) or 5.26 accepted at
114 steps/s (+52 %), or a mix.

---

## 0. TL;DR — the three levers that matter, and the ceiling above them

**The ceiling first.** The served GPUs are RTX PRO 6000 **Server Edition**, not the Moet reference
workstation cards: `nvidia-smi -q -d SUPPORTED_CLOCKS` lists exactly one memory clock (**12481 MHz**,
the one in use), SM 2422 of max 2430 MHz, 285-340 W of a 600 W limit, no throttle reason ever active,
ECC **enabled**, PCIe Gen5 x16. On the identical k=7 Fraser recipe the reference host gets 127.8
steps/s and we get 102 (k=5: ~143 vs 111), i.e. **the card costs 25-29 % and no flag recovers it**.
~50 % of our step is DRAM-bound weight streaming (§2.5), so budget against 111-118 steps/s, not 127.8.

**Lever 1 — accepted length (moves accepted; the only big controllable term).**
3.45 accepted against ~0.85-0.88 per-token acceptance on the same cards at temperature 0 (Fraser: 4.8
accepted at k=5). Measured per-position conditionals at T=1.0 (drafter owner's baseline, C1 workload,
k=5): **c = [0.812, 0.759, 0.730, 0.672, 0.725]**, i.e. AL 3.40, and the AL sensitivity is
`dAL/dc_1 = (AL-1)/c_1 = 2.96` — position 1 is worth 3x the tail. At T=1.0 the target distribution is
flat, and the verifier's p is top_p=0.95-truncated (`rejection_sampler.py:264`) while the draft
proposal q is not (`rejection_sampler_utils.py:169`) — every draft sampled in p's tail is
deterministically rejected.
**Exact change:** apply the same top_p (or a top_k=512) to the draft logits in
`DSparkSpeculator._sample_logits` (`dspark/speculator.py:116-154`) and cache the truncated logits, so
the rejection test sees a consistent q. Quality-neutral by construction (the proposal never changes the
output distribution), ~20 lines, ~2 h including a 6-run codebench A/B. Second variant: scale the draft
logits by 1/tau (tau in 0.8-1.0) — same place, same guarantee. Widening k is **not** one of these at
T=1.0: the served k=5 is the checkpoint's trained block size (`dspark_block_size: 5`), the workspace
measured k=7 at +2 % accepted against +12 % step on codebench (= -8..9 % net tok/s), and the measured
per-position curve shows why (positions 6-7 run below the in-block acceptance rate; §5).

**Lever 2 — expert bytes (moves step time; the only way to cut the dominant term).**
The routed-expert stream is 4.2 GB per rank per step (22.5 distinct experts x 4.7 MB x 40 layers,
measured routing) = **3.0 ms of the 8.77 ms** at the DRAM floor. Nothing in the patch set can shrink
it: the count of distinct experts is set by the 6 verified rows (k), and TP vs EP does not change the
total bytes (each rank streams 1/4 of every touched expert under TP, or all of 1/4 of the experts under
EP — same total). **Exact change:** 2-bit routed experts — the repo's own v0.24 line stores the experts
as 2-bit planes plus an FP4 "delta" recovery pool (`docs/v024-port.md:1-6,229-241`, knobs
`VLLM_MOE_W2_DELTA_GB`), which halves the expert bytes and takes the MoE to ~1.5 ms (**-17 % of the
step**). Caveat: that line is vLLM v0.24.0 + DeepSeek-V4-Flash, not our nightly + V4.1, so porting it
means re-quantizing the 552B checkpoint into 2-bit planes, porting the expert kernels onto the b12x
compact path, tuning the delta pool and running the quality gate — weeks, not days. It is still the
single largest step-time item that is not already at a hardware floor.

**Lever 3 — the 80 mHC boundary all-reduces (moves step time; largest non-floor item).**
80 boundaries x (5.7 us P2P push + 7.7 us fused boundary kernel) = **1.1 ms, 12.5 %**, and the push is
a separate kernel that cannot start before the sublayer that produces its input. **Exact change:**
issue the push from the producing kernel itself (the `wo_b` GEMV / b12x fused-MoE epilogue writes its
three peers' partial buffers), so the boundary kernel's flag wait is the only cost left: **-5.7 us x 80
= -0.46 ms, -5 %**. 2-3 days of CUDA in `mhc_overlap/mhc_post_norm_sm120.cu` + the two producers.

If none of the three lands, 600 tok/s is out of reach on these cards: the floor is ~4.3-4.6 ms of DRAM
streaming + ~1.1 ms of PCIe all-reduce + ~0.5 ms of attention latency = ~6.1 ms with the drafter, the
launch gaps and every eager step free, i.e. ~164 steps/s x 3.45 = ~566 tok/s. **600 needs -10..-15 % of
step time and +20..30 % of accepted length together.** The full ranked list is §5.

---

## 1. Ground truth: what is actually served

From `sudo docker exec ds41 env` and the startup log of the production container
(`/data/logs/ds41-r21-image.log`), 2026-11-08:

* image `vllm-moet-sm120:dsv41-nightly-df3bd094-r21` = vLLM nightly `df3bd094a`
  (`v0.32.1rc1.dev117+gdf3bd094a`) + Dockerfile steps 1-37, model `/data/models/ds41-abl`.
* `--tensor-parallel-size 4 --max-model-len 1048576 --max-num-seqs 8 --max-num-batched-tokens 8192
  --max-cudagraph-capture-size 128 --gpu-memory-utilization 0.94 --moe-backend b12x
  --kv-cache-dtype fp8_ds_mla --enable-prefix-caching`
* `--attention-config {"indexer_tp_shard_min_rows":2048}` (no `indexer_kv_dtype` key: INDEXER_KV_DTYPE=fp8
  so the served indexer cache is FP8, 132 B/key, not MXFP4's 68 B).
* `--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic",
  "rejection_sample_method":"block","enable_adaptive_verification":false,"moe_backend":"b12x"}`
* `--override-generation-config '{"temperature":1.0,"top_p":0.95}'`; `bench/codebench.py:5` sends **no**
  sampling parameters, so the served 376 tok/s is measured at temperature 1.0.
* env: `VLLM_MOET_B12X_MOE_FUSED=1 SPARSE_MLA_LDG=1 TARGET_LM_HEAD_MXFP8=1 P2P_AR=1
  P2P_AR_FUSED_MAX_ROWS=8 P2P_AG=0 DFLASH_PREP_GROUPS=1 CANDIDATE_DET_TOPK=1 SWA_DECODE_BUILD=1
  MOE_TOPKSUM_ADD=1 SWA_DRAFT_BUILD=1`, `NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS`, `OMP_NUM_THREADS=1`.
  Everything else the patch set reads is at its Python default (§4).
* log confirms each served switch took effect: `mHC overlap ... all-reduce runs right before the
  boundary kernel`; `B12X_MXFP4_MXFP8 Mxfp4 MoE backend` + `B12xExperts`; `BF16 small-M GEMM ...
  indexer,router,wk`; `wo_a stays MXFP8`; `DSpark draft model loaded: 97 params`; `DSpark fused Markov
  sampling: 5 steps, vocabulary 129280, rank 256, 505 blocks`; `drafter lm_head (32320, 5120) as an
  MXFP8 GEMV for <= 16 rows`; `target lm_head (32320, 5120) as an MXFP8 GEMV`; `Engram wkv split over
  TP=4 up to 64 tokens`; `P2P all-reduce ... one-shot push for M <= 0 rows, two-shot up to 64 rows`;
  `NCCL serialEvent records in captured graphs reduced to the last one per event`.
* memory: weights + non-torch 76.48 GiB/GPU, peak activation 3.99 GiB, CUDAGraph memory 0.80 GiB
  (plus 1.79 GiB during the first capture round), GPU KV 4,192,362 tokens at 8.8 GiB.
* live decode right now (1-2 requests in flight): 319-406 tok/s generation, i.e. the 376 number is
  reproducible on the box today.

Model shape (served `config.json`): 40 layers, hidden 5120, `moe_intermediate_size` 2304, 384 routed
experts top-6 + 1 shared, FP4 experts / FP8 dense with UE8M0 32x32 blocks, MLA `q_lora_rank` 1280,
`o_groups` 8 / `o_lora_rank` 1024, 8 index-source layers, `index_topk` 512, candidate source layer 20
(2048 blocks of 8), 2 Engram layers (1, 14), `hc_mult` 4; DSpark drafter = 3 layers taken from the
target class, 128 routed experts top-3, `dspark_block_size` 5, markov rank 256, full 129280 vocab,
aux hidden states from target layers 37/38/39.

### 1.1 The one number that explains the reference-vs-us gap

| | reference host (`-r21`) | served VM |
|---|---|---|
| Fraser bench, k=7 (same image, same flags) | 680 median / 741 best tok/s, 127.8 steps/s, accepted 5.34 | 555 tok/s, **102** steps/s, accepted 5.45 (recipe: k7 + FP4 KV + offload) |
| Fraser bench, k=5 | — (127.8 x 1.12 = ~143 steps/s scaled to k=5) | 534 tok/s, **111** steps/s, accepted 4.8 (served config) |
| memory clock | 12481 MHz + 5000 MHz afterburner OC | **12481 MHz, the only value in `nvidia-smi -q -d SUPPORTED_CLOCKS`** |
| SM clock | full + OC | 2422 of max 2430 MHz, no throttle reason active |
| power | 400 W | 600 W limit, 285-340 W drawn, never throttled |
| ECC | (workstation card, presumably on) | **Enabled** on all four |
| PCIe | Gen5 x16 | Gen5 x16, x16 width |

Accepted length travels by 2 % (5.45 vs 5.34): the drafter is fine. Steps/s travel by **25-29 %**
(102 vs 127.8 on the identical k=7 recipe; 111 vs ~143 scaled to k=5). The memory clock is the largest
single cause (the workstation SKU runs 28 Gbps GDDR7 at 1.79 TB/s, the Server Edition reports a single
lower memory clock and a 2407-2430 MHz SM boost against the workstation's 2617 MHz), but 25-29 % is
more than the memory-clock ratio alone explains, so **ECC and the sustained-clock difference are open
candidates**. Treat 127.8 steps/s as unreachable; 111-118 is the honest band here, and disabling ECC
(`nvidia-smi -e 0`, needs a GPU reset) is a test-VM experiment worth one cold start (+3-6 % if GDDR7
ECC costs what it does on GDDR6).

---

## 2. One decode verify step, in order (batch = 1, k = 5, 6 verify rows)

### 2.1 Host loop (one `execute_model` call = one step)

`work/engine/src/vllm/vllm/v1/worker/gpu/model_runner.py`:

1. Build `InputBatch`, attention metadata, block tables; the per-step *eager* metadata builders run
   here (§2.4).
2. **Target verify forward** on `[last_token, d1..d5]` (**6 rows, no padding**). FULL CUDA graph, one
   replay (`model_runner.py` execute_model -> `cudagraph_manager.run_fullgraph`). Why it is 6 and not
   a padded 8: we run Model Runner V2, whose manager sizes decode FULL graphs as
   `rounded_num_tokens = round_up(capture_size, decode_query_len)` with
   `decode_query_len = num_speculative_steps + num_new_sampled_tokens_per_step = 5 + 1 = 6`
   (`model_runner.py:442-446`, `:763-772`; `cudagraph_utils.py:321-345`), taking the
   `separate_decode_routine and not varlen_decode` branch because `varlen_decode =
   self.adaptive_verification is not None` is False for the served config (`model_runner.py:769`).
   The raw `cudagraph_capture_sizes [1,2,4,8,...,128]` printed in the EngineCore startup config is
   **not** the decode graph list — MRV2 rounds it inside the worker and does not rewrite the printed
   config (`adjust_cudagraph_sizes_for_spec_decode` is MRV1-only, `compilation.py:1495-1510`).
   The live log agrees: the real capture round prints `Capturing CUDA graphs (FULL): 0/7` = the 7
   unique rounded sizes {6, 12, 18, 24, 36, 42, 48}, bounded by
   `max_num_reqs * decode_query_len = 8 * 6 = 48`; the earlier `0/2` FULL round in the same log is the
   graph-memory estimation pass, which deliberately captures only "the largest few" FULL graphs
   (`cudagraph_utils.py:930-931`). The candidate map range-fills every token count up to each captured
   size (`cudagraph_utils.py:384-391`) and `_is_compatible` only requires `desc.num_tokens >=
   num_tokens` (`:125-138`), so a batch that falls between two captured sizes pads up to the next one
   (5 requests = 30 tokens runs the 36-token graph with 6 pad rows) — but at batch=1, 6 tokens is
   exactly the 6-token graph, **zero pad rows**. Consequence for every per-row number in this
   document: the step's cost is paid by **6 real rows**, and each +1 in k is a real +1 row (k=6 -> a
   7-row graph, k=7 -> an 8-row graph), which is why the measured k-sweep penalty is monotone.
3. **Sampling / rejection**, eager, outside the graph: `apply_sampling_params` then `rejection_sample`
   (`model_runner.py:1674` passes `self.speculator.draft_logits`; the sampler is
   `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:441-479`).
4. `postprocess_sampled` (updates request state, host side).
5. **Drafter for the NEXT step** — `self.speculator.propose(...)` at `model_runner.py:2271`, i.e. the
   draft is enqueued *after* the current step's sampler, at the end of the same call. GPU order is
   therefore: target graph -> sampler kernels -> drafter graph, and the next iteration's target graph
   only needs the drafter's output on the device (no host round trip).
6. `draft_tokens_handler.set_draft_tokens` for the next step.

So the GPU critical path per step = target verify graph + sampler kernels + drafter graph, and the
host must enqueue all of it in less than that (the reference trace says GPU busy 99 % at 7.7 ms, so
there is ~0.1 ms of host slack; at 8.77 ms our host has more).

### 2.2 Phase A — the drafter (DSpark) step

`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (DSpark) on top of
`.../dflash/speculator.py` (DFlash base):

| # | what | where | cost |
|---|---|---|---|
| A1 | `prepare_dflash_inputs` for the 3 draft KV groups — input ids, positions, query_start_loc, seq_lens, sample indices/positions, context+query slot mappings | `dflash/speculator.py:487-530` (called once per group), kernel at `dflash/speculator.py:629`; Moet step 30 fuses the 3 groups into 1 launch | 45.2 -> 18.6 us per step (eager, **outside** the graph) |
| A2 | stage aux hidden states: `torch.cat([h[start:] for h in aux_hidden_states])` + `combine_hidden_states` (main_proj 3x5120 -> 5120 + main_norm) for rows past the graph's staged rows; at batch=1 (6 rows <= 6 staged) it is the `copy_` only | `dflash/speculator.py:405-430` | eager; the main_proj itself runs inside the graph at batch=1 (`_prepare_graph_forward`, `dflash/speculator.py:190-206`) |
| A3 | drafter FULL graph replay (`DSpark speculator` capture: 6 sizes) | `dflash/speculator.py:608-610` | whole graph ~0.75 ms on the reference host |
| A3a | `combine_hidden_states` (main_proj, ReplicatedLinear 15360->5120) for the staged 6 rows | `dflash/speculator.py:198` | **replicated on every rank**: 78.6 M params; bf16 157 MB or FP8 78 MB streamed per step per rank -> ~50-100 us **estimate** |
| A3b | `prepare_context_anchor` (no-op for DeepSeek-V4.1) | `dflash/speculator.py:431-436` | 0 |
| A3c | `precompute_and_store_context_kv`: per draft layer `fused_wqa_wkv(main_x)` (q part discarded) + `kv_norm` + `fused_deepseek_v4_kv_rope_insert` | `vllm/models/deepseek_v41/nvidia/dspark.py:175-215`, `_insert_context_kv` at `:238` | 3 x (2 launches); writes the 6 verify rows' KV into the draft SWA cache |
| A3d | `_update_draft_decode_metadata` (3 SWA builders, non-causal) — Moet step 36 fuses each builder into 1 launch | `patch_vllm_swa_draft_build_sm120.py` | 9 launches -> 3; ~9 x 1 us inside the graph |
| A3e | **one** parallel backbone forward over 5 query slots `[anchor, noise, noise, noise, noise]` through the 3 draft decoder layers | `dspark/speculator.py:258-276` -> `_run_model` -> `dspark.py:197-266` | 3 x DeepseekV4DecoderLayer, i.e. 3 x (2 mHC boundaries + attention + MoE 128/top-3) |
| A3f | `compute_draft_logits` = `lm_head(norm(hidden))` for the 5 slots (shared MXFP8 copy of the target's lm_head, [32320,5120] per rank) | `dspark.py:339-345`; Moet step 18 | 32320 x 5120 FP8 = 165 MB per rank per step, ~104 us (measured, Dockerfile step 18) |
| A3g | sequential Markov sampling: 5 x (`markov_embed` + `markov_bias` over the full 129280 vocab + gumbel sample), Moet step 20 = **1 launch per draft step** | `dspark/speculator.py:156-207`; log `DSpark fused Markov sampling: 5 steps, vocabulary 129280, rank 256` | 36.6 us cold + 4 x 12.9 us from L2 = ~88 us (measured, Dockerfile step 20) |

**Answers to the questions in the task:**

* *Is the DSpark draft chain running TP4?* **Yes.** `draft_tensor_parallel_size` defaults to the
  target's TP (`vllm/config/speculative.py:1731-1745`) and is honoured for `method=dspark` — the
  drafter is built with `draft_parallel_config` (`speculative.py:1770-1790`) and the drafter's layers
  are the same `DeepseekV4DecoderLayer` class, so their `wo_b`/MoE all-reduces run over the TP4 group.
  It accepts only 1 or the target TP (no TP2).
* *All-reduce per MTP layer per draft token?* **Per layer, not per draft token.** DSpark is
  **semi-autoregressive**: one parallel forward over all 5 slots (`dspark/speculator.py:3-24,
  258-276`), so the backbone's all-reduces are 2 per draft layer = 6, plus 1 for the final collapse
  (`patch_vllm_dspark_mhc_stream_sm120.py` adds the last layer's FFN all-reduce in front of the
  drafter's final `mhc_post_tilelang`) = **7 all-reduces per draft step**, all on 5-8 rows. The
  per-draft-token part is only the Markov head: 5 launches of a rank-256 bias GEMV + sample, **no
  collective** in them.
* *Where is the MTP forward?* `vllm/models/deepseek_v41/nvidia/dspark.py:216-267` (backbone, 3 layers)
  and `:337-341` (`compute_draft_logits`); the step driver is
  `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:258-276`.

### 2.3 Phase B — the target verify graph (6 rows, 40 layers)

One FULL-graph replay. Per decoder layer, in order (`vllm/models/deepseek_v41/nvidia/model.py:483-660`
is `DeepseekV4DecoderLayer.forward`; the served wiring has `fuse_mhc_all_reduce=True` and an mHC side
stream, `model.py:749-753`):

| # | kernel(s) | where | cost |
|---|---|---|---|
| B1 | **attention boundary**: P2P push kernel (reduce-scatter + all-gather of the sublayer output, 6 rows x 5120 bf16 = 60 KB) then `mhc_post_norm` with the all-reduce consumer fused in (waits on 3 peers' flags, adds the 4 partials in NCCL's order, post-mix, collapse, RMSNorm, draft aux); the fn-projection + 20 Sinkhorn iterations run on `mhc_stream` behind it | `model.py:599-612`; `mhc_shifted_post_pre` in `vllm/model_executor/kernels/mhc/ops.py`; Moet steps 10 + 23 + 24 | push 5.7 us + fused kernel 7.7 us (measured, Dockerfile step 24) |
| B2 | `fused_wqa_wkv` MXFP8 GEMV (q_lora 1280 + kv_lora 576 out of 5120) on the main stream; **in parallel on 2 aux streams**: compressor `kv_score` and indexer `weights_proj` | `vllm/models/deepseek_v41/attention.py:969` (`_run_parallel_input_projections`), `:674-703` (`forward`) | ~10-15 us **estimate** |
| B3 | `fused_q_kv_rmsnorm_quant` (RMSNorm q + kv, MXFP8 quantize q in the same launch) | `attention.py:788-805`; upstream vllm#57679 in the base | 1 launch |
| B4 | `wq_b` MXFP8 GEMV (4096 x 1280 per rank) + `fused_qnorm_rope_kv_insert` (q norm/RoPE + KV RoPE/quant/insert into the compressed + SWA caches) | `attention.py:875` (`_fused_qnorm_rope_kv_insert`), `:872` (`_wq_b_proj`); Moet step 3 (`patch_vllm_mxfp8_gemv.py`) | 2 launches, ~10 us |
| B5 | indexer (only the 8 `index_source_layer_ids` 2,8,14,20,24,28,32,36): `_produce_k`/`forward_q` -> DeepGEMM paged-MQA logits -> top-512 (`persistent_topk`; on layer 20 the candidate-source path `select_candidate_blocks` = Moet step 31's `topk_prefill` over the score row) | `vllm/model_executor/layers/sparse_attn_indexer.py`, `.../dsa/candidate_blocks.py`; Moet step 31 | 4.9-12.7 us per layer for the top-k at 4K-300K columns + the MQA logits; ~45 us/step for the candidate selection before step 31, 3.2-10.2 us with it |
| B6 | **sparse MLA decode**: FlashInfer 0.7.2 `sparse_mla_decode_dsv4_kernel<DSV4_1,...>` (128 SWA + 512 compressed candidates, 10 splits at the served shape) with Moet step 28's LDG gather + XV `ldmatrix`, then the merge kernel | `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:602-700` (`_forward_sparse_impl`); Moet step 28 | decode 7.07 us + merge 2.1 us per layer (measured in the served graph) |
| B7 | `dsv41_o_proj`: `wo_a` grouped MXFP8 GEMV (2 groups/rank) + `fused_inv_rope_fp8_quant` + `wo_b` row-parallel GEMV with `reduce_results=False` | `flashinfer_sparse.py:629-631` (`_o_proj`), Moet step 3's grouped GEMV (`patch_vllm_wo_a_sm120.py`) | 2-3 launches, ~10 us |
| B8 | **FFN boundary**: same push + fused `mhc_post_norm` as B1 | `model.py:633-647` | 13.4 us |
| B9 | router: bf16 5120->384 GEMM (Moet step 11 spread kernel with PDL) + `dsv4_topk` sqrt-softplus top-6 (Moet step 19, plain launch) | `patch_vllm_bf16_gemm_sm120.py`, `patch_vllm_dsv4_topk_sm120.py`; log shows `M 6-8 spread 16x320` | ~6.6 + 3.9 us |
| B10 | b12x compact rows quantize (MXFP8) + **fused decode MoE** (FC1 -> SwiGLU/activation -> FC2 in one persistent kernel, 2 CTAs/SM) + `W4A16TopKSum` + shared-expert add (Moet step 33) | `patch_flashinfer_b12x_moe_fused_sm120.py` (step 27), `patch_vllm_moe_topksum_add_sm120.py` (step 33); gate `>= 32 pairs` per launch | 73.2 us per layer vs 75.7 for b12x's three kernels (measured in a graph at 22.5 distinct experts); **at the DRAM floor** |

After the 40 layers: final `mhc_post_tilelang` + `hc_collapse_triton` (draft aux hidden states for
layers 37/38/39 are captured by the `write_aux` variant of the boundary kernel), final RMSNorm, then
`lm_head` — **outside the CUDA graphs** — as the Moet MXFP8 GEMV (`patch_vllm_target_lm_head_mxfp8_sm120.py`,
step 21): [8, 32320] per rank, ~104-136 us (BF16 was ~214 us), then the vocab all-gather of
[8, 32320] bf16 logits (NCCL all-gather 54.6-62.1 us from a graph; `VLLM_MOET_P2P_AG=1` would take it
to ~54.6, measured ~15 us better, off by default).

### 2.4 Phase C — sampling / rejection (all eager, outside every graph)

`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:441-479` then the Triton kernels in
`rejection_sampler_utils.py`:

1. `apply_sampling_params` on [8, 129280] (temperature, top_p=0.95, penalties) — this is the p that
   the verifier uses (`rejection_sampler.py:264`).
2. `gather_draft_sampled` (`:129-141`).
3. block verification (`USE_BLOCK_VERIFICATION=True`, `rejection_sampler_utils.py:503-700`): Sun et
   al. block scheme — cumulative joint ratio `log_p = sum(log p_i - log q_i)` capped at 0
   (`:322-400`), residual mass per vocab block (`:403-500`), then
   `h = residual_mass / (residual_mass + 1 - prefix_joint_ratio)` and accept while `u <= h`.
   Note the **draft logits are cached pre-temperature** and scaled by `temp` inside the kernel
   (`rejection_sampler_utils.py:169`), and **top_p/top_k are applied to p only, never to q**.
4. `_resample_kernel` for the rejected position (`:804`), `get_num_sampled_and_rejected`,
   `postprocess_sampled` (host).

Per-step eager metadata work that is *also* outside the graphs (all measured in the repo):
8 target SWA builders 100.3 us with Moet step 32 (175.9 without), 3 drafter SWA builders with step 36,
drafter input prep 18.6 us with step 30, candidate selection 3.2-10.2 us with step 31, target lm_head
104-136 us, logits all-gather 55-62 us, sampler kernels ~30-60 us **estimate**. Total ~0.35-0.45 ms
per step (4-5 %) that pays full launch latency and cannot use PDL.

### 2.5 The per-step budget

Measured on the reference host at `-r21`, k=7, 7.7 ms under the profiler
(`docs/dsv41-sm120-port.md`, section "The df3bd094 line", last paragraph); our cards' column is that
number scaled by the DRAM-clock ratio for the bandwidth-bound rows (**estimate**, ratio 1.15-1.2 from
111 vs 127.8 steps/s on the same Fraser prompt).

| item | reference host, k=7 | our cards, k=5 (**estimate**) | bounded by |
|---|---|---|---|
| fused decode MoE, 40 layers | 2.7 ms | ~3.0 ms | **DRAM**: 4.2 GB/rank/step of routed-expert bytes |
| dense MXFP8 GEMVs (q_a, kv_a, q_b, wo_a, wo_b, shared expert, Engram wkv) | 0.9 ms | ~1.0 ms | **DRAM**: ~1.8 GB/rank/step |
| mHC boundaries (80 x 13.4 us) | 1.1 ms | 1.1 ms | **PCIe egress + flag latency** (not DRAM) |
| drafter graph (A3a-A3g) | 0.75 ms | ~0.8 ms | mix: 165 MB lm_head + 66 MB markov + 3 layers |
| sparse MLA + indexer + merge (40 layers) | ~0.5 ms | ~0.5 ms | latency, ~3x its KV-byte floor |
| target lm_head + logits all-gather | ~0.17 ms | ~0.20 ms | DRAM (165 MB) + PCIe |
| router + top-k + quantizers + norms + TopKSum, 40 layers | ~0.4 ms | ~0.4 ms | launch/DRAM mix |
| eager per-step metadata (2.4) | ~0.35 ms | ~0.4 ms | launch latency |
| launch gaps (~1500 kernels x ~0.35 us) | ~0.5 ms | ~0.5 ms | graph node-to-node |
| sampler + host idle | ~0.3 ms | ~0.3 ms | latency |
| **sum** | **7.7 ms** | **~8.2 ms** | the measured 8.77 leaves ~0.5 ms unattributed (rank skew, host enqueue, sub-profiler-resolution items) |

Byte budget per rank per verify step (k=5, 6 rows, arithmetic from the served `config.json`):

| stream | bytes | note |
|---|---|---|
| routed experts | **4.2 GB** | 22.5 distinct experts/layer (measured on the served model, Dockerfile step 27) x 4.7 MB (w13 2.95 MB + w2 1.47 MB + scales) x 40 layers |
| dense + shared weights | ~1.8 GB | q_a 6.55 MB + kv_a 2.95 MB are **replicated** (every rank streams all of it, `disable_tp=True`); wo_a 8.4 MB (2 groups/rank), wo_b 10.5 MB (5120x2048/rank), shared expert 8.85 MB, router 3.9 MB (bf16), q_b 1.3 MB, Engram wkv 39 MB x 2 |
| lm_heads | 0.33 GB | drafter 165 MB + target 165 MB, the same MXFP8 copy read twice per step |
| drafter backbone + main_proj + markov | ~0.3 GB | main_proj 78-157 MB is **replicated**; markov head 66 MB read 5x (L2 after the first) |
| **total** | **~6.6 GB/rank/step** | at 1.45 TB/s achieved = **4.5 ms**; at the card's 1.6 TB/s peak = 4.1 ms |

So ~50 % of the step is weight streaming, and **the single largest term is the routed-expert bytes**
(4.2 GB of the 6.6 GB). Its scaling factor is the number of *distinct* experts touched by the 6
verified rows, which is why every k change moves the step, and why nothing in the patch set can
shrink it except a smaller expert format.

---

## 3. The drafter, costed

| question | answer | evidence |
|---|---|---|
| how many forwards per step? | **one** parallel backbone forward over 5 query slots, not 5 forwards | `dspark/speculator.py:3-24` (module docstring), `:258-276` (`_generate_draft` = `_run_model` once then `_sample_sequential`) |
| what is the query layout? | `[anchor, noise, noise, noise, noise]` (`sample_from_anchor=True`, N = num_speculative_tokens), every slot predicts the next token | `dspark/speculator.py:49-55`, `dflash/speculator.py:629` (`_prepare_dflash_inputs_kernel`), config `dspark_noise_token_id` |
| TP size | 4, inherited from the target; `draft_tensor_parallel_size` is honoured and may only be 1 or the target TP | `vllm/config/speculative.py:1731-1745`, `:1770-1790`; drafter built from the target's `DeepseekV4DecoderLayer` (`dspark.py:120-133`) |
| all-reduces | **7 per draft step**: 2 per draft layer (attention boundary, FFN boundary) + 1 for the final collapse the Moet patch adds | `patch_vllm_dspark_mhc_stream_sm120.py`; port doc "The drafter's boundaries" (1,031 us graph with the TileLang pair on the model stream) |
| per-draft-token collectives | none. The 5 Markov steps are 5 launches of a rank-256 bias GEMV + gumbel sample, no collective | `dspark/speculator.py:156-207`; log `fused Markov sampling: 5 steps, vocabulary 129280, rank 256, 505 blocks` |
| cost | ~0.75 ms/step on the reference host: lm_head 104 us (165 MB/rank) + markov ~88 us (66 MB x 5, L2 after the first) + main_proj ~50-100 us (replicated) + 3 layers x (attention + 128-expert top-3 MoE + 2 boundaries) | Dockerfile steps 18, 20; port doc |
| where the drafter's inputs come from | the target's aux hidden states at layers 37/38/39 (`dspark_target_layer_ids`), concatenated to 15360 wide and projected by `main_proj`; at batch=1 the projection runs **inside** the drafter graph | `dflash/speculator.py:190-206`; `dspark.py:154-162` |

Two consequences worth stating plainly:

1. **The drafter is not the problem for step time** (9 % of the step) but it is the whole problem for
   accepted length. Its 7 all-reduces are ~90 us; its lm_head and markov head are ~200 us of pure
   weight streaming that could be cut by ~60 us with a FP4 lm_head copy (drafts only change).
2. **The draft for step N+1 is enqueued at the end of step N** (`model_runner.py:2271`), after the
   sampler, so the GPU sees target graph -> sampler -> drafter graph and the host has one full step of
   slack. Any patch that removes host work per step buys nothing until the GPU is not the limiter
   (it is: GPU busy 99 % on the reference trace).

---

## 4. Every switch the patch set reads

`default` = what the code does when the variable is unset. `served` = the value the production
container runs (from `sudo docker exec ds41 env`; "-" = unset, i.e. default).

### 4.1 Runtime switches that are on the served decode path

| switch | default | served | effect |
|---|---|---|---|
| `VLLM_MOET_B12X_MOE_FUSED` | 0 | **1** | step 27: whole decode MoE (FC1->act->FC2) in one persistent kernel; needs >= 32 pairs/launch, so the drafter's MoE stays on b12x's kernels |
| `VLLM_MOET_B12X_MOE_FUSED_PF` / `_SPIN_NS` | 4 / 512 | - | FC1 prefetch slices / CTA spin backoff inside that kernel |
| `VLLM_MOET_B12X_FC2_GROUPED` | 1 | - | step 22: FC2 grouped by expert (bit-identical); =0 restores b12x phase 2 |
| `VLLM_MOET_SPARSE_MLA_LDG` | 0 | **1** | step 28: sparse-MLA decode with plain-load gather, shared fp4->fp8 conversion, 4-barrier tile; 10.5 -> 7.07 us/layer |
| `VLLM_MOET_SPARSE_MLA_LDG_MERGE` | 1 | - | also use the rewritten merge kernel (2.1 us) |
| `VLLM_MOET_TARGET_LM_HEAD_MXFP8` | 0 | **1** | step 21: target lm_head as MXFP8 GEMV, [8,32320]/rank, ~214 -> ~104-136 us. **Changes the target logits** (~3.8 % rel error, top-5 overlap 0.96) -> owner's quality decision, gated by GSM8K/needle |
| `VLLM_MOET_DRAFTER_LM_HEAD_MXFP8` | 1 | - | step 18: drafter lm_head on the same MXFP8 copy (shared by address); only drafts change |
| `VLLM_MOET_P2P_AR` | 0 | **1** | step 23: decode TP4 all-reduces as a two-shot push over PCIe P2P in NCCL's summation order |
| `VLLM_MOET_P2P_AR_MAX_ROWS` | 64 | - | rows above this stay on NCCL |
| `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS` | **0** | - | one-shot push for rows <= this; **0 means the one-shot path never fires** (log: "one-shot push for M <= 0 rows") |
| `VLLM_MOET_P2P_AR_FUSED_MAX_ROWS` | 0 | **8** | step 24: rows <= this fuse the all-reduce into `mhc_post_norm` (push 5.7 + kernel 7.7 us instead of NCCL 11.8 + post_norm 3.9) |
| `VLLM_MOET_P2P_AG` | 0 | **0** | step 25: lm_head vocab all-gather on the P2P push instead of NCCL (~15 us/step better, measured, off) |
| `VLLM_MOET_DFLASH_PREP_GROUPS` | 0 | **1** | step 30: the drafter's input prep in one launch for all 3 draft KV groups (45.2 -> 18.6 us) |
| `VLLM_MOET_CANDIDATE_DET_TOPK` | 0 | **1** | step 31: the DSA candidate-block selection on det_topk instead of ~20-launch torch.topk (41.3 -> 3.2 us at 2.7K ctx, 39.8 -> 10.2 at 60K) |
| `VLLM_MOET_SWA_DECODE_BUILD` | 0 | **1** | step 32: each of the 8 target SWA builders' decode build in one launch (175.9 -> 100.3 us) |
| `VLLM_MOET_SWA_DRAFT_BUILD` | 0 | **1** | step 36: the drafter's 3 non-causal SWA metadata updates in one launch each |
| `VLLM_MOET_MOE_TOPKSUM_ADD` | 0 | **1** | step 33: shared-expert add fused into b12x's top-k sum (1.80 -> 1.03 us/layer, 40 fewer graph nodes) |
| `VLLM_MOET_MHC_OVERLAP` | 1 | - | step 10: mHC boundary off the critical path (post-mix+collapse+RMSNorm on the model stream, projection+Sinkhorn on a side stream) |
| `VLLM_MOET_MHC_FUSE_ALLREDUCE` | 1 | - | move the sublayer's all-reduce out of `wo_b`/the MoE to right in front of the boundary kernel (needed for the P2P fused consumer) |
| `VLLM_MOET_MHC_PDL` | 1 | - | PDL launch of `mhc_post_norm` |
| `VLLM_MOET_MHC_OVERLAP_MAX_TOKENS` | 64 | - | token limit for the overlap path (decode only) |
| `VLLM_MOET_MHC_PROJ` | "ours" | - | boundary projection kernel: `ours` / `fused` / `tf32` (>32 tokens uses the TF32 GEMM) |
| `VLLM_MOET_BF16_GEMM` | 1 | - | step 11: small-M BF16 GEMMs (indexer weights_proj, wk, router) with cuBLAS's bits in one launch |
| `VLLM_MOET_BF16_GEMM_SITES` | `indexer,wk,router` | - | which sites; `compressor` is opt-in (faster alone, slower in the served graph) |
| `VLLM_MOET_BF16_GEMM_PDL` | `router` | - | PDL launches per site |
| `VLLM_MOET_BF16_GEMM_SPLITK_MAX_M` | 8 | - | split-K kernels up to this many rows |
| `VLLM_MOET_BF16_CHAIN` | `v1` | - | `v2` = chain_ring_kernel (opt-in, calibrated like the rest) |
| `VLLM_MOET_ENGRAM_WKV_TP` | 1 | - | step 12: Engram wkv split over TP up to `VLLM_MOET_ENGRAM_WKV_TP_MAX_TOKENS` (64) |
| `VLLM_MOET_NCCL_GRAPH_RECORDS` | 1 | - | step 13: drop all but the last NCCL serialEvent record per event in captured graphs (gap in front of an all-reduce 2.8 -> 0.4 us) |
| `VLLM_MOET_MOE_FC2_DISCARD` | 1 | - | step 14: invalidate FC2's padding rows' L2 lines; `_MIN_TOKENS` 2, `_MAX_TOKENS` 64, `_SIDE` 0 (side-stream variant measured slower in the served graph) |
| `VLLM_MOET_MOE_QUANT_SCATTER` | 1 | - | step 8: MoE input quantize+permute in one launch (bit-identical) |
| `VLLM_MOET_MOE_QUANT_SCATTER_PDL` | 1 | - | PDL launch for it |
| `VLLM_MOET_DSV4_TOPK` | 1 | - | step 19: router top-6 in a plain CUDA kernel instead of Triton (-136 us critical path) |
| `VLLM_MOET_DSV4_TOPK_PDL` | 0 | - | PDL launch for the top-k (measured worse) |
| `VLLM_MOET_FUSED_MARKOV` | 1 | - | step 20: DSpark Markov sampling one launch per draft step |
| `VLLM_MOET_SM120_GEMV` | 1 | - | step 3: dense MXFP8 GEMV for M <= 16 |
| `VLLM_MOET_SM120_GEMV_BMM` | 1 | - | grouped `wo_a` GEMV |
| `VLLM_MOET_GEMV_PDL` | 1 | - | PDL for the GEMVs |
| `VLLM_MOET_GEMV_EVICT_FIRST` | 1 | - | evict-first weight loads in v3 |
| `VLLM_MOET_GEMV_IMPL` | `v3` | - | `v1` / `scalar` fallbacks |
| `VLLM_MOET_GEMV_V3_R` | 0 (table) | - | force units-per-round 1/2/4 |
| `VLLM_MOET_SM120_LL_GEMM` | 1 | - | (Qwen3.8 line, not this model) |

### 4.2 Runtime switches that exist but are off / not on the decode path

| switch | default | effect |
|---|---|---|
| `VLLM_MOET_MOE_PREQUANT` | 0 | step 34: MoE rows quantize on a side stream before the router; pb-deep -82 us/step, the real model +0.08 % |
| `VLLM_MOET_ROUTER_TOPK_FUSED` | 0 | step 35: router GEMM + top-6 in one launch; measured neutral (serial selection tail) |
| `VLLM_MOET_MOE_MEGA` / `_PDL` | 0 / 0 | step 37: routing folded into the MoE megakernel; measured +2.7 % worse. Needs the build arg `VLLM_MOET_MOE_MEGA_PATCH=1` |
| `VLLM_MOET_DET_TOPK` | 0 | step 26: deterministic indexer top-512 (changes which tied positions are kept) |
| `VLLM_MOET_DECODER_REPLAY` | 1 | prefill-only (upstream `--swa-bounded-replay` in this base) |
| `VLLM_MOET_MHC_VERBOSE`, `_GEMV_VERBOSE`, `_MOE_QS_VERBOSE`, `_MOE_L2_VERBOSE`, `_BF16_GEMM_VERBOSE`, `_MOE_GATE_VERBOSE`, `_DSV4_TOPK_CHECK`, `_SM120_MOE_GEMV_VERBOSE` | 0 | logging / self-checks |
| `*_DIR`, `*_ARCH`, `*_PTXAS`, `VLLM_MOET_HOME` | `/opt/vllm-moet/...`, `12.0`/`12.0a`/`12.0f` | where the JIT extensions and cubins live; arch strings |
| `VLLM_MOET_KV_RECORD` | `fp8_ds_mla` | build-time record layout for the FP4 KV patch |
| `VLLM_MOET_KV_FP4_FAKE`, `_KV_GATHER_ROWS` 64, `_KV_PREFILL_POOL_STATES` 0, `_NVFP4_KV` 1 | - | FP4-KV experiment knobs (not served) |
| `VLLM_MOET_ALLOW_RESULT_OVERWRITE` | 0 | bench harness only |
| `VLLM_MOET_PREFILL_DBO` / `_DBO_SCHEDULE` / `_DBO_COMM_PRIORITY` | 0 / yield_first / -1 | prefill dual-batch overlap (needs `--enable-dbo`; not served) |
| `VLLM_MOET_SM120_MOE_GEMV*` (7 vars) | 1 / 320 pairs / 512 / 1 | the older DeepGEMM MoE GEMV path (superseded by b12x here) |
| `VLLM_MOET_GEMV_*` (splitk/stage/ukt/ablate/ptxas/ref_cu) | - | GEMV tuning/measurement |

### 4.3 Dockerfile build args (gate whether a patch is applied; not visible at runtime)

`VLLM_MOET_REASONING_EFFORT` 1, `VLLM_MOET_NVFP4_KV` 1, `VLLM_MOET_OFFLOAD_DEDUP` 1,
`VLLM_MOET_MHC_OVERLAP` 1, `VLLM_MOET_DSPARK_MHC_STREAM` 1, `VLLM_MOET_BF16_GEMM` 1,
`VLLM_MOET_ENGRAM_WKV_TP` 1, `VLLM_MOET_NCCL_GRAPH_RECORDS` 1, `VLLM_MOET_MOE_FC2_DISCARD` 1,
`VLLM_MOET_PREFILL_DBO` **0**, `VLLM_MOET_INDEXER_SHARD_ROWS` 1, `VLLM_MOET_B12X_COMPACT_BIND` 1,
`VLLM_MOET_DRAFTER_LM_HEAD_MXFP8` 1, `VLLM_MOET_DSV4_TOPK` 1, `VLLM_MOET_FUSED_MARKOV` 1,
`VLLM_MOET_TARGET_LM_HEAD_MXFP8_PATCH` 1, `VLLM_MOET_B12X_FC2_GROUPED` 1, `VLLM_MOET_DET_TOPK_PATCH` 1,
`VLLM_MOET_B12X_MOE_FUSED_PATCH` 1, `VLLM_MOET_SPARSE_MLA_LDG_PATCH` 1,
`VLLM_MOET_DFLASH_PREP_GROUPS_PATCH` 1, `VLLM_MOET_CANDIDATE_DET_TOPK_PATCH` 1,
`VLLM_MOET_SWA_DECODE_BUILD_PATCH` 1, `VLLM_MOET_MOE_TOPKSUM_ADD_PATCH` 1,
`VLLM_MOET_MOE_PREQUANT_PATCH` 1, `VLLM_MOET_SWA_DRAFT_BUILD_PATCH` 1, `VLLM_MOET_MOE_MEGA_PATCH` **0**,
`VLLM_MOET_V41_KV_RECORDS` interleaved.

### 4.4 CLI flags and config fields that matter for the decode step

| flag | served | why it matters |
|---|---|---|
| `--speculative-config.num_speculative_tokens` | 5 | the trained block size (`dspark_block_size: 5`); k>5 drafts out-of-distribution positions |
| `--speculative-config.draft_sample_method` | `probabilistic` | gumbel sampling of drafts (vs `greedy` = argmax, which sets `HAS_DRAFT_LOGITS=False` and changes the acceptance test to `p(x)` only) |
| `--speculative-config.rejection_sample_method` | `block` | Sun et al. block verification (`rejection_sampler.py:163-180`); `standard` is the token-wise Leviathan test; `synthetic`/`fly` are other schemes |
| `--speculative-config.enable_adaptive_verification` | false | measured -14 % on single-stream code |
| `--speculative-config.draft_tensor_parallel_size` | unset (=4) | see §3; only 1 or 4 allowed |
| `--speculative-config.moe_backend` | `b12x` | the drafter's MoE backend |
| `--moe-backend` | `b12x` | target MoE: FlashInfer b12x W4A8 compact micro path |
| `--attention-config.indexer_tp_shard_min_rows` | 2048 | prefill only |
| `--attention-config.indexer_kv_dtype` | absent (fp8) | `mxfp4` halves the indexer cache bytes (68 vs 132 B/key) and the indexer's DRAM read at 120k |
| `--kv-cache-dtype` | `fp8_ds_mla` | 584 B/state; `nvfp4_ds_mla` is 288 B/state (2.5x KV pool, ~-50 us/step at 120k, +8 % prefill) |
| `--max-cudagraph-capture-size` | 128 | captures 19 sizes; a 6-token verify runs the **8**-token graph either way |
| `--max-num-seqs` / `--max-num-batched-tokens` | 8 / 8192 | decode unaffected; prefill chunking |
| `--gpu-memory-utilization` | 0.94 | with CUDA-graph memory profiling this is effectively 0.9184; the log itself suggests 0.9616 for the same KV as before (capacity only) |
| `--override-generation-config` | `temperature 1.0, top_p 0.95` | **the single biggest acceptance factor** (see §0 L2) |
| `--enable-prefix-caching`, `--enable-prompt-tokens-details`, `--enable-force-include-usage`, `--enable-request-id-headers` | on | no decode cost |
| env `NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS` | on | required for the P2P all-reduce to be P2P at all (the hypervisor hides P2P otherwise) |
| env `OMP_NUM_THREADS=1` | on | keeps the host threads off each other |

Nothing on the served decode path is left at a default that the repo's own measurements call
non-optimal, with two exceptions: `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS` is 0 (the one-shot path is dead
code as served) and `VLLM_MOET_P2P_AG` is 0 despite a measured ~15 us/step gain. Both are §5 items.

---

## 5. Top 10 experiments, ranked by expected tok/s per hour

### The quantitative model everything below is ranked against

The step time is **linear in the number of distinct experts touched per layer**, with a fixed
remainder. Two independent measurements pin the slope: the reference host's `-r21` step is 7.7 ms with
22.5 distinct experts (36 pairs, k=5) and 2.7 ms of MoE; widening the draft to k=7 (48 pairs, ~30
distinct) costs it 12 % of the step (-12 % steps/s, port doc table). That gives
**~120 us per extra distinct expert** on the reference host, i.e. 40 layers x 4.7 MB / 1.57 TB/s.
On our cards the same slope is ~145 us (1.6/1.45 TB/s ratio). So

    step(k=5, our cards) ~= 5.7 ms fixed + 3.0 ms MoE(22.5 experts)
    tok/s = accepted(k) / step(k)

and **600 tok/s at the current accepted length 3.45 needs a 5.75 ms step, i.e. the whole MoE for
free**. Since the MoE cannot go below ~1.6 ms without a smaller expert format, 600 tok/s requires
**both** -15..20 % of expert bytes **and** +20..30 % of accepted length. Plan accordingly: no single
switch gets there, and anything that only moves one of the two is a 2-5 % item.

Cross-check on our cards, from the Lead's fresh measurement (k=5 110 steps/s -> k=7 97 steps/s, i.e.
9.09 -> 10.31 ms): +1.2 ms for 2 extra verify rows = **~0.6 ms/row**, which is ~7.5 extra distinct
experts x 145 us. Same slope, so the model holds on both machines.

### What one extra verify row costs, split (for the drafter's row budget)

The row is almost entirely expert bytes, and that is measurable from the repo's own data rather than
from a new trace:

* **MoE**: the fused b12x kernel costs 65.9 us/layer at 22.5 distinct experts and 95.4 us/layer at 36
  distinct (measured, Dockerfile step 27) = **2.2 us per extra distinct expert per layer** (2.6 us on
  our cards). One extra verify row adds ~5.5 distinct experts per layer (union saturation:
  `E[new] = 384 x (6/384) x (1 - 6/384)^n`, i.e. 5.5-6 at n = 6-8), so **+14 us/layer = +0.57 ms/row**.
  Against the measured 0.55-0.69 ms/row that is 85-100 % of the row.
* **Everything else that scales with rows** (source-derived, and small because it is latency-bound with
  parallelism headroom at these shapes): mHC push payload 60 -> 80 KB (+0.4 us x 80 = 32 us), router +
  top-6 (+0.6 us x 40 = 24 us), sparse MLA + merge (+0.3 us x 40 = 12 us; the 60 split-tasks of a
  6-row step occupy 60 of 188 SMs, so the marginal is near zero), the indexer's per-row top-k over 131K
  columns (8 layers x 2.5 us = 20 us) plus its QK logits, the sampler over [rows, 129280] (~7 us).
* **Rows cost nothing at all** in the weight-bound parts: the dense MXFP8 GEMVs (the m16 tile loop
  covers up to 16 rows in one pass), the lm_head, the Engram wkv, the shared expert.

So the row floor without touching the expert format is **~0.47-0.55 ms**, and driving 0.61 -> 0.45 ms
requires cutting the expert bytes ~25 % (the 2-bit work, §5 E9), not cleaning up row overhead.
Decisive check if a trace is available: profile one decode step at k=5 and one at k=7 and diff the
per-kernel totals — kernels whose total does not move are weight-bound, and the ones that do are the
row cost, split by kernel.

Measurement protocol for every row below: `bench/ecc.sh bench TAG` on the VM (thinkbench + codebench
C1 at 30/60/120k, two sets each; codebench sends **no** sampling parameters so it measures the served
temperature 1.0), `bench/ecc.sh gate TAG` for quality, and `tools/sm120_perf/spec_matrix.py
--metrics-url http://127.0.0.1:8000/metrics` for steps/s + accepted/step at temperature 0. Always
compare medians of >= 6 codebench C1 runs at 120k; the run-to-run spread is +-10 %.

| # | change | exact edit | moves | expected | effort |
|---|---|---|---|---|---|
| **E1** | **truncate the draft proposal q the same way p is truncated (top_p 0.95), or drop top_p from the target** | see below | accepted +3..8 % | +3..8 % tok/s | 2 h |
| **E2** | **draft-side logit scale** (sharpen/flatten the draft before sampling) | `dspark/speculator.py:116-154` (`_sample_logits`): `logits = logits * FLOAT` before `sampler(...)`, env `VLLM_MOET_DRAFT_LOGIT_SCALE`, default 1.0. The cached `draft_logits` gets the scaled values, so the verifier's q is consistent and the output distribution stays exactly p | accepted 0..+10 % | 0..+10 % | 2 h |
| **E3** | **re-run the k sweep on r21 at temperature 1.0** (k=4,5,6) | `--speculative-config.num_speculative_tokens` 4/5/6, everything else frozen | both | 0..+4 % | 1 h |
| **E4** | **env bundle A: the four cheap switches nobody measured on r21** | `EXTRA_ENV="-e VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8 -e VLLM_MOET_P2P_AG=1 -e VLLM_MOET_DET_TOPK=1 -e VLLM_MOET_MOE_PREQUANT=1"` — one at a time | step -0..3 % | 0..+3 % | 1 h |
| **E5** | **`dspark_draft_topk`** (draft from the top-k of its own logits) | remove the Qwen3-only assertion at `vllm/config/speculative.py:1546-1553`, set `"dspark_draft_topk": 64` in the draft model config / `--speculative-config`; the path is `dspark/speculator.py:209-256` | accepted 0..+8 % | 0..+8 % | 3 h |
| **E6** | **`rejection_sample_method`: block vs standard** | `--speculative-config.rejection_sample_method standard` | accepted +-3 % | 0..+3 % | 30 min |
| **E7** | **issue the mHC all-reduce push from the producing kernel** (wo_b / the MoE epilogue writes its 3 peers' partial buffers instead of a separate push kernel) | new kernel work in `mhc_overlap/mhc_post_norm_sm120.cu` + the wo_b GEMV / b12x fused MoE epilogue; removes the 5.7 us push kernel x 80 boundaries | step -5 % | +5 % | 2-3 days |
| **E8** | **TP-shard the drafter's `main_proj`** (ReplicatedLinear 15360->5120, 78-157 MB per rank per step) | `dspark.py:98-105`: ColumnParallelLinear + one all-reduce of [5, 5120] (~7 us) — the same trade as step 12's Engram wkv | step -0.5..1 % | +0.5..1 % | 3 h |
| **E9** | **2-bit routed experts** (2-bit planes + FP4 delta pool, the repo's other line) | the v0.24 2-bit recipe (`docs/v024-port.md:1-6,229-241`, `Dockerfile.sm120-v024`) ported to our nightly + b12x compact path: re-quantize the 552B checkpoint's experts, port the 2-bit expert kernels, size the delta pool | step -17 % | +15..17 % | 2-4 weeks + quality gate |
| **E10** | **move the target lm_head + logits all-gather inside the FULL graph** | the lm_head is eager today (`patch_vllm_target_lm_head_mxfp8_sm120.py`, step 21, "runs once per decode step outside the CUDA graphs"), so it pays a launch gap and the all-gather cannot use PDL | step -1 % | +1 % | 1 day |

**E1 in detail** (the one I would run first). `apply_sampling_params` processes the target logits with
temperature **and top_p=0.95** (`rejection_sampler.py:264`) before the verifier uses them as p, but the
draft logits are only scaled by temperature inside the kernel (`rejection_sampler_utils.py:169`). So q
has mass where p is zero and every draft sampled there is rejected deterministically. At temperature
1.0 the target distribution is flat, the top-95 % set can be small, and that is exactly the regime
where the served 0.77 acceptance lives. Two ways to test it, both quality-neutral (the proposal never
changes the output distribution):

* (a) truncate q with the same top_p in `DSparkSpeculator._sample_logits` before the sampler and let
  the cached `draft_logits` hold the truncated logits. Cost: 5 x a top-p pass over 129280 logits in
  the drafter graph (~10-20 us each, **estimate**) — measure it, it may eat the gain.
* (b) cheaper: `top_k` truncation of q (one pass, no sort of the tail) with k = 512 or 1024; that is
  E5's mechanism, and it removes the same tail mass.

**Bound the gain before writing the patch (30 min, no code, but read the caveat).** The mismatch can
only cost what the top_p truncation of p removes, so boot once with
`--override-generation-config '{"temperature":1.0,"top_p":1.0}'`, everything else frozen, and read the
per-position conditional acceptances and AL from the same bench. `dAL = AL(top_p 1.0) -
AL(top_p 0.95)` bounds what truncating q can recover; the per-position `dc_i` says where it sits.
**Caveat (drafter owner's): the arm is confounded and one-sided** — raising top_p to 1.0 removes the
support mismatch (raises acceptance) *and* raises the target's entropy (lowers acceptance), so a
near-zero or negative dAL is ambiguous; only `dAL > 0.05` is strong evidence *for* E1. A mechanism-only
probe without the entropy confound: `top_k=1024` (removed p-mass ~0, so p is essentially unchanged
while its support is still truncated) vs `top_k` off — but that prices a 1024-token support, not the
served top-95 % set, so the definitive test remains the patch itself (E1b, 2 h). The drafter owner's
baseline on r21 (k=5, T=1.0, C1 workload) is `c = [0.812, 0.759, 0.730, 0.672, 0.725]`, AL 3.40; that
vector is exactly self-consistent (`1 + sum_i prod_{j<=i} c_j = 3.3997`) and `dAL/dc_1 = (AL-1)/c_1 =
2.955` exactly, so +0.05 on `c_1` is +4.3 % AL. **The metric for this patch is `c_1`, not AL.**

**What I would not spend time on** (already measured by the repo, do not re-litigate):
`enable_adaptive_verification` (-14 % single-stream code), k=7 (measured -9 % net on codebench at
T=1.0: +2 % accepted, +12 % step), `VLLM_MOET_MOE_MEGA` (+2.7 % worse), `VLLM_MOET_MOE_FC2_DISCARD_SIDE`
(+70 us in the served graph), `VLLM_MOET_ROUTER_TOPK_FUSED` (neutral), `VLLM_MOET_GEMV_V3_R` (neutral),
`VLLM_MOET_MHC_PROJ=tf32` at decode, `--enable-dbo` for decode (it is a prefill feature).

---

## 6. Latent bugs and wasteful work

1. **The one-shot all-reduce path is unreachable as served.** `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS`
   defaults to 0 and is unset in the container, so the log line says `one-shot push for M <= 0 rows`
   and every decode boundary pays the two-shot path (2 phases, 6 peer-hops) instead of 1 phase, 3
   peer-hops. The repo measured "neutral in the full C1 graph", but that was on the full-depth dummy
   model on 21d93d0d; 80 boundaries x 2-3 us is 0.16-0.24 ms/step (2-3 %) and nobody has measured it
   on r21 with 6 rows. Cheap test (E4).
2. **q is untruncated while p is top_p-truncated** (§5 E1). This is the only place in the sampler where
   acceptance is lost for free; the code is not wrong (the scheme still yields exactly p), but the
   proposal is systematically mismatched to the verification distribution.
3. **`main_proj` (drafter) and `fused_wqa_wkv` (target, `disable_tp=True` at
   `vllm/models/deepseek_v41/attention.py:371-377`) are replicated**: every TP rank streams the whole
   weight. `fused_wqa_wkv` is 9.5 MB/layer x 40 = **380 MB per rank per step (~0.26 ms, 3 %)** of
   redundant streaming — the same defect step 12 fixed for Engram's wkv. It is not cheap to fix here
   (the q_lora output feeds q_b and both indexer wq_b's, so sharding it needs an all-gather per layer,
   ~40 x 5-10 us, which mostly cancels the 196 MB it would save); worth knowing before someone
   "optimizes" it.
4. **The drafter's lm_head and the target's lm_head are the same weight read twice per step**
   (2 x 165 MB FP8 = 0.23 ms). The MXFP8 copy is already shared by address
   (`patch_vllm_target_lm_head_mxfp8_sm120.py`), but the traffic is not. There is no way to fuse them
   across the graph boundary (the drafter runs after the sampler, the target before), but the
   *drafter's* copy could be FP4 (82 MB, -60 us, drafts only, no target-logit change).
5. **`get_num_nans` and the logprobs path run over [8, 129280] logits per step** when enabled; with
   `--enable-prompt-tokens-details` the top-k logprob gather is per request. Small, but it is eager
   work on the critical path and worth checking in a trace.
6. **`draft_tensor_parallel_size` cannot be 2** (`speculative.py:1737-1745` accepts only 1 or TP). At
   TP1 the drafter's 7 all-reduces vanish (~90 us) but its per-rank weight streaming quadruples
   (lm_head 165 -> 662 MB), so TP4 is right — just note the restriction exists if someone tries to
   tune it.
7. **The decode graph sizes are rounded up to multiples of `decode_query_len`** (§2.1 item 2:
   `cudagraph_utils.py:321-345`, MRV2), so the served k=5 captures graphs for 6, 12, ... 48 tokens and
   a one-request step runs exactly 6 rows — no pad rows anywhere in the step. A k=7 sweep therefore
   adds 2 real rows (attention, sampler, lm_head, and 12 more (token, expert) pairs) on top of the
   ~7.5 extra distinct experts per layer. Read every per-step cost in §2.5 as per-6-rows.
8. **`--gpu-memory-utilization 0.94` is effectively 0.9184** under CUDA-graph memory profiling (the log
   says so and suggests 0.9616 for the same KV as before). Not a speed item, but the KV pool number
   (4.19 M tokens) is 1.5 GiB below what the card would allow.

---

## 7. What the r21 recipe leaves on the table vs the Moet README

* **The card.** README's 680 tok/s / 127.8 steps/s is a 400 W workstation card with GDDR7 +5000 MHz.
  The served Server Edition has one supported memory clock (12481 MHz) and 600 W of headroom it does
  not use (285 W drawn). ~15-20 % of step time is not recoverable. This is the single largest item in
  the gap and it is not a knob.
* **DSpark k = 5 instead of the README's 7.** Deliberate and correct for the agent workload at
  temperature 1.0 (measured: k=7 = +2 % accepted / +12 % step), but it does mean the README's Fraser
  row (680/741) is not comparable to our 376: the served config is tuned for codebench, not Fraser.
  If the team's 600 target is measured on the Fraser prompt, k=7 + FP4 KV is worth re-running.
* **`VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=8` instead of the README's 16** — served is *better* (C2 -1.3 %
  measured when it was lowered), nothing left there.
* **`VLLM_MOET_P2P_AG=0`** despite a measured ~15 us/step gain (step 25). Left off.
* **`VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS` never set** (dead path, §6.1).
* **`VLLM_MOET_MOE_PREQUANT`, `VLLM_MOET_DET_TOPK`, `VLLM_MOET_ROUTER_TOPK_FUSED`, `VLLM_MOET_MOE_MEGA`,
  `VLLM_MOET_MOE_FC2_DISCARD_SIDE`** all off; three of the five were measured *worse* on the real model,
  two are neutral — nothing obviously left on the table, but prequant/det_topk were measured on r18/pb-deep
  and never on r21.
* **FP4 KV + MXFP4 indexer (`nvfp4_ds_mla`) not served.** The README recipe uses it. The workspace
  measured it as decode-neutral and prefill-costly; at 120k context the indexer's FP8 cache (132 B/key)
  reads ~126 MB/step against MXFP4's 65 MB (~40-50 us/step, 0.5 %), so there is a small decode win
  available if the KV pool trade is acceptable.
* **Everything else the r21 recipe sets is a superset of the README** (steps 30-33 and 36 on top of
  the README's 17-28), and each of those is bit-identical to the path it replaces.

---

## 8. Hand-back: the one-line experiment recipes for the Lead

1. **Trace (gates E7/E10, 40 min):** on the test VM, served config, one stream at 120k context:
   `python3 tools/sm120_perf/profile_capture.py http://127.0.0.1:8000 <served-model-name> '{"thinking":true,"reasoning_effort":"max"}' 96 1`
   then `tools/sm120_perf/trace_agg.py` / `trace_gaps.py` on the rank-0 trace; report the top-20
   kernels by total time and the critical-path DAG span of the target graph vs the drafter graph.
2. **E1 (highest expected value, 2 h):** patch `DSparkSpeculator._sample_logits` to apply the same
   top_p=0.95 (or top_k=512) to the draft logits before sampling, cache the truncated logits, A/B six
   codebench C1 runs at 120k; accept if accepted length rises > 3 % with identical greedy outputs on
   the quality gate.
3. **E2 (2 h):** same function, `logits *= scale`, sweep scale in {0.8, 0.9, 1.0, 1.1}; report accepted
   length and steps/s. This one is free of quality risk by construction.
4. **E4 (1 h, 4 cold starts):** four single-env A/Bs, each against the same-state baseline:
   `VLLM_MOET_P2P_AR_ONESHOT_MAX_ROWS=8`, `VLLM_MOET_P2P_AG=1`, `VLLM_MOET_DET_TOPK=1`,
   `VLLM_MOET_MOE_PREQUANT=1`; report steps/s (codebench C1 at 120k, 6 runs).
5. **E3 (1 h):** k=4/5/6 on r21 at temperature 1.0, 6 runs each at 120k.

---

## 9. Answers to the Lead's four source questions + the profiler recipe

### 9.1 `monokernel` — real, but AMD-only; on CUDA the flag is silently ignored

* It is the **DeepSeek-V4.1 ROCm mono decode layer**: `vllm/models/deepseek_v41/amd/mono_decode.py:1-25`
  — "A decode step's backbone layer — the attention seam, the attention, its TP all-reduce, the FFN
  seam, the MoE and its all-reduce — runs as **two persistent FlyDSL launches** (`mono`) on the layer's
  loaded weights, the TP reductions inside the kernels." So it is **per decoder layer**, not the whole
  step: K1 = attention seam + attention + TP all-reduce, K2 = FFN seam + MoE + TP all-reduce
  (`mono_decode.py:330-400`). 40 layers x 2 launches instead of ~1000 kernels.
* Constraints: CDNA3/CDNA4 only (`CDNA_VERSIONS = (3, 4)`, i.e. gfx942 MI300X and gfx950 MI355X,
  `mono_decode.py:71`), TP 2 or 4, FlyDSL + AITER helpers, enough CUs for every CTA to stay resident;
  decode-only steps of at most `MAX_ROWS = 48` rows (42 on gfx942, `mono_decode.py:63-66`) — our 6-row
  step fits. A deployment that cannot run it is **refused at startup** (`raise ValueError("--monokernel
  {why}.")`, `mono_decode.py:378-385`).
* Enable: `--monokernel` (`vllm/engine/arg_utils.py:1826`) or `--kernel-config '{"monokernel":true}'`;
  the field is `KernelConfig.monokernel` (`vllm/config/kernel.py:295-300`).
* **On CUDA it does nothing.** The only call site is `vllm/models/deepseek_v41/amd/model.py:277`
  (`self.mono = MonoDecodeLayer.create(...)`) — the ROCm model. The CUDA model
  (`.../nvidia/model.py`) never reads the flag, so `--monokernel` is parsed and ignored, with no
  startup refusal. Nothing on sm_120 has been measured because the code path does not exist there.
* Verdict for us: **not a lever** unless someone ports the two kernels from FlyDSL to CUDA (weeks to
  months). The idea is the same one Moet step 37 tried on the MoE alone (measured +2.7 % *worse*,
  because the phases are serial and the PDL overlap with the boundary kernel is lost) — the AMD win
  comes from doing the seams, the attention and the MoE in one persistent grid with the TP reductions
  in-kernel, which is exactly what the 80 PCIe all-reduces + 1000 launches we pay for would need.

### 9.2 `fuse_allreduce_rms` / `fuse_rope_kvcache_cat_mla` / `fuse_attn_quant` — inert with `mode=NONE`

All three are **inductor post-grad fusion passes** (`CompilationConfig.pass_config`,
`vllm/config/compilation.py:114-160`, consumed by `vllm/compilation/fusion.py`). The served config has
`compilation_config.mode = NONE` (log: `'mode': <CompilationMode.NONE: 0>`), i.e. the model is **not
torch.compiled**; the FULL/PIECEWISE graphs are captured over eager execution (with breakable CUDA
graphs, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` auto-enabled). So the pass flags have no effect. The
served `pass_config` is the O0 preset with `fuse_norm_quant`/`fuse_act_quant` left True
(`vllm/config/vllm.py:251-270`); both are equally inert.

* To turn them on: a compile mode, e.g. `-O2` / `-O3` or `--compilation-config '{"mode":"VLLM_COMPILE"}'`
  (`vllm/config/vllm.py:293-334` = OPTIMIZATION_LEVEL_02/03, which set `fuse_allreduce_rms`,
  `fuse_attn_quant` and `fuse_rope_kvcache_cat_mla` from the platform helpers).
* Two non-inductor side effects exist but do not apply here: `fuse_allreduce_rms` adds compile ranges
  for the AITER/FlashInfer custom all-reduce at config time (`vllm/config/vllm.py:2665-2680`, a no-op
  without AITER), and `fuse_attn_quant` gates a FlashInfer kernel registration
  (`vllm/v1/attention/backends/flashinfer.py:1945-1959`) that requires SM100 + trtllm-gen XQA — not our
  sparse-MLA path.
* Note the Moet line deliberately runs `mode=NONE` (port doc: "torch.compile -O3 is off for this
  model"), because the model's forward is a stack of custom ops. Turning compile on would be a
  project, not a flag.

### 9.3 `sparse_indexer_topk_backend`, `linear_backend`, `moe_backend`: legal values and what auto picks

**`sparse_indexer_topk_backend`** (`vllm/config/kernel.py:342-357`): `auto | deep_select | cooperative |
persistent | per_row | flashinfer | torch | aiter`. `auto` = the pre-existing chain
cooperative -> persistent -> per_row. On sm_120 with our model the decode top-512 runs
**`persistent_topk`** (the Moet tools README says so explicitly, and step 26's notes call it "the decode
`auto` backend"). `deep_select` is SM100a/SM103a only. The faster option is **not in the enum**: Moet
step 26's deterministic radix-select kernel behind `VLLM_MOET_DET_TOPK=1`, which beats persistent_topk
at every width (4K columns 4.9 vs 7.4 us, 32K 7.8 vs 15.5, 131K 9.4 vs 15.1, 300K 12.7 vs 15.8, 512K
20.5 vs 29.4). At 120k context x 8 index layers that is **~-46 us/step (-0.5 %)** plus run-to-run
determinism. It changes which tied positions are kept, so it needs the quality gate. (The candidate-block
selection at layer 20 already uses the same kernel via `VLLM_MOET_CANDIDATE_DET_TOPK=1`, served.)

**`linear_backend`** (`kernel.py:359-380`): `auto | cutlass | flashinfer_cutlass | flashinfer_cutedsl |
flashinfer_trtllm | flashinfer_cudnn | flashinfer_b12x | b12x | marlin | triton | deep_gemm | torch |
aiter | machete | fbgemm | conch | exllama | emulation | xpu | xpu_woq`, plus
`linear_backend_per_quant`. On sm_120 `auto` picks **FlashInfer CUTLASS** for the MXFP8 dense linears
(`FlashInferCutlassMxfp8LinearKernel`) — which is exactly what Moet step 3's GEMV patch hooks
(`patch_vllm_mxfp8_gemv.py` patches `apply_weights` on that class) and where step 3's grouped `wo_a`
GEMV is installed. **Warning: `--linear-backend b12x` or `flashinfer_b12x` (the SM12x native FP8/FP4
linear kernels, which `auto` does not pick) would bypass the Moet GEMVs** — the small-M decode GEMMs
would fall back to the CUTLASS 128-row tile. Only worth an A/B for prefill, and it must be re-measured
at decode.

**`moe_backend`** (`kernel.py:301-340`): served `b12x` explicitly and it must stay explicit. `auto`
for MXFP4 experts resolves through `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:436-462`:
priority `[FLASHINFER_TRTLLM_MXFP4_MXFP8, DEEPGEMM_MXFP4, MARLIN, BATCHED_MARLIN, HUMMING]`, filtered
by `is_supported_config`. TRTLLM-GEN MXFP4 is SM100-only, so **on sm_120 `auto` would land on
DeepGEMM's MXFP4 grouped MoE** — the path the Moet line replaced: it pads every touched expert to 128
rows (2x slower at decode) and, before step 17's bind patch, cost the b12x/TP4 port 3.80M -> 9.90M KV
tokens. `b12x` = FlashInfer 0.7.2's W4A8 compact micro path, and the image's
`patch_flashinfer_b12x_compact_bind.py` is what makes it bind at TP4 with 576 intermediate rows per
rank (576 % 128 == 64 -> the compact N64 micro kernels).

### 9.4 The drafter: one block forward, TP4 honoured, 7 all-reduces

Short answer (full detail in §3): **one** parallel forward over all `num_speculative_tokens` slots per
verify step (`[anchor, noise, noise, noise, noise]`, non-causal over the block), then 5 cheap Markov
launches; **not** k sequential forwards through the 3 MTP layers. `draft_tensor_parallel_size` **is**
honoured for `method=dspark` (default = target TP = 4; only 1 or 4 allowed,
`vllm/config/speculative.py:1731-1745`), so the drafter does run TP4 with 2 all-reduces per MTP layer
(attention boundary, FFN boundary) + 1 for the final collapse = **7 per draft step**, ~90 us of PCIe
all-reduce in a ~0.75 ms drafter graph (~1 % of the step). TP1 would delete the 7 all-reduces but
quadruple the drafter's per-rank weight streaming (its lm_head alone is 165 MB/rank at TP4), so TP4 is
the right choice; the 7 all-reduces are not worth attacking.

### 9.5 The profiler recipe (exact, for vLLM 0.32.1rc1.dev117+gdf3bd094a)

Launch the server with the torch profiler enabled (this is the Moet launcher's own path,
`docker/sm120/run-dsv41.sh:218-223`, `PROFILER=1 PROFILE_DIR=<host dir>`):

```
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /profiles \
--profiler-config.torch_profiler_with_stack false
```
(bind-mount the host dir at `/profiles`; gzip is on by default, so the trace is
`/profiles/<prefix>_<rank_suffix>.pt.trace.json.gz` — add `--profiler-config.torch_profiler_use_gzip false`
for a plain json that `trace_*.py` can read directly.)

Then, to profile **exactly one decode step**, start the profile *while a decode is in flight* and let
`max_iterations` stop it (`vllm/config/profiler.py:161-165`; the endpoint is
`vllm/entrypoints/serve/profile/api_router.py:35-58`, `POST /start_profile` accepts
`profile_prefix`, `delay_iterations`, `max_iterations`):

```bash
# 1. one long request at 120k context (bench/mkctx.sh builds the context), 4096 output tokens, streaming
python3 bench/codebench.py http://127.0.0.1:8000 /tmp/prof "120:1x1" 4096 &
sleep 25                                    # prefill done, decode under way
# 2. profile exactly one engine iteration (= one verify step)
curl -s -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"profile_prefix":"c1_decode_120k","delay_iterations":0,"max_iterations":1}'
sleep 5
curl -s -X POST http://127.0.0.1:8000/stop_profile
```
The Moet repo's own wrapper does the same thing with a fixed window and is worth reusing:
`python3 tools/sm120_perf/profile_capture.py http://127.0.0.1:8000 <served-model-name> '{"thinking":true,"reasoning_effort":"max"}' 96 1`.

**Do not** use `--observability-config '{"enable_layerwise_nvtx_tracing":true}'` for the decode step:
the config docstring says it "doesn't work with CUDA graphs enabled"
(`vllm/config/observability.py:86-88`), and our decode is a FULL graph replay — the NVTX ranges fire
during capture, not on replay. Two usable variants: (a) profile the **capture** with
`--profiler-config.capture_torch_profiler true`, which runs the model eagerly with the hooks and gives
per-layer attribution (timings are capture-shaped, useful for attribution only); (b) run one
`--enforce-eager` (or `--compilation-config '{"cudagraph_mode":"NONE"}'`) start with layerwise NVTX to
get per-layer attribution, accepting that eager step time is not the served step time.

For the per-kernel split inside the graphs, the trace + the Moet trace tools are the right pair
(`tools/sm120_perf/`): `trace_agg.py` (top kernels by total time), `trace_gaps.py` (kernel-to-kernel
gaps), `trace_graphs.py` (per-graph span), `trace_step_skew.py` (rank skew), and
`nccl_graph/graph_topology.py` for the DAG. That is how the repo's own "critical path 7,367 -> 7,231 us,
rank-0 DAG over 24 replays" numbers were produced.
