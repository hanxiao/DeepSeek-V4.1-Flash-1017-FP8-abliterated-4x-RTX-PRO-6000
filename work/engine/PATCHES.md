# P1 / P2: what to write, what they are worth, and the one thing to test first

Written after re-reading the kernels. **Two of my round-2 numbers were too kind to these patches**; the
corrections are the most useful part of this note, so they come first.

## Correction 1: the mHC push is at the PCIe egress floor, and the fused consumer moves 2.3x the bytes

`p2p_ar/p2p_allreduce_sm120.cu:73-105` (`push_phased_kernel`): grid = 3 destinations x 64 CTAs, and
**phase p waits for phase p-1** (`if (p > 0) while (phase[p-1] < ctas) {}`) because "a kernel writing to
several peers concurrently is ~16x slower" on this host. So one push = 3 sequential phases x 60 KB =
**180 KB of PCIe egress**, then `__threadfence_system()` + a release flag per destination. At the 31.6 GB/s
that 5.7 us implies, that is the floor, not latency.

And the fused consumer (step 24, `wait` in `mhc_post_norm`'s kAR template) then **reads back 4 partials =
240 KB** (180 KB of it over PCIe). Per boundary the fused path therefore moves 180 KB out + 240 KB in =
**420 KB**, against the two-shot path's 90 KB out + 90 KB in = **180 KB** (`push_twoshot_kernel` +
`reduce_slice_kernel` + `assemble_kernel`). The served choice (fused one-shot) was made on the
full-depth dummy model (-1.25 % step period) and never A/B'd against the two-shot at 6 rows.

**So the first experiment is one env var, zero code: `VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=0`** turns the
fused consumer off, which puts the boundary back on `_moet_p2p_all_reduce` = the two-shot push +
`mhc_post_norm`. If the two-shot wins (plausible: 2.3x less traffic, 3 extra launches), it is worth
0.5-1.5 % and it makes P1 moot. Run it before spending days on P1.

## Correction 2: a CUDA graph does not make kernels faster — P2 is worth ~0.1 %, not 1.9 %

P2's premise ("the lm_head + all-gather are eager, so a graph would absorb 0.17 ms") is wrong in
magnitude: capturing a kernel removes only its launch gap (~1-2 us) and the host enqueue (hidden while
the GPU is the limiter). The lm_head's 104-136 us and the all-gather's 55-62 us are real GPU work and stay
real inside a graph. P2 is therefore **~5-10 us (0.1 %)**. The 180 us can only be cut by making those
kernels cheaper: the all-gather has a cheaper variant already (`VLLM_MOET_P2P_AG=1`, 54.6 vs 62.1 us), and
the lm_head is at its bandwidth floor. **Recommendation: skip P2** unless it rides along with a bigger
graph change; the two env vars below are worth more for zero code.

---

## P1 — issue the mHC push from the producing kernel

**What it buys, honestly: 1.3-2.2 % (-0.12..-0.20 ms), not 5 %.** The push's 5.7 us is 180 KB of egress
at the floor; fusing it into the producer cannot remove those bytes, it can only (a) remove the launch
(~0.35 us), (b) overlap the first phase's writes with the producer's own tail (~1-2 us of DRAM work), and
(c) let the producer's `__threadfence_system()` replace the push kernel's. 1.5-2.5 us x 80.

**Which producers (answer to the Lead's question: two different kernels, not one):**
* **40 boundaries are the FFN boundary** (`model.py:633`, `mhc_shifted_post_pre(..., self.hc_ffn_fn)`) and
  consume the **attention output**: producer = `wo_b`, i.e. the Moet MXFP8 GEMV
  (`sm120_gemv/mxfp8_gemv_sm120.cu`, `mxfp8_mma_gemv_v3_kernel`, installed by `patch_vllm_mxfp8_gemv.py`)
  or CUTLASS above 16 rows. CUDA extension -> needs a rebuilt/JIT-built kernel.
* **39 boundaries are the attention boundary** (`model.py:599`, `mhc_shifted_post_pre(..., self.hc_attn_fn)`,
  every layer after layer 0) and consume the **previous layer's FFN output**: producer = the MoE's
  epilogue, i.e. Moet step 33's fused TopKSum + shared add Triton kernel
  (`patch_vllm_moe_topksum_add_sm120.py`). Triton -> easiest to change, but it needs raw peer pointers as
  arguments, which Triton cannot take directly from `ipc_open` ints (you would wrap them with
  `torch.frombuffer`-style device views or keep the write in the extension).
* **1 site is the final collapse** (`model.py:749`, `_moet_p2p_all_reduce`), which uses the two-kernel
  path and needs no producer change (and is a candidate to switch to the fused consumer for free).

**The change, in three pieces:**
1. `p2p_ar/p2p_allreduce_sm120.cu`: factor `push_phased_kernel`'s store loop into a device function
   `push_rows(const __nv_bfloat16* x, int M, int rank, Peers3 peers, unsigned* ctrl)` that a producer can
   call after its last store, keeping the phase counters and the `__threadfence_system()` + release flag.
   Export `push_into(...)` so the producer kernels can be passed `peers_data/peers_flags/ctrl` as args.
2. `p2p_ar/p2p_allreduce_sm120.py`: add `fused_push_args()` returning the raw ints (peers_data,
   peers_flags, ctrl, rank) and a `mark_pushed(x)` that returns the consumer dict **without launching the
   push**; `push_for_fused` keeps working for the sites whose producer is unchanged.
3. The two producer kernels take those args and call `push_rows` at the end. `mhc_post_norm` is unchanged:
   it already waits on the flags and adds the 4 partials.

**Bit-exactness: safe.** The summation order is applied by the *consumer* (`wait` in kAR adds in NCCL's
order from the measured table), so moving the transport into the producer cannot change any bit — the
only failure mode is ordering (the flag must be release-ordered after the stores, which is exactly what
`push_phased_kernel` already does).

**Effort: 2-4 days** (two kernel epilogues + a JIT rebuild path; the GEMV side is the expensive half
because the extension is compiled into the image — an overlay would have to `torch.utils.cpp_extension.load`
a modified `.cu` at import, +1-3 min of boot). That is over the Lead's one-day rule, so: **run the
`FUSED_MAX_ROWS=0` A/B first; write P1 only if that A/B does not already give the 0.5-1.5 %.**

**Mount (once written):** `OVERLAYS="/data/patches/p2p_allreduce_sm120.cu:/opt/vllm-moet/dsv41_sm120/p2p_ar/p2p_allreduce_sm120.cu /data/patches/p2p_allreduce_sm120.py:/opt/vllm-moet/dsv41_sm120/p2p_ar/p2p_allreduce_sm120.py"` (the .cu needs the JIT to recompile: clear `TORCH_EXTENSIONS_DIR`'s `p2p_allreduce_sm120` subdir, or change the `load(name=...)` to a new name so a fresh build happens).

**10-minute check after boot:** the log line `P2P all-reduce: rank 0 of 4, NCCL's summation order
reproduced for 64 of 64 row counts` must still appear (same count -> the order table is intact); then one
greedy 128-token request must match the reference token-for-token (the push is bit-exact, so any
difference means an ordering bug). Step rate: `vllm:generation_tokens_total` / accepted over 60 s, or one
codebench C1 run — but 1.5 us x 80 is 0.12 ms of 9.1 ms, i.e. inside the run-to-run noise of one run, so
verify it with the trace kernel count (80 `push_phased_kernel` launches -> 0), not with tok/s.

---

## P2 — lm_head + vocab all-gather inside the FULL decode graph

**What it buys: ~5-10 us (0.1 %).** See correction 2. The mechanism if it is still wanted:

* Where: `model_runner.py:1637-1638` (`sample()`: `logits = self.model.compute_logits(sample_hidden_states)`)
  and `:993` (the dummy run). The FULL graph is captured around the *model* forward only
  (`cudagraph_manager.capture(create_forward_fn)`, `model_runner.py:1104`), so the gather of the logits
  rows + norm + lm_head + vocab all-gather all sit outside it.
* The change: extend the captured forward to `compute_logits(model_out[logits_indices_buf])` on static
  buffers — a persistent logits buffer sized to the largest captured decode size (32320 columns x 8 rows
  bf16 = 0.5 MB), a static `logits_indices` buffer (the input buffers' is already static), and the vocab
  all-gather captured inside (NCCL collectives are already captured in this graph; the P2P all-gather is
  graph-capturable too — the repo measured it "from a graph").
* What breaks if done wrong: the sampler reads the logits **after** the replay, so the graph's output
  buffer must be stable (it is: allocated in the graph's private pool); `sample_hidden_states` must be
  gathered inside the graph with the static indices buffer, not with `input_batch.logits_indices` contents
  read on the host; and the drafter's lm_head (step 18) must keep its own path (it is called inside the
  drafter graph already).
* Effort: ~half a day, but it touches the central runner; **>30 % chance of a subtle break** (graph
  capture of a collective whose output buffer is reallocated per call) and 0.1 % payoff. My advice: only
  as part of a bigger graph change (e.g. if the sampler is ever captured too).

**The two env vars that beat both patches for zero code, in order:**
1. `VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=0` — the two-shot vs fused one-shot A/B (correction 1), 0.5-1.5 %.
2. `VLLM_MOET_P2P_AG=1` + `VLLM_MOET_DET_TOPK=1` — the all-gather and the indexer top-k, ~-60 us at 120k.

---

## Trace plan (ready for when the retrain frees the VM)

```bash
# k=5 (served), decode only, 120k context
PROF=2 GATE=0 THINK=0 bash /var/lib/ds41/x/exp.sh r2dec-k5 /etc/ds41/r21.env "120:1x1" ""
# k=7, same bench
sed 's/^: "${SPEC_TOKENS:=5}"/: "${SPEC_TOKENS:=7}"/' /etc/ds41/r21.env > /tmp/k7.env
PROF=2 GATE=0 THINK=0 bash /var/lib/ds41/x/exp.sh r2dec-k7 /tmp/k7.env "120:1x1" ""
python3 vm/traceparse.py /var/lib/ds41/x/r2dec-k5/profiles/*rank0*.json.gz --top 40 > /tmp/k5.txt
python3 vm/traceparse.py /var/lib/ds41/x/r2dec-k7/profiles/*rank0*.json.gz --top 40 > /tmp/k7.txt
```
Then the per-kernel diff of the two traces is the marginal row cost split by kernel: kernels whose total
does not move are weight-bound (lm_head, dense GEMVs, mHC push), the ones that move are the row cost
(expect ~0.55-0.6 ms of it in the b12x MoE kernels, per §5). Hand me both traces (or the two traceparse
outputs) and I will produce the split table.
