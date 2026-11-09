# Fine-tuning the DSpark MTP head of DeepSeek-V4.1-Flash-1017-abliterated

Goal: raise **position-1 draft acceptance from 0.812 to 0.88-0.90** on the daily coding-agent workload
(thinking at max effort, T=1.0) and **train the block out to k=9**, which is what the row economics
need: alpha 0.88-0.90 with a trained block of 8-9 is 563-616 tok/s single stream, and alpha alone
inside the shipped block of 5 caps at ~534 (`work/leads/notes/dspark-drafter-training.md` §3).

Acceptance is a **lossless** speedup: verification is exact rejection sampling, so the served
distribution is unchanged and GSM8K-100 / needle / gates must not move. That also means a quality
regression is a bug, not a trade-off.

Everything below is grounded in the actual artefacts, not in the papers:

| claim | where it was checked |
| --- | --- |
| the drafter is 2401 tensors, 7.933 GB, 3 full V4.1 decoder layers with 128 experts top-3 | safetensors headers of `model-00044..46-of-00048.safetensors`, `config.json` `dspark_*` |
| the reference forward path (block, Markov head, confidence head, position convention) | `/data/models/ds41-abl/inference/model.py:1022-1160`, `:1264-1285` |
| the engine's serving path (block length = `num_speculative_tokens`, sequential Markov sampling, weight remap) | `vllm/models/deepseek_v41/nvidia/dspark.py:474-632`, `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:156-276`, `dflash/speculator.py:700-780` |
| the training objective, sampler, position weights, hyperparameters | `deepseek-ai/DeepSpec` @ 005e03b: `deepspec/modeling/dspark/{loss,common,markov_head}.py`, `config/dspark/dspark_qwen3_4b.py` |
| the quantisation conventions (FP8 32x32 ue8m0, FP4 row-32 ue8m0, nibble order) | `inference/kernel.py:40-204`, `inference/convert.py:13-53`; `quantize.py selftest` reproduces them |

Deliverables in this directory: `PLAN.md`, `run.sh`, and the conversion step (`quantize.py` +
`tools/load_mtp.py` + `tools/export_mtp.py`), plus the code `run.sh` drives.

---

## 1. What the drafter is, exactly

`mtp.{0,1,2}` in the checkpoint, 7.933 GB, 2.653 / 2.574 / 2.706 GB per layer, **~14.1 B params**
(experts are 4.53 B of each layer's 4.69 B). Per layer: MLA attention (one shared KV head, K == V,
partial interleaved RoPE on the trailing 64 of 512 dims, per-head sink, grouped low-rank output),
128-expert top-3 MoE plus one shared expert, sqrtsoftplus routing with `e_score_correction_bias`,
mHC residual streams (`hc_mult=4`, 20 Sinkhorn iterations), and:

* `mtp.0` additionally owns `main_proj` (15360 -> 5120) and `main_norm` — the combiner for the target
  hidden states at **layers 37/38/39**;
* `mtp.2` additionally owns the final `norm`, the **Markov head** (rank 256: `embed` [129280, 256],
  `head` [129280, 256], whose output is added to the logits) and the **confidence head**
  (`proj` [1, 5376] = hidden + markov rank).

The serving forward (`dspark.py`, `speculator.py`):

1. the target emits, per token, the mean over the hc copies of the hidden state entering layers
   37/38/39 (`model.py:576`, `:1106-1109`); the drafter projects the concat with `main_norm(main_proj(x))`
   into `main_x` and every draft layer derives its **context KV** from that same `main_x` with its own
   `wkv`/`kv_norm` (`dspark.py:193-224`);
2. one parallel forward over `N = num_speculative_tokens` query slots: slot 0's embedding is the last
   real token, slots 1..N-1 are the noise token 128799; each slot attends to the last 128 context
   positions **plus every slot of its own block, bidirectionally** (the DSpark mask);
3. logits = shared LM head; then a sequential Markov chain adds `markov_head(prev_token)` and samples
   left to right (`speculator.py:187-198`). **Position 1 is the parallel backbone's own prediction
   plus the anchor token's Markov bias; positions 2..N inherit position 1's token.**

Position convention (verified twice, and the reason it matters): for an anchor at token index `p`,
query slot `j` carries RoPE position **`p + 1 + j`** and predicts the token at index `p + 1 + j`
(`inference/model.py:1057` with `start_pos = p`; `dflash/speculator.py:742` `query_pos =
last_valid_pos + 1 + query_off`). DeepSpec's Qwen3 code uses `anchor + j` — off by one for this
checkpoint. `deepspec_v41/modeling.py` uses `position_offset=1`, and `tests/test_local.py` asserts
the label/markov-prefix alignment.

Measured today: `c = [0.812, 0.759, 0.730, 0.672, 0.725]`, AL 3.40 at k=5. Those are **conditional**
acceptances; `1 + 0.812 + 0.812*0.759 + ... = 3.40` reproduces the measured AL, which is the sanity
check that we are training the right quantity.

---

## 2. What DeepSpec gives us, and what it does not

`deepseek-ai/DeepSpec` (MIT, commit `005e03b`, cloned to `vendor/DeepSpec`) is DeepSeek's own
full-stack drafter-training code: data prep, three drafter families (DSpark / DFlash / Eagle3),
training, evaluation over nine benchmarks. **It does not support DeepSeek-V4.1**: the draft model
implementations are `deepspec/modeling/dspark/{qwen3,gemma4}/` and the trainer builds them from a
`transformers` config of the target family. There is no V4.1 module, no MoE draft, no mHC.

What we reuse, verbatim in design:

| DeepSpec artefact | how it is used here |
| --- | --- |
| `modeling/dspark/loss.py` | the objective: CE + `sum|p-q|` (= 2*TV) + confidence BCE, position weights `exp(-j/4)`, 0.1 / 0.9 / 1.0 mix, all-reduced denominators. Reimplemented in `deepspec_v41/loss.py` with the TVP generalisation and a top-K target support. |
| `modeling/dspark/common.py` | the anchor sampler (`sample_anchor_positions`), `build_eval_mask`'s contiguous-prefix rule, `create_noise_embed`. Ported to `V41DSparkModel.sample_anchors`/`hidden_states`. |
| `config/dspark/dspark_qwen3_4b.py` | hyperparameters: `num_anchors=512`, `markov_rank=256`, `confidence_head_with_markov`, `loss_decay_gamma=4.0`, `ce:l1 = 0.1:0.9`, warmup 4%, grad clip 1.0. Kept except where §5 says otherwise. |
| `data/target_cache_dataset.py` + `scripts/data/prepare_target_cache.py` | the *idea* of a precomputed target cache: per-token hidden states + the target's final hidden state, so training never runs the 510 GB target. Ours stores the target's **top-K distribution** instead of its final hidden state (see §4) because the 129280-way head is 6.1 TFLOP per sample and we would rather pay it once offline. |
| `scripts/eval/*` | the acceptance-length evaluation. Ours uses the VM's own battery (§8) because the engine, not a HF model, is what has to improve. |

What we had to write (all in this directory):

* `deepspec_v41/modeling.py` — the V4.1 DSpark draft: mHC, MLA-with-DSpark-context, 128-expert MoE,
  Markov/confidence heads, block sampler, and the `p+1+j` position convention. Reuses
  `transformers==5.20.2`'s `DeepseekV4RMSNorm`-equivalent math only as a reference; the mHC, MoE and
  attention are implemented here because the checkpoint's own reference (`inference/model.py`) is the
  authority and the training path must not depend on inference kernels.
* `deepspec_v41/loss.py` — DSpark/TVP objective on a top-K support.
* `tools/load_mtp.py`, `tools/export_mtp.py`, `quantize.py` — checkpoint <-> model plumbing in the
  native FP8/FP4/ue8m0 format (see §7).
* `dump/*` — the engine-side hidden-state dump and the cache builder.

---

## 3. Why the previous fine-tune was flat, and what is different here

On record: `Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft` fine-tuned exactly these three layers on
the served model's own outputs and reported "about flat in thinking mode at temperature 1". And the
published DSpark paper's open-ended-chat AL at T=1.0 is 3.49, i.e. **our 3.40-3.45 is already the
published level for prose** — the drafter is not broken, prose at T=1.0 is the hard case.

So the plan does not bet on "fine-tune on own outputs" alone. Five differences, each independently
justified:

1. **In-domain data, thinking-on, with the agent's tool protocol** (DeepSpec's released drafters are
   trained on non-thinking open-perfectblend; its README says to fine-tune for thinking mode, and the
   TVP paper attributes part of its gain over released heads to the same shift).
2. **TVP instead of fixed TV** (§5): +1.5% AL over TV when fine-tuning, +9.8% over CE, and the
   margin grows with draft length (+1.5-1.9% over TV at longer drafts) — which is the regime we need.
3. **Train the block out to 9**, so k=9 rows are in-distribution. At today's alpha the extra rows do
   not pay (0.6 ms/row vs 0.23 accepted tokens needed); at alpha 0.88 they do.
4. **The target is the abliterated model.** The ablation removed a rank-2 subspace from
   `attn.wo_b`/`ffn.shared_experts.w2` of all 40 layers, so the hidden states at 37/38/39 that
   `main_proj` consumes are *not* the distribution the shipped drafter was trained on. Refitting
   `main_proj`/`main_norm` on the abliterated states is free (3 of the 2398 trained tensors).
5. **All 14.1 B parameters are trainable**, including the routed experts (the previous fine-tune
   touched the same layers, so this is a difference of budget and objective rather than architecture).

Honest ceiling risk: if the three-layer head simply cannot separate the target's distribution better
than TV≈0.19 on this domain, no objective fixes it. §6 puts a cheap early read on that (train-set
accept rate after 200 steps) before the expensive stages are repeated.

---

## 4. Data pipeline

### 4.1 What we need per training token

For an anchor at index `p` we need (a) the target hidden states at 37/38/39 for every position in the
128-token window `[p-127, p]`, and (b) the target's next-token distribution at `p+1 .. p+k` for the
loss. (b) is 129280 floats per position if stored raw; we store the **top-256** support plus the
residual mass, which is a valid coarsening of the same distribution and can only *under*-estimate TV
(TV is a contraction under coarsening; the TVP paper finds a top-16 support loses little).

Per token on disk: aux 3x5120 bf16 = 30 KB, top-256 ids + logprobs + residual = 2 KB, tokens/mask 5 B
→ **~32 KB/token**. 40 M tokens = 1.3 TB, and the raw dump before compaction is 1.6 TB. `/data` is
6 TB of RAID0 with 510 GB of checkpoint, so one run fits with room; the dump is deleted after the
cache is built.

### 4.2 Stage A — generate in-domain traces *and* dump hidden states in one pass

The dump is a ~60-line insertion into the image's DFlash speculator, mounted through the launcher's
existing `OVERLAYS` mechanism (`vm/ds41-start-r21.sh`: `OVERLAYS="host:container"`), so no image
change. It records, for every token the target forwards (prefill chunks and decode steps):

```
int32 n ; n x (int32 token_id, int32 position, int32 seq_id) ; n x (bf16 aux[3*5120], bf16 last[5120])
```

* `aux` is the tensor the served drafter consumes (`previous_post.mean(dim=1)` at layers 37/38/39,
  `model.py:576`, `:1185`), so the training input is byte-for-byte the serving input;
* `last` is the target's final normed hidden state, from which the offline pass recomputes the
  target's exact next-token distribution with the checkpoint's own `head.weight`;
* only rank 0 writes (TP4 without sequence parallelism replicates the hidden states).

Generation uses the production workload shape: a coding-agent prompt with a repository context, three
tools offered, `thinking` enabled with the full budget, T=1.0 / top_p=0.95 (the served generation
config), max_tokens 20000, 32 concurrent requests. Prompts come from `dump/gen_traces.py` (ten
repository-scale coding tasks, optional `--context-file` for the 30 k-120 k context of the real
traffic). Each request gets a unique system prompt so prefix caching does not swallow the prompt.

Volume: **40 M generated tokens** by default (`TOKENS=40000000`, i.e. 2000 requests x 20 k). At the
measured C8 aggregate (688-765 tok/s) that is 14 h; with `MNS=32` for the dump run it should be
2-3 k tok/s, i.e. **5-6 h**. `run.sh` prints the running tok/s, so the first 100 requests tell you
whether to cut `TOKENS` in half. Prompt tokens are a bonus, not part of the budget: after the first
request the shared context is a prefix-cache hit and is only dumped once.

### 4.3 Stage B — reassemble and compact (`dump/reassemble.py`)

Two passes, bounded memory. The dump is **71.7 kB per row** (12 int bytes + 3 aux blocks + the
model's 4 HC blocks of 5120 bf16), so the 10.2 M rows of the first run are 729 GB: nothing may hold
the file in RAM.

1. **pass 1** reads only the 12-byte integer block of every record and remembers each row's float
   offset (~200 MB of index for 10 M rows). 5 s per million records.
2. sort by `(seq, position)`, keep the **last** record per position (rejected speculative rows are
   re-forwarded with a different token), cut contiguous runs into samples of at most `MAXLEN=8192`;
3. **pass 2** preads the rows a sample needs (8 us/row warm), mean-collapses the HC stream, recomputes
   the target distribution from `head.weight` (fp32, chunked), and writes flat memmaps the trainer
   reads with zero parsing: `tokens.bin`, `loss_mask.bin`, `lengths.bin`, `aux.bin` (bf16 bits),
   `topk_ids.bin`, `topk_logp.bin`, `residual.bin`, `meta.json`.

Two guards learned the hard way: the row size is **detected from the file** (candidates
`12 + (3+k)*5120*2`, the one whose header walk consumes ≥99 % of the file wins) instead of assumed,
and a truncated tail record is tolerated. With the wrong row size the walk desynchronises after the
first record and the stage fails loudly instead of writing garbage.

The target's next-token distribution needs the hidden state that feeds `head.weight`, i.e.
`norm(hc_collapse(streams, pre_mix))`. With MTP enabled the model keeps the full HC stream and the
speculator only sees that, so the dump has **4 HC copies of 5120**, not the collapsed 5120, and
`pre_mix` (the last layer's per-token FFN mix) is not in the dump. We stand in with the **mean over
the 4 copies**: RMSNorm makes that exact whenever the four mix weights are equal (a uniform scale
cancels), so the residual error is the per-token non-uniformity of `pre_mix` — a rotation of the
collapsed vector, not a scale error. `meta.json` records the rule, and the raw dump is kept so an
exact collapse can be added later without re-dumping.

---

## 5. Loss

Per supervised position `j` (0-based slot), with target `p` (the stored top-K distribution plus the
residual bucket), draft `q = softmax(logits)`, and position weight `w_j = exp(-j/gamma)`:

```
L = ce_alpha * CE  +  tvp_alpha * D_beta(p, q)  +  conf_alpha * BCE(confidence, 1 - TV(p, q))
```

* **CE** = `-log q[target token]`, computed on the full 129280-way draft logits (exact).
* **`D_beta`** is the TVP family (arXiv 2611.00588, eq. 4):

  ```
  D_beta(p, q) = (1/beta) * sum_v p(v) * [ 1 - min(1, q(v)/p(v))**beta ]
  D_0  = sum_{q<p} p log(p/q)      (coverage / forward KL on under-covered tokens)
  D_1  = TV(p, q) = 0.5 * sum_v |p(v) - q(v)|
  ```

  with `beta` **annealed linearly 0 -> 1 over the first 80 % of training**. `D_0` keeps a gradient on
  tokens the draft under-predicts (TV's gradient vanishes there, since TV reads the under-covered side
  with weight 1); `D_1` ends on exactly the quantity rejection sampling cares about, because a draft
  token is accepted with probability `1 - TV`.
* **Why TVP and not the shipped L1 term.** DSpark's `l1_loss_alpha=0.9` term *is* `2*TV`, so TVP is a
  drop-in replacement for it, one parameter, no new hypernetwork: at `beta=1` the loss is the shipped
  objective exactly (asserted in `tests/test_local.py::test_tvp_limits`), and annealing from 0 buys
  the reported +1.5 % AL over TV / +9.8 % over CE on fine-tuning, with a larger margin at longer
  drafts. It costs one extra `log`/`exp` per token.
* **Top-K coarsening** keeps the objective tractable: `p` and `q` are compared on the stored 256-token
  support plus one residual bucket, `q_res = 1 - sum_K q`. `TV` on the coarsened pair is a lower bound
  of the true TV, so the loss is a conservative version of the served acceptance.
* **Confidence head**: BCE against `1 - TV`. Trained but not on the critical path — `dspark`'s
  adaptive verification is unsupported on sm_120 (`DeepseekV41IndexerBackend` cannot trim device
  side), so the head is exported and unused. `conf_alpha=0.1` (DeepSpec uses 1.0) to keep it from
  competing with the acceptance term.
* **Position weights and masks** follow DeepSpec exactly: `gamma=4.0`, and supervision stops at the
  first position whose label leaves the sequence or the loss mask (contiguous prefix).

Draft logits are computed in fp32 in anchor chunks of 64 (576 rows x 129280 x 4 B = 300 MB), because
one sample's full logits at k=9, 512 anchors would be 2.4 GB.

---

## 6. Hyperparameters

| knob | value | source / why |
| --- | --- | --- |
| draft layers / experts / top-k | 3 / 128 / 3 | checkpoint `dspark_n_routed_experts`, `dspark_num_experts_per_tok` |
| target layers | 37, 38, 39 | `dspark_target_layer_ids` |
| noise token | 128799 | `dspark_noise_token_id` |
| **block_size** | **9** | must cover the k we serve; the row budget only pays for k>5 at alpha >= 0.85 |
| position offset | +1 | `inference/model.py:1057`, `dflash/speculator.py:742` |
| num anchors / sample | 512 | DeepSpec DSpark recipe |
| sample length | 8192 tokens | the draft only ever sees a 128-token window; longer samples buy nothing but I/O |
| epochs | 4 (10 M-token cache) | ~1245 samples / 32 per step = 38 steps/epoch, so 4 epochs = 152 steps, the budget the 40 M-token plan assumed in one pass |
| global batch | 32 samples = 262 k tokens | 4 GPUs x 1 sample x 8 accumulation steps |
| lr | 1e-4, cosine to 10 %, warmup 4 % | fine-tune value (DeepSpec's 6e-4 is from scratch); sweep {5e-5, 1e-4, 2e-4} if the first run under-shoots |
| weight decay / grad clip | 0.0 / 1.0 | DeepSpec |
| precision | bf16 params, fp32 reduce, fp32 Adam states | FSDP `MixedPrecision` |
| `ce_alpha` / `tvp_alpha` / `conf_alpha` | 0.1 / 0.9 / 0.1 | DeepSpec's 0.1 / 0.9, confidence scaled down |
| `loss_decay_gamma` | 4.0 | DeepSpec |
| `tvp_anneal_frac` | 0.8 | TVP paper |
| top-K target support | 256 | TVP finds top-16 loses little; 256 is cheap insurance |
| frozen | `embed.weight`, `head.weight` | shared with the target, must stay identical |

Trainable: 14.1 B params (~28 GB bf16). FSDP FULL_SHARD over 4 GPUs, one FSDP unit per draft block:
params 7 GB + grads 7 GB + Adam 28 GB + one all-gather peak 9.4 GB + frozen buffers 2.7 GB +
activations ~4 GB ≈ **60 GB of 96 GB per GPU**.

---

## 7. Export and loading without rebuilding the image

Two separate problems: (a) write the trained tensors back in the checkpoint's native format,
(b) get the engine to read them while `/model` is a read-only bind mount.

### 7.1 Requantise to native (`quantize.py`, `tools/export_mtp.py`)

`tools/export_mtp.py` walks the original 2401 `mtp.*` tensors as a template (so the file is complete,
including the three unused `ffn.gate.bias_vl`), replaces the 2398 the model owns, and writes them in
the checkpoint's own format:

* dense weights: FP8 E4M3, **32x32** blocks, scale `2**ceil(log2(max(amax, 1e-4)/448))` stored as
  F8_E8M0 (exponent only) — `inference/kernel.py:40-124`;
* expert weights: FP4 e2m1, blocks of 32 **along the input dim only**, scale
  `2**ceil(log2(max(amax, 6*2**-126)/6))`, two values per byte with the **even column in the low
  nibble** — `inference/kernel.py:127-204`, `inference/convert.py:31-34`;
* everything else (norms, router, Markov/confidence, sink) in its stored bf16/fp32 dtype.

Correctness is checkable without a GPU: `python3 quantize.py roundtrip --model /data/models/ds41-abl`
dequantises and requantises all 2398 tensors and asserts the values are unchanged. It is exact
because the stored values are `code * 2**k`, which is closed under the same rule (the scale may
tighten for a block whose largest code is < 3, which doubles every code and preserves every value).
That is Gate 0, and it is the reason we can trust the export path before spending a GPU-hour.

### 7.2 Serve it: overlay checkpoint directory (chosen)

`/model` is `-v /data/models/ds41-abl:/model:ro`, a **directory** mount, so a single-file overlay
cannot reach inside the 7.4 GB shards that hold the `mtp.*` tensors. Three candidates:

| option | cost | risk |
| --- | --- | --- |
| **A. overlay checkpoint dir** (chosen) | 22 GB written once (~30 s on RAID0) | none structural: `config.json` and `model.safetensors.index.json` are byte-identical, 45 of 48 shards are **hardlinks**, only the three shards holding `mtp.*` are rewritten |
| B. separate draft dir + `"model": <dir>` in `--speculative-config` | 8 GB | the draft `ModelConfig` is then built from that directory, so its `config.json` must resolve the same `hf_config` the dspark drafter reads (`config.hidden_size`, `dspark_target_layer_ids`, `n_mtp_layers`, ...) — a different code path from the one production uses |
| C. rebuild the image | 40 min + a new tag | unnecessary; the weights are data, not code |

`quantize.py build` implements A: hardlink every shard without `mtp.*`, copy the small side files,
rewrite the three `model-0004{4,5,6}-of-00048.safetensors` with the same tensor names, order and
dtypes, and leave the index untouched. The target model reads the same 45 hardlinked shards, so the
**backbone weights are bit-identical**; only the drafter differs. Serving needs no config change at
all: `method=dspark` with no `model` uses the target path, so pointing the launcher's `M` at the
overlay is enough (`run.sh` makes `M` overridable once with a `sed` on the launcher, idempotent).

Rollback: `M=/data/models/ds41-abl` and restart. Nothing in the image, `/model`, or the index changes.

---

## 8. Eval gate

Everything runs through the VM's existing battery (`/usr/local/bin/exp.sh`), so the numbers are
directly comparable to the baseline in `work/RESULTS.md`:

```bash
M=/data/models/ds41-abl-mtp-v2 SPEC_TOKENS=9 THINK=1 GATE=1 \
  bash /usr/local/bin/exp.sh mtp-block9-k9 /etc/ds41/r21.env "30:1x3,120:1x2" ""
```

| measurement | artefact | pass |
| --- | --- | --- |
| accepted length, codebench C1 @30k (the graded workload) | `/var/lib/ds41/x/<tag>/cb.log` (`C1 n=...`) | **>= 3.70 at k=9** (baseline 3.40 at k=5) |
| per-position conditional acceptance | `think.jsonl` via `tools/report_think.py` | **c[0] >= 0.86**, and c[1..k] no worse than today's decay (`c[j] >= c_today[j]` for j<=5) |
| same at k=5 (isolates the block-length effect from the accuracy effect) | second eval run with `SPEC_TOKENS=5` | AL >= 3.55 at k=5, i.e. the accuracy gain alone is real |
| decode tok/s | `cb.log` | reported, not gated: step time grows with k, so 600 tok/s is a `k=9` + step-time-work result, not a drafter-only result |
| quality (lossless) | `/opt/bench/gate.sh`, `gate2.py` (image), GSM8K-100 | no regression beyond noise; a real drop means the drafter is broken |
| train-set accept rate | trainer log (`acc`) | early read at step 200: if `acc < 0.85` after a full epoch the 3-layer ceiling is probably real — stop and reconsider before repeating stages |

Kill criteria: if `c[0] < 0.83` after one epoch, the objective/data change did not move the backbone;
the next lever is capacity or a longer block, not more of the same data.

---

## 9. Cost

| stage | wall clock | GPU-hours | note |
| --- | --- | --- | --- |
| gate0 | 10 min | 0.2 | quantisation roundtrip + model load + forward |
| dump (40 M tokens) | 5-14 h | 20-56 | the long pole; 2-3 k tok/s at MNS=32, 700 tok/s at MNS=8. Includes one container start |
| cache | 30 min | 2 | I/O bound (1.6 TB read, 1.3 TB write) |
| train (1 epoch) | 1-1.5 h | 4-6 | 152 optimizer steps x ~12-20 s |
| export | 15 min | 1 | 22 GB of writes |
| eval | 40 min | 3 | one container start + codebench + thinkbench + gates + GSM8K-100 |
| **total** | **8-17 h** | **30-70** | dominated by generation; halving `TOKENS` halves the run |

If the first epoch shows movement, the second run is data-scaling only (no code change): 150 M tokens
needs 4.9 TB of cache, so either use `MAXLEN=4096` samples or store `main_x` (5120 instead of 15360)
and freeze `main_proj`.

---

## 10. Order of operations on the VM

`run.sh` runs these in order; each stage is idempotent and can be re-run alone.

```
bash run.sh gate0     # 10 min, no GPU contention with the engine (it stops it)
bash run.sh dump      # 5-14 h, engine up with the patched speculator
bash run.sh cache     # 30 min, GPUs idle
bash run.sh train     # 1-1.5 h, engine stopped
bash run.sh export    # 15 min
bash run.sh eval      # 40 min, engine up on the overlay
```

Time-sharing: the dump and the eval need the served engine (all four GPUs, one container at a time);
training needs the same four GPUs. So it is strictly sequential: engine up -> dump -> engine down ->
cache -> train -> export -> engine up -> eval. Two container starts, ~6 min each. The 510 GB
checkpoint is only read (Gate 0, the cache's `head.weight`, the export template), never rewritten.

All Python runs inside the serving image (`--entrypoint python3`), so the host needs no torch.

---

## 11. Risks, and what is not yet verified

**Verified locally (CPU, `tests/test_local.py`, 9 tests):** TVP limits (`beta=1` == TV, `beta=0` ==
coverage KL, monotone in `beta`, residual bucket counted once), the anneal schedule, the anchor
sampler, the `p+1+j` label/markov-prefix convention, the 128-token causal window, finite
forward/backward, the FP8/FP4/ue8m0 conventions (value-exact requantisation), and a full
native-format checkpoint round trip through the model (2398/2401 tensors mapped and re-exported; the
three unmapped ones are the unused `ffn.gate.bias_vl`).

**Verified against the real checkpoint (read-only ssh):** the 2401-tensor inventory and per-layer
counts, all shapes/dtypes, the name mapping round trip on the real names, and `head.weight` /
`embed.weight` locations.

**Not yet verified — check these on the first VM run:**

1. **No numerical parity test against the engine's drafter.** A parity check would need the image's
   kernel stack; instead the export path is proven value-exact (§7.1) and the eval gate is an
   accepted-length measurement, which is the number that matters. If the first eval shows AL far
   *below* baseline (e.g. < 2), suspect the model forward, not the training: build an identity
   overlay (`quantize.py build --model ... --out ...` with no `--weights`, which copies the mtp
   tensors unchanged) and re-measure; that isolates the overlay path from the training path.
2. **The mHC/MoE/attention reimplementation** follows `inference/model.py` and
   `transformers 5.20.2`'s `DeepseekV4DecoderLayer` (which has the same tensor names and shapes);
   the parity spot-check that is still cheap to add is one layer's output against
   `transformers.models.deepseek_v4.DeepseekV4DecoderLayer` loaded from the same weights.
3. **The speculator patch** is generated from the production container's exact file
   (`vendor/vllm/.../dflash/speculator.py`, copied from the image the test VM runs) and
   `dump/make_patch.py` asserts on each of its three anchors, so a different image fails loudly
   instead of silently dumping nothing. Verify the first dump file is non-empty and that
   `reassemble.py` reports ≈ `REQUESTS x 20000` generated tokens.
4. **Memory at k=9, 512 anchors**: ~60 GB/GPU estimated. If the first training step OOMs, drop
   `num_anchors` to 256 (halves the block rows and the head chunks) before changing anything else.
5. **The 40 M token budget** is calibrated on an assumed 2-3 k tok/s aggregate with `MNS=32`; the
   stage prints live tok/s, so cut `TOKENS` at the first sign of a slower rate.
6. **The target distribution uses a mean-collapsed HC stream** (§4.3). TV is 1-Lipschitz, so if the
   mean-collapsed target sits at TV = delta from the true one, the achievable acceptance is short by
   at most delta; the symptom is a train accept rate that saturates below what the eval implies. If
   that happens, the fix is the exact collapse (needs `pre_mix`, so one more 2 h dump), not more data.
7. **`enable_adaptive_verification` stays off** (unsupported on sm_120), so the confidence head is
   exported but unused; nothing in the gate depends on it.

Fallbacks if alpha does not move: (a) freeze the routed experts and train only attention + shared
expert + `main_proj` (cheap, 20x less memory, tests whether the mismatch is in the input projection);
(b) sweep lr {5e-5, 1e-4, 2e-4}; (c) block 5 (reproduce the shipped block exactly, isolating the
block-length change); (d) more data.
