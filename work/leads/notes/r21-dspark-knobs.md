# The r21 image's DSpark knobs, the confidence head, and FLy

Verified 2026-11-08 **on the running container**, not from the source tree:
`sudo docker exec ds41 python3 -c "import dataclasses, vllm.config.speculative as s; ..."` on the
production box. The source tree (`work/engine/src/vllm` at df3bd094) is the *base* nightly; the image is
that base plus the Moet patch set, and the two differ in exactly the places below.

## 1. The served image's SpeculativeConfig, all 41 fields

```
enforce_eager, num_speculative_tokens, model, method, draft_tensor_parallel_size, tensor_parallel_size,
quantization, moe_backend, attention_backend, kv_cache_dtype, max_model_len, revision, code_revision,
index_share_for_mtp_iteration, disable_padded_drafter_batch, disable_eagle_block_drop,
draft_lm_head_quantization, use_local_argmax_reduction, use_heterogeneous_vocab, prompt_lookup_max,
prompt_lookup_min, parallel_drafting, target_model_config, target_parallel_config,
num_speculative_tokens_per_batch_size, draft_model_config, draft_parallel_config,
suffix_decoding_max_tree_depth, suffix_decoding_max_cached_requests, suffix_decoding_max_spec_factor,
suffix_decoding_min_token_prob, draft_load_config, rejection_sample_method, synthetic_acceptance_rates,
synthetic_acceptance_length, enable_adaptive_verification, draft_sample_method, dspark_draft_topk,
fly_window_size, fly_entropy_threshold, fly_entropy_top_k
```

Consequences:

- **There is no `dspark_scheduler`, `dspark_per_request`, `dspark_pad_to_bucket`,
  `dspark_confidence_threshold` or `dspark_budget_frac` in our image.** Those names are in
  `patch/vllm-moet-v0.24.0.patch` in the Moet clone (the v0.24.0 line), not in the nightly base we serve.
  The confidence head can therefore only be exercised through `enable_adaptive_verification`.
- **`dspark_draft_topk` is a SpeculativeConfig field** (settable through `--speculative-config`), and the
  speculator also reads it from the drafter's `hf_config` (`speculator.py:78-79`). It is the split-vocabulary
  top-k of the **greedy** draft path, so it does nothing for our probabilistic drafting.
- `rejection_sample_method` accepts `standard`, `synthetic`, `block`, **`fly`** - FLy is available and unused.
- Also present and unused: `num_speculative_tokens_per_batch_size` (dynamic k by batch size),
  `synthetic_acceptance_rates/length`, `prompt_lookup_min/max`, `suffix_decoding_*`.

## 2. FLy: the one acceptance lever in the image that needs no training

`rejection_sample_method: "fly"` (arXiv 2511.22972, "Accepting Semantically Correct Drafts Beyond Exact
Match"): an entropy gate decides whether the current position admits several plausible continuations, and a
mismatched draft token is accepted anyway, with the decision deferred over a lookahead window of the next W
native decisions. vLLM's kernel:

```
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:1115-1117   fly_window_size / threshold / top_k
:718-722   entropy = load(fly_entropy_ptr); if (... and entropy >= fly_entropy_threshold): accept
vllm/config/speculative.py:619-627        defaults: window min(6, k-1), threshold 0.3, top_k 3
:1906-1909  warning: "FLy verification is a lossy speculative decoding method and may degrade model outputs"
```

Why it should pay here specifically: the measured per-position curve (`work/leads/LEADS.md` §0) shows our
rejections are spread over positions where the target is uncertain, which is exactly where the entropy gate
fires. It is lossy, so gate it with GSM8K-100 (we are at 96% on r21 k5 FP8), the 9/9 smoke test, and a greedy
agreement check against the served baseline.

## 3. The confidence head IS in the checkpoint, and we never run it

`mtp.2.confidence_head.proj.weight` is a tensor in the released checkpoint (96,085 tensors total; 2,401
`mtp.*`; exactly one confidence-head tensor, plus `mtp.2.markov_head.embed.weight` and
`mtp.2.markov_head.head.weight`, all in `model-00046-of-00048`). Matches the DSpark paper: "a single linear
projection plus sigmoid that reads the backbone hidden state and the Markov feature" (arXiv 2607.05147).

vLLM loads it by default for our model class (`vllm/models/deepseek_v41/nvidia/dspark.py:147-152`,
`enable_confidence_head` defaults True) and runs it only with adaptive verification:

```
vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:106-111
  self.use_confidence_head = self.enable_adaptive_verification and model.model.confidence_head is not None
  if self.use_confidence_head: self.use_acceptance_estimator = False
compute_confidence -> torch.sigmoid(confidence_head(hidden, markov_embed))   dspark.py:356-361
```

Served config has `enable_adaptive_verification: false`, so the head is loaded and never executed. It is a
per-position acceptance predictor - useful as a diagnostic (calibration vs capacity) and, in principle, for
scheduling, but the scheduling idea is measured at -14% on code / -2.6% on prose on this hardware, and the
Moet scheduler implementation is not in our image anyway.

## 4. Two sampler facts worth knowing before patching acceptance

- The draft logits are cached **pre-temperature** and scaled by `temp` inside the rejection kernel
  (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:169`).
- `top_p` / `top_k` are applied to the target distribution p only, never to the draft q
  (`rejection_sampler_utils.py:267`, `apply_sampling_params`). q therefore has mass where p has none, and
  every such proposal is an outright rejection. Applying the same truncation to q is lossless (q may be any
  distribution) and is the cheapest untested acceptance patch in the stack.

## 5. Block size is a hard constraint, not a tuning knob

`dspark_block_size: 5` in the checkpoint config, with `num_nextn_predict_layers: 3`,
`dspark_target_layer_ids: [37,38,39]`, `dspark_markov_rank: 256`, `dspark_n_routed_experts: 128` top-3,
`dspark_noise_token_id: 128799`. Drafting beyond the trained block is out of distribution, which is why
k=6/7 buy almost nothing today. Any plan that wants accepted length above ~4 must train a longer block, and
DeepSpec (deepseek-ai/DeepSpec, MIT) is the code that does it - `block_size` is a config field, and its
released DSpark checkpoints are block 7.
