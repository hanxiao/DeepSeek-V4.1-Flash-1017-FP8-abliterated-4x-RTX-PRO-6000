"""Fine-tune config for the V4.1 DSpark head. Numbers in comments are the source of the value.

DeepSpec's own DSpark recipe (config/dspark/dspark_qwen3_4b.py) is the starting point; the
differences are the ones this checkpoint and this workload force:

  * block_size 9 instead of 7 -- the engine drafts ``num_speculative_tokens`` slots per step, so the
    trained block must cover the k we intend to serve. The row budget only pays for k>5 once the
    per-position acceptance is above ~0.85 (work/leads/notes/dspark-drafter-training.md, section 3).
  * target_layer_ids (37, 38, 39) -- dspark_target_layer_ids of the checkpoint; the draft reads the
    attention input of those layers, not every 8th layer.
  * 3 draft layers, 128 experts top-3, markov rank 256, confidence head with markov -- fixed by the
    checkpoint.
  * loss: TVP (arXiv 2611.00588) in place of DSpark's fixed L1 = 2*TV term, annealed beta 0->1 over
    the first 80% of the run, same 0.9/0.1 TV:CE mix and the same exp(-j/4) position decay.
  * learning rate 1e-4 for a one-epoch fine-tune (DeepSpec's 6e-4 is a from-scratch value).
"""

cfg = dict(
    # --- io (overridden on the command line by run.sh) ---
    model_dir="/data/models/ds41-abl",
    ckpt_config="/data/models/ds41-abl/config.json",
    cache="/data/mtp/cache",
    out="/data/mtp/runs/block9",

    seed=42,
    # 4.14 M unique tokens from the first dump -> ~505 samples of 8192 (runs packed across holes)
    # -> 126 steps/epoch at global batch 4. One epoch is the "did acceptance move at all" arm;
    # scale with --epochs N once the train accept rate moves.
    epochs=1,
    # 4 samples of up to 8192 tokens = 33k tokens per optimizer step. DeepSpec's DSpark recipe uses
    # 32; this first read runs 4 so that a 4.14 M-token dump still gives ~126 optimizer steps
    # (batch 32 would give 16). Scale once the train accept rate moves.
    global_batch_size=4,
    local_batch_size=1,
    lr=1.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    max_grad_norm=1.0,

    # --- objective ---
    tvp_alpha=0.9,             # DSpark: l1_loss_alpha = 0.9
    ce_alpha=0.1,              # DSpark: ce_loss_alpha = 0.1
    confidence_head_alpha=0.1, # DSpark: 1.0; the head is unused on sm_120 (adaptive verification off)
    loss_decay_gamma=4.0,      # DSpark: exp(-pos / 4)
    tvp_anneal_frac=0.8,       # TVP paper: beta 0 -> 1 over the first 80% of training
    anchor_chunk=16,           # 129280-way head in chunks; 16 anchors = 144 rows per chunk (~1.2 GiB less peak)
    log_every=5,
    save_every=0,

    model=dict(
        block_size=9,
        num_anchors=512,
        num_draft_layers=3,
        target_layer_ids=(37, 38, 39),
        noise_token_id=128799,
        markov_rank=256,
        enable_confidence_head=True,
    ),
)
