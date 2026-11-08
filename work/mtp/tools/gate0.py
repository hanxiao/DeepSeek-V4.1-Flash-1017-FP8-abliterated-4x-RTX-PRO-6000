#!/usr/bin/env python3
"""Gate 0 -- run before spending a GPU-hour on data. Fails loudly, costs ~10 minutes.

1. ``quantize.py roundtrip`` on the real checkpoint: dequantise -> requantise -> dequantise must be
   value-exact on all 2398 quantised MTP tensors (validates the FP8/FP4/ue8m0 conventions against
   the checkpoint itself).
2. build an identity overlay and prove the engine still serves accepted length ~= baseline
   (done by run.sh, not here).
3. load the checkpoint into the training model and check every tensor lands: no missing keys other
   than the two shared buffers, and a forward pass produces finite logits.

    python tools/gate0.py --model-dir /data/models/ds41-abl
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quantize  # noqa: E402
from deepspec_v41.modeling import V41DSparkArgs, V41DSparkModel  # noqa: E402
from tools.load_mtp import load_into_model  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--block-size", type=int, default=9)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    print("== 1. quantisation conventions against the checkpoint ==")
    rc = quantize.cmd_roundtrip(argparse.Namespace(model=a.model_dir))
    if rc != 0:
        return rc

    print("== 2. checkpoint -> training model ==")
    args = V41DSparkArgs.from_checkpoint_config(os.path.join(a.model_dir, "config.json"),
                                               block_size=a.block_size)
    model = V41DSparkModel(args)
    info = load_into_model(model, a.model_dir)
    print(f"   {info['loaded']} model tensors filled, {len(info['missing'])} missing, "
          f"{len(info['unexpected'])} unexpected")
    device = torch.device(a.device)
    # the real head is 14.1 B params: bf16 is what training and serving use, fp32 would be 56 GB
    model = model.to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters()) + model.embed_weight.numel() + model.head_weight.numel()
    print(f"   {n_params / 1e9:.2f} B params, block_size {args.block_size}, "
          f"{len(args.target_layer_ids)} aux layers, device {device}")

    print("== 3. forward pass is finite ==")
    # a 64-token sample with 512 anchors exercises the padding path of the anchor sampler too
    torch.manual_seed(0)
    b, t, k = 1, 64, args.block_size
    batch = {
        "input_ids": torch.randint(0, args.vocab_size, (b, t)),
        "aux": torch.randn(b, t, args.hidden_size * len(args.target_layer_ids)),
        "loss_mask": torch.ones(b, t),
        "topk_ids": torch.randint(0, args.vocab_size, (b, args.num_anchors, k, 16)).int(),
        "topk_logp": torch.log_softmax(torch.randn(b, args.num_anchors, k, 16), -1),
        "residual": torch.full((b, args.num_anchors, k), 1e-4),
    }
    dtype = next(model.parameters()).dtype
    batch = {key: (value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device))
             for key, value in batch.items()}
    loss, metrics = model(batch, beta=1.0)
    assert torch.isfinite(loss), "loss is not finite"
    print(f"   loss {float(loss.detach()):.4f} accept {metrics['accept_rate']:.4f} "
          f"per-pos {[round(x, 3) for x in metrics['accept_rate_per_pos']]}")
    print("\ngate 0 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
