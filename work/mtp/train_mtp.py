#!/usr/bin/env python3
"""Train the DSpark MTP head on a prepared cache.

    torchrun --nproc_per_node 4 train_mtp.py --config config/dspark_v41_block9.py \
        --cache /data/mtp/cache --out /data/mtp/runs/block9
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("mtp_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.cfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--ckpt-config", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--tvp-alpha", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="debug: cap the step count")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    for key, value in (("model_dir", a.model_dir), ("ckpt_config", a.ckpt_config), ("cache", a.cache),
                       ("out", a.out), ("epochs", a.epochs), ("lr", a.lr),
                       ("global_batch_size", a.global_batch_size), ("tvp_alpha", a.tvp_alpha)):
        if value is not None:
            cfg[key] = value
    if a.block_size is not None:
        cfg.setdefault("model", {})["block_size"] = a.block_size
    if a.max_steps is not None:
        cfg["_max_steps"] = a.max_steps

    from deepspec_v41.trainer import MtpTrainer

    trainer = MtpTrainer(cfg)
    if a.max_steps is not None:
        trainer.total_steps = min(trainer.total_steps, a.max_steps)
    trainer.train()
    if trainer.world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
