"""Training loop for the V4.1 DSpark head.

Deliberately small: FSDP over the three draft blocks, AdamW with fp32 optimizer state, the TVP
schedule from ``loss.beta_at_step``, and the DSpark/TVP objective computed inside the model's
``forward`` (FSDP only materialises parameters for the module's own forward).

The frozen embedding and LM head are registered as *buffers* of the model, so FSDP keeps them
replicated and every rank can use them without an all-gather: they are shared with the target
checkpoint and never trained.
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from .data import MtpCache, collate
from .loss import beta_at_step
from .modeling import DSparkBlock, V41DSparkArgs, V41DSparkModel


def _dist_info():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def _cosine(step: int, total: int, warmup: float, base_lr: float, min_ratio: float = 0.1) -> float:
    w = max(1, int(warmup * total))
    if step < w:
        return base_lr * (step + 1) / w
    prog = (step - w) / max(1, total - w)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class MtpTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rank, self.world, self.local = _dist_info()
        self.device = torch.device("cuda", self.local)
        self.is_main = self.rank == 0
        torch.manual_seed(int(cfg.get("seed", 42)))

        self.args = V41DSparkArgs.from_checkpoint_config(cfg["ckpt_config"], **cfg.get("model", {}))
        self.model = V41DSparkModel(self.args)
        self._load_checkpoint(cfg["model_dir"])
        self.model = self.model.to(self.device)

        self.fsdp = FSDP(
            self.model,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                           buffer_dtype=torch.bfloat16),
            auto_wrap_policy=ModuleWrapPolicy({DSparkBlock}),
            device_id=self.device,
        )

        trainable = [p for p in self.fsdp.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        self.opt = torch.optim.AdamW(trainable, lr=float(cfg["lr"]), betas=(0.9, 0.95),
                                     eps=1e-8, weight_decay=float(cfg.get("weight_decay", 0.0)))

        self.cache = MtpCache(cfg["cache"])
        self.steps_per_epoch = max(1, len(self.cache) // int(cfg["global_batch_size"]))
        self.total_steps = int(cfg.get("epochs", 1)) * self.steps_per_epoch
        self.grad_accum = int(cfg["global_batch_size"]) // (self.world * int(cfg["local_batch_size"]))
        assert self.grad_accum >= 1, "global_batch_size must be >= world * local_batch_size"
        if self.is_main:
            print(f"[mtp-train] {len(self.cache)} samples, {n_train/1e9:.2f} B trainable params, "
                  f"{self.steps_per_epoch} steps/epoch, {self.total_steps} steps, "
                  f"grad_accum {self.grad_accum}", flush=True)

    def _load_checkpoint(self, model_dir: str):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tools.load_mtp import load_into_model

        load_into_model(self.model, model_dir, verbose=self.rank == 0)

    def _batch(self, index: int) -> dict:
        samples = [self.cache[i] for i in index]
        batch = collate(samples)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train(self):
        cfg = self.cfg
        gbs = int(cfg["global_batch_size"])
        lbs = int(cfg["local_batch_size"])
        order = torch.randperm(len(self.cache), generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))))
        cursor = 0
        t0 = time.time()
        for step in range(self.total_steps):
            lr = _cosine(step, self.total_steps, float(cfg.get("warmup_ratio", 0.04)), float(cfg["lr"]))
            for g in self.opt.param_groups:
                g["lr"] = lr
            beta = beta_at_step(step, self.total_steps, float(cfg.get("tvp_anneal_frac", 0.8)))
            self.opt.zero_grad(set_to_none=True)
            metrics = None
            for micro in range(self.grad_accum):
                idx = []
                for _ in range(lbs):
                    idx.append(int(order[cursor % len(order)]))
                    cursor += 1
                batch = self._batch(idx)
                loss, metrics = self.fsdp(
                    batch,
                    beta=beta,
                    ce_alpha=float(cfg.get("ce_alpha", 0.1)),
                    tvp_alpha=float(cfg.get("tvp_alpha", 0.9)),
                    confidence_head_alpha=float(cfg.get("confidence_head_alpha", 0.1)),
                    loss_decay_gamma=float(cfg.get("loss_decay_gamma", 4.0)),
                    anchor_chunk=int(cfg.get("anchor_chunk", 64)),
                    world_size=self.world,
                )
                (loss / self.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.fsdp.parameters() if p.requires_grad],
                                           float(cfg.get("max_grad_norm", 1.0)))
            self.opt.step()
            if self.is_main and (step % int(cfg.get("log_every", 5)) == 0 or step == self.total_steps - 1):
                per_pos = " ".join(f"{x:.3f}" for x in metrics["accept_rate_per_pos"])
                print(f"step {step}/{self.total_steps} lr {lr:.2e} beta {beta:.2f} "
                      f"loss {metrics['loss']:.4f} ce {metrics['ce_loss']:.4f} "
                      f"tvp {metrics['tvp_loss']:.4f} acc {metrics['accept_rate']:.4f} "
                      f"pos [{per_pos}] {(time.time()-t0)/(step+1):.2f}s/step", flush=True)
            if int(cfg.get("save_every", 0)) and step and step % int(cfg["save_every"]) == 0:
                self.save(step)
        self.save(self.total_steps)

    def save(self, step: int):
        """Rank 0 writes the bf16 state dict; ``tools/export_mtp.py`` turns it into the checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        out = os.path.join(self.cfg["out"], f"step_{step:06d}")
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp, StateDictType.FULL_STATE_DICT, cfg):
            state = self.fsdp.state_dict()
        if self.is_main:
            os.makedirs(out, exist_ok=True)
            from safetensors.torch import save_file
            clean = {k: v.contiguous().cpu() for k, v in state.items()
                     if not k.startswith(("embed_weight", "head_weight"))}
            save_file(clean, os.path.join(out, "model.safetensors"))
            json.dump({"step": step, "args": self.args.__dict__},
                      open(os.path.join(out, "train_meta.json"), "w"), indent=2)
            print(f"[mtp-train] saved {out} ({len(clean)} tensors)", flush=True)
        if self.world > 1:
            dist.barrier()


__all__ = ["MtpTrainer"]
