"""Reader for the flat memmap cache written by ``dump/reassemble.py``.

Nothing is parsed per sample: the cache is a handful of ``.bin`` files with fixed shapes, opened
with ``np.memmap`` so the OS page cache does the work and a sample costs one read of
``lengths[i]`` rows.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


class MtpCache:
    def __init__(self, cache_dir: str):
        self.dir = os.path.abspath(cache_dir)
        self.meta = json.load(open(os.path.join(self.dir, "meta.json")))
        n, t = self.meta["num_samples"], self.meta["max_length"]
        k = self.meta["topk"]
        hidden = self.meta["hidden_size"] * self.meta["num_aux_layers"]
        self.arrays = {
            "tokens": np.memmap(os.path.join(self.dir, "tokens.bin"), dtype=np.int32, mode="r", shape=(n, t)),
            "loss_mask": np.memmap(os.path.join(self.dir, "loss_mask.bin"), dtype=np.uint8, mode="r", shape=(n, t)),
            "positions": np.memmap(os.path.join(self.dir, "positions.bin"), dtype=np.int32, mode="r", shape=(n, t)),
            "lengths": np.memmap(os.path.join(self.dir, "lengths.bin"), dtype=np.int32, mode="r", shape=(n,)),
            "aux": np.memmap(os.path.join(self.dir, "aux.bin"), dtype=np.uint16, mode="r", shape=(n, t, hidden)),
            "topk_ids": np.memmap(os.path.join(self.dir, "topk_ids.bin"), dtype=np.int32, mode="r", shape=(n, t, k)),
            "topk_logp": np.memmap(os.path.join(self.dir, "topk_logp.bin"), dtype=np.float32, mode="r", shape=(n, t, k)),
            "residual": np.memmap(os.path.join(self.dir, "residual.bin"), dtype=np.float32, mode="r", shape=(n, t)),
        }

    def __len__(self) -> int:
        return self.meta["num_samples"]

    def __getitem__(self, i: int) -> dict:
        n = int(self.arrays["lengths"][i])
        aux = torch.from_numpy(np.ascontiguousarray(self.arrays["aux"][i, :n])).view(torch.bfloat16)
        return {
            "input_ids": torch.from_numpy(np.ascontiguousarray(self.arrays["tokens"][i, :n])).long(),
            "positions": torch.from_numpy(np.ascontiguousarray(self.arrays["positions"][i, :n])).long(),
            "loss_mask": torch.from_numpy(np.ascontiguousarray(self.arrays["loss_mask"][i, :n])).float(),
            "aux": aux,
            "topk_ids": torch.from_numpy(np.ascontiguousarray(self.arrays["topk_ids"][i, :n])).long(),
            "topk_logp": torch.from_numpy(np.ascontiguousarray(self.arrays["topk_logp"][i, :n])),
            "residual": torch.from_numpy(np.ascontiguousarray(self.arrays["residual"][i, :n])),
        }


def collate(samples: list[dict]) -> dict:
    """Right-pad to the longest sample in the micro-batch."""
    T = max(s["input_ids"].shape[0] for s in samples)
    out = {}
    for key, pad in (("input_ids", 0), ("positions", 0), ("loss_mask", 0.0), ("aux", 0.0),
                     ("topk_ids", 0), ("topk_logp", -1e4), ("residual", 0.0)):
        rows = []
        for s in samples:
            x = s[key]
            if x.shape[0] < T:
                shape = (T - x.shape[0],) + x.shape[1:]
                x = torch.cat([x, x.new_full(shape, pad)])
            rows.append(x)
        out[key] = torch.stack(rows)
    return out


__all__ = ["MtpCache", "collate"]
