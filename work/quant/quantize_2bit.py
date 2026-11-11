#!/usr/bin/env python3
"""Re-quantise the checkpoint's routed experts to 2-bit, in the checkpoint's own layout.

The checkpoint stores each FP4 expert as packed int8 (two nibbles per byte) plus a per-32-column
`e8m0` scale. A 2-bit code is a **subset of the same nibbles**, so the 2-bit expert is
the same container, the same loader, with half the distinct codes. That is why the 2-bit
*quality* can be measured on the real model **today**, without a 2-bit kernel.

The 4-bit magnitudes are `e2m1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]`; the 2-bit magnitudes are the
two-level code `{0, 2}` with the block scale multiplied by 4, which makes the 2-bit levels
`{0, 2, 4, 6} x s` — the same dynamic range the 4-bit had, in the 4 magnitudes.

    python3 quantize_2bit.py --src /data/models/ds41-abl --dst /data/models/ds41-abl-2bit
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

# 4-bit e2m1 magnitude code -> 2-bit kernel code (the {0,2,4,6} of the 4x scale).
LUT = np.array([0, 0, 1, 2, 2, 3, 3, 3], dtype=np.uint8)


def is_expert(code: str) -> bool:
    return ".experts." in code and "shared_experts" not in code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/models/ds41-abl")
    ap.add_argument("--dst", default="/data/models/ds41-abl-2bit")
    ap.add_argument("--limit", type=int, default=0, help="only the first N shards (a smoke test)")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index = json.load(open(f"{args.src}/model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values()))
    if args.limit:
        shards = shards[: args.limit]
    os.makedirs(args.dst, exist_ok=True)
    if not os.path.exists(f"{args.dst}/model.safetensors.index.json"):
        for f in os.listdir(args.src):
            if f.endswith((".json", ".py", ".txt", ".model")):
                shutil.copy(f"{args.src}/{f}", f"{args.dst}/{f}")

    total = 0
    for f in shards:
        want = {k for k, v in index["weight_map"].items() if v == f}
        tensors = {}
        with safe_open(f"{args.src}/{f}", framework="pt", device="cpu") as h:
            for k in want:
                t = h.get_tensor(k)
                if is_expert(k) and k.endswith("weight"):
                    u = t.cpu().numpy().astype(np.uint8)
                    lo = u & 0x0F
                    hi = (u >> 4) & 0x0F
                    nl = (lo & 0x8) | LUT[lo & 0x7]
                    nh = (hi & 0x8) | LUT[hi & 0x7]
                    tensors[k] = torch.from_numpy(((nl << 4) | nh).astype(np.uint8))
                    total += 1
                elif is_expert(k) and k.endswith("scale"):
                    # the 2-bit codes are the 4x levels, so the e8m0 exponent goes up by 2
                    s = t.view(torch.uint8).cpu().numpy()          # the e8m0 raw byte
                    s = np.clip(s.astype(np.uint16) + 2, 0, 254).astype(np.uint8)
                    tensors[k] = torch.from_numpy(s)
                else:
                    tensors[k] = t
        save_file(tensors, f"{args.dst}/{f}")
        print(f"  {f}: {len(tensors)} tensors, {total} re-quantised so far", flush=True)
    json.dump({"weight_map": index["weight_map"]}, open(f"{args.dst}/model.safetensors.index.json", "w"))
    print(f"done: {total} weights + 2-bit scales -> {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
