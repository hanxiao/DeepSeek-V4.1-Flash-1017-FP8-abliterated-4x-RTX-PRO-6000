#!/usr/bin/env python3
"""What the 2-bit re-quant costs, measured on one routed expert.

The checkpoint stores each routed expert as FP4 (E2M1) packed two int8 bytes per weight, a
`ue8m0` block scale per 32 columns, and nothing else.  A 2-bit re-quant keeps the same layout and
the same block scales but only the 4 levels (`0.5, 1.5` with signs), so the checkpoint becomes half
the size and the b12x kernel reads 2.1 GB/rank instead of 4.23.

This script measures, on real weights, the *reconstruction* error of the 4-bit versus the 2-bit
quantiser on the same block grid, so the output can be judged before an hour of training:
by 4E's -- "per the block scale" the two levels versions the 4-level, the 4E--.

    python3 re-quant2.py --model-dir /data/models/ds41-abl --layer 6 --experts 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# E2M1 (float4) positive values, one per 3-bit code (exp 0..3, mant 0..1).
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# The 2-bit levels for signed 4-level quantisation: the midpoints of the positive range.
# 4 levels, signed: {-1.5, -0.5, 0.5, 1.5} * scale, so the code is the sign bit3.
_Q2 = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


def dequant_fp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """packed int8 [rows, cols/2] + e8m0 scale [rows, cols/32] -> [rows, cols] float32.

    The checkpoint stores the fp4 as a packed nibble (two per byte) and the block scale as an
    e8m0 power of two; this is the expanded weight the b12x kernel computes before.
    """
    lo = (packed & 0x0F).astype(np.int32)          # columns 0,2,4...
    hi = (packed >> 4).astype(np.int32)             # columns 1,3,5...
    code = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], -1)
    mag = _E2M1[code & 0x7]
    sign = np.where(code & 0x8, -1.0, 1.0)
    vals = mag * sign
    return vals * np.repeat(scale.astype(np.float32), 32, axis=1)


def quant_uniform(vals: np.ndarray, bits: int, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Per-32-block signed uniform quantisation of `bits` bits (n = 2^bits levels)."""
    rows, cols = vals.shape
    flat = vals.reshape(rows, cols // block, block)
    levels = 2 ** bits
    amax = np.abs(flat).max(axis=2, keepdims=True)
    scale = np.where(amax > 0, amax / (levels / 2), 1.0)
    norm = flat / scale
    q = np.clip(np.round(norm), -levels / 2, levels / 2 - 1)
    return (q * scale).reshape(rows, cols), scale.reshape(rows, cols // block)


def quant_lloyd(vals: np.ndarray, levels: int, block: int = 32, iters: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Per-32-block Lloyd quantiser (the optimal 4/8-level codebook for that block).

    The uniform half-width quantiser over-reads the heavy; the tail of the e2m1 distribution
    Lloyd's loftsthe per-block on the optimum, so each block gets the 4/8 levels it.
    """
    rows, cols = vals.shape
    flat = vals.reshape(rows, cols // block, block)
    nb = flat.shape[0] * flat.shape[1]
    flat = flat.reshape(nb, block).astype(np.float32)
    q = np.quantile(flat, (np.arange(levels) + 0.5) / levels, axis=1).T
    for _ in range(iters):
        d = np.abs(flat[:, :, None] - q[:, None, :])
        idx = d.argmin(axis=2)
        for k in range(levels):
            m = idx == k
            cnt = m.sum(axis=1)
            q[:, k] = np.where(cnt > 0, (flat * m).sum(axis=1) / np.maximum(cnt, 1), q[:, k])
    q = np.ascontiguousarray(q)
    d = np.abs(flat[:, :, None] - q[:, None, :])
    out = q[np.arange(nb)[:, None], d.argmin(axis=2)]   # the level value, not the flat value
    return out.reshape(rows, cols), None


def quant2(vals: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Per-32-block signed 4-level quantisation. Returns (dequantised, scale)."""
    rows, cols = vals.shape
    flat = vals.reshape(rows, cols // block, block)
    amax = np.abs(flat).max(axis=2, keepdims=True)
    scale = np.where(amax > 0, amax / 1.5, 1.0)
    norm = flat / scale
    q = np.where(norm > 0, np.where(norm > 1.0, 1.5, np.where(norm > 0.0, 0.5, 0.0)),
                 np.where(norm < -1.0, -1.5, np.where(norm < 0.0, -0.5, 0.0)))
    return (q * scale).reshape(rows, cols), scale.reshape(rows, cols // block)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/data/models/ds41-abl")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--experts", type=int, default=0)
    args = ap.parse_args()

    from safetensors import safe_open

    index = json.load(open(f"{args.model_dir}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for key in ("w1", "w3"):
        name = f"layers.{args.layer}.ffn.experts.{args.experts}.{key}.weight"
        sname = f"layers.{args.layer}.ffn.experts.{args.experts}.{key}.scale"
        # pt: safe_open's numpy loader cannot decode the e8m0 scales
        with safe_open(f"{args.model_dir}/{index[name]}", framework="pt", device="cpu") as f:
            w = f.get_tensor(name).numpy()
        with safe_open(f"{args.model_dir}/{index[sname]}", framework="pt", device="cpu") as f:
            import torch
            s = f.get_tensor(sname).float().numpy()
        out[key] = (w, s)
    for key, (w, s) in out.items():
        v4 = dequant_fp4(w.astype(np.int32), s.astype(np.float32))
        row = [f"{key}: 4-bit fp4"]
        for bits in (2, 3):
            q, _ = quant_uniform(v4, bits)
            err = float(np.sqrt(((v4 - q) ** 2).mean()) / v4.std())
            row.append(f"u{bits} {err*100:.2f} %")
        for bits in (2, 3):
            q, _ = quant_lloyd(v4, 2 ** bits)
            err = float(np.sqrt(((v4 - q) ** 2).mean()) / v4.std())
            row.append(f"L{bits} {err*100:.2f} %")
        print("  |  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
