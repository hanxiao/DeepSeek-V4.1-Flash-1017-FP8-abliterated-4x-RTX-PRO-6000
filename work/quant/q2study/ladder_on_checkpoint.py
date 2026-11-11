#!/usr/bin/env python3
"""Measure the 2-bit codebook / scale ladder on the real checkpoint, on the checkpoint's own grid.

Same measurement as work/quant/re-quant2.py (mean per-block relative RMS against the FP4 weights),
extended to every codebook the b12x E2M1 MMA path can actually represent.

    python3 ladder_on_checkpoint.py --model-dir /data/models/ds41-abl --layer 6 --experts 0
    python3 ladder_on_checkpoint.py --model-dir /data/models/ds41-abl --layer 6 --experts 0 --imatrix /data/imatrix

--imatrix DIR: optional, one <tensor key>.npy per tensor holding the per-input-channel importance
(diag H).  When present the H-weighted error is printed next to the weight RMS and the per-block
scale is also fitted to the H objective.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from q2_ops import (CB_CURRENT, CB_RATIO4, GRID, dequant_fp4, fit_global, lloyd_perblock, q_ls,
                    q_pow2, rel_rms)


def load_tensor(model_dir, key):
    import torch
    from safetensors import safe_open
    index = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    wkey, skey = f"{key}.weight", f"{key}.scale"
    with safe_open(f"{model_dir}/{index[wkey]}", framework="pt", device="cpu") as f:
        w = f.get_tensor(wkey).numpy()
    with safe_open(f"{model_dir}/{index[skey]}", framework="pt", device="cpu") as f:
        st = f.get_tensor(skey)
        if st.dtype in (torch.uint8, torch.int8):
            s = np.exp2(st.numpy().astype(np.float64) - 127.0)
        else:
            s = st.float().numpy().astype(np.float64)
    return w, s


def q_pow2_h(W, s0, c, hd, jrange=range(-8, 9)):
    """per-block pow2 scale chosen to minimise the H-weighted block error."""
    rows, cols = W.shape
    nb = cols // 32
    Wb = W.reshape(rows, nb, 32)
    s0b = s0.reshape(rows, nb, 1)
    wgt = np.broadcast_to(hd.reshape(1, nb, 32), Wb.shape)
    best_q = best = None
    for j in jrange:
        lev = c[None, None, :] * (s0b * 2.0**j)
        a = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
        q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
        sse = (wgt * (Wb - q) ** 2).sum(2)
        if best_q is None:
            best_q, best = q, sse
        else:
            m = sse < best
            best_q = np.where(m[:, :, None], q, best_q)
            best = np.where(m, sse, best)
    return best_q.reshape(W.shape)


def h_err(W, q, hd):
    d = W - q
    return 100 * np.sqrt((d * d * hd).sum() / (W * W * hd).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/data/models/ds41-abl")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--keys", default="w1,w3")
    ap.add_argument("--imatrix", default="")
    ap.add_argument("--max-blocks", type=int, default=4000, help="subsample for the fitted codebooks")
    args = ap.parse_args()

    for k in args.keys.split(","):
        key = f"layers.{args.layer}.ffn.experts.{args.experts}.{k}"
        packed, s = load_tensor(args.model_dir, key)
        W4 = dequant_fp4(packed.astype(np.int32), s.astype(np.float32)).astype(np.float64)
        hd = None
        p = os.path.join(args.imatrix, key + ".npy")
        if args.imatrix and os.path.exists(p):
            hd = np.load(p).astype(np.float64).ravel()
            assert hd.shape[0] == W4.shape[1], (hd.shape, W4.shape)

        nb = W4.shape[1] // 32
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(nb, min(args.max_blocks, nb), replace=False))
        cols = np.concatenate([np.arange(b * 32, b * 32 + 32) for b in idx])
        rows_f = min(64, W4.shape[0])          # the codebook is global, so a row subsample is enough
        Ws, ss = W4[:rows_f, cols], s[:rows_f, idx]

        print(f"\n=== {key}  {W4.shape}  kurtosis {float(((W4 - W4.mean())**4).mean() / W4.var()**2):.2f}"
              f"  sigma/s0 {float((W4.reshape(W4.shape[0], -1, 32).std(2) / s).mean()):.2f}"
              f"  amax/s0 {float((np.abs(W4).reshape(W4.shape[0], -1, 32).max(2) / s).mean()):.2f}", flush=True)

        def add(name, q, note=""):
            e, eg = rel_rms(W4, q)
            line = f"  {e:6.2f} %  (global {eg:5.2f})  {name:46s} {note}"
            if hd is not None:
                line += f"   H-wtd {h_err(W4, q, hd):5.2f} %"
            print(line, flush=True)

        rows_, nb_ = W4.shape[0], W4.shape[1] // 32
        u2s = (np.abs(W4).reshape(rows_, nb_, 32).max(2, keepdims=True) / 2).repeat(32, axis=2).reshape(W4.shape)
        add("u2 uniform amax/2 {-2,-1,0,1}", np.clip(np.round(W4 / u2s), -2, 1) * u2s, "= 40.26 measured")
        add("CURRENT {0.5,1.5} x 2^j s0, j 0..3", q_pow2(W4, s, CB_CURRENT, range(0, 4)), "= 37.94 measured")
        add("CURRENT {0.5,1.5} x 2^j s0, j -8..8", q_pow2(W4, s, CB_CURRENT), "wide j buys nothing")
        add("RATIO4 {1,4} x 2^j s0", q_pow2(W4, s, CB_RATIO4), "<- recommended, free")
        # the 21 sign-symmetric pairs span every ratio the E2M1 grid can express; the full 1365
        # 4-subsets never beat the best symmetric one (checked offline), so this is the cheap sweep
        pairs = [(rel_rms(Ws, q_pow2(Ws, ss, np.array([-q, -p, p, q])))[0], (p, q))
                 for p, q in itertools.combinations([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], 2)]
        pairs.sort()
        print("    ratio ladder: " + "  ".join(f"{{{p},{q}}}:{e:.2f}" for e, (p, q) in pairs[:6]))
        b4 = (pairs[0][0], (-pairs[0][1][1], -pairs[0][1][0], pairs[0][1][0], pairs[0][1][1]))
        add(f"best pair {{+-{b4[1][2]}, +-{b4[1][3]}}} x 2^j s0", q_pow2(W4, s, np.array(b4[1])), "best of the 21 pairs")
        e, c = fit_global(Ws, ss, "pow2", symmetric=True)
        add(f"global sym {np.round(c, 3)} x 2^j s0", q_pow2(W4, s, c), "fit on a block subsample")
        add("RATIO4 {1,4} x LS scale (fp8 mantissa)", q_ls(W4, s, CB_RATIO4), "")
        e, c = fit_global(Ws, ss, "ls", symmetric=True)
        add(f"global sym {np.round(c, 3)} x LS scale", q_ls(W4, s, c), "")
        e, c = fit_global(Ws, ss, "ls", symmetric=False)
        add(f"global asym {np.round(c, 3)} x LS scale", q_ls(W4, s, c), "asymmetry is worth ~0")
        if hd is not None:
            add("RATIO4 x 2^j s0, j fitted to H", q_pow2_h(W4, s, CB_RATIO4, hd), "")
            e, c = fit_global(Ws, ss, "pow2", symmetric=True, hd=hd[cols])
            add(f"global sym {np.round(c, 3)} x 2^j s0, H-fitted", q_pow2_h(W4, s, c, hd), "")
        e, _ = rel_rms(Ws, lloyd_perblock(Ws))
        print(f"  {e:6.2f} %  (subsample)  per-block Lloyd 4-level, free scale        = 31.47 measured, the expensive option")


if __name__ == "__main__":
    raise SystemExit(main())
