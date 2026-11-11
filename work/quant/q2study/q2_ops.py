#!/usr/bin/env python3
"""2-bit quantiser primitives for an MXFP4 (E2M1 + E8M0/32) weight source.

Shared by the study scripts and by ladder_on_checkpoint.py so the numbers in the report and the
numbers measured on the real checkpoint come from the same code.

Conventions
  W   : (rows, cols) float64/float32 absolute weights, already dequantised from FP4
  s0  : (rows, cols/32) the checkpoint's own E8M0 block scale, as a float
  c   : (4,) sorted codebook in units of the per-block scale
"""
from __future__ import annotations

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
GRID = np.array([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# the two codebooks that matter, as E2M1 nibble constants
CB_CURRENT = np.array([-1.5, -0.5, 0.5, 1.5])      # nib = 1 + 2*(b&1) + 8*(b>>1)
CB_RATIO4 = np.array([-4.0, -1.0, 1.0, 4.0])       # nib = 2 + 4*(b&1) + 8*(b>>1)

J_WIDE = range(-8, 9)


def dequant_fp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """packed uint8 [rows, cols/2] + e8m0 scale [rows, cols/32] -> [rows, cols] float32."""
    lo = (packed & 0x0F).astype(np.int32)
    hi = (packed >> 4).astype(np.int32)
    code = np.stack([lo, hi], axis=-1).reshape(packed.shape[0], -1)
    mag = E2M1[code & 0x7]
    sign = np.where(code & 0x8, -1.0, 1.0)
    return mag * sign * np.repeat(scale.astype(np.float32), 32, axis=1)


def rel_rms(W: np.ndarray, Wq: np.ndarray) -> tuple[float, float]:
    sse = ((W - Wq) ** 2).sum(axis=1)
    ss = (W**2).sum(axis=1)
    return float(np.mean(np.sqrt(sse / ss)) * 100), float(np.sqrt(sse.sum() / ss.sum()) * 100)


def _levels(c, s):
    return c[None, None, :] * s.reshape(s.shape[0], s.shape[1], 1)   # (rows, nblk, 4)


def _assign(Wb, lev):
    a = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
    q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
    return a, q


def q_pow2(W, s0, c, jrange=J_WIDE):
    """per-block scale = 2**j * s0, j chosen to minimise that block's SSE."""
    rows, cols = W.shape
    nb = cols // 32
    Wb = W.reshape(rows, nb, 32)
    s0b = s0.reshape(rows, nb, 1)
    best_q = best = None
    for j in jrange:
        _, q = _assign(Wb, _levels(c, s0b * 2.0**j))
        sse = ((Wb - q) ** 2).sum(2)
        if best is None:
            best_q, best = q, sse
        else:
            m = sse < best
            best_q = np.where(m[:, :, None], q, best_q)
            best = np.where(m, sse, best)
    return best_q.reshape(W.shape)


def q_ls(W, s0, c, iters=6):
    """per-block scale fitted by least squares (fp8-with-mantissa equivalent), coarse search first."""
    rows, cols = W.shape
    nb = cols // 32
    Wb = W.reshape(rows, nb, 32)
    s0b = s0.reshape(rows, nb, 1)
    grid = np.linspace(-1.5, 1.5, 25)
    cand = np.stack([((Wb - _assign(Wb, _levels(c, s0b * 2.0**u))[1]) ** 2).sum(2) for u in grid], 2)
    s = s0b * 2.0 ** grid[cand.argmin(2)][:, :, None]
    base = s.copy()
    for _ in range(iters):
        a, q = _assign(Wb, _levels(c, s))
        cc = np.take_along_axis(np.broadcast_to(c[None, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
        num = (Wb * cc).sum(2)[:, :, None]
        den = (cc * cc).sum(2)[:, :, None]
        s = np.where(den > 0, num / np.maximum(den, 1e-30), s)
    if ((Wb - _assign(Wb, _levels(c, s))[1]) ** 2).sum() > ((Wb - _assign(Wb, _levels(c, base))[1]) ** 2).sum():
        s = base
    return _assign(Wb, _levels(c, s))[1].reshape(W.shape)


def fit_global(W, s0, mode="pow2", symmetric=True, iters=25, restarts=3, seed=0, hd=None):
    """Fit one global 4-level codebook (pooled over all blocks) for the pow2 or free scale rule.

    hd: optional per-column importance (diag H).  When given, the scale and the codebook are fitted
    to the H-weighted block error instead of plain MSE.
    """
    rows, cols = W.shape
    nb = cols // 32
    Wb = W.reshape(rows, nb, 32)
    s0b = s0.reshape(rows, nb, 1)
    wgt = np.ones_like(Wb) if hd is None else np.broadcast_to(hd.reshape(1, nb, 32), Wb.shape)
    rng = np.random.default_rng(seed)
    flat = (W / np.repeat(s0.reshape(W.shape[0], -1), 32, axis=1)).ravel()
    inits = [np.quantile(flat, (np.arange(4) + 0.5) / 4)]
    for _ in range(restarts - 1):
        inits.append(np.quantile(flat, (np.arange(4) + rng.random(4)) / 4))
    best = None
    for c in inits:
        c = np.sort(np.asarray(c, dtype=np.float64))
        for _ in range(iters):
            a, ss = _fit_scale(Wb, s0b, c, mode, wgt)
            for m in range(4):
                mask = a == m
                num = (wgt * ss * Wb * mask).sum()
                den = (wgt * ss * ss * mask).sum()
                if den > 0:
                    c[m] = num / den
            c = np.sort(c / np.linalg.norm(c))
            if symmetric:
                c = np.array([-c[3], -c[2], c[2], c[3]])
        a, ss = _fit_scale(Wb, s0b, c, mode, wgt)
        q = _assign(Wb, _levels(c, ss[:, :, :1]))[1].reshape(W.shape)
        e = rel_rms(W, q)[0]
        if best is None or e < best[0]:
            best = (e, c.copy())
    return best


def _fit_scale(Wb, s0b, c, mode, wgt):
    if mode == "pow2":
        cand = []
        for j in J_WIDE:
            s = s0b * 2.0**j
            a, q = _assign(Wb, _levels(c, s))
            cand.append(((wgt * (Wb - q) ** 2).sum(2), a, s))
    else:
        cand = []
        for u in np.linspace(-1.5, 1.5, 25):
            s = s0b * 2.0**u
            a, q = _assign(Wb, _levels(c, s))
            cand.append(((wgt * (Wb - q) ** 2).sum(2), a, s))
    S = np.stack([x[0] for x in cand], 2)
    k = S.argmin(2)
    rows, nb = S.shape[0], S.shape[1]
    a = np.stack([x[1] for x in cand], 3)[np.arange(rows)[:, None], np.arange(nb)[None, :], :, k]
    s = np.stack([x[2] for x in cand], 3)[np.arange(rows)[:, None], np.arange(nb)[None, :], 0, k][:, :, None]
    if mode != "pow2":
        for _ in range(6):
            cc = np.take_along_axis(np.broadcast_to(c[None, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
            num = (wgt * Wb * cc).sum(2)[:, :, None]
            den = (wgt * cc * cc).sum(2)[:, :, None]
            s2 = np.where(den > 0, num / np.maximum(den, 1e-30), s)
            a2, q2 = _assign(Wb, _levels(c, s2))
            if (wgt * (Wb - q2) ** 2).sum() < (wgt * (Wb - _assign(Wb, _levels(c, s))[1]) ** 2).sum():
                a, s = a2, s2
            else:
                break
    return a, s


def lloyd_perblock(W, k=4, iters=15):
    rows, cols = W.shape
    nb = cols // 32
    Wb = W.reshape(rows, nb, 32)
    q = np.moveaxis(np.quantile(Wb, (np.arange(k) + 0.5) / k, axis=2), 0, -1)
    for _ in range(iters):
        a = np.abs(Wb[:, :, :, None] - q[:, :, None, :]).argmin(3)
        for m in range(k):
            mask = a == m
            cnt = mask.sum(2)
            q[:, :, m] = np.where(cnt > 0, (Wb * mask).sum(2) / np.maximum(cnt, 1), q[:, :, m])
    a = np.abs(Wb[:, :, :, None] - q[:, :, None, :]).argmin(3)
    return np.take_along_axis(np.broadcast_to(q[:, :, None, :], Wb.shape + (k,)), a[..., None], 3)[..., 0].reshape(W.shape)
