#!/usr/bin/env python3
"""2-bit codebook / block-scale study for an MXFP4 (E2M1 + E8M0/32) weight source.

Everything here is a *derivation*, not a measurement on the checkpoint: the checkpoint lives on
the VM (/data/models/ds41-abl).  The simulator is calibrated against the two numbers that were
measured on the real weights:

    uniform 2-bit (scale = amax/2, integer grid)          40.26 %  mean block rel-RMS
    {+-0.5, +-1.5} * 2**j * s0, j in {0..3}               37.94 %
    free per-block Lloyd 4-level, free scale              31.47 %

The metric is the same one used there: mean over blocks of sqrt(SSE_b / sum(w_b^2)).

Run:  .venv-q/bin/python q2_core.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
GRID = np.array([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---------------------------------------------------------------- source model
def mx_block_scale(amax: np.ndarray) -> np.ndarray:
    """OCP MX: shared scale = 2**(floor(log2(amax)) - emax_elem), emax_elem = 2 for E2M1."""
    a = np.where(amax > 0, amax, 1.0)
    return 2.0 ** (np.floor(np.log2(a)) - 2.0)


def round_e2m1(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    idx = np.clip(np.searchsorted(E2M1, ax), 1, 7)
    lo, hi = E2M1[idx - 1], E2M1[idx]
    pick = np.where(ax - lo <= hi - ax, idx - 1, idx)
    return np.sign(x) * E2M1[pick]


def fp4_block(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FP4-quantise blocks (nb, B) -> (dequantised weights, s0).  This is the 'FP4 weights' truth."""
    s0 = mx_block_scale(np.abs(W).max(axis=1))
    q = round_e2m1(W / s0[:, None]) * s0[:, None]
    return q, s0


def sample(dist: str, nb: int, B: int, rng: np.random.Generator, sigma_het: float = 0.0) -> np.ndarray:
    n = nb * B
    if dist == "gauss":
        x = rng.standard_normal(n)
    elif dist == "laplace":
        x = rng.laplace(size=n) / np.sqrt(2.0)
    elif dist == "t4":
        x = rng.standard_t(4, size=n) / np.sqrt(2.0)
    elif dist == "t3":
        x = rng.standard_t(3, size=n) / np.sqrt(3.0)
    else:
        raise ValueError(dist)
    x = x.reshape(nb, B)
    if sigma_het > 0:  # extra per-block scale heterogeneity (real tensors are not i.i.d.)
        x = x * np.exp(rng.normal(0.0, sigma_het, size=(nb, 1)))
    return x


# ---------------------------------------------------------------- metrics
def rel_rms(W: np.ndarray, Wq: np.ndarray) -> tuple[float, float]:
    sse = ((W - Wq) ** 2).sum(axis=1)
    ss = (W**2).sum(axis=1)
    return float(np.mean(np.sqrt(sse / ss)) * 100), float(np.sqrt(sse.sum() / ss.sum()) * 100)


# ---------------------------------------------------------------- 2-bit quantisers
def q_uniform2(W: np.ndarray) -> np.ndarray:
    """The measured 40.26 % baseline: scale = amax/2, integer grid {-2,-1,0,1}, clip."""
    amax = np.abs(W).max(axis=1, keepdims=True)
    s = np.where(amax > 0, amax / 2.0, 1.0)
    q = np.clip(np.round(W / s), -2, 1)
    return q * s


def q_pow2(W: np.ndarray, s0: np.ndarray, levels: np.ndarray, jrange=range(0, 4)) -> np.ndarray:
    """levels (4,) sorted; per block s = 2**j * s0, j chosen to minimise that block's SSE."""
    best = None
    best_sse = None
    for j in jrange:
        s = s0 * (2.0**j)
        L = levels[None, :] * s[:, None]                      # (nb, 4)
        d = np.abs(W[:, :, None] - L[:, None, :])
        a = d.argmin(axis=2)
        q = np.take_along_axis(L[:, None, :].repeat(W.shape[1], 1), a[:, :, None], axis=2)[:, :, 0]
        sse = ((W - q) ** 2).sum(axis=1)
        if best is None:
            best, best_sse = q, sse
        else:
            m = sse < best_sse
            best[m] = q[m]
            best_sse = np.where(m, sse, best_sse)
    return best


def q_free_scale(W: np.ndarray, s0: np.ndarray, levels: np.ndarray, scale_mode: str,
                 jrange=range(-6, 7), iters: int = 3) -> np.ndarray:
    """levels (4,) sorted.  scale_mode: 'pow2' (2**j*s0, j searched) or 'free' (LS per block)."""
    L = levels
    if scale_mode == "pow2":
        return q_pow2(W, s0, L, jrange)
    s = s0.copy()
    a = None
    for _ in range(iters):
        d = np.abs(W[:, :, None] - (L[None, :] * s[:, None])[:, None, :])
        a = d.argmin(axis=2)
        c = L[a]                                              # (nb, B)
        num = (W * c).sum(axis=1)
        den = (c * c).sum(axis=1)
        s = np.where(den > 0, num / np.maximum(den, 1e-30), s)
    d = np.abs(W[:, :, None] - (L[None, :] * s[:, None])[:, None, :])
    a = d.argmin(axis=2)
    return L[a] * s[:, None]


def q_lloyd_perblock(W: np.ndarray, k: int = 4, iters: int = 12) -> np.ndarray:
    nb, B = W.shape
    q = np.quantile(W, (np.arange(k) + 0.5) / k, axis=1).T
    for _ in range(iters):
        d = np.abs(W[:, :, None] - q[:, None, :])
        idx = d.argmin(axis=2)
        for m in range(k):
            mask = idx == m
            cnt = mask.sum(axis=1)
            q[:, m] = np.where(cnt > 0, (W * mask).sum(axis=1) / np.maximum(cnt, 1), q[:, m])
    d = np.abs(W[:, :, None] - q[:, None, :])
    return q[np.arange(nb)[:, None], d.argmin(axis=2)]


def fit_global_codebook(W: np.ndarray, s0: np.ndarray, k: int = 4, scale_mode: str = "pow2",
                        jrange=range(-6, 7), symmetric: bool = False,
                        iters: int = 40, restarts: int = 6, seed: int = 0) -> np.ndarray:
    """Pooled k-means over all blocks, with the per-block scale rule inside the loop."""
    rng = np.random.default_rng(seed)
    flat = (W / s0[:, None]).ravel()
    best = None
    for r in range(restarts):
        if symmetric:
            # parametrise as (-b,-a,a,b); init from quantiles of |w|
            aw = np.abs(flat)
            qs = np.quantile(aw, [0.55, 0.92])
            c = np.array([-qs[1], -qs[0], qs[0], qs[1]])
        else:
            c = np.quantile(flat, (np.arange(k) + 0.5) / k)
            if r:
                c = c + rng.normal(0, 0.02 * flat.std(), k)
        for _ in range(iters):
            q = q_free_scale(W, s0, c, scale_mode, jrange, iters=2)
            # recompute assignment + scale, then the closed-form level update
            if scale_mode == "free":
                d = np.abs(W[:, :, None] - (c[None, :] * 1.0)[:, None, :])
                s = np.ones(W.shape[0])
                for _ in range(2):
                    d = np.abs(W[:, :, None] - (c[None, :] * s[:, None])[:, None, :])
                    a = d.argmin(axis=2)
                    cc = c[a]
                    s = (W * cc).sum(axis=1) / np.maximum((cc * cc).sum(axis=1), 1e-30)
                d = np.abs(W[:, :, None] - (c[None, :] * s[:, None])[:, None, :])
                a = d.argmin(axis=2)
                ss = s[:, None] * np.ones_like(W)
            else:
                sse_best = None
                for j in jrange:
                    sj = s0 * 2.0**j
                    d = np.abs(W[:, :, None] - (c[None, :] * sj[:, None])[:, None, :])
                    aj = d.argmin(axis=2)
                    qj = c[aj] * sj[:, None]
                    sse = ((W - qj) ** 2).sum(axis=1)
                    if sse_best is None:
                        sse_best, a, ss = sse, aj, sj[:, None] * np.ones_like(W)
                    else:
                        m = sse < sse_best
                        sse_best = np.where(m, sse, sse_best)
                        a = np.where(m[:, None], aj, a)
                        ss = np.where(m[:, None], sj[:, None] * np.ones_like(W), ss)
            # closed-form level update:  c_m = sum(s*w) / sum(s^2)
            for m in range(k):
                mask = a == m
                num = (ss * W * mask).sum()
                den = (ss * ss * mask).sum()
                if den > 0:
                    c[m] = num / den
            c = np.sort(c)
            if symmetric:
                c = np.array([-c[3], -c[2], c[2], c[3]])
        err, _ = rel_rms(W, q_free_scale(W, s0, c, scale_mode, jrange))
        if best is None or err < best[0]:
            best = (err, c.copy())
    return best[1]


def best_subset_of_grid(W: np.ndarray, s0: np.ndarray, jrange=range(-6, 7)) -> list[tuple]:
    """Every 4-subset of the 15 E2M1 values (the format's reachable codebooks), ranked."""
    out = []
    for sub in itertools.combinations(GRID, 4):
        c = np.array(sub)
        q = q_pow2(W, s0, c, jrange)
        e, _ = rel_rms(W, q)
        out.append((e, sub))
    out.sort(key=lambda t: t[0])
    return out


def main() -> int:
    rng = np.random.default_rng(7)
    NB, B = 20000, 32
    res = {}
    for dist in ("gauss", "laplace", "t4", "t3"):
        for het in (0.0, 0.4):
            W = sample(dist, NB, B, rng, sigma_het=het)
            W4, s0 = fp4_block(W)
            row = {}
            row["u2"] = rel_rms(W4, q_uniform2(W4))[0]
            row["sym_j03"] = rel_rms(W4, q_pow2(W4, s0, np.array([-1.5, -0.5, 0.5, 1.5])))[0]
            row["sym_jwide"] = rel_rms(W4, q_pow2(W4, s0, np.array([-1.5, -0.5, 0.5, 1.5]), range(-8, 9)))[0]
            row["lloyd_pb"] = rel_rms(W4, q_lloyd_perblock(W4))[0]
            res[f"{dist}_het{het}"] = row
            print(dist, het, {k: round(v, 2) for k, v in row.items()}, flush=True)
    json.dump(res, open("calib.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
