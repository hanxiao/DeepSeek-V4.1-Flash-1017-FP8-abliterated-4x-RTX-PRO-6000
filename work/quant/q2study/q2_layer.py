#!/usr/bin/env python3
"""Layer-level experiment: what does the 2-bit codebook buy once the objective is
activation-weighted (diag H) and once error feedback (GPTQ) is allowed?

Note the identity this experiment relies on: for a linear layer, ||X dW^T||_F^2 = n * tr(dW H dW^T)
with H = X^T X / n, so the "imatrix objective" *is* the output error of the calibration set, not an
approximation of it.  The two columns below are therefore algebraically identical and the second is
printed only as a check.

The source is FP4(MX, block 32) and the 2-bit error is measured against those FP4 weights.

Run: .venv-q/bin/python q2_layer.py
"""
from __future__ import annotations

import numpy as np

from q2_core import fp4_block, mx_block_scale, rel_rms

OUT, IN, NS, BLK = 256, 1024, 96, 32


def fp4_matrix(W, B=BLK):
    out, inn = W.shape
    nb = inn // B
    q, s = fp4_block(W.reshape(out * nb, B))
    return q.reshape(out, inn), s.reshape(out, nb)


def mx_scale_of(W, B=BLK):
    out, inn = W.shape
    nb = inn // B
    return mx_block_scale(np.abs(W).reshape(out * nb, B).max(1)).reshape(out, nb)


def make_problem(seed=0, act_sigma=0.6, rho=0.35, spikes=8, spike_gain=20.0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((OUT, IN))
    W4, s0 = fp4_matrix(W)
    gam = np.exp(rng.normal(0, act_sigma, IN))
    if spikes:
        gam[rng.choice(IN, spikes, replace=False)] *= spike_gain
    R = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (IN, IN)) + 1e-6 * np.eye(IN)
    X = (rng.standard_normal((NS, IN)) @ np.linalg.cholesky(R).T) * gam[None, :]
    H = X.T @ X / NS
    return W4, s0, X, H


# ------------------------------------------------------------------ quantisers
def _blocks(W):
    return W.reshape(W.shape[0], -1, BLK)


def quant(W, s0, c, hd=None, mode="pow2", codebook_per_block=False, iters=12):
    """mode: 'pow2' (j searched), 'ls' (continuous scale, grid+LS), 'none' (s = s0).
    hd: per-column importance (H diagonal) or None for plain MSE."""
    out, inn = W.shape
    Wb = _blocks(W)
    _, nb, _ = Wb.shape
    wgt = np.ones_like(Wb) if hd is None else np.broadcast_to(hd.reshape(1, nb, BLK), Wb.shape)

    if codebook_per_block:
        q = np.moveaxis(np.quantile(Wb, (np.arange(4) + 0.5) / 4, axis=2), 0, -1)
        for _ in range(iters):
            a = np.abs(Wb[:, :, :, None] - q[:, :, None, :]).argmin(3)
            for m in range(4):
                mask = (a == m) * wgt
                num, den = (mask * Wb).sum(2), mask.sum(2)
                q[:, :, m] = np.where(den > 0, num / np.maximum(den, 1e-30), q[:, :, m])
        a = np.abs(Wb[:, :, :, None] - q[:, :, None, :]).argmin(3)
        return np.take_along_axis(np.broadcast_to(q[:, :, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0].reshape(W.shape)

    s0b = s0.reshape(out, nb, 1)

    def sse_of(s):                                    # s: (out, nblk, 1)
        lev = c[None, None, :] * s                     # (out, nblk, 4)
        a = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
        q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
        return (wgt * (Wb - q) ** 2).sum(2), q, a

    if mode == "none":
        best = s0b
    elif mode == "pow2":
        cand = np.stack([sse_of(s0b * 2.0**j)[0] for j in range(-8, 9)], 2)   # (out,nblk,17)
        best = s0b * 2.0 ** (cand.argmin(2)[:, :, None] - 8)
    elif mode == "ls":
        # coarse search over +-1.5 octaves, then closed-form least squares, keep the best
        grid = np.linspace(-1.5, 1.5, 25)
        cand = np.stack([sse_of(s0b * 2.0**u)[0] for u in grid], 2)
        best = s0b * 2.0 ** grid[cand.argmin(2)][:, :, None]
        for _ in range(4):
            _, _, a = sse_of(best)
            lev = c[None, None, :] * np.ones_like(best)
            cc = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
            num = (wgt * Wb * cc).sum(2)[:, :, None]
            den = (wgt * cc * cc).sum(2)[:, :, None]
            best = np.where(den > 0, num / np.maximum(den, 1e-30), best)
        base = s0b * 2.0 ** grid[cand.argmin(2)][:, :, None]
        if sse_of(best)[0].sum() > sse_of(base)[0].sum():
            best = base
    else:
        raise ValueError(mode)
    _, q, _ = sse_of(best)
    return q.reshape(W.shape)


def gptq(W, c, X, act_order=False, damp=0.01, per_block_codebook=False):
    out, inn = W.shape
    H = X.T @ X / X.shape[0]
    perm = np.argsort(-np.diag(H)) if act_order else np.arange(inn)
    Wp = W[:, perm].copy()
    Hp = H[np.ix_(perm, perm)] + damp * np.diag(H).mean() * np.eye(inn)
    U = np.linalg.cholesky(np.linalg.inv(Hp)).T
    Wq = np.zeros_like(Wp)
    nb = inn // BLK
    s0b = mx_scale_of(Wp)
    hd = np.diag(H)[perm]
    for bi in range(nb):
        cols = np.arange(bi * BLK, (bi + 1) * BLK)
        Wblk = Wp[:, cols]
        wj = hd[cols][None, :]
        sse = np.empty((out, 17))
        for k, j in enumerate(range(-8, 9)):
            lev = c[None, :] * (s0b[:, bi] * 2.0**j)[:, None]
            a_ = np.abs(Wblk[:, :, None] - lev[:, None, :]).argmin(2)
            q = np.take_along_axis(np.broadcast_to(lev[:, None, :], (out, BLK, 4)), a_[..., None], 2)[..., 0]
            sse[:, k] = (wj * (Wblk - q) ** 2).sum(1)
        sj = s0b[:, bi] * 2.0 ** (sse.argmin(1) - 8)
        levb = None
        if per_block_codebook:
            levb = np.quantile(Wblk, (np.arange(4) + 0.5) / 4, axis=1).T      # (out, 4)
            for _ in range(6):
                aa = np.abs(Wblk[:, :, None] - levb[:, None, :]).argmin(2)
                for m in range(4):
                    mask = (aa == m) * wj
                    num, den = (mask * Wblk).sum(1), mask.sum(1)
                    levb[:, m] = np.where(den > 0, num / np.maximum(den, 1e-30), levb[:, m])
        for k, jc in enumerate(cols):
            w = Wp[:, jc]
            if per_block_codebook:
                a_ = np.abs(w[:, None] - levb).argmin(1)
                q = np.take_along_axis(levb, a_[:, None], 1)[:, 0]
            else:
                a_ = np.abs(w[:, None] - c[None, :] * sj[:, None]).argmin(1)
                q = c[a_] * sj
            Wq[:, jc] = q
            err = (w - q) / U[jc, jc]
            if jc + 1 < inn:
                Wp[:, jc + 1:] -= err[:, None] * U[jc, jc + 1:][None, :]
    if act_order:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(inn)
        Wq = Wq[:, inv]
    return Wq


def fit_codebook_h(W, s0, hd, weighted=True, symmetric=False, iters=40, seed=0):
    """global 4-level codebook for the (optionally H-weighted) objective, pow2 per-block scale."""
    out, inn = W.shape
    Wb = _blocks(W)
    _, nb, _ = Wb.shape
    wgt = np.ones_like(Wb) if not weighted else np.broadcast_to(hd.reshape(1, nb, BLK), Wb.shape)
    flat = (W / s0.repeat(BLK, axis=1)).ravel()
    c = np.quantile(flat, (np.arange(4) + 0.5) / 4)
    for _ in range(iters):
        sse = None
        for j in range(-8, 9):
            s = s0.reshape(out, nb, 1) * 2.0**j
            lev = c[None, None, :] * s
            a = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
            q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape + (4,)), a[..., None], 3)[..., 0]
            e = (wgt * (Wb - q) ** 2).sum(2)
            if sse is None:
                sse, aa, ss = e, a, np.broadcast_to(s, Wb.shape)
            else:
                m = e < sse
                sse = np.where(m, e, sse)
                aa = np.where(m[:, :, None], a, aa)
                ss = np.where(m[:, :, None], np.broadcast_to(s, Wb.shape), ss)
        for m in range(4):
            mask = (aa == m) * wgt
            num, den = (mask * ss * Wb).sum(), (mask * ss * ss).sum()
            if den > 0:
                c[m] = num / den
        c = np.sort(c / np.linalg.norm(c))
        if symmetric:
            c = np.array([-c[3], -c[2], c[2], c[3]])
    return c


def main():
    for tag, kw in (("benign H: act sigma 0.6, 8 spikes x20", dict(act_sigma=0.6, spikes=8)),
                    ("hostile H: act sigma 1.2, 20 spikes x40", dict(act_sigma=1.2, spikes=20))):
        W4, s0, X, H = make_problem(**kw)
        hd = np.diag(H)
        print("=" * 100)
        print(f"{tag}   out={OUT} in={IN} block={BLK}   H p99/p50={np.percentile(hd,99)/np.median(hd):.1f} "
              f"max/med={hd.max()/np.median(hd):.0f}")
        print("=" * 100)
        c_cur = np.array([-1.5, -0.5, 0.5, 1.5])
        c_r4 = np.array([-4.0, -1.0, 1.0, 4.0])
        c_h = fit_codebook_h(W4, s0, hd, weighted=True)
        c_p = fit_codebook_h(W4, s0, hd, weighted=False)
        print(f"  codebook fitted to plain MSE : {np.round(c_p,3)}")
        print(f"  codebook fitted to H-weighted: {np.round(c_h,3)}")
        rows = []

        def row(name, q):
            rows.append((name, rel_rms(W4, q)[1], 100 * np.sqrt(((W4 - q) @ H * (W4 - q)).sum() / (W4 @ H * W4).sum())))

        row("A  {0.5,1.5}  pow2 scale, plain", quant(W4, s0, c_cur))
        row("B  {0.5,1.5}  pow2 scale, H-fitted", quant(W4, s0, c_cur, hd))
        row("C  {1,4}      pow2 scale, plain", quant(W4, s0, c_r4))
        row("D  {1,4}      pow2 scale, H-fitted", quant(W4, s0, c_r4, hd))
        row("E  {1,4}      LS scale,  plain", quant(W4, s0, c_r4, None, mode="ls"))
        row("F  {1,4}      LS scale,  H-fitted", quant(W4, s0, c_r4, hd, mode="ls"))
        row("G  cb(H)      pow2 scale, H-fitted", quant(W4, s0, c_h, hd))
        row("H  cb(plain)  pow2 scale, H-fitted", quant(W4, s0, c_p, hd))
        row("I  per-block cb, H-fitted", quant(W4, s0, None, hd, codebook_per_block=True))
        row("O  {0.5,1.5}  LS scale,  H-fitted", quant(W4, s0, c_cur, hd, mode="ls"))
        row("P  cb(H)      LS scale,  H-fitted", quant(W4, s0, c_h, hd, mode="ls"))
        row("Q  cb(plain)  LS scale,  H-fitted", quant(W4, s0, c_p, hd, mode="ls"))
        row("J  GPTQ {0.5,1.5}", gptq(W4, c_cur, X))
        row("J2 GPTQ {0.5,1.5} act-order", gptq(W4, c_cur, X, act_order=True))
        row("K  GPTQ {1,4}", gptq(W4, c_r4, X))
        row("L  GPTQ cb(H)", gptq(W4, c_h, X))
        row("M  GPTQ {1,4} act-order", gptq(W4, c_r4, X, act_order=True))
        row("N  GPTQ per-block cb", gptq(W4, c_r4, X, per_block_codebook=True))
        print(f"  {'scheme':40s} {'weight rel-RMS':>14} {'H-weighted / output':>20}")
        for n, a, b in rows:
            print(f"  {n:40s} {a:13.2f}% {b:19.2f}%")
        print()


if __name__ == "__main__":
    main()
