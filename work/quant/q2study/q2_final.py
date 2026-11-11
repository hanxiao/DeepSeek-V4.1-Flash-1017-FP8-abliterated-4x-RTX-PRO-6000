#!/usr/bin/env python3
"""Final numbers: symmetric-pair ladder, robust global-codebook fit with a free scale,
and a distribution-robustness check."""
import itertools, json
import numpy as np
from q2_core import E2M1, GRID, fp4_block, q_lloyd_perblock, rel_rms, sample
from q2_layer import quant as robust_quant   # has the grid+LS scale

J = range(-8, 9)
MAGS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def q_pow2(W, s0, c, jr=J):
    best = bq = None
    for j in jr:
        L = c[None, :] * (s0 * 2.0**j)[:, None]
        a = np.abs(W[:, :, None] - L[:, None, :]).argmin(2)
        q = np.take_along_axis(L[:, None, :].repeat(W.shape[1], 1), a[:, :, None], 2)[:, :, 0]
        sse = ((W - q)**2).sum(1)
        if best is None: best, bq = sse, q
        else:
            m = sse < best; bq[m] = q[m]; best = np.where(m, sse, best)
    return bq

def fit_global_robust(W, s0, mode, iters=30, seed=0, symmetric=True):
    """global codebook, per-block scale by (grid+LS) or pow2, robust to the LS local optimum."""
    out, inn = W.shape
    nb = inn // 32
    Wb = W.reshape(out, nb, 32)
    s0b = s0.reshape(out, nb, 1)
    def scale_and_assign(c):
        if mode == "pow2":
            cand = []
            for j in J:
                s = s0b * 2.0**j
                lev = c[None, None, :] * s
                a = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
                q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape+(4,)), a[..., None], 3)[..., 0]
                cand.append((((Wb - q)**2).sum(2), a, np.broadcast_to(s, Wb.shape)))
            S = np.stack([x[0] for x in cand], 2)
            k = S.argmin(2)
            a = np.stack([cand[j][1] for j in range(len(cand))], 3)[np.arange(out)[:, None], np.arange(nb)[None, :], :, k]
            ss = np.stack([cand[j][2] for j in range(len(cand))], 3)[np.arange(out)[:, None], np.arange(nb)[None, :], :, k]
            return a, ss
        grid = np.linspace(-1.5, 1.5, 25)
        cand = []
        for u in grid:
            s = s0b * 2.0**u
            lev = c[None, None, :] * s
            a = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
            q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape+(4,)), a[..., None], 3)[..., 0]
            cand.append((((Wb - q)**2).sum(2), a, np.broadcast_to(s, Wb.shape)))
        S = np.stack([x[0] for x in cand], 2); k = S.argmin(2)
        a = np.stack([cand[j][1] for j in range(len(cand))], 3)[np.arange(out)[:, None], np.arange(nb)[None, :], :, k]
        ss = np.stack([cand[j][2] for j in range(len(cand))], 3)[np.arange(out)[:, None], np.arange(nb)[None, :], :, k]
        for _ in range(5):
            cc = np.take_along_axis(np.broadcast_to(c[None, None, :], Wb.shape+(4,)), a[..., None], 3)[..., 0]
            num = (Wb*cc).sum(2)[:, :, None]; den = (cc*cc).sum(2)[:, :, None]
            ss = np.where(den > 0, num/np.maximum(den, 1e-30), ss)
            lev = c[None, None, :] * ss[:, :, :1]
            a2 = np.abs(Wb[:, :, :, None] - lev[:, :, None, :]).argmin(3)
            e1 = ((Wb - np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape+(4,)), a[..., None], 3)[..., 0])**2).sum()
            e2 = ((Wb - np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape+(4,)), a2[..., None], 3)[..., 0])**2).sum()
            if e2 < e1: a = a2
            else: break
        return a, ss
    rng = np.random.default_rng(seed)
    flat = (W / s0[:, None]).ravel()
    inits = [np.quantile(flat, (np.arange(4)+0.5)/4)]
    for _ in range(4):
        inits.append(np.quantile(flat, (np.arange(4)+rng.random(4))/4))
    best = None
    for c in inits:
        c = np.sort(np.asarray(c, float))
        for _ in range(iters):
            a, ss = scale_and_assign(c)
            for m in range(4):
                mask = (a == m)
                num = (ss*Wb*mask).sum(); den = (ss*ss*mask).sum()
                if den > 0: c[m] = num/den
            c = c/np.linalg.norm(c); c = np.sort(c)
            if symmetric: c = np.array([-c[3], -c[2], c[2], c[3]])
        a, ss = scale_and_assign(c)
        lev = c[None, None, :]*ss[:, :, :1]
        q = np.take_along_axis(np.broadcast_to(lev[:, :, None, :], Wb.shape+(4,)), a[..., None], 3)[..., 0]
        e = rel_rms(W, q.reshape(W.shape))[0]
        if best is None or e < best[0]: best = (e, c.copy())
    return best

def main():
    rng = np.random.default_rng(7)
    for dist in ("gauss", "laplace"):
        W = sample(dist, 6000, 32, rng)
        W4, s0 = fp4_block(W)
        print("="*92); print(f"distribution: {dist} (calibrated: gauss reproduces the measured anchors)")
        print("="*92)
        print("  sign-symmetric 2-magnitude codebooks, pow2 per-block scale (the reachable set):")
        rows = []
        for p, q in itertools.combinations(MAGS, 2):
            c = np.array([-q, -p, p, q])
            e = rel_rms(W4, q_pow2(W4, s0, c))[0]
            rows.append((e, p, q, q/p))
        for e, p, q, r in sorted(rows):
            print(f"    {{+-{p}, +-{q}}}  ratio {r:5.2f}  -> {e:5.2f} %")
        best4 = min((rel_rms(W4, q_pow2(W4, s0, np.array(s)))[0], s) for s in itertools.combinations(GRID, 4))
        print(f"  best 4-of-15 (asymmetric allowed): {best4[1]} -> {best4[0]:.2f} %")
        for mode in ("pow2", "ls"):
            e, c = fit_global_robust(W4, s0, mode, symmetric=True)
            print(f"  global free real codebook, {mode:4s} scale: {np.round(c,4)} ratio {c[3]/c[2]:.2f} -> {e:.2f} %")
            e2, c2 = fit_global_robust(W4, s0, mode, symmetric=False)
            print(f"  global free asymmetric  , {mode:4s} scale: {np.round(c2,4)} -> {e2:.2f} %")
        print(f"  per-block codebook + free scale (Lloyd): {rel_rms(W4, q_lloyd_perblock(W4))[0]:.2f} %")
        print()
main()
