#!/usr/bin/env python3
"""Derivations for 2-bit codebook + block-scale choice on an MXFP4 weight source.

Sections
  A  optimal 4-level quantisers for continuous Gaussian / Laplacian  (validated against Max 1960)
  B  the ladder of 2-bit schemes on the E2M1 grid, block 32
  C  block size 16 / 32 / 64
  D  discrete-support check: is the optimum on the E2M1 grid or off it

All numbers are this script's own derivation.  Calibration against the two measured anchors is
printed at the top of section B.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from q2_core import (E2M1, GRID, fp4_block, q_lloyd_perblock, q_pow2, q_uniform2, rel_rms, sample)

NB, B = 12000, 32
J_ALL = range(-8, 9)


# ------------------------------------------------------------------ A: continuous optima
def lloyd_pdf(pdf, k=4, lo=-14.0, hi=14.0, n=400001, iters=300):
    x = np.linspace(lo, hi, n)
    w = pdf(x)
    w = w / w.sum()
    c = np.quantile(np.repeat(x, 1), 1) * np.ones(k)
    # init at quantiles of the discrete measure
    cdf = np.cumsum(w)
    c = np.array([x[np.searchsorted(cdf, (i + 0.5) / k)] for i in range(k)])
    for _ in range(iters):
        # boundaries = midpoints (valid for MSE)
        b = 0.5 * (c[1:] + c[:-1])
        idx = np.searchsorted(b, x)
        newc = np.array([(w[idx == i] * x[idx == i]).sum() / max(w[idx == i].sum(), 1e-300) for i in range(k)])
        if np.allclose(newc, c, atol=1e-12):
            c = newc
            break
        c = newc
    b = 0.5 * (c[1:] + c[:-1])
    idx = np.searchsorted(b, x)
    mse = sum((w[idx == i] * (x[idx == i] - c[i]) ** 2).sum() for i in range(k))
    return c, mse


def section_a():
    print("=" * 100)
    print("A. optimal 4-level (2-bit) quantisers, continuous distributions, unit variance")
    print("=" * 100)
    out = {}
    g = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    lap = lambda x: np.exp(-np.abs(x) * np.sqrt(2)) * np.sqrt(2) / 2
    for name, pdf in (("gaussian", g), ("laplacian", lap)):
        c, mse = lloyd_pdf(pdf)
        # sign-symmetric constrained optimum: levels (-b,-a,a,b), boundaries at 0 and +-(a+b)/2
        x = np.linspace(-14, 14, 400001)
        w = pdf(x)
        w = w / w.sum()
        a, bb = 0.4, 1.4
        for _ in range(500):
            bnd = np.array([-0.5 * (a + bb), 0.0, 0.5 * (a + bb)])
            idx = np.searchsorted(bnd, x)
            m = [(idx == 0), (idx == 1), (idx == 2), (idx == 3)]
            na = ((w[m[1]] * x[m[1]]).sum() + (w[m[2]] * x[m[2]]).sum()) / (w[m[1]].sum() + w[m[2]].sum())
            nb = ((w[m[0]] * x[m[0]]).sum() + (w[m[3]] * x[m[3]]).sum()) / (w[m[0]].sum() + w[m[3]].sum())
            if abs(na - a) < 1e-13 and abs(nb - bb) < 1e-13:
                a, bb = na, nb
                break
            a, bb = na, nb
        bnd = np.array([-0.5 * (a + bb), 0.0, 0.5 * (a + bb)])
        idx = np.searchsorted(bnd, x)
        lv = np.array([-bb, -a, a, bb])
        mse_sym = sum((w[idx == i] * (x[idx == i] - lv[i]) ** 2).sum() for i in range(4))
        print(f"  {name:10s} free optimum levels {np.round(c,4)}  MSE {mse:.5f}  rel-RMS {100*np.sqrt(mse):.2f} %")
        print(f"  {name:10s} sign-symmetric  {np.round(lv,4)}  MSE {mse_sym:.5f}  rel-RMS {100*np.sqrt(mse_sym):.2f} %"
              f"   ratio b/a = {bb/a:.3f}")
        out[name] = dict(free=c.tolist(), mse_free=mse, sym=lv.tolist(), mse_sym=mse_sym, ratio=bb / a)
    print("  reference: Max (1960) tabulates MSE 0.1175 for the 4-level Gaussian quantiser,")
    print("             i.e. 34.28 % RMS; our Lloyd run is the check on that number.")
    return out


# ------------------------------------------------------------------ generic scheme eval
def assign_pow2(W, s0, c, jrange=J_ALL):
    best_q = best_sse = None
    for j in jrange:
        sj = s0 * (2.0**j)
        L = c[None, :] * sj[:, None]
        a = np.abs(W[:, :, None] - L[:, None, :]).argmin(2)
        q = np.take_along_axis(L[:, None, :].repeat(W.shape[1], 1), a[:, :, None], 2)[:, :, 0]
        sse = ((W - q) ** 2).sum(1)
        if best_q is None:
            best_q, best_sse = q, sse
        else:
            m = sse < best_sse
            best_q[m] = q[m]
            best_sse = np.where(m, sse, best_sse)
    return best_q


def assign_free(W, s0, c, iters=6):
    s = s0.copy()
    for _ in range(iters):
        a = np.abs(W[:, :, None] - (c[None, :] * s[:, None])[:, None, :]).argmin(2)
        cc = c[a]
        s = (W * cc).sum(1) / np.maximum((cc * cc).sum(1), 1e-30)
    a = np.abs(W[:, :, None] - (c[None, :] * s[:, None])[:, None, :]).argmin(2)
    return c[a] * s[:, None]


def fit_global(W, s0, k=4, mode="pow2", symmetric=False, iters=60, seed=0):
    """Alternating minimisation of the pooled SSE over one global codebook."""
    rng = np.random.default_rng(seed)
    flat = (W / s0[:, None]).ravel()
    if symmetric:
        qs = np.quantile(np.abs(flat), [0.6, 0.93])
        inits = [np.array([-qs[1], -qs[0], qs[0], qs[1]])]
    else:
        inits = [np.quantile(flat, (np.arange(k) + 0.5) / k)]
        for _ in range(5):
            inits.append(np.quantile(flat, (np.arange(k) + rng.random(k)) / k))
    best = None
    for c in inits:
        c = np.sort(np.asarray(c, dtype=np.float64))
        for _ in range(iters):
            if mode == "pow2":
                sse = {}
                a = s = None
                best_j = None
                for j in J_ALL:
                    sj = s0 * 2.0**j
                    aj = np.abs(W[:, :, None] - (c[None, :] * sj[:, None])[:, None, :]).argmin(2)
                    qj = c[aj] * sj[:, None]
                    ssej = ((W - qj) ** 2).sum(1)
                    if best_j is None:
                        best_j, aj_b, sse_b = sj, aj, ssej
                    else:
                        m = ssej < sse_b
                        best_j = np.where(m, sj, best_j)
                        aj_b = np.where(m[:, None], aj, aj_b)
                        sse_b = np.where(m, ssej, sse_b)
                a = aj_b
                s = best_j[:, None] * np.ones_like(W)
            else:
                s1 = s0.copy()
                for _ in range(4):
                    a = np.abs(W[:, :, None] - (c[None, :] * s1[:, None])[:, None, :]).argmin(2)
                    cc = c[a]
                    s1 = (W * cc).sum(1) / np.maximum((cc * cc).sum(1), 1e-30)
                a = np.abs(W[:, :, None] - (c[None, :] * s1[:, None])[:, None, :]).argmin(2)
                s = s1[:, None] * np.ones_like(W)
            for m_ in range(k):
                mask = a == m_
                num = (s * W * mask).sum()
                den = (s * s * mask).sum()
                if den > 0:
                    c[m_] = num / den
            if symmetric:
                n = np.linalg.norm(c)
                c = c / n
                c = np.array([-c[3], -c[2], c[2], c[3]])
            else:
                c = c / np.linalg.norm(c)
                c = np.sort(c)
        q = assign_pow2(W, s0, c) if mode == "pow2" else assign_free(W, s0, c)
        e = rel_rms(W, q)[0]
        if best is None or e < best[0]:
            best = (e, c.copy())
    return best


def section_b(W4, s0):
    print()
    print("=" * 100)
    print("B. the 2-bit ladder on the E2M1 source, block 32 (i.i.d.-Gaussian calibration)")
    print("=" * 100)
    rows = []

    def add(name, q, note=""):
        e, eg = rel_rms(W4, q)
        rows.append((e, name, eg, note))
        print(f"  {e:6.2f} %  {name:52s} (global {eg:5.2f}) {note}")

    add("u2: scale=amax/2, integer grid {-2,-1,0,1}", q_uniform2(W4), "= measured 40.26")
    add("sym {0.5,1.5} x 2^j s0, j in {0..3}", q_pow2(W4, s0, np.array([-1.5, -0.5, 0.5, 1.5]), range(0, 4)),
        "= measured 37.94")
    add("sym {0.5,1.5} x 2^j s0, j in {-8..8}", q_pow2(W4, s0, np.array([-1.5, -0.5, 0.5, 1.5]), J_ALL),
        "wide exponent range")
    add("per-block Lloyd, free scale", q_lloyd_perblock(W4), "= measured 31.47")

    # every 4-subset of the 15 E2M1 values, pow2 scale
    sub_rows = []
    for sub in itertools.combinations(GRID, 4):
        c = np.array(sub)
        sub_rows.append((rel_rms(W4, assign_pow2(W4, s0, c))[0], sub))
    sub_rows.sort()
    print("\n  best 4-of-15 E2M1 subsets (pow2 scale, the reachable codebooks of the kernel):")
    for e, sub in sub_rows[:8]:
        symm = "sym" if abs(sub[0] + sub[3]) < 1e-9 and abs(sub[1] + sub[2]) < 1e-9 else "   "
        print(f"    {e:6.2f} %  {str(sub):40s} {symm}")
    best_sym_sub = next(r for r in sub_rows if abs(r[1][0] + r[1][3]) < 1e-9 and abs(r[1][1] + r[1][2]) < 1e-9)
    print(f"    best sign-symmetric subset: {best_sym_sub[1]} at {best_sym_sub[0]:.2f} %")
    best_asym = sub_rows[0]
    print(f"    best asymmetric subset:     {best_asym[1]} at {best_asym[0]:.2f} %")

    e, c = fit_global(W4, s0, mode="pow2", symmetric=True)
    print(f"\n  global sign-symmetric {np.round(c,4)} x 2^j s0 -> {e:.2f} %")
    e_sym_free, c_sym_free = e, c
    e, c = fit_global(W4, s0, mode="pow2", symmetric=False)
    print(f"  global free 4-level {np.round(c,4)} x 2^j s0 -> {e:.2f} %")
    e, c = fit_global(W4, s0, mode="free", symmetric=True)
    print(f"  global sign-symmetric {np.round(c,4)} x free scale -> {e:.2f} %")
    e, c = fit_global(W4, s0, mode="free", symmetric=False)
    print(f"  global free 4-level {np.round(c,4)} x free scale -> {e:.2f} %")
    return rows, sub_rows


def section_c():
    print()
    print("=" * 100)
    print("C. block size at 2 bits (free scale = fp8 mantissa; pow2 = E8M0)")
    print("=" * 100)
    rng = np.random.default_rng(21)
    print(f"  {'B':>4} {'scale bits/weight':>18} {'u2':>8} {'sym pow2':>9} {'sym free':>9} "
          f"{'best4of15 pow2':>15} {'best4of15 free':>15} {'pbLloyd free':>13}")
    for Bx in (8, 16, 32, 64, 128):
        W = sample("gauss", 48000 // Bx * 4, Bx, rng)
        W4, s0 = fp4_block(W)
        c = np.array([-1.5, -0.5, 0.5, 1.5])
        e_u2 = rel_rms(W4, q_uniform2(W4))[0]
        e_sym_p = rel_rms(W4, q_pow2(W4, s0, c, J_ALL))[0]
        e_sym_f = rel_rms(W4, assign_free(W4, s0, c))[0]
        # best 4-of-15 subset, evaluated on a subsample for speed
        idx = rng.choice(W4.shape[0], min(1500, W4.shape[0]), replace=False)
        sub = [(rel_rms(W4[idx], assign_pow2(W4[idx], s0[idx], np.array(s)))[0], s)
               for s in itertools.combinations(GRID, 4)]
        sub.sort()
        bs = np.array(sub[0][1])
        e_b4p = rel_rms(W4, assign_pow2(W4, s0, bs))[0]
        e_b4f = rel_rms(W4, assign_free(W4, s0, bs))[0]
        e_pb = rel_rms(W4, q_lloyd_perblock(W4))[0]
        print(f"  {Bx:>4} {8.0/Bx:>18.3f} {e_u2:8.2f} {e_sym_p:9.2f} {e_sym_f:9.2f} {e_b4p:15.2f} "
              f"{e_b4f:15.2f} {e_pb:13.2f}")
    print("  (block 8/16/64 rows use a different sample count; the columns are comparable within a row)")


def section_d(W4, s0):
    print()
    print("=" * 100)
    print("D. is the optimum on the E2M1 grid, and does asymmetry ever pay?")
    print("=" * 100)
    v = W4 / s0[:, None]
    u, cnt = np.unique(np.round(v, 6), return_counts=True)
    p = cnt / cnt.sum()
    print("  pooled within-block pmf over the 15 E2M1 values (in units of the block scale):")
    for val, pr in zip(u, p):
        print(f"    {val:+5.1f}  {pr*100:5.2f} %")
    print(f"  mass at exactly 0: {p[np.argmin(np.abs(u))]*100:.2f} %")
    print(f"  kurtosis of the FP4 weights: {float(((W4 - W4.mean())**4).mean()/W4.var()**2):.2f}")


def main():
    a = section_a()
    rng = np.random.default_rng(7)
    W = sample("gauss", NB, B, rng)
    W4, s0 = fp4_block(W)
    b, subs = section_b(W4, s0)
    section_d(W4, s0)
    section_c()
    json.dump({"A": a}, open("out_a.json", "w"), indent=1)


if __name__ == "__main__":
    main()
