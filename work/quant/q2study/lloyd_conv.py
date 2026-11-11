import numpy as np
from q2_core import *

def quant_lloyd_their(vals, levels, block=32, iters=6):
    """verbatim re-implementation of work/quant/re-quant2.py:quant_lloyd"""
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
    d = np.abs(flat[:, :, None] - q[:, None, :])
    out = q[np.arange(nb)[:, None], d.argmin(axis=2)]
    return out.reshape(rows, cols)

rng = np.random.default_rng(5)
NB,B=8000,32
for nu in (None, 12):
    for kappa in (0.0, 0.2):
        z = rng.standard_normal((NB,B))
        if nu is not None:
            g = rng.chisquare(nu, size=(NB,1))/nu; z = z/np.sqrt(g); z/=np.sqrt((z**2).mean())
        if kappa>0:
            z = z + rng.normal(0,kappa,size=(NB,1)); z/=z.std()
        W4,s0 = fp4_block(z)
        e_their = rel_rms(W4, quant_lloyd_their(W4,4))[0]
        e_mine6 = rel_rms(W4, q_lloyd_perblock(W4,iters=6))[0]
        e_mine  = rel_rms(W4, q_lloyd_perblock(W4,iters=40))[0]
        print(f"nu={nu} k={kappa}: their-6it {e_their:.2f}  mine-6it {e_mine6:.2f}  mine-40it {e_mine:.2f}")
