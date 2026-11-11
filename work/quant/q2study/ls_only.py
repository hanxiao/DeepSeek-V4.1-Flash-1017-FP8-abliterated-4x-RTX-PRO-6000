import numpy as np
from q2_core import fp4_block, q_lloyd_perblock, rel_rms, sample
from q2_final import fit_global_robust
rng = np.random.default_rng(7)
for dist in ("gauss","laplace","t4"):
    W = sample(dist, 5000, 32, rng); W4,s0 = fp4_block(W)
    print(f"--- {dist}")
    for sym in (True, False):
        e,c = fit_global_robust(W4, s0, "ls", symmetric=sym)
        r = (c[3]/c[2]) if sym else (c[3]/c[1])
        print(f"    global codebook {'sym ' if sym else 'asym'} + free scale: {np.round(c,3)} ratio {r:.2f} -> {e:.2f} %")
    print(f"    per-block codebook + free scale: {rel_rms(W4,q_lloyd_perblock(W4))[0]:.2f} %")
