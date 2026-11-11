import numpy as np, itertools
from q2_core import *

TARGET = np.array([40.26, 37.94, 31.47])
rng0 = np.random.default_rng(11)
NB,B = 12000,32
base = rng0.standard_normal((NB,B))

def mk(nu, kappa, rng):
    z = base
    if nu is not None:
        g = rng.chisquare(nu, size=(NB,1))/nu
        z = z/np.sqrt(g)
        z = z/np.sqrt((z**2).mean())
    if kappa>0:
        d = rng.normal(0,kappa,size=(NB,1))
        z = z + d
        z = z/z.std()
    return z

rows=[]
for nu in (None, 30, 12, 8):
    for kappa in (0.0, 0.1, 0.2, 0.3, 0.45):
        rng = np.random.default_rng(3)
        W = mk(nu,kappa,rng)
        W4,s0 = fp4_block(W)
        e1 = rel_rms(W4,q_uniform2(W4))[0]
        e2 = rel_rms(W4,q_pow2(W4,s0,np.array([-1.5,-0.5,0.5,1.5]),range(-8,9)))[0]
        e3 = rel_rms(W4,q_lloyd_perblock(W4))[0]
        v = np.array([e1,e2,e3])
        rows.append((float(np.abs(v-TARGET).mean()), nu, kappa, v))
        print(f"nu={str(nu):>4} kappa={kappa:.2f}  u2 {e1:6.2f}  sym {e2:6.2f}  pbL {e3:6.2f}   L1 {np.abs(v-TARGET).mean():.2f}", flush=True)
rows.sort()
print("\nbest:", rows[0])
