import numpy as np
from q2_core import *
TARGET = np.array([40.26, 37.94, 31.47])
NB,B=12000,32
rows=[]
for p in (0.0,0.05,0.1,0.2,0.3):
    for A in (3.0,5.0,8.0,12.0):
        rng = np.random.default_rng(9)
        z = rng.standard_normal((NB,B))
        m = rng.random((NB,B))<p
        z = z + m*rng.normal(0,A,size=(NB,B))
        z = z/z.std()
        W4,s0 = fp4_block(z)
        e1 = rel_rms(W4,q_uniform2(W4))[0]
        e2 = rel_rms(W4,q_pow2(W4,s0,np.array([-1.5,-0.5,0.5,1.5]),range(-8,9)))[0]
        e3 = rel_rms(W4,q_lloyd_perblock(W4))[0]
        v=np.array([e1,e2,e3]); L=np.abs(v-TARGET).mean()
        rows.append((L,p,A,v))
        print(f"p={p:.2f} A={A:4.0f}  u2 {e1:6.2f} sym {e2:6.2f} pbL {e3:6.2f}  L1 {L:.2f}",flush=True)
rows.sort(); print("\nBEST", rows[0])
