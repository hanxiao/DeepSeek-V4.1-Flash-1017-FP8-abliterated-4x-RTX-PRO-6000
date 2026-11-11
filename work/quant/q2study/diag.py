import numpy as np
from q2_core import *

rng = np.random.default_rng(7)
NB,B = 20000,32
for dist in ("gauss","laplace","t4","t3"):
    W = sample(dist,NB,B,rng); W4,s0 = fp4_block(W)
    c = np.array([-1.5,-0.5,0.5,1.5])
    sse = {}
    for j in range(-6,7):
        sj = s0*2.0**j
        L = c[None,:]*sj[:,None]
        a = np.abs(W4[:,:,None]-L[:,None,:]).argmin(2)
        q = np.take_along_axis(L[:,None,:].repeat(B,1), a[:,:,None],2)[:,:,0]
        sse[j] = ((W4-q)**2).sum(1)
    S = np.stack([sse[j] for j in range(-6,7)],1)
    jb = np.array(range(-6,7))[S.argmin(1)]
    print(dist, "j hist:", {int(k):int(v) for k,v in zip(*np.unique(jb,return_counts=True))},
          "sigma/s0 mean %.2f"%float((W4.std(1)/s0).mean()),
          "amax/s0 mean %.2f"%float((np.abs(W4).max(1)/s0).mean()))
