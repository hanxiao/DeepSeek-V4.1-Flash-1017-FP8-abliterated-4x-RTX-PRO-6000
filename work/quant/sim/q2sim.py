import numpy as np
rng = np.random.default_rng(0)

E2M1 = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0], dtype=np.float32)
NB, BLK = 400_000, 32

def make_blocks(tau, df=None):
    if df is None:
        x = rng.standard_normal((NB,BLK), dtype=np.float32)
    else:
        x = (rng.standard_t(df,(NB,BLK))).astype(np.float32)
    if tau>0:
        s = np.exp(rng.normal(0,tau,(NB,1))).astype(np.float32)
        x = x*s
    return x

def mxfp4(x):
    amax = np.abs(x).max(axis=1,keepdims=True)
    s0 = np.exp2(np.floor(np.log2(np.maximum(amax,1e-30)))-2).astype(np.float32)
    y = x/s0
    idx = np.abs(np.abs(y)[...,None]-E2M1[None,None,:]).argmin(-1)
    q = np.sign(y)*E2M1[idx]
    return (q*s0).astype(np.float32), s0

def rel_rms(w, q):
    num = np.sqrt(((w-q)**2).mean(1)); den = np.sqrt((w**2).mean(1))
    return float(np.mean(num/np.maximum(den,1e-12)))

def quant_fixed(w, s0, levels, jmax=3):
    best=None; beste=None
    for j in range(jmax+1):
        s = s0*(2.0**j)
        L = np.asarray(levels,dtype=np.float32)[None,:]*s
        d = np.abs(w[...,None]-L[:,None,:]); idx=d.argmin(-1)
        q = L[np.arange(len(L))[:,None], idx]
        e = ((w-q)**2).sum(1)
        if best is None: best=q.copy(); beste=e.copy()
        else:
            m = e<beste; best[m]=q[m]; beste[m]=e[m]
    return best

def quant_scale_cont(w, levels, gran=None):
    """levels shape (k,) positive/negative template; scale chosen per block, optional log2 granularity"""
    tmpl = np.asarray(levels,dtype=np.float32)
    # golden-section-ish grid search over scale
    amax = np.abs(w).max(1,keepdims=True)
    base = amax/ np.abs(tmpl).max()
    if gran is None:
        cands = base*np.exp(np.linspace(np.log(0.4),np.log(2.5),120)).astype(np.float32)
        cands = cands[:, :, None]  # (NB,120,1)
        L = cands*tmpl[None,None,:]
        d = np.abs(w[:,None,:,None]-L[:,:,None,:])
        idx = d.argmin(-1); q = np.take_along_axis(L[:,:,None,:].repeat(1,1), idx[...,None], -1)[...,0]
        err = ((w[:,None,:]-q)**2).sum(-1)
        bi = err.argmin(1)
        return q[np.arange(len(w)),bi]
    else:
        k = np.round(np.log2(base)/gran)*gran
        L = (2.0**k)*tmpl[None,:]
        d = np.abs(w[...,None]-L[:,None,:]); idx=d.argmin(-1)
        return L[np.arange(len(L))[:,None], idx]

def lloyd_perblock(w, k=4, iters=12):
    q = np.quantile(w, (np.arange(k)+0.5)/k, axis=1).T.astype(np.float32)
    for _ in range(iters):
        d = np.abs(w[:,:,None]-q[:,None,:]); idx = d.argmin(2)
        for m in range(k):
            mask = idx==m; cnt = mask.sum(1)
            q[:,m] = np.where(cnt>0,(w*mask).sum(1)/np.maximum(cnt,1),q[:,m])
    d = np.abs(w[:,:,None]-q[:,None,:]); idx=d.argmin(2)
    return q[np.arange(len(w))[:,None], idx]

def global_lloyd(w, k=4, iters=30):
    q = np.quantile(w.ravel(), (np.arange(k)+0.5)/k).astype(np.float32)
    for _ in range(iters):
        d = np.abs(w.ravel()[:,None]-q[None,:]); idx=d.argmin(1)
        for m in range(k):
            mask=idx==m
            if mask.any(): q[m]=w.ravel()[mask].mean()
    return np.sort(q)

for tau,df in [(0.0,None),(0.5,None),(1.0,None),(0.0,5)]:
    x = make_blocks(tau,df); w4,s0 = mxfp4(x)
    # anchors
    u2_scale = np.abs(w4).max(1,keepdims=True)/2
    L = (np.array([-2,-1,0,1],dtype=np.float32)[None,:]*u2_scale)
    d=np.abs(w4[...,None]-L[:,None,:]); idx=d.argmin(-1); qu=L[np.arange(len(L))[:,None],idx]
    sym = quant_fixed(w4,s0,[-1.5,-0.5,0.5,1.5])
    print(f"tau={tau} df={df}: u2={rel_rms(w4,qu)*100:.2f}%  sym_j={rel_rms(w4,sym)*100:.2f}%  lloyd_pb={rel_rms(w4,lloyd_perblock(w4))*100:.2f}%  fp4_vs_x={rel_rms(x,w4)*100:.2f}%")
