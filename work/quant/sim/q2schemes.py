import numpy as np
rng=np.random.default_rng(1)
E2M1=np.array([0.5,1.0,1.5,2.0,3.0,4.0,6.0],np.float32)
NB,BLK=20_000,32
def make(tau=0.5):
    x=rng.standard_normal((NB,BLK),dtype=np.float32)
    if tau>0: x=x*np.exp(rng.normal(0,tau,(NB,1))).astype(np.float32)
    return x
def mxfp4(x):
    amax=np.abs(x).max(1,keepdims=True)
    s0=np.exp2(np.floor(np.log2(np.maximum(amax,1e-30)))-2).astype(np.float32)
    y=x/s0; grid=np.array([0,0.5,1,1.5,2,3,4,6],np.float32)
    idx=np.abs(np.abs(y)[...,None]-grid[None,None,:]).argmin(-1)
    return (np.sign(y)*grid[idx]*s0).astype(np.float32),s0
def rel(w,q): return float(np.mean(np.sqrt(((w-q)**2).mean(1))/np.maximum(np.sqrt((w**2).mean(1)),1e-12)))
def pick_pair(z,A,B):
    """z: (NB,32) in s0 units. A,B: (K,) multipliers. returns chosen idx per block."""
    az=np.abs(z); best=np.full(len(z),np.inf,np.float32); bi=np.zeros(len(z),np.int32)
    for k in range(len(A)):
        e=np.minimum((az-A[k])**2,(az-B[k])**2).sum(1)
        m=e<best; best[m]=e[m]; bi[m]=k
    return bi
def build(z,s0,A,B,bi):
    az=np.abs(z); a=A[bi][:,None]; b=B[bi][:,None]
    return (np.where(np.abs(az-a)<np.abs(az-b),a,b)*np.sign(z)*s0).astype(np.float32)

x=make(); w,s0=mxfp4(x); z=w/s0
print("fp4 vs fp32:",f"{rel(x,w)*100:.2f}%   (their fp4 reference)")

A4=np.array([0.5*2**j for j in range(4)],np.float32); B4=np.array([1.5*2**j for j in range(4)],np.float32)
bi=pick_pair(z,A4,B4); q=build(z,s0,A4,B4,bi); print("1 sym{0.5,1.5} + j in 0..3    ",f"{rel(w,q)*100:.2f}%   <- their current")

pairs=[(a,b) for a in E2M1 for b in E2M1 if b>a]
A=np.array([a*2**j for a,b in pairs for j in range(4)],np.float32)
B=np.array([b*2**j for a,b in pairs for j in range(4)],np.float32)
bi2=pick_pair(z,A,B); q2=build(z,s0,A,B,bi2); print("2 sym, any E2M1 pair + j      ",f"{rel(w,q2)*100:.2f}%")
c=np.bincount(bi2,minlength=len(A)); 
for t in c.argsort()[::-1][:5]:
    a_,b_=A[t],B[t]; print(f"     A={a_:<5g} B={b_:<5g} ratio={b_/a_:.2f} share={c[t]/NB*100:5.1f}%")
print("   best single fixed pair by share-weighted scan:")
best=None
for a_,b_ in pairs:
    AA=np.array([a_*2**j for j in range(4)],np.float32); BB=np.array([b_*2**j for j in range(4)],np.float32)
    bb=pick_pair(z,AA,BB); qq=build(z,s0,AA,BB,bb); r=rel(w,qq)
    if best is None or r<best[0]: best=(r,a_,b_)
print(f"     best fixed ratio a={best[1]} b={best[2]} -> {best[0]*100:.2f}%")

# 3 sym per-block continuous levels
def sym_lloyd(z,iters=15):
    az=np.abs(z); a=np.quantile(az,0.35,1); b=np.quantile(az,0.85,1)
    for _ in range(iters):
        ma=np.abs(az-a[:,None])<np.abs(az-b[:,None]); mb=~ma
        a=np.where(ma.any(1),np.where(ma,az,0).sum(1)/np.maximum(ma.sum(1),1),a)
        b=np.where(mb.any(1),np.where(mb,az,0).sum(1)/np.maximum(mb.sum(1),1),b)
    a,b=np.minimum(a,b),np.maximum(a,b)
    q=np.where(np.abs(az-a[:,None])<np.abs(az-b[:,None]),a[:,None],b[:,None])*np.sign(z)
    return q*s0
print("3 sym, per-block continuous   ",f"{rel(w,sym_lloyd(z))*100:.2f}%")
# 4 per-block free 4-level Lloyd
def lloyd(z,k=4,iters=15):
    q=np.quantile(z,(np.arange(k)+0.5)/k,axis=1).T.astype(np.float32)
    for _ in range(iters):
        i=np.abs(z[:,:,None]-q[:,None,:]).argmin(2)
        for m in range(k):
            mk=i==m; cc=mk.sum(1); q[:,m]=np.where(cc>0,(z*mk).sum(1)/np.maximum(cc,1),q[:,m])
    i=np.abs(z[:,:,None]-q[:,None,:]).argmin(2)
    return (q[np.arange(len(z))[:,None],i]*s0).astype(np.float32)
q4=lloyd(z); print("4 per-block free Lloyd        ",f"{rel(w,q4)*100:.2f}%   <- round 66's 31.47%")
# 5 current cb, continuous scale
tmpl=np.array([-1.5,-0.5,0.5,1.5],np.float32)
base=np.abs(z).max(1,keepdims=True)/1.5
cands=(base*np.exp(np.linspace(np.log(0.5),np.log(2.0),64))).astype(np.float32)  # (NB,64)
best=np.full(NB,np.inf,np.float32); bs=np.ones(NB,np.float32)
for c_ in range(cands.shape[1]):
    L=cands[:,c_][:,None,None]*tmpl[None,None,:]
    e=((z[:,None,:,None]-L)**2).sum(-2)[:,0,0]
    m=e<best; best[m]=e[m]; bs[m]=cands[m,c_]
L=bs[:,None,None]*tmpl[None,None,:]
i=np.abs(z[:,None,:,None]-L).argmin(-1)[:,0]
q5=(L[np.arange(NB)[:,None],i]*s0).astype(np.float32)
print("5 sym{0.5,1.5}, continuous scale",f"{rel(w,q5)*100:.2f}%")
# 6 fine power-of-2 scale granularity
for g in (1/2,1/4,1/8,1/16):
    kk=np.round(np.log2(np.maximum(base,1e-9))/g)*g
    L=(2.0**kk)[:,:,None]*tmpl[None,None,:]
    i=np.abs(z[:,None,:,None]-L).argmin(-1)[:,0]; qq=(L[np.arange(NB)[:,None],i]*s0).astype(np.float32)
    print(f"6 sym{{0.5,1.5}}, scale 2^(k/{int(1/g)})",f"{rel(w,qq)*100:.2f}%")
# 7 global free Lloyd + j
flat=z.ravel(); q=np.quantile(flat,(np.arange(4)+0.5)/4).astype(np.float32)
for _ in range(30):
    i=np.abs(flat[:,None]-q[None,:]).argmin(1)
    for m in range(4):
        mk=i==m
        if mk.any(): q[m]=flat[mk].mean()
q=np.sort(q); print("   global free-Lloyd levels (x s0):",np.round(q,3),"  (sign-sym would be symmetric)")
for j in range(4):
    L=(q*2**j)[None,:]
    i=np.abs(z[:,None,:,None]-L).argmin(-1)[:,0]; qq=(L[np.arange(NB)[:,None],i]*s0).astype(np.float32)
    print(f"7 global free Lloyd, j={j}      ",f"{rel(w,qq)*100:.2f}%")
# 8 global free Lloyd + fine pow2
for g in (1/4,1/8):
    kk=np.round(np.log2(np.maximum(base,1e-9))/g)*g
    L=(2.0**kk)[:,:,None]*q[None,None,:]
    i=np.abs(z[:,None,:,None]-L[:,:,None,:]).argmin(-1)
    qq=(np.take_along_axis(L[:,:,None,:],i[...,None],-1)[...,0]*s0).astype(np.float32)
    print(f"8 global free Lloyd, scale 2^(k/{int(1/g)})",f"{rel(w,qq)*100:.2f}%")
print("9 3-bit per-block 8-level Lloyd",f"{rel(w,lloyd(z,8))*100:.2f}%")
