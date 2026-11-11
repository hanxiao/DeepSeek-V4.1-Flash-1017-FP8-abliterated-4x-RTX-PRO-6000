import numpy as np
rng=np.random.default_rng(11); NB,BLK=6000,32
G=np.array([0,0.5,1,1.5,2,3,4,6],np.float32)
x=rng.standard_normal((NB,BLK),dtype=np.float32)
s0=np.exp2(np.floor(np.log2(np.abs(x).max(1,keepdims=True)))-2).astype(np.float32)
y=x/s0; i0=np.abs(np.abs(y)[...,None]-G[None,None,:]).argmin(-1)
w=(np.sign(y)*G[i0]*s0).astype(np.float32); z=(w/s0).astype(np.float32); az=np.abs(z)
def rel(q): return float(np.mean(np.sqrt(((w-q)**2).mean(1))/np.sqrt((w**2).mean(1))))
def q_pair(A,B,bi):
    a=A[bi][:,None]; b=B[bi][:,None]
    pos=np.where(np.abs(az-a)<np.abs(az-b),a,b)
    return pos*np.sign(z)*s0
def pick_pair(A,B):
    best=np.full(NB,np.inf,np.float32); bi=np.zeros(NB,np.int32)
    for k in range(len(A)):
        e=np.minimum((az-A[k])**2,(az-B[k])**2).sum(1)
        m=e<best; best[m]=e[m]; bi[m]=k
    return bi,best.sum()
def pair_scan(pairs,jmax=3):
    A=np.array([a*2**j for a,b in pairs for j in range(jmax+1)],np.float32)
    B=np.array([b*2**j for a,b in pairs for j in range(jmax+1)],np.float32)
    return A,B,pick_pair(A,B)
def rep(name,q): print(f"{name:42s} {rel(q)*100:6.2f}%")
A4=np.array([0.5,1.0,2.0,4.0],np.float32); B4=np.array([1.5,3.0,6.0,12.0],np.float32)
bi,_=pick_pair(A4,B4); base=rel(q_pair(A4,B4,bi)); rep("current {0.5,1.5} + j0..3",q_pair(A4,B4,bi))
for nm,a,b in [("ratio 4 {0.5,2.0}",0.5,2.0),("ratio 4 {1.0,4.0}",1.0,4.0),("ratio 2 {0.5,1.0}",0.5,1.0),("ratio 6 {0.5,3.0}",0.5,3.0)]:
    AA=np.array([a*2**j for j in range(4)],np.float32); BB=np.array([b*2**j for j in range(4)],np.float32)
    bii,_=pick_pair(AA,BB); q=q_pair(AA,BB,bii); print(f"{nm:42s} {rel(q)*100:6.2f}%   delta {(rel(q)/base-1)*100:+5.1f}%")
pairs=[(a,b) for a in G[G>0] for b in G[G>0] if b>a]
Ap,Bp,(bip,_)=pair_scan(pairs); q=q_pair(Ap,Bp,bip)
print(f"{'per-block (a,b) pair + j':42s} {rel(q)*100:6.2f}%   delta {(rel(q)/base-1)*100:+5.1f}%")
for g in (1/2,1/4,1/8):
    ks=np.arange(-2,3+g/2,g); cands=np.stack([np.full(NB,2.0**k,np.float32) for k in ks],1)
    best=np.full(NB,np.inf,np.float32); bs=np.ones(NB,np.float32)
    for k in range(cands.shape[1]):
        sv=cands[:,k]; lv=sv[:,None]*np.array([0.5,1.5],np.float32)[None,:]
        ii=np.abs(az[:,:,None]-lv[:,None,:]).argmin(-1)
        pos=np.take_along_axis(np.broadcast_to(lv[:,None,:],(NB,BLK,2)),ii[...,None],-1)[:,:,0]
        e=((az-pos)**2).sum(1); m=e<best; best[m]=e[m]; bs[m]=sv[m]
    lv=bs[:,None]*np.array([0.5,1.5],np.float32)[None,:]
    ii=np.abs(az[:,:,None]-lv[:,None,:]).argmin(-1)
    pos=np.take_along_axis(np.broadcast_to(lv[:,None,:],(NB,BLK,2)),ii[...,None],-1)[:,:,0]
    q=pos*np.sign(z)*s0
    print(f"{'scale 2^(k/'+str(int(1/g))+') {0.5,1.5}':42s} {rel(q)*100:6.2f}%   delta {(rel(q)/base-1)*100:+5.1f}%")
def lloyd(k=4,iters=15):
    q=np.quantile(z,(np.arange(k)+0.5)/k,axis=1).T.astype(np.float32)
    for _ in range(iters):
        ii=np.abs(z[:,:,None]-q[:,None,:]).argmin(2)
        for m in range(k):
            mk=ii==m; c=mk.sum(1); q[:,m]=np.where(c>0,(z*mk).sum(1)/np.maximum(c,1),q[:,m])
    ii=np.abs(z[:,:,None]-q[:,None,:]).argmin(2)
    return (q[np.arange(NB)[:,None],ii]*s0).astype(np.float32)
q=lloyd(); print(f"{'per-block free Lloyd 4 levels':42s} {rel(q)*100:6.2f}%   delta {(rel(q)/base-1)*100:+5.1f}%")
q=lloyd(8); print(f"{'per-block free Lloyd 8 levels (3-bit)':42s} {rel(q)*100:6.2f}%   delta {(rel(q)/base-1)*100:+5.1f}%")
