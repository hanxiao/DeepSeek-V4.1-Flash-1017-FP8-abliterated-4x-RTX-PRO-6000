import numpy as np
rng=np.random.default_rng(7); NB,BLK=5000,32
G=np.array([0,0.5,1,1.5,2,3,4,6],np.float32)
x=rng.standard_normal((NB,BLK),dtype=np.float32)
s0=np.exp2(np.floor(np.log2(np.abs(x).max(1,keepdims=True)))-2).astype(np.float32)
y=x/s0; i0=np.abs(np.abs(y)[...,None]-G[None,None,:]).argmin(-1)
w=(np.sign(y)*G[i0]*s0).astype(np.float32); z=(w/s0).astype(np.float32); az=np.abs(z)
def rel(q): return float(np.mean(np.sqrt(((w-q)**2).mean(1))/np.sqrt((w**2).mean(1))))
def q_sym(sv,a,b):
    lv=sv[:,None]*np.array([a,b],np.float32)[None,:]
    d=np.abs(az[:,:,None]-lv[:,None,:]); ii=d.argmin(-1)
    pos=np.take_along_axis(np.broadcast_to(lv[:,None,:],(NB,BLK,2)),ii[...,None],-1)[:,:,0]
    return pos*np.sign(z)
def pick(cands,a,b):
    best=np.full(NB,np.inf,np.float32); bs=np.ones(NB,np.float32)
    for k in range(cands.shape[1]):
        sv=cands[:,k]; q=q_sym(sv,a,b); e=((z-q)**2).sum(1)
        m=e<best; best[m]=e[m]; bs[m]=sv[m]
    return bs
print("--- power-of-2 scale, j in 0..3 (their scheme) ---")
for a,b in [(0.5,1.5),(0.5,2.0),(1.0,4.0),(0.5,1.0),(1.0,1.5),(0.5,3.0)]:
    j4=np.stack([np.full(NB,2.0**j,np.float32) for j in range(4)],1)
    print(f"  {{ {a}, {b} }} + j0..3 : {rel(q_sym(pick(j4,a,b),a,b)*s0)*100:6.2f}%")
print("--- scale range 2^-2..2^3, granularity 2^(1/g) ---")
for a,b in [(0.5,1.5),(0.5,2.0),(1.0,4.0)]:
    for g in (1,1/2,1/4,1/8):
        ks=np.arange(-2,3+g/2,g); cands=np.stack([np.full(NB,2.0**k,np.float32) for k in ks],1)
        print(f"  {{ {a}, {b} }} g=2^(1/{int(1/g)}) : {rel(q_sym(pick(cands,a,b),a,b)*s0)*100:6.2f}%   ({len(ks)} cands)")
