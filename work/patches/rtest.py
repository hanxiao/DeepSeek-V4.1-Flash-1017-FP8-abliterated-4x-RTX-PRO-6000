import json, re, torch, sys
sys.path.insert(0, "/data/patches")
from safetensors import safe_open
from requant import requant_2bit, E2M1
idx = json.load(open("/data/models/ds41-abl/model.safetensors.index.json"))["weight_map"]
k = "layers.6.ffn.experts.0.w1.weight"; sk = k[:-7] + ".scale"
with safe_open("/data/models/ds41-abl/" + idx[k], framework="pt", device="cpu") as f:
    w = f.get_tensor(k); s = f.get_tensor(sk)
print("w", w.shape, w.dtype, "s", s.shape, s.dtype, idx[k])
s8 = s.view(torch.uint8)
print("scale bytes min/max", int(s8.min()), int(s8.max()))
nw, ns, err = requant_2bit(w, s8, dev="cpu")
print("err", round(err*100, 2), "%", "new scale bytes", int(ns.min()), int(ns.max()), "shift", int(ns.float().mean()-s8.float().mean()))
# verify: dequant both and compare
def deq(p, sc):
    p=p.to(torch.int32); lo=p&0xF; hi=(p>>4)&0xF
    code=torch.stack([lo,hi],dim=-1).reshape(p.shape[0],-1)
    mag=E2M1[(code&0x7).long()]; sign=torch.where((code&0x8)!=0,-1.0,1.0)
    s0=torch.pow(2.0, sc.float()-127.0)
    return (mag*sign)*s0.repeat_interleave(32,dim=1)
a=deq(w,s8); b=deq(nw,ns)
print("rel rms err (dequantised)", round(float(torch.sqrt(((a-b)**2).mean())/a.std())*100,2), "%")
# how many distinct magnitudes does the new tensor use?
p=nw.to(torch.int32); code=torch.stack([p&0xF,(p>>4)&0xF],dim=-1).reshape(p.shape[0],-1)
print("distinct codes", torch.unique(code).tolist())
