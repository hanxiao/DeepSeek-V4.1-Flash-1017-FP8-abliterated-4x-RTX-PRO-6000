import numpy as np, glob, json
p = "/data/mtp/cache/meta.json"
m = json.load(open(p))
print("meta:", {k: m[k] for k in ("num_samples", "total_tokens", "max_length", "topk") if k in m})
aux = np.memmap("/data/mtp/cache/aux.bin", dtype=np.uint16, mode="r", shape=(m["num_samples"], m["max_length"], m["num_hidden"] if "num_hidden" in m else 15360))
print("aux shape", aux.shape)
sel = aux[:4].reshape(-1)[:2621440]
f = sel.view(np.float32) if False else None
# bf16 -> bits: NaN are exp 0x7F80 and mantissa != 0
bits = sel.astype(np.uint32)
nonzero = (bits & 0x7FFF) != 0
inf = (bits & 0x7FFF) == 0x7F80 & nonzero
nan = (bits & 0x7FF0) == 0x7F80
print("rows read", len(sel), "nan", int(nan.sum()), "inf", int(inf.sum()), "zero", int((bits==0).sum()))
tg = np.memmap("/data/mtp/cache/topk_logp.bin", dtype=np.float32, mode="r", shape=(m["num_samples"], m["max_length"], m["topk"]))
a = tg[:256].reshape(-1)
print("topk_logp nan", int(np.isnan(a).sum()), "min", float(np.nanmin(a)), "max", float(np.nanmax(a)))
