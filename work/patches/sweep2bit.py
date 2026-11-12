"""Real-weight sweep for the 2-bit expert format: codebook ratio x scale grid."""
import json, sys
import numpy as np

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
MODEL = "/data/models/ds41-abl"


def load(layer, expert, key):
    from safetensors import safe_open
    import torch
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    n = f"layers.{layer}.ffn.experts.{expert}.{key}.weight"
    sn = f"layers.{layer}.ffn.experts.{expert}.{key}.scale"
    with safe_open(f"{MODEL}/{idx[n]}", framework="pt", device="cpu") as f:
        w = f.get_tensor(n).numpy().astype(np.int32)
    with safe_open(f"{MODEL}/{idx[sn]}", framework="pt", device="cpu") as f:
        s = f.get_tensor(sn).float().numpy()
    lo = w & 0x0F
    hi = (w >> 4) & 0x0F
    code = np.stack([lo, hi], axis=-1).reshape(w.shape[0], -1)
    mag = _E2M1[code & 0x7]
    sign = np.where(code & 0x8, -1.0, 1.0)
    return (mag * sign) * np.repeat(s, 32, axis=1)


def err(v, q):
    return float(np.sqrt(((v - q) ** 2).mean()) / v.std()) * 100.0


def sym_quant(flat, ratio, scale_mode):
    """sign-symmetric 4-level {0.5r, 1.5r} x s with s from the given grid; returns (dequant, rms)."""
    hi = 1.5 * ratio            # the outer level, in units of s
    inner = 0.5 * ratio
    amax = np.abs(flat).max(axis=1, keepdims=True)
    s_cont = np.maximum(amax / hi, 1e-30)
    if scale_mode == "e8m0":
        s = np.exp2(np.round(np.log2(s_cont)))
    elif scale_mode == "e4m3":
        e = np.floor(np.log2(s_cont))
        m = np.round((s_cont / np.exp2(e) - 1.0) * 8.0) / 8.0 + 1.0
        m = np.clip(m, 1.0, 1.875)
        s = np.exp2(e) * m
    else:                        # continuous
        s = s_cont
    n = flat / s
    q = np.where(n > ratio, hi, np.where(n > 0, inner, np.where(n < -ratio, -hi, np.where(n < 0, -inner, 0.0))))
    return q * s, s


def lloyd(flat, iters=8):
    levels = np.quantile(flat, (np.arange(4) + 0.5) / 4, axis=1).T
    for _ in range(iters):
        d = np.abs(flat[:, :, None] - levels[:, None, :])
        idx = d.argmin(axis=2)
        for k in range(4):
            m = idx == k
            cnt = m.sum(axis=1)
            levels[:, k] = np.where(cnt > 0, (flat * m).sum(axis=1) / np.maximum(cnt, 1), levels[:, k])
    d = np.abs(flat[:, :, None] - levels[:, None, :])
    return levels[np.arange(flat.shape[0])[:, None], d.argmin(axis=2)]


rows = []
for layer, expert, key in [(6, 0, "w1"), (6, 0, "w3"), (20, 5, "w1"), (20, 5, "w2"), (33, 100, "w3")]:
    v = load(layer, expert, key)
    flat = v.reshape(-1, 32)
    ref = np.sqrt((flat ** 2).mean())
    r = {"tensor": f"L{layer} e{expert} {key}", "std": float(v.std())}
    for ratio in (3.0, 4.0):
        for mode in ("e8m0", "e4m3", "cont"):
            q, _ = sym_quant(flat, ratio, mode)
            r[f"sym r{int(ratio)} {mode}"] = round(err(flat, q), 2)
    r["lloyd4"] = round(err(flat, lloyd(flat)), 2)
    rows.append(r)
    print(r, flush=True)
print("\nMEANS")
keys = [k for k in rows[0] if k not in ("tensor", "std")]
for k in keys:
    print(f"  {k:20s} {np.mean([r[k] for r in rows]):.2f} %")
