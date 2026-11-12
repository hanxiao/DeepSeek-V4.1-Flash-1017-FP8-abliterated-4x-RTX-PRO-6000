#!/usr/bin/env python3
"""Requantise the routed experts of the DeepSeek-V4.1-Flash-1017 checkpoint to a SIGN-SYMMETRIC 2-BIT
codebook while keeping the checkpoint's own 4-bit storage, so the stock b12x kernel can serve it and the
only thing that changes is the information in the weights. That makes the 2-bit quality cost measurable
without writing a 2-bit kernel.

Per 32-element block (the checkpoint's own ue8m0 block grid) the codebook is {+-0.5 s, +-1.5 s} with
s = 2**j * s0, j chosen per block from {0,1,2,3} to minimise the block's squared error; the four levels
are E2M1 codes 1 and 3 with the sign bit, and the scale exponent is bumped by j. This is exactly what a
2-bit kernel with a per-block power-of-two scale would store.

usage: requant.py MODEL_DIR OUT_DIR [--bits 2] [--limit N]
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
CODE_LO, CODE_HI = 1, 3  # E2M1 codes for 0.5 and 1.5

# [vllm-moet] imatrix: per-input-channel importance E[x_k^2] from /data/patches/h_rank*.pt (offline
# calibration, 8 prompts, 4 TP ranks). Used to weight the per-block scale/code search for w1/w3, whose
# K is the 5120-wide hidden state. w2's K is the 576-wide intermediate, for which we have no importance
# vector yet, so it keeps the plain MSE objective.
_HIMP = None


def _load_h():
    global _HIMP
    import glob
    fs = sorted(glob.glob("/data/patches/h_rank*.pt"))
    if not fs:
        return None
    acc = None
    for f in fs:
        d = torch.load(f, map_location="cpu")
        acc = d["h"].float() if acc is None else acc + d["h"].float()
    acc = acc / len(fs)
    _HIMP = acc
    print("[h] importance loaded from %d ranks, len %d, min %.3e max %.3e" % (len(fs), acc.numel(), acc.min(), acc.max()), flush=True)
    return acc


def requant_2bit(packed: torch.Tensor, scale: torch.Tensor, dev="cuda", hvec=None):
    """packed int8 [R, C/2], scale e8m0 bytes [R, C/32] -> (packed int8 [R, C/2], scale bytes) at 2 bits."""
    p = packed.to(dev)
    s = scale.to(dev)
    R, half = p.shape
    C = half * 2
    lo = (p & 0x0F).to(torch.int16)
    hi = ((p.to(torch.int16) >> 4) & 0x0F)
    code = torch.stack([lo, hi], dim=-1).reshape(R, C)              # [R, C] 0..15
    mag = E2M1.to(dev)[(code & 0x7).long()]
    sign = torch.where((code & 0x8) != 0, -1.0, 1.0)
    # e8m0: value = 2**(byte-127)
    s0 = torch.pow(2.0, s.to(torch.float32) - 127.0)                 # [R, C/32]
    val = mag * sign
    v = val.reshape(R, C // 32, 32)                                  # in units of s0
    a = v.abs()
    best_err = None
    best = None
    wgt = None
    if hvec is not None and hvec.numel() == C:
        wgt = hvec.to(dev).float().view(1, C // 32, 32)
    for j in (0, 1, 2, 3):
        lo_lvl, hi_lvl = 0.5 * (2 ** j), 1.5 * (2 ** j)
        thr = 1.0 * (2 ** j)
        use_hi = a >= thr
        rec = torch.where(use_hi, torch.full_like(a, hi_lvl), torch.full_like(a, lo_lvl))
        if wgt is None:
            err = ((a - rec) ** 2).sum(dim=2)
        else:
            err = (((a - rec) ** 2) * wgt).sum(dim=2)
        if best_err is None:
            best_err = err
            best = (use_hi, torch.full((1,), j, dtype=torch.int32, device=dev))
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best = (torch.where(take.unsqueeze(-1), use_hi, best[0]), torch.where(take, j, best[1]))
    use_hi, j = best[0], best[1]
    codes = torch.where(use_hi, torch.full_like(a, CODE_HI, dtype=torch.int16),
                        torch.full_like(a, CODE_LO, dtype=torch.int16))
    neg = (v < 0) | ((v == 0) & (torch.arange(32, device=dev) % 2 == 0))
    codes = torch.where(neg, codes | 0x8, codes).reshape(R, C).to(torch.int16)
    out = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    new_scale = (s.to(torch.int16) + j.to(torch.int16)).clamp(0, 254).to(torch.uint8)
    err = torch.sqrt(best_err.sum() / (a ** 2).sum()).item()
    return out.to(torch.int8).cpu(), new_scale.cpu(), err


def main() -> int:
    _load_h()
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    src, dst = args.model_dir, args.out_dir
    os.makedirs(dst, exist_ok=True)
    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    routed = [k for k in idx if re.match(r"^(layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.", k)]
    shards = sorted({idx[k] for k in routed})
    new_map = dict(idx)
    tot_err, n = 0.0, 0
    for si, shard in enumerate(shards):
        keys = [k for k in routed if idx[k] == shard]
        out_t = {}
        with safe_open(f"{src}/{shard}", framework="pt", device="cpu") as f:
            for k in keys:
                if k.endswith(".scale"):
                    continue
                sk = k[:-len(".weight")] + ".scale"
                w = f.get_tensor(k)
                s = f.get_tensor(sk).view(torch.uint8)
                hv = _HIMP if (k.endswith("w1.weight") or k.endswith("w3.weight")) else None
                nw, ns, err = requant_2bit(w, s, hvec=hv)
                out_t[k] = nw
                out_t[sk] = ns.view(torch.float8_e8m0fnu)
                tot_err += err; n += 1
        name = f"q2-{shard}"
        save_file(out_t, f"{dst}/{name}", metadata={"format": "pt"})
        for k in out_t:
            new_map[k] = name
        print(f"[{si+1}/{len(shards)}] {shard} -> {name}: {len(keys)//2} tensors, rms err so far {tot_err/max(n,1)*100:.2f} %", flush=True)
        if args.limit and si + 1 >= args.limit:
            break
    json.dump({"metadata": {"total_size": 0}, "weight_map": new_map},
              open(f"{dst}/model.safetensors.index.json", "w"))
    for fn in os.listdir(src):
        if fn == "model.safetensors.index.json" or fn.startswith("q2-"):
            continue
        p = f"{dst}/{fn}"
        if not os.path.exists(p):
            os.symlink(os.path.abspath(f"{src}/{fn}"), p)
    print(f"DONE mean block rms err {tot_err/max(n,1)*100:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
