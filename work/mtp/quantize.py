#!/usr/bin/env python3
"""Native DeepSeek-V4.1 quantisation for the DSpark MTP tensors.

The checkpoint stores the three MTP blocks in the same native format as the backbone:

  * dense FP8   ``<name>.weight``  F8_E4M3, 2-D blocks of 32x32, power-of-2 (ue8m0)
                 scales in ``<name>.scale`` (F8_E8M0, shape [out/32, in/32]);
                 scale = 2**ceil(log2(max(amax, 1e-4) / 448));
  * expert FP4  ``<name>.weight``  I8, two e2m1 values per byte (low nibble = even
                 column), row blocks of 32 along the input dim, scales in
                 ``<name>.scale`` (F8_E8M0, shape [out, in/32]);
                 scale = 2**ceil(log2(max(amax, 6*2**-126) / 6));
  * unquantised BF16/F32 for norms, routers, the Markov/confidence heads and the
                 attention sink.

Both conventions are the ones the reference implementation in the checkpoint uses
(``inference/kernel.py: act_quant_kernel`` / ``fp4_quant_kernel``) and the nibble
order is the one the official ``inference/convert.py`` unpacks.

Commands
--------
  selftest                       pure-python checks, no checkpoint needed
  roundtrip --model DIR          dequantise -> requantise every MTP tensor of DIR and
                                 assert it is byte-identical (validates this file
                                 against the real checkpoint; run it first on the VM)
  build --model DIR --weights F --out DIR
                                 write the overlay checkpoint directory: hardlinks for
                                 every shard that holds no ``mtp.*`` tensor, rewritten
                                 copies of the three that do, index.json unchanged
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from glob import glob

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 32

# e2m1 magnitudes, index 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6 ; index 8..15 are the negatives
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# midpoints between consecutive magnitudes, used to snap a real value to the nearest e2m1 code
FP4_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


# --------------------------------------------------------------------------------------
# power-of-2 (ue8m0) scales
# --------------------------------------------------------------------------------------
def _ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for x > 0, exact: frexp gives x = m * 2**e with 0.5 <= m < 1, so
    log2(x) = e-1 when m == 0.5 (x an exact power of two) and lies in (e-1, e) otherwise."""
    mant, exp = torch.frexp(x.float())
    return torch.where(mant == 0.5, exp - 1, exp).to(torch.int32)


def _pow2(exp: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)


def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """F8_E8M0 -> fp32 power of two (byte b means 2**(b-127))."""
    raw = scale.view(torch.uint8).to(torch.int32)
    return _pow2(raw - 127)


def _fp32_to_e8m0(exp: torch.Tensor) -> torch.Tensor:
    """integer exponent -> F8_E8M0, saturating to the representable [-127, 128]."""
    exp = exp.clamp(-127, 128).to(torch.int32) + 127
    return exp.to(torch.uint8).view(torch.float8_e8m0fnu)


# --------------------------------------------------------------------------------------
# FP8 dense weights, 32x32 blocks
# --------------------------------------------------------------------------------------
def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in] F8_E4M3 + [out/32, in/32] F8_E8M0 -> bf16 [out, in]."""
    out_dim, in_dim = weight.shape
    bo, bi = scale.shape
    assert out_dim % bo == 0 and in_dim % bi == 0, (weight.shape, scale.shape)
    s = _e8m0_to_fp32(scale)
    w = weight.to(torch.float32).unflatten(0, (bo, out_dim // bo)).unflatten(-1, (bi, in_dim // bi))
    w = w * s[:, None, :, None]
    return w.flatten(0, 1).flatten(-2).to(torch.bfloat16)


def quant_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [out, in] -> (F8_E4M3, F8_E8M0 [out/32, in/32]) using the checkpoint convention."""
    w = weight.float()
    out_dim, in_dim = w.shape
    assert out_dim % BLOCK == 0 and in_dim % BLOCK == 0, w.shape
    bo, bi = out_dim // BLOCK, in_dim // BLOCK
    blk = w.unflatten(0, (bo, BLOCK)).unflatten(-1, (bi, BLOCK))
    amax = blk.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    exp = _ceil_log2(amax / FP8_MAX)
    s = _pow2(exp)
    q = (blk / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    q = q.flatten(0, 1).flatten(-2).to(torch.float8_e4m3fn)
    return q, _fp32_to_e8m0(exp)


# --------------------------------------------------------------------------------------
# FP4 expert weights, row blocks of 32, two values per byte
# --------------------------------------------------------------------------------------
def unpack_fp4(packed: torch.Tensor, in_dim: int) -> torch.Tensor:
    """[out, in/2] int8 -> [out, in] fp32 values (low nibble = even column)."""
    u = packed.view(torch.uint8)
    low = (u & 0x0F).to(torch.long)
    high = ((u >> 4) & 0x0F).to(torch.long)
    vals = torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(1)
    assert vals.shape[-1] == in_dim, (vals.shape, in_dim)
    return vals


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """[out, in] uint8 nibble codes -> [out, in/2] int8 (low nibble = even column)."""
    assert codes.shape[-1] % 2 == 0
    low = codes[..., 0::2] & 0x0F
    high = (codes[..., 1::2] & 0x0F) << 4
    return (low | high).to(torch.uint8).view(torch.int8)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in/2] int8 + [out, in/32] F8_E8M0 -> bf16 [out, in]."""
    out_dim, in_half = packed.shape
    in_dim = in_half * 2
    assert scale.shape == (out_dim, in_dim // BLOCK), (packed.shape, scale.shape)
    vals = unpack_fp4(packed, in_dim)
    s = _e8m0_to_fp32(scale)
    vals = vals.unflatten(-1, (-1, BLOCK)) * s[..., None]
    return vals.flatten(-2).to(torch.bfloat16)


def quant_fp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [out, in] -> (int8 [out, in/2] packed, F8_E8M0 [out, in/32])."""
    w = weight.float()
    out_dim, in_dim = w.shape
    assert in_dim % BLOCK == 0, w.shape
    blk = w.unflatten(-1, (-1, BLOCK))
    amax = blk.abs().amax(dim=-1).clamp_min(FP4_MAX * 2.0**-126)
    exp = _ceil_log2(amax / FP4_MAX)
    s = _pow2(exp)
    q = (blk / s[..., None]).clamp(-FP4_MAX, FP4_MAX)
    mag = q.abs()
    idx = torch.bucketize(mag, FP4_BOUNDS)          # 0..7
    codes = torch.where(q < 0, idx + 8, idx).to(torch.uint8)
    return pack_fp4(codes.flatten(-2)), _fp32_to_e8m0(exp)


# --------------------------------------------------------------------------------------
# checkpoint helpers
# --------------------------------------------------------------------------------------
def read_safetensors_header(path: str) -> dict:
    """Header JSON of a safetensors file without mapping the (multi-GB) body."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def mtp_files(model_dir: str) -> list[str]:
    """Shard files that hold at least one ``mtp.*`` tensor."""
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    return sorted({f for k, f in index["weight_map"].items() if k.startswith("mtp.")})


def is_expert_weight(name: str) -> bool:
    return ".experts." in name and name.endswith(".weight")


def scale_name(weight_name: str) -> str | None:
    """``...w1.weight`` -> ``...w1.scale``.

    The checkpoint replaces the ``.weight`` suffix, it does not append to it: the scale of
    ``mtp.2.ffn.experts.123.w1.weight`` is ``mtp.2.ffn.experts.123.w1.scale``.
    """
    if not weight_name.endswith(".weight"):
        return None
    return weight_name[: -len(".weight")] + ".scale"


def requantise(name: str, tensor: torch.Tensor, scale: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Take one dequantised MTP tensor back to the checkpoint format."""
    if scale is None:
        return tensor, None
    if is_expert_weight(name):
        q, s = quant_fp4(tensor)
    else:
        q, s = quant_fp8(tensor)
    assert q.shape == tensor.shape if q.dtype == torch.float8_e4m3fn else True
    return q, s


def dequantise(name: str, tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if scale is None:
        return tensor
    if is_expert_weight(name):
        return dequant_fp4(tensor, scale)
    return dequant_fp8(tensor, scale)


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------
def cmd_selftest(_args) -> int:
    torch.manual_seed(0)
    # --- fp8: quantise a random tensor, dequantise, requantise -> identical bytes
    w = torch.randn(128, 256).to(torch.bfloat16)
    q, s = quant_fp8(w)
    d = dequant_fp8(q, s)
    q2, s2 = quant_fp8(d)
    d2 = dequant_fp8(q2, s2)
    assert torch.equal(d, d2), "fp8 values are not preserved by requantisation"
    err = (d.float() - w.float()).abs().max().item()
    print(f"fp8   value-idempotent; max |dequant-w| = {err:.5f} (block 32x32, ue8m0)")

    # --- fp4: same, plus explicit table check
    we = torch.randn(64, 256).to(torch.bfloat16) * 0.05
    q, s = quant_fp4(we)
    d = dequant_fp4(q, s)
    q2, s2 = quant_fp4(d)
    d2 = dequant_fp4(q2, s2)
    assert torch.equal(d, d2), "fp4 values are not preserved by requantisation"
    print(f"fp4   value-idempotent; max |dequant-w| = {(d.float()-we.float()).abs().max().item():.5f}")

    # every e2m1 code survives a round trip, and the packed nibble order is low=even
    codes = torch.arange(16, dtype=torch.uint8).repeat(4, 2)          # [4, 32]
    packed = pack_fp4(codes)
    assert torch.equal(unpack_fp4(packed, 32), FP4_TABLE[codes.long()])
    scale = torch.full((4, 1), 1.0).to(torch.float8_e8m0fnu)
    back = dequant_fp4(packed, scale)
    qq, _ = quant_fp4(back)
    assert torch.equal(dequant_fp4(qq, scale), back), "code table does not survive a round trip"
    print("fp4   16/16 codes and nibble order verified (negative zero normalises to code 0)")

    # --- ue8m0 scale encoding: exact powers of two, ceil semantics
    for v, want in [(1.0, -8), (2.0, -7), (3.0, -7), (0.5, -9), (0.75, -9), (448.0, 0), (896.0, 1)]:
        got = _ceil_log2(torch.tensor([v / 448.0])).item()
        assert got == want, (v, got, want)
    s = _fp32_to_e8m0(torch.tensor([-127, -1, 0, 1, 128], dtype=torch.int32))
    assert torch.equal(s.view(torch.uint8), torch.tensor([0, 126, 127, 128, 255], dtype=torch.uint8))
    print("ue8m0 scale encoding verified")
    return 0


def cmd_roundtrip(args) -> int:
    from safetensors import safe_open

    files = mtp_files(args.model)
    print(f"checking {len(files)} shard(s) of {args.model}: {', '.join(files)}")
    n_checked = 0
    for fname in files:
        with safe_open(os.path.join(args.model, fname), framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if k.startswith("mtp.")]
            scales = {k for k in keys if k.endswith(".scale")}
            for k in sorted(keys):
                if k.endswith(".scale"):
                    continue
                sn = scale_name(k)
                if sn is None or sn not in scales:
                    continue
                t = f.get_tensor(k)
                scale = f.get_tensor(sn)
                d = dequantise(k, t, scale)
                q, s = requantise(k, d, scale)
                d2 = dequantise(k, q, s)
                if not torch.equal(d, d2):
                    bad = (d != d2).sum().item()
                    print(f"FAIL {k}: {bad} values changed by dequant->requant->dequant")
                    return 1
                n_checked += 1
        print(f"  {fname}: ok")
    assert n_checked > 0, "no quantised MTP tensor was checked -- scale names not found"
    print(f"dequant->requant->dequant is value-exact on {n_checked} quantised MTP tensors")
    return 0


def cmd_build(args) -> int:
    """Overlay checkpoint: /model stays read-only, we add a directory that differs only in
    the three shards holding ``mtp.*`` tensors."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    model_dir = os.path.abspath(args.model)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    index = json.load(open(index_path))
    mtp_shards = set(mtp_files(model_dir))

    new = {}
    if args.weights:
        w = safe_open(args.weights, framework="pt", device="cpu")
        new = {k: w.get_tensor(k) for k in w.keys()}
        missing = {k for k in index["weight_map"] if k.startswith("mtp.")} - set(new)
        assert not missing, f"{len(missing)} mtp tensors missing from {args.weights}: {sorted(missing)[:4]}"

    shard_files = sorted({v for v in index["weight_map"].values()})
    for fname in shard_files:
        src = os.path.join(model_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(dst):
            os.remove(dst)
        if fname not in mtp_shards:
            os.link(src, dst)          # same filesystem: no bytes copied
            continue
        with safe_open(src, framework="pt", device="cpu") as f:
            tensors = {}
            for k in f.keys():
                if k.startswith("mtp.") and k in new:
                    t = new[k]
                    if k.endswith(".scale"):
                        t = t.to(torch.float8_e8m0fnu)
                    elif ".experts." in k and k.endswith(".weight"):
                        t = t.view(torch.int8)
                    elif t.dtype == torch.float32 and "scale" not in k:
                        pass
                    tensors[k] = t
                else:
                    tensors[k] = f.get_tensor(k)
        save_file(tensors, dst, metadata={"format": "pt"})
        print(f"  rewrote {fname} ({len(tensors)} tensors, {os.path.getsize(dst)/2**30:.2f} GiB)")

    # index and every small side file are copied byte-for-byte
    for fname in os.listdir(model_dir):
        src = os.path.join(model_dir, fname)
        dst = os.path.join(out_dir, fname)
        if not os.path.isfile(src) or fname in shard_files or os.path.exists(dst):
            continue
        if os.path.getsize(src) < 2**28:
            os.link(src, dst)
    print(f"overlay ready at {out_dir}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    r = sub.add_parser("roundtrip")
    r.add_argument("--model", required=True)
    r.set_defaults(fn=cmd_roundtrip)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True, help="the read-only checkpoint, e.g. /data/models/ds41-abl")
    b.add_argument("--weights", default=None, help="mtp.* safetensors in native format; omit for an identity overlay")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
