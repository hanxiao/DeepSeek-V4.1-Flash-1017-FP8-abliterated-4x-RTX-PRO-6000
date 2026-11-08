#!/usr/bin/env python3
"""Checkpoint <-> training-model weight plumbing for the DSpark MTP head.

``load``   dequantise the checkpoint's ``mtp.*`` tensors (FP8 dense, FP4 experts, ue8m0 scales) into
           the bf16 training model.
``export`` take the trained model, requantise every tensor it owns back to the checkpoint's native
           format and write a complete ``mtp.*`` safetensors file (2401 tensors: the ones the model
           owns are rewritten, the rest -- e.g. the unused vision bias -- are copied from the
           original) ready for ``quantize.py build``.

The rename table is the only V4.1-specific thing here; every shape comes from the checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize import (  # noqa: E402
    dequant_fp4, dequant_fp8, is_expert_weight, quant_fp4, quant_fp8, scale_name,
)


def _mtp_index(model_dir: str) -> dict[str, str]:
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    return {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}


def read_mtp_tensors(model_dir: str) -> dict[str, torch.Tensor]:
    """All 2401 ``mtp.*`` tensors of the checkpoint, as stored (quantised)."""
    weight_map = _mtp_index(model_dir)
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in weight_map:
                    out[k] = f.get_tensor(k)
    assert len(out) == len(weight_map), (len(out), len(weight_map))
    return out


def read_shared_embed_head(model_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``embed.weight`` and ``head.weight`` (bf16, shared with the target)."""
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    out = {}
    for name in ("embed.weight", "head.weight"):
        shard = index["weight_map"][name]
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(name)
    return out["embed.weight"], out["head.weight"]


# --------------------------------------------------------------------------------------
# name mapping
# --------------------------------------------------------------------------------------
# checkpoint stem -> model parameter name.  A stem is the ``mtp.{i}.`` suffix with any
# ``.weight`` / ``.scale`` removed; the checkpoint stores the quantised weights under
# ``<stem>.weight`` + ``<stem>.scale``, and the unquantised tensors under the bare stem.
_LAYER_MAP = {
    "attn_norm": "attn_norm.weight",
    "ffn_norm": "ffn_norm.weight",
    "attn.wq_a": "attn.wq_a.weight",
    "attn.q_norm": "attn.q_norm.weight",
    "attn.wq_b": "attn.wq_b.weight",
    "attn.wkv": "attn.wkv.weight",
    "attn.kv_norm": "attn.kv_norm.weight",
    "attn.wo_a": "attn.wo_a.weight",
    "attn.wo_b": "attn.wo_b.weight",
    "attn.attn_sink": "attn.attn_sink",
    "ffn.gate": "ffn.gate.weight",
    "ffn.gate.bias": "ffn.gate_bias",
    "ffn.shared_experts.w1": "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2": "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3": "ffn.shared_experts.w3.weight",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
}
_HEAD_ONLY = {          # last draft layer only
    "norm": "norm.weight",
    "markov_head.embed": "markov_embed.weight",
    "markov_head.head": "markov_head.weight",
    "confidence_head.proj": "confidence_head.weight",
}
_FIRST_ONLY = {         # first draft layer only
    "main_norm": "main_norm.weight",
    "main_proj": "main_proj.weight",
}
_MODEL_TO_CKPT = {m: c for c, m in {**_LAYER_MAP, **_HEAD_ONLY, **_FIRST_ONLY}.items()}



def _split_suffix(rest: str) -> tuple[str, str]:
    for suffix in (".scale", ".weight"):
        if rest.endswith(suffix):
            return rest[: -len(suffix)], suffix[1:]
    return rest, ""


def checkpoint_to_model(name: str, num_layers: int) -> tuple[str, str] | None:
    """``mtp.{i}.<stem>[.weight|.scale]`` -> (model parameter name, 'weight'|'scale'), or None."""
    if not name.startswith("mtp."):
        return None
    _, stage, rest = name.split(".", 2)
    stage = int(stage)
    stem, kind = _split_suffix(rest)
    if stem in _FIRST_ONLY:
        return None if stage != 0 else (_FIRST_ONLY[stem], kind or "weight")
    if stem in _HEAD_ONLY:
        return None if stage != num_layers - 1 else (_HEAD_ONLY[stem], kind or "weight")
    if stem.startswith("ffn.experts."):
        _, _, expert_id, leaf = stem.split(".", 3)
        return f"blocks.{stage}.ffn.experts.{expert_id}.{leaf}.weight", kind or "weight"
    if stem in _LAYER_MAP:
        return f"blocks.{stage}.{_LAYER_MAP[stem]}", kind or "weight"
    return None


def model_to_checkpoint(param: str, num_layers: int) -> tuple[str, str] | None:
    """Model parameter name -> (checkpoint tensor name, 'weight').  Scales are re-derived on export."""
    if param in ("embed_weight", "head_weight", "embed_tokens.weight", "lm_head.weight"):
        return None
    if param.startswith("blocks."):
        _, stage, rest = param.split(".", 2)
        m = re.match(r"ffn\.experts\.(\d+)\.(w[123])\.weight$", rest)
        if m:
            return f"mtp.{stage}.ffn.experts.{m.group(1)}.{m.group(2)}.weight", "weight"
        if rest in _MODEL_TO_CKPT:
            stem_ck = _MODEL_TO_CKPT[rest]
            return f"mtp.{stage}.{stem_ck}" + (".weight" if rest.endswith(".weight") else ""), "weight"
        return None
    if param in _MODEL_TO_CKPT:
        stem_ck = _MODEL_TO_CKPT[param]
        stage = 0 if stem_ck in _FIRST_ONLY else num_layers - 1
        return f"mtp.{stage}.{stem_ck}" + (".weight" if param.endswith(".weight") else ""), "weight"
    return None


# --------------------------------------------------------------------------------------
# load / export
# --------------------------------------------------------------------------------------
def load_into_model(model, model_dir: str, verbose: bool = True) -> dict:
    """Dequantise the checkpoint's MTP tensors into ``model`` in place."""
    tensors = read_mtp_tensors(model_dir)
    embed, head = read_shared_embed_head(model_dir)
    state = {"embed_weight": embed.to(torch.bfloat16), "head_weight": head.to(torch.bfloat16)}
    n_layers = len(model.blocks)
    param_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    consumed = set()
    for name, t in tensors.items():
        mapped = checkpoint_to_model(name, n_layers)
        if mapped is None:
            continue
        target, kind = mapped
        if kind == "scale":
            continue
        sn = scale_name(name)
        if sn is not None and sn in tensors:
            scale = tensors[sn]
            value = dequant_fp4(t, scale) if is_expert_weight(name) else dequant_fp8(t, scale)
        else:
            assert not is_expert_weight(name), f"{name} is FP4-packed but {sn} is missing from the checkpoint"
            value = t
        if value.is_floating_point() and value.dtype != torch.float32:
            value = value.to(torch.bfloat16)
        if target in param_shapes:
            assert tuple(value.shape) == param_shapes[target], (
                f"{name} -> {target}: dequantised shape {tuple(value.shape)} != model {param_shapes[target]}"
                f" (quantised checkpoint shape was {tuple(t.shape)})")
        state[target] = value
        consumed.add(name)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        print(f"loaded {len(state)} tensors from {model_dir} "
              f"({len(consumed)} checkpoint weights dequantised); "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", sorted(missing)[:8])
        if unexpected:
            print("  unexpected:", sorted(unexpected)[:8])
    assert not [m for m in missing if m not in ("embed_weight", "head_weight")], missing
    return {"loaded": len(state), "missing": missing, "unexpected": unexpected}


@torch.no_grad()
def export_mtp(model, model_dir: str, out_path: str, verbose: bool = True) -> str:
    """Write a complete ``mtp.*`` safetensors file in the checkpoint's native quantised format."""
    template = read_mtp_tensors(model_dir)
    n_layers = len(model.blocks)
    owned = {}
    for param, tensor in model.state_dict().items():
        mapped = model_to_checkpoint(param, n_layers)
        if mapped is not None:
            owned[mapped[0]] = (tensor, mapped[1])

    out = {}
    n_written = 0
    for name, t in template.items():
        if name.endswith(".scale"):
            continue
        sn = scale_name(name)
        scale = template.get(sn) if sn is not None else None
        value = owned.get(name, (None, None))[0]
        if value is None:
            out[name] = t
            if scale is not None:
                out[sn] = scale
            continue
        if scale is None:
            out[name] = value.to(t.dtype) if t.dtype in (torch.float32, torch.bfloat16) else value.to(torch.bfloat16)
        else:
            value = value.to(torch.bfloat16)
            q, s = quant_fp4(value) if is_expert_weight(name) else quant_fp8(value)
            out[name] = q
            out[sn] = s.to(torch.float8_e8m0fnu)
        n_written += 1
    missing = set(owned) - set(out)
    assert not missing, f"{len(missing)} trained tensors had no template slot: {sorted(missing)[:5]}"
    assert set(template) - set(out) == set(), f"missing {sorted(set(template) - set(out))[:5]}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_file(out, out_path, metadata={"format": "pt", "source": "mtp-train"})
    if verbose:
        print(f"exported {n_written} trained tensors + {len(out) - n_written} copied, "
              f"{len(out)} total -> {out_path} ({os.path.getsize(out_path) / 2**30:.2f} GiB)")
    return out_path


__all__ = ["load_into_model", "export_mtp", "read_mtp_tensors", "checkpoint_to_model", "model_to_checkpoint"]
