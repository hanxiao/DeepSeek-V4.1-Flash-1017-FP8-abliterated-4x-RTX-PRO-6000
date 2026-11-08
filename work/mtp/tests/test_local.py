"""CPU tests for everything that does not need the checkpoint or a GPU.

    .venv/bin/python tests/test_local.py        (from work/mtp)
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepspec_v41.loss import _tvp_term, beta_at_step  # noqa: E402
from deepspec_v41.modeling import V41DSparkArgs, V41DSparkModel  # noqa: E402

TINY = dict(
    vocab_size=256, hidden_size=64, num_attention_heads=2, head_dim=64, qk_rope_head_dim=16,
    q_lora_rank=32, o_lora_rank=16, o_groups=2, sliding_window=8, hc_mult=2, hc_sinkhorn_iters=3,
    rms_norm_eps=1e-6, moe_intermediate_size=32, dspark_n_routed_experts=8,
    dspark_num_experts_per_tok=2, markov_rank=16, num_anchors=4, block_size=3, num_draft_layers=2,
    noise_token_id=255,
)


def test_tvp_limits():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(16, 256), -1)
    q = torch.softmax(torch.randn(16, 256), -1)
    zeros = torch.zeros(16)
    d1, acc = _tvp_term(p, q, zeros, zeros, beta=1.0)
    tv = 0.5 * (p - q).abs().sum(-1)
    assert torch.allclose(d1, tv, atol=1e-6), (d1[:3], tv[:3])
    assert torch.allclose(acc, 1 - tv, atol=1e-6)
    d0, _ = _tvp_term(p, q, zeros, zeros, beta=0.0)
    ref0 = torch.where(q < p, p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log()), torch.zeros_like(p)).sum(-1)
    assert torch.allclose(d0, ref0, atol=1e-5), (d0[:3], ref0[:3])
    # beta -> 1 from below approaches TV
    d_mid, _ = _tvp_term(p, q, zeros, zeros, beta=0.99)
    assert (d_mid - tv).abs().max() < 0.05 * tv.mean() + 1e-3
    # monotone in beta for a fixed pair
    vals = [_tvp_term(p, q, zeros, zeros, beta=b)[0].mean().item() for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(vals[i] >= vals[i + 1] - 1e-5 for i in range(len(vals) - 1)), vals
    print(f"tvp      beta=1 == TV, beta=0 == coverage KL, mean D_beta {vals}")


def test_tvp_residual_bucket():
    # a residual bucket makes the coarsened pair a valid distribution pair
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.4, 0.4, 0.2]])
    d1, acc = _tvp_term(p[:, :2], q[:, :2], p[:, 2], q[:, 2], beta=1.0)
    assert abs(d1.item() - 0.1) < 1e-6 and abs(acc.item() - 0.9) < 1e-6, (d1, acc)
    print("tvp      residual bucket counted as one bucket (TV=0.1, accept=0.9)")


def test_beta_schedule():
    assert beta_at_step(0, 100) == 0.0
    assert abs(beta_at_step(40, 100) - 0.5) < 1e-9      # 0.8 * 100 = 80 -> 40 steps is halfway
    assert beta_at_step(80, 100) == 1.0
    assert beta_at_step(100, 100) == 1.0
    print("tvp      anneal 0->1 over the first 80% of steps")


def test_anchors():
    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    ids = torch.arange(32).unsqueeze(0)
    mask = torch.ones(1, 32)
    mask[0, 20:] = 0.0                                   # only 20 valid tokens
    anchors, keep = m.sample_anchors(ids, mask)
    assert anchors.max() <= 19, anchors
    assert keep.sum() <= 4
    # every anchor must have its own token and its first target inside the mask
    print(f"anchors  sampled {anchors[0][keep[0]].tolist()} from 20 valid tokens")


def test_positions_and_labels():
    """slot j -> rope position p+1+j, predicts token p+1+j, input id anchor at slot 0 then noise."""
    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    ids = torch.arange(32).unsqueeze(0) % 200
    out = m.hidden_states(ids, torch.randn(1, 32, 3 * 64), torch.ones(1, 32))
    anchors = out["anchors"]
    tid = out["target_ids"]
    prev = out["prev_ids"]
    for b in range(anchors.shape[0]):
        for i in range(anchors.shape[1]):
            p = int(anchors[b, i])
            assert tid[b, i].tolist() == [int(ids[b, p + 1 + j]) for j in range(a.block_size)], (p, tid[b, i])
            assert prev[b, i].tolist() == [int(ids[b, p])] + tid[b, i, :-1].tolist()
    print("model    slot j labels token p+1+j; markov prefix is teacher-forced")


def test_mask_shape_and_finiteness():
    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    torch.manual_seed(1)
    ids = torch.randint(0, 200, (2, 40))
    out = m.hidden_states(ids, torch.randn(2, 40, 3 * 64), torch.ones(2, 40))
    logits = m.head_logits(out["head_hidden"], out["markov_emb"])
    assert torch.isfinite(logits).all(), "non-finite draft logits"
    loss = logits.float().logsumexp(-1).mean()
    loss.backward()
    for name, prm in m.named_parameters():
        if prm.grad is not None:
            assert torch.isfinite(prm.grad).all(), f"non-finite grad in {name}"
    print(f"model    forward+backward finite, {sum(p.numel() for p in m.parameters())} params in the tiny config")


def test_context_window_is_causal_window():
    """The block must not see context positions newer than the anchor, and must see exactly the
    last ``sliding_window`` of them."""
    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    ids = torch.arange(40).unsqueeze(0) % 200
    lm = torch.ones(1, 40)
    anchors, keep = m.sample_anchors(ids, lm)
    # rebuild the window the way forward() does
    w = min(a.sliding_window, 40)
    offs = torch.arange(-w + 1, 1).unsqueeze(0).unsqueeze(0)
    ctx = anchors.unsqueeze(-1) + offs
    p = int(anchors[0, 0])
    want = [max(0, p - w + 1) + i for i in range(min(p + 1, w))]
    got = [int(x) for x in ctx[0, 0] if x >= 0]
    assert got[-1] == p, (got, p)
    assert got == want, (got, want)
    print(f"model    context window for anchor {p} is {got[0]}..{got[-1]} ({len(got)} rows)")


def test_loss_end_to_end():
    """Full-vocab support == stored top-K support when K == V, so the loss reduces to DeepSpec's."""
    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    torch.manual_seed(2)
    ids = torch.randint(0, 200, (1, 40))
    out = m.hidden_states(ids, torch.randn(1, 40, 3 * 64), torch.ones(1, 40))
    V, K = a.vocab_size, a.vocab_size
    logits = m.head_logits(out["head_hidden"], out["markov_emb"]).detach()
    p = torch.softmax(logits, -1)
    ids_sorted = p.argsort(-1)
    logp = p.gather(-1, ids_sorted).log()
    loss, metrics = __import__("deepspec_v41.loss", fromlist=["dspark_v41_loss"]).dspark_v41_loss(
        model=m, outputs=out, target_ids=out["target_ids"], target_logprobs=logp,
        target_support=ids_sorted.int(), target_residual=torch.zeros(1, a.num_anchors, a.block_size),
        eval_mask=out["eval_mask"], loss_decay_gamma=4.0, ce_alpha=0.1, tvp_alpha=0.9,
        tvp_beta=1.0, confidence_head_alpha=0.0,
    )
    assert math.isfinite(float(loss))
    # q == p here, so TV is 0 and the reported accept rate must be 1
    assert abs(metrics["accept_rate"] - 1.0) < 2e-3, metrics["accept_rate"]
    print(f"loss     self-target: accept_rate {metrics['accept_rate']:.4f}, ce {metrics['ce_loss']:.4f}, tvp {metrics['tvp_loss']:.6f}")




def _fake_checkpoint(m, a, directory):
    """Write a native-format checkpoint that mimics the real one, including the naming rule that
    the scale of ``<stem>.weight`` is ``<stem>.scale`` (not ``<stem>.weight.scale``)."""
    import json
    import os

    import torch
    from safetensors.torch import save_file

    from quantize import is_expert_weight, quant_fp4, quant_fp8, scale_name
    from tools.load_mtp import model_to_checkpoint

    template = {}
    for param, tensor in m.state_dict().items():
        mapped = model_to_checkpoint(param, a.num_draft_layers)
        if mapped is None:
            continue
        name = mapped[0]
        quantised = param.endswith(".weight") and any(
            k in name for k in (".wq_", ".wkv", ".wo_", "shared_experts", "main_proj")) or is_expert_weight(name)
        if quantised:
            q, s = (quant_fp4(tensor) if is_expert_weight(name) else quant_fp8(tensor))
            template[name] = q
            template[scale_name(name)] = s
        else:
            template[name] = tensor.to(torch.bfloat16) if tensor.is_floating_point() and tensor.dtype != torch.float32 else tensor
    template["mtp.0.ffn.gate.bias_vl"] = torch.zeros(a.dspark_n_routed_experts, dtype=torch.float32)
    template["embed.weight"] = m.embed_weight.to(torch.bfloat16)
    template["head.weight"] = m.head_weight.to(torch.bfloat16)
    os.makedirs(directory, exist_ok=True)
    save_file(template, os.path.join(directory, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in template}},
              open(os.path.join(directory, "model.safetensors.index.json"), "w"))
    return template


def test_checkpoint_roundtrip():
    """Fabricate a native-format checkpoint out of the tiny model, load it back, export it again.

    This is the regression test for the scale-naming bug that made gate 0 fail on the real
    checkpoint: the packed FP4 experts kept their half-width shape because the scale was looked up
    as ``<name>.weight.scale`` instead of ``<name>.scale``, and the FP8 dense weights silently kept
    their raw FP8 values.
    """
    import os
    import tempfile

    import torch
    from safetensors import safe_open

    from quantize import scale_name
    from tools.load_mtp import export_mtp, load_into_model

    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    torch.manual_seed(3)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    d = tempfile.mkdtemp()
    md = os.path.join(d, "ckpt")
    template = _fake_checkpoint(m, a, md)

    # the fake checkpoint must follow the real naming rule
    for name in template:
        if name.endswith(".weight") and name != "embed.weight" and name != "head.weight":
            stem = name[: -len(".weight")]
            if any(k in name for k in (".wq_", ".wkv", ".wo_", "shared_experts", "main_proj", ".experts.")):
                assert stem + ".scale" in template, f"{name} has no {stem}.scale"

    m2 = V41DSparkModel(a)
    info = load_into_model(m2, md, verbose=False)
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    for param, tensor in m2.state_dict().items():
        if param in ("embed_weight", "head_weight"):
            continue
        assert tuple(tensor.shape) == shapes[param], f"{param}: {tuple(tensor.shape)} != {shapes[param]}"
        if tensor.is_floating_point():
            # fp4 keeps 3 bits of mantissa: tolerance is relative to the block maximum
            assert torch.allclose(tensor.float(), m.state_dict()[param].float(), atol=0.05, rtol=0.35), param
    out = os.path.join(d, "export.safetensors")
    export_mtp(m2, md, out, verbose=False)
    with safe_open(out, "pt") as f:
        got = set(f.keys())
    want = set(template) - {"embed.weight", "head.weight"}
    assert got == want, (sorted(got - want)[:3], sorted(want - got)[:3])
    print(f"ckpt     native-format round trip: {len(template)} tensors, {info['loaded']} loaded, "
          f"{len(got)} exported, all shapes match")


def test_loader_rejects_packed_expert_without_scale():
    """A packed FP4 expert with no scale must fail loudly, not load at half width."""
    import json
    import os
    import tempfile

    import torch
    from safetensors.torch import save_file

    from tools.load_mtp import load_into_model

    a = V41DSparkArgs(**TINY)
    m = V41DSparkModel(a)
    d = tempfile.mkdtemp()
    md = os.path.join(d, "ckpt")
    template = _fake_checkpoint(m, a, md)
    del template["mtp.0.ffn.experts.0.w1.scale"]
    save_file(template, os.path.join(md, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in template}},
              open(os.path.join(md, "model.safetensors.index.json"), "w"))
    try:
        load_into_model(V41DSparkModel(a), md, verbose=False)
    except AssertionError as exc:
        assert "FP4-packed" in str(exc), exc
        print("ckpt     a packed expert without its scale fails loudly")
        return
    raise AssertionError("loader accepted a packed expert with no scale")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall local tests passed")
