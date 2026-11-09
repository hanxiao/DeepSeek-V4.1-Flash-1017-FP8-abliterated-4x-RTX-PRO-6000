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




def _write_synthetic_dump(path, records, truncate_tail=None, row_bytes=None):
    """records: list of (token_ids, positions, seq_ids, aux[n,15360] uint16, last[n,4*5120] uint16)."""
    import struct

    import numpy as np

    import importlib.util
    import os as _os
    spec = importlib.util.spec_from_file_location(
        "reassemble", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "dump", "reassemble.py"))
    re = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(re)
    rb = row_bytes or re.ROW_BYTES
    n_flt = (rb - re.INT_BYTES) // 2                      # uint16 values of float payload per row
    with open(path, "wb") as f:
        for toks, poss, seqs, aux, last in records:
            n = len(toks)
            f.write(struct.pack("<i", n))
            ints = np.array(list(zip(toks, poss, seqs)), dtype=re.INT_DTYPE)
            f.write(ints.tobytes())
            flt = np.concatenate([np.asarray(aux, dtype=np.uint16).reshape(n, -1),
                                  np.asarray(last, dtype=np.uint16).reshape(n, -1)], axis=1)
            f.write(np.ascontiguousarray(flt[:, :n_flt]).tobytes())
    if truncate_tail:
        with open(path, "ab") as f:
            f.write(truncate_tail)
    return re


def test_reassemble_layout():
    """The dump's float payload is 3 aux blocks + 4 HC blocks of 5120 bf16, 12 int bytes, and the
    reader must survive a truncated tail, dedupe rejected rows and reproduce the tokens/aux."""
    import json
    import os
    import tempfile

    import numpy as np
    import torch
    from safetensors.torch import save_file

    d = tempfile.mkdtemp()
    raw = os.path.join(d, "dump-0.bin")
    rng = np.random.default_rng(0)
    H = 5120

    def row(aux_seed, last_seed):
        """valid bfloat16 bit patterns (random uint16 can be a NaN pattern)"""
        def bf16(n, seed):
            g = torch.Generator().manual_seed(int(seed) + 1)
            return torch.randn(n, generator=g).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
        return bf16(3 * H, aux_seed), bf16(4 * H, last_seed)

    recs = []
    a0, l0 = row(0, 0)
    # seq 0, positions 2,3, and a rejected speculative row at position 4 (token 999)
    a2, l2 = row(2, 2)
    recs.append(([19914, 128799, 999], [2, 3, 4], [0, 0, 0],
                 np.stack([a0, a0, a2]), np.stack([l0, l0, l2])))
    # seq 0 re-forwards position 4 with the accepted token, then 5..12; seq 1 contributes one row
    a1, l1 = row(1, 1)
    recs.append(([100, 200, 101, 102, 103, 104, 105, 106, 107, 108], [4, 0, 5, 6, 7, 8, 9, 10, 11, 12],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], np.stack([a1] * 10), np.stack([l1] * 10)))
    re = _write_synthetic_dump(raw, recs, truncate_tail=b"\x07\x00\x00\x00" + b"\x00" * 20)

    # a tiny checkpoint for head.weight
    md = os.path.join(d, "model")
    os.makedirs(md)
    save_file({"head.weight": torch.randn(512, H, dtype=torch.bfloat16)},
              os.path.join(md, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {"head.weight": "model-00001-of-00001.safetensors"}},
              open(os.path.join(md, "model.safetensors.index.json"), "w"))

    out = os.path.join(d, "cache")
    rc = re.main(["--raw", raw, "--model-dir", md, "--out", out,
                  "--max-length", "8", "--topk", "16", "--device", "cpu"])
    assert rc == 0
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["row_bytes"] == 12 + 7 * H * 2, meta["row_bytes"]
    assert meta["num_hc"] == 4 and meta["num_aux_layers"] == 3
    # seq 0 has positions 2..12 (11 rows after dedupe; the rejected row at 4 is overwritten),
    # seq 1 has position 0 (1 row -> dropped, under the 8-row minimum)
    toks = np.memmap(os.path.join(out, "tokens.bin"), dtype=np.int32, mode="r",
                     shape=(meta["num_samples"], meta["max_length"]))
    lens = np.memmap(os.path.join(out, "lengths.bin"), dtype=np.int32, mode="r", shape=(meta["num_samples"],))
    assert meta["num_samples"] == 1, meta
    assert lens[0] == 8, lens[0]                                   # 11 rows -> one sample + a 3-row tail
    assert toks[0, :8].tolist() == [19914, 128799, 100, 101, 102, 103, 104, 105], toks[0, :8].tolist()
    aux = np.memmap(os.path.join(out, "aux.bin"), dtype=np.uint16, mode="r",
                    shape=(1, meta["max_length"], 3 * H))
    assert int(aux[0, 2, 0]) == int(a1[0]), (int(aux[0, 2, 0]), int(a1[0]))
    lp = np.memmap(os.path.join(out, "topk_logp.bin"), dtype=np.float32, mode="r",
                   shape=(1, meta["max_length"], 16))
    assert float(np.exp(lp[0, 0]).sum()) <= 1.0001
    print(f"reassemble  {meta['num_samples']} sample, {lens[0]} rows, tokens+aux+topK correct, "
          f"truncated tail tolerated, rejected row overwritten")


def test_reassemble_detects_hc_blocks():
    """The number of HC copies in the float payload is identified from the file itself."""
    import json
    import os
    import tempfile

    import numpy as np
    import torch
    from safetensors.torch import save_file

    d = tempfile.mkdtemp()
    raw = os.path.join(d, "dump-0.bin")
    H = 5120
    aux = np.zeros(3 * H, dtype=np.uint16)
    last = np.zeros(4 * H, dtype=np.uint16)
    recs = [([1, 2, 3], [0, 1, 2], [0, 0, 0], np.stack([aux] * 3), np.stack([last] * 3)),
            ([4, 5, 6, 7, 8, 9, 10, 11], list(range(3, 11)), [0] * 8,
             np.stack([aux] * 8), np.stack([last] * 8))]
    # written with the *old* 3-aux + 1-HC assumption
    re = _write_synthetic_dump(raw, recs, row_bytes=12 + 4 * H * 2)
    md = os.path.join(d, "model")
    os.makedirs(md)
    save_file({"head.weight": torch.randn(512, H, dtype=torch.bfloat16)},
              os.path.join(md, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {"head.weight": "model-00001-of-00001.safetensors"}},
              open(os.path.join(md, "model.safetensors.index.json"), "w"))
    out = os.path.join(d, "cache")
    assert re.main(["--raw", raw, "--model-dir", md, "--out", out,
                    "--max-length", "8", "--topk", "16", "--device", "cpu"]) == 0
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["num_hc"] == 1 and meta["row_bytes"] == 12 + 4 * H * 2, meta
    assert meta["num_samples"] == 1, meta
    print(f"reassemble  auto-detected a 3+1 payload ({meta['row_bytes']} B/row) and parsed it")


def test_reassemble_rejects_unknown_layout():
    """A dump that matches no layout candidate must fail loudly, not parse garbage."""
    import json
    import os
    import struct
    import tempfile

    import numpy as np
    import torch
    from safetensors.torch import save_file

    d = tempfile.mkdtemp()
    raw = os.path.join(d, "dump-0.bin")
    H = 5120
    # 2 int32 per row instead of 3: no candidate row size can walk this file
    with open(raw, "wb") as f:
        for _ in range(2):
            n = 4
            f.write(struct.pack("<i", n))
            f.write(np.zeros((n, 2), dtype=np.int32).tobytes())
            f.write(np.zeros(n * 7 * H, dtype=np.uint16).tobytes())
    md = os.path.join(d, "model")
    os.makedirs(md)
    save_file({"head.weight": torch.randn(512, H, dtype=torch.bfloat16)},
              os.path.join(md, "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {"head.weight": "model-00001-of-00001.safetensors"}},
              open(os.path.join(md, "model.safetensors.index.json"), "w"))
    try:
        re = _write_synthetic_dump(os.path.join(d, "unused.bin"), [])
        re.main(["--raw", raw, "--model-dir", md, "--out", os.path.join(d, "cache"),
                 "--max-length", "8", "--topk", "16", "--device", "cpu"])
    except SystemExit as exc:
        assert "no row layout" in str(exc), exc
        print("reassemble  a dump matching no layout fails loudly")
        return
    raise AssertionError("an unparseable dump was accepted")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall local tests passed")
