"""DeepSeek-V4.1 DSpark draft head, training-side implementation.

This is the same module the engine serves (the checkpoint's ``mtp.0/1/2`` tensors), written so it
can be trained:

  * it reads the target's aux hidden states at layers 37/38/39 and projects them with
    ``main_proj``/``main_norm`` into the single ``main_x`` every draft layer uses for its context
    KV -- exactly like ``precompute_and_store_context_kv`` in
    ``vllm/models/deepseek_v41/nvidia/dspark.py``;
  * each draft layer is a V4.1 decoder block with mHC residual streams, MLA (one shared KV head,
    K == V, partial RoPE on the trailing 64 dims, per-head sink, grouped low-rank output) and a
    128-expert top-3 MoE;
  * the block of ``block_size`` query slots per anchor attends to the last ``sliding_window``
    (128) context positions plus the whole block, bidirectionally -- the non-causal DSpark mask;
  * the Markov head adds a rank-256 logit bias chained over the block, and the confidence head
    predicts the per-position acceptance probability.

Position convention (verified against both the checkpoint's reference implementation and the
engine's prepare-inputs kernel): anchor at token index ``p`` -> query slot ``j`` has RoPE position
``p + 1 + j`` and predicts the token at index ``p + 1 + j``. Slot 0's input embedding is the anchor
token, slots 1..k-1 are the noise token.

Tensor names mirror the checkpoint so the loader in ``tools/load_mtp.py`` is a rename table.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------
@dataclass
class V41DSparkArgs:
    # backbone geometry (from the checkpoint's text_config)
    vocab_size: int = 129280
    hidden_size: int = 5120
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-20
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    rope_theta: float = 10000.0
    # draft geometry
    num_draft_layers: int = 3
    target_layer_ids: tuple[int, ...] = (37, 38, 39)
    block_size: int = 9
    noise_token_id: int = 128799
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    moe_intermediate_size: int = 2304
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    markov_rank: int = 256
    enable_confidence_head: bool = True
    # training knobs
    num_anchors: int = 512
    position_offset: int = 1          # slot j -> rope position anchor + position_offset + j

    @classmethod
    def from_checkpoint_config(cls, path: str, **overrides) -> "V41DSparkArgs":
        raw = json.load(open(path))
        text = raw.get("text_config", raw)
        args = cls()
        for f in args.__dataclass_fields__:
            if f in text:
                setattr(args, f, text[f])
        for k, v in overrides.items():
            if not hasattr(args, k):
                raise KeyError(f"unknown V41DSparkArgs field {k}")
            setattr(args, k, v)
        args.target_layer_ids = tuple(int(i) for i in args.target_layer_ids)
        return args


# ----------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.float32)


class HyperConnection(nn.Module):
    """Manifold-constrained hyper-connections (mHC).

    ``fn``/``base``/``scale`` produce the collapse (pre), expand (post) and mix (comb) weights from
    the ``hc_mult`` residual streams; comb is projected onto doubly-stochastic matrices with
    Sinkhorn-Knopp.  Shapes match the checkpoint: ``fn`` [ (2+hc)*hc, hc*hidden ], ``base``
    [(2+hc)*hc], ``scale`` [3].
    """

    def __init__(self, args: V41DSparkArgs):
        super().__init__()
        hc = args.hc_mult
        self.hc_mult = hc
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.input_norm = UnweightedRMSNorm(args.rms_norm_eps)
        self.fn = nn.Parameter(torch.empty((2 + hc) * hc, hc * args.hidden_size))
        self.base = nn.Parameter(torch.empty((2 + hc) * hc))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, streams: torch.Tensor):
        b, s = streams.shape[:2]
        hc = self.hc_mult
        flat = self.input_norm(streams.reshape(b, s, -1))
        mixes = F.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)
        comb_b = comb_b.view(hc, hc)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
        return post, comb, collapsed, pre


class DSparkAttention(nn.Module):
    """MLA with a DSpark context: the block of ``block_size`` queries attends to a 128-token
    sliding window of ``main_x``-derived KV plus every slot of its own block."""

    def __init__(self, args: V41DSparkArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.scaling = self.head_dim**-0.5
        self.rope_theta = args.rope_theta

        self.wq_a = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, args.rms_norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.o_groups, self.o_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, args.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))

    # --- rope (interleaved pairs, trailing rope_dim channels) ---------------------------
    @staticmethod
    def _rope_tables(positions: torch.Tensor, dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim))
        freqs = positions.float().unsqueeze(-1) * inv
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Interleaved RoPE on the trailing rope slice of ``x``.

        ``cos``/``sin`` carry one entry per interleaved pair and must broadcast against
        ``x[..., :-rope_dim]`` (pass a trailing singleton dim for the head axis).
        """
        half = cos.shape[-1]
        rope = x[..., -2 * half:]
        c = cos.repeat_interleave(2, dim=-1)
        s = sin.repeat_interleave(2, dim=-1)
        if inverse:
            s = -s
        rot = torch.stack((-rope[..., 1::2], rope[..., 0::2]), dim=-1).flatten(-2)
        out = rope.float() * c + rot.float() * s
        return torch.cat([x[..., : -2 * half], out.to(x.dtype)], dim=-1)

    def context_kv(self, main_x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, head_dim] rotated context KV (K == V in V4)."""
        kv = self.kv_norm(self.wkv(main_x))
        cos, sin = self._rope_tables(positions, self.rope_dim, self.rope_theta, kv.dtype)
        return self._rotate(kv, cos, sin)

    def forward(
        self,
        x: torch.Tensor,              # [B, A, k, D] block hidden states
        ctx_kv: torch.Tensor,         # [B, T, head_dim] context KV for the whole sequence
        ctx_index: torch.Tensor,      # [B, A, W] indices into T, -1 = pad
        block_positions: torch.Tensor,  # [B, A, k] rope positions of the query slots
        ctx_positions: torch.Tensor,  # [B, T] rope positions of the context tokens
    ) -> torch.Tensor:
        b, a, k, d = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(b, a, k, self.n_heads, self.head_dim)
        qcos, qsin = self._rope_tables(block_positions, self.rope_dim, self.rope_theta, q.dtype)
        q = self._rotate(q, qcos.unsqueeze(-2), qsin.unsqueeze(-2))      # [B,A,k,H,dim]

        kv_blk = self.kv_norm(self.wkv(x)).view(b, a, k, self.head_dim)
        kv_blk = self._rotate(kv_blk, qcos, qsin)                        # block KV at query positions

        # gather the 128-token window per anchor: [B, A, W, dim]
        w = ctx_index.shape[-1]
        safe = ctx_index.clamp_min(0)
        kv_ctx = torch.gather(ctx_kv.unsqueeze(1).expand(-1, a, -1, -1), 2,
                              safe.unsqueeze(-1).expand(-1, -1, -1, ctx_kv.shape[-1]))
        ctx_valid = ctx_index >= 0

        # attention: query [B,A,k,H,dim] against [B,A,W+k,dim] (single KV head)
        kv = torch.cat([kv_ctx, kv_blk], dim=2)                          # [B,A,W+k,dim]
        scores = torch.einsum("baknd,bamd->baknm", q.float(), kv.float()) * self.scaling
        mask = torch.cat([
            ctx_valid.unsqueeze(2).expand(-1, -1, k, -1),
            torch.ones(b, a, k, k, dtype=torch.bool, device=x.device),
        ], dim=-1)
        scores = scores.masked_fill(~mask.unsqueeze(3), float("-inf"))
        sink = self.attn_sink.float().view(1, 1, 1, self.n_heads, 1)
        combined = torch.cat([scores, sink.expand(b, a, k, -1, -1)], dim=-1)
        combined = combined - combined.amax(dim=-1, keepdim=True)
        probs = torch.softmax(combined, dim=-1)[..., :-1]                # drop the sink
        out = torch.einsum("baknm,bamd->baknd", probs, kv.float()).to(x.dtype)

        out = self._rotate(out, qcos.unsqueeze(-2), qsin.unsqueeze(-2), inverse=True)
        out = out.reshape(b, a, k, self.o_groups, -1)
        wa = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        out = torch.einsum("bakgd,grd->bakgr", out, wa)
        return self.wo_b(out.flatten(3))


class Expert(nn.Module):
    def __init__(self, hidden: int, inter: int, swiglu_limit: float):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)
        self.w3 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.w2(F.silu(gate) * up)


class MoE(nn.Module):
    """128 routed experts top-3 (sqrtsoftplus scores, correction bias for selection only) plus one
    shared expert.  Experts are stored as ModuleLists so the checkpoint's per-expert tensors load
    without a stacking step."""

    def __init__(self, args: V41DSparkArgs):
        super().__init__()
        self.hidden = args.hidden_size
        self.n_experts = args.dspark_n_routed_experts
        self.top_k = args.dspark_num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(self.hidden, self.n_experts, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(self.n_experts, dtype=torch.float32))
        self.experts = nn.ModuleList(
            [Expert(self.hidden, args.moe_intermediate_size, args.swiglu_limit) for _ in range(self.n_experts)]
        )
        self.shared_experts = Expert(self.hidden, args.moe_intermediate_size, args.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden)
        scores = F.softplus(F.linear(flat.float(), self.gate.weight.float())).sqrt()
        indices = (scores + self.gate_bias.float()).topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = (weights * self.route_scale).to(x.dtype)
        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            rows, slots = torch.where(indices == e)
            if rows.numel() == 0:
                continue
            y = self.experts[e](flat[rows]) * weights[rows, slots, None]
            out.index_add_(0, rows, y)
        return (out + self.shared_experts(flat)).view(shape)


class DSparkBlock(nn.Module):
    """One draft layer. Hidden state is [B, A, k, hc, D]: A anchors x k query slots x hc streams."""

    def __init__(self, args: V41DSparkArgs):
        super().__init__()
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)
        self.attn_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.attn = DSparkAttention(args)
        self.ffn = MoE(args)

    def forward(self, streams, ctx_kv, ctx_index, block_positions, ctx_positions):
        """streams [B, A, k, hc, D]; mHC runs on the flattened [B, A*k, hc, D] view."""
        b, a, k, hc, d = streams.shape
        dtype = streams.dtype
        flat = streams.reshape(b, a * k, hc, d)

        post, comb, collapsed, _ = self.attn_hc(flat)
        attn_out = self.attn(self.attn_norm(collapsed).view(b, a, k, d),
                             ctx_kv, ctx_index, block_positions, ctx_positions).reshape(b, a * k, d)
        flat = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), flat
        )

        post, comb, collapsed, pre = self.ffn_hc(flat)
        mlp_out = self.ffn(self.ffn_norm(collapsed))
        flat = post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), flat
        )
        return flat.view(b, a, k, hc, d), pre.view(b, a, k, hc)


# ----------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------
class V41DSparkModel(nn.Module):
    def __init__(self, args: V41DSparkArgs):
        super().__init__()
        self.args = args
        # shared with the target and frozen: buffers, so FSDP replicates them instead of sharding
        self.register_buffer("embed_weight", torch.empty(args.vocab_size, args.hidden_size))
        self.register_buffer("head_weight", torch.empty(args.vocab_size, args.hidden_size))
        self.main_proj = nn.Linear(args.hidden_size * len(args.target_layer_ids), args.hidden_size, bias=False)
        self.main_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.blocks = nn.ModuleList([DSparkBlock(args) for _ in range(args.num_draft_layers)])
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.markov_embed = nn.Embedding(args.vocab_size, args.markov_rank)
        self.markov_head = nn.Linear(args.markov_rank, args.vocab_size, bias=False)
        self.confidence_head = (
            nn.Linear(args.hidden_size + args.markov_rank, 1, bias=False) if args.enable_confidence_head else None
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Same initialisation as the target checkpoint's own modules (all weights are replaced by
        the checkpoint before training; this only keeps a randomly initialised model finite)."""
        std = 0.02
        self.embed_weight.normal_(0, std)
        self.head_weight.normal_(0, std)
        for module in self.modules():
            if isinstance(module, HyperConnection):
                module.fn.normal_(0, std)
                module.base.zero_()
                module.scale.fill_(1.0)
            elif isinstance(module, RMSNorm):
                module.weight.fill_(1.0)
            elif isinstance(module, nn.Linear):
                module.weight.normal_(0, std)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(0, std)
        for blk in self.blocks:
            blk.attn.attn_sink.zero_()
            blk.ffn.gate_bias.zero_()

    # -- block sampling ---------------------------------------------------------------
    def sample_anchors(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        """DeepSpec's sampler: anchors are token indices whose own token and whose first target are
        both enabled by the loss mask; ``num_anchors`` of them are drawn uniformly per sample."""
        b, t = input_ids.shape
        k = self.args.block_size
        n = self.args.num_anchors
        device = input_ids.device
        valid = (loss_mask[:, : t - 1] > 0.5) & (loss_mask[:, 1:] > 0.5)
        cand = torch.arange(t - 1, device=device).unsqueeze(0).expand(b, -1)
        keys = torch.where(valid, torch.rand(b, t - 1, device=device), torch.full_like(valid, 2.0, dtype=torch.float))
        order = keys.sort(dim=1).indices
        gathered = torch.gather(torch.where(valid, cand, torch.full_like(cand, t + 1)), 1, order)
        keep = torch.arange(n, device=device).unsqueeze(0) < valid.sum(1).clamp(max=n).unsqueeze(1)
        if gathered.shape[1] < n:
            gathered = torch.cat([gathered, torch.full((b, n - gathered.shape[1]), t + 1, device=device)], 1)
        anchors = gathered[:, :n]
        anchors = torch.where(keep, anchors, torch.zeros_like(anchors))
        return anchors.sort(dim=1).values, keep

    def hidden_states(self, input_ids: torch.Tensor, target_hidden: torch.Tensor, loss_mask: torch.Tensor):
        args = self.args
        b, t = input_ids.shape
        k = args.block_size
        device = input_ids.device
        a = args.num_anchors

        anchors, keep = self.sample_anchors(input_ids, loss_mask)

        # --- main_x: target aux hidden states -> the single draft context feature --------
        main_x = self.main_norm(self.main_proj(target_hidden))                      # [B,T,D]
        positions = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # --- per-anchor 128-token context window ---------------------------------------
        w = min(args.sliding_window, t)
        offs = torch.arange(-w + 1, 1, device=device).unsqueeze(0).unsqueeze(0)     # [1,1,W]
        ctx_index = anchors.unsqueeze(-1) + offs                                    # [B,A,W]
        ctx_index = torch.where(ctx_index < 0, torch.full_like(ctx_index, -1), ctx_index)
        ctx_index = torch.where(keep.unsqueeze(-1), ctx_index, torch.full_like(ctx_index, -1))

        # --- query slots: rope position anchor + 1 + j, input = anchor token then noise --
        block_positions = anchors.unsqueeze(-1) + args.position_offset + torch.arange(k, device=device)
        noise = torch.full((b, a, k), args.noise_token_id, dtype=torch.long, device=device)
        noise[:, :, 0] = torch.gather(input_ids, 1, anchors)
        x = F.embedding(noise, self.embed_weight.to(self.embed_weight.dtype))      # [B,A,k,D]
        streams = x.unsqueeze(-2).repeat(1, 1, 1, args.hc_mult, 1)

        pre = None
        for blk in self.blocks:
            ctx_kv = blk.attn.context_kv(main_x, positions)                          # [B,T,dim]
            streams, pre = blk(streams, ctx_kv, ctx_index, block_positions, positions)

        # v4.1 has no learned hc head: collapse with the last block's ffn pre-mix
        h = (pre.unsqueeze(-1) * streams).sum(dim=3).to(streams.dtype)               # [B,A,k,D]
        head_hidden = self.norm(h)

        # --- labels --------------------------------------------------------------------
        label_idx = anchors.unsqueeze(-1) + torch.arange(1, k + 1, device=device)    # [B,A,k]
        in_range = label_idx < t
        safe = label_idx.clamp(max=t - 1)
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, a, -1), 2, safe)
        target_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, a, -1), 2, safe) > 0.5
        eval_mask = (in_range & target_mask & keep.unsqueeze(-1)).to(torch.int32).cumprod(-1).bool()
        target_ids = torch.where(eval_mask, target_ids, torch.zeros_like(target_ids))

        # --- markov bias: teacher-forced on the true prefix ------------------------------
        prev_ids = torch.cat([torch.gather(input_ids, 1, anchors).unsqueeze(-1), target_ids[:, :, :-1]], -1)
        markov_emb = self.markov_embed(prev_ids)                                     # [B,A,k,rank]

        return {
            "head_hidden": head_hidden,
            "target_ids": target_ids,
            "eval_mask": eval_mask,
            "block_keep_mask": keep,
            "prev_ids": prev_ids,
            "markov_emb": markov_emb,
            "anchors": anchors,
        }

    def forward(self, batch: dict, *, beta: float = 1.0, ce_alpha: float = 0.1, tvp_alpha: float = 0.9,
                confidence_head_alpha: float = 0.1, loss_decay_gamma: float = 4.0,
                anchor_chunk: int = 64, world_size: int = 1):
        """Training step: block forward + 129280-way head + DSpark/TVP objective.

        The head lives inside this call because FSDP only materialises parameters for the module's
        own ``forward``; computing logits outside would read sharded weights.
        """
        from .loss import dspark_v41_loss

        outputs = self.hidden_states(batch["input_ids"], batch["aux"], batch["loss_mask"])
        return dspark_v41_loss(
            model=self,
            outputs=outputs,
            target_ids=outputs["target_ids"],
            target_logprobs=batch["topk_logp"],
            target_support=batch["topk_ids"],
            target_residual=batch["residual"],
            eval_mask=outputs["eval_mask"],
            loss_decay_gamma=loss_decay_gamma,
            ce_alpha=ce_alpha,
            tvp_alpha=tvp_alpha,
            tvp_beta=beta,
            confidence_head_alpha=confidence_head_alpha,
            anchor_chunk=anchor_chunk,
            world_size=world_size,
        )

    def head_logits(self, head_hidden: torch.Tensor, markov_emb: torch.Tensor) -> torch.Tensor:
        """Draft logits for the whole vocabulary: base logits plus the Markov bias."""
        base = F.linear(head_hidden, self.head_weight.to(head_hidden.dtype))
        bias = self.markov_head(markov_emb.float())
        return base.float() + bias.float()


__all__ = ["V41DSparkArgs", "V41DSparkModel", "DSparkBlock", "DSparkAttention", "HyperConnection", "MoE"]
