"""Training objectives for the V4.1 DSpark head.

The shipped DSpark recipe (``deepspec/modeling/dspark/loss.py``) is already a
distribution-matching loss, not cross-entropy:

    L = ce_alpha * CE  +  l1_alpha * sum_v |p_target(v) - p_draft(v)|  +  conf_alpha * BCE(conf, 1-TV)

with exponential position weights ``exp(-j / gamma)``.  Its L1 term is ``2 * TV``, and under
lossless speculative sampling a draft token is accepted with probability ``1 - TV``, so the recipe
optimises acceptance directly.

TVP (arXiv 2611.00588, "TV is Almost All You Need for Speculative Decoding") generalises that term
to a one-parameter family

    D_beta(p, q) = (1/beta) * sum_v p(v) * [1 - min(1, q(v)/p(v))**beta]

with ``D_0 = sum_{q<p} p log(p/q)`` (coverage / forward-KL restricted to undercovered tokens) and
``D_1 = TV(p, q)``.  ``beta`` is annealed linearly 0 -> 1 over the first 80% of training: the
coverage phase keeps a gradient on tokens the draft under-predicts (TV's gradient vanishes there,
because TV only reads the under-covered side with weight 1), the acceptance phase ends on TV.
Reported gains over TV are +1.5% acceptance length when fine-tuning, and +1.9% at longer drafts.

We compute the target distribution on a stored top-K support plus one residual bucket.  Coarsening
the vocabulary like that can only *under*-estimate TV (TV is a contraction under coarsening), and
the TVP paper finds a top-16 support loses little; we keep K=256 by default.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _tvp_term(
    p: torch.Tensor,
    q: torch.Tensor,
    p_res: torch.Tensor,
    q_res: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """D_beta over the top-K support plus the residual bucket.

    ``p``/``q`` are [N, K] probabilities on the stored support, ``p_res``/``q_res`` [N] the mass
    outside it.  Returns the per-row loss and the implied accept rate (1 - TV) for the same
    coarsening, which is what the confidence head regresses.
    """
    eps = 1e-12
    p = p.clamp_min(0.0)
    q = q.clamp_min(0.0)
    p_all = torch.cat([p, p_res.unsqueeze(-1)], dim=-1)
    q_all = torch.cat([q, q_res.unsqueeze(-1)], dim=-1)

    # accept rate of the coarsened pair: 1 - TV, TV = sum_v [p - q]_+
    tv = (p_all - q_all).clamp_min(0.0).sum(-1)
    accept = 1.0 - tv

    if beta <= 0.0:
        # D_0 = sum_{q < p} p * log(p / q): the coverage objective, with TV's vanishing
        # gradient on under-covered tokens replaced by a log penalty.
        covered = q_all >= p_all
        ratio_log = torch.log(p_all.clamp_min(eps)) - torch.log(q_all.clamp_min(eps))
        d = torch.where(covered | (p_all <= eps), torch.zeros_like(p_all), p_all * ratio_log)
        return d.sum(-1), accept

    # D_beta = (1/beta) * (sum_v p_v - sum_v p_v^{1-beta} * min(q_v, p_v)^beta)
    log_p = torch.log(p_all.clamp_min(eps))
    log_q = torch.log(torch.minimum(q_all, p_all).clamp_min(eps))
    t = torch.exp((1.0 - beta) * log_p + beta * log_q)
    d = (p_all.sum(-1) - t.sum(-1)) / beta
    return d, accept


def dspark_v41_loss(
    *,
    model,
    outputs: dict,
    target_ids: torch.Tensor,        # [B, A, k]
    target_logprobs: torch.Tensor,   # [B, A, k, K]
    target_support: torch.Tensor,    # [B, A, k, K] int32
    target_residual: torch.Tensor,   # [B, A, k]     mass outside the support
    eval_mask: torch.Tensor,         # [B, A, k]
    loss_decay_gamma: float = 4.0,
    ce_alpha: float = 0.1,
    tvp_alpha: float = 0.9,
    tvp_beta: float = 1.0,
    confidence_head_alpha: float = 0.1,
    anchor_chunk: int = 32,
    world_size: int = 1,
) -> tuple[torch.Tensor, dict]:
    """Returns (backward loss scaled for DDP, metrics).

    The head (129280-way) is applied in anchor chunks: the full logits of one sample at
    k=9, A=512 would be 2.4 GB in fp32.  Only the *gathered* support probabilities are kept in the
    graph -- the full-vocabulary softmax is reduced to a log-sum-exp scalar per row, so a chunk costs
    one logits tensor (149 MB at the default chunk of 32) instead of three.
    """
    head_hidden = outputs["head_hidden"]            # [B, A, k, D]
    markov_emb = outputs["markov_emb"]              # [B, A, k, rank]
    b, a, k, _ = head_hidden.shape
    device = head_hidden.device

    pos = torch.arange(k, device=device, dtype=torch.float32)
    decay = torch.exp(-pos / float(loss_decay_gamma)) if loss_decay_gamma > 0 else torch.ones_like(pos)
    weight_mask = eval_mask.to(torch.float32) * decay.view(1, 1, k)

    ce_num = tvp_num = tvp_den = ce_den = conf_num = conf_den = head_hidden.new_zeros((), dtype=torch.float32)
    accept_sum = None
    pos_accept_sum = torch.zeros(k, device=device, dtype=torch.float32)
    pos_weight_sum = torch.zeros(k, device=device, dtype=torch.float32)

    for start in range(0, a, anchor_chunk):
        sl = slice(start, min(start + anchor_chunk, a))
        h = head_hidden[:, sl].reshape(-1, head_hidden.shape[-1])
        m = markov_emb[:, sl].reshape(-1, markov_emb.shape[-1])
        ids = target_ids[:, sl].reshape(-1)
        logp = target_logprobs[:, sl].reshape(-1, target_logprobs.shape[-1])
        sup = target_support[:, sl].reshape(-1, target_support.shape[-1])
        pres = target_residual[:, sl].reshape(-1)
        mask = weight_mask[:, sl].reshape(-1)

        # bf16 GEMM (fp32 accumulate), fp32 reduction: the same precision serving uses, and no
        # 2.6 GB fp32 copy of the 129280x5120 head per chunk (that copy was held by autograd)
        w_head, w_markov = model.head_weight, model.markov_head.weight
        hh = h.to(w_head.dtype) if h.dtype != w_head.dtype else h
        mm = m.to(w_markov.dtype) if m.dtype != w_markov.dtype else m
        logits = (F.linear(hh, w_head) + F.linear(mm, w_markov)).float()                # [N, V] fp32
        log_z = torch.logsumexp(logits, dim=-1)                                         # [N]
        ce = log_z - logits.gather(1, ids.unsqueeze(1)).squeeze(1)
        ce_num = ce_num + (ce * mask).sum()
        ce_den = ce_den + mask.sum()

        # q on the stored support, from logits only: exp(logit - logZ)
        q_k = (logits.gather(1, sup.long()) - log_z.unsqueeze(1)).exp()
        q_res = (1.0 - q_k.sum(-1)).clamp_min(0.0)
        p = logp.exp()
        d, accept = _tvp_term(p, q_k, pres, q_res, tvp_beta)
        tvp_num = tvp_num + (d * mask).sum()
        tvp_den = tvp_den + mask.sum()

        if model.confidence_head is not None and confidence_head_alpha > 0:
            feats = torch.cat([h.float(), m.float()], dim=-1)
            w_conf = model.confidence_head.weight
            conf = F.linear(feats.to(w_conf.dtype), w_conf).float().squeeze(-1)
            conf_num = conf_num + (F.binary_cross_entropy_with_logits(conf, accept.detach(), reduction="none") * mask).sum()
            conf_den = conf_den + mask.sum()

        with torch.no_grad():
            acc = (accept.detach() * mask).sum()
            accept_sum = acc if accept_sum is None else accept_sum + acc
            m3 = mask.view(b, -1, k)
            pos_accept_sum = pos_accept_sum + (accept.detach().view(b, -1, k) * m3).sum(dim=(0, 1))
            pos_weight_sum = pos_weight_sum + m3.sum(dim=(0, 1))

    # all-reduce the denominators so every rank normalises identically
    denoms = torch.stack([ce_den, tvp_den, conf_den]).detach()
    pos_stats = torch.stack([pos_accept_sum.detach(), pos_weight_sum.detach()])
    if world_size > 1:
        torch.distributed.all_reduce(denoms)
        torch.distributed.all_reduce(pos_stats)
    ce_den_g, tvp_den_g, conf_den_g = denoms

    ce_loss = ce_num / (ce_den_g + 1e-6)
    tvp_loss = tvp_num / (tvp_den_g + 1e-6)
    conf_loss = conf_num / (conf_den_g + 1e-6)
    loss = ce_alpha * ce_loss + tvp_alpha * tvp_loss + confidence_head_alpha * conf_loss

    with torch.no_grad():
        # REPORTED metrics use the *local* denominators: the numerators are local sums, so dividing
        # them by the all-reduced denominator (world_size x bigger) under-reports every number by
        # world_size. The backward loss keeps the global denominators (that is what makes the
        # gradient identical on every rank), hence the separate "_local" values here.
        ce_rep = ce_num.detach() / (ce_den + 1e-6)
        tvp_rep = tvp_num.detach() / (tvp_den + 1e-6)
        conf_rep = conf_num.detach() / (conf_den + 1e-6)
        accept_rep = accept_sum.detach() / (tvp_den + 1e-6)
        metrics = {
            "loss": float((ce_alpha * ce_rep + tvp_alpha * tvp_rep
                           + confidence_head_alpha * conf_rep).detach()),
            "ce_loss": float(ce_rep),
            "tvp_loss": float(tvp_rep),
            "tvp_beta": float(tvp_beta),
            "confidence_loss": float(conf_rep),
            # acceptance the verifier will actually see: 1 - TV(p_target, p_draft) on the stored support
            "accept_rate": float(accept_rep),
            "tv_distance": float(1.0 - accept_rep),
            "accept_rate_per_pos": [float(x) for x in (pos_stats[0] / (pos_stats[1] + 1e-6))],
        }
    return loss * world_size, metrics


def beta_at_step(step: int, total_steps: int, anneal_frac: float = 0.8) -> float:
    """TVP schedule: linear 0 -> 1 over the first ``anneal_frac`` of training, then 1."""
    if total_steps <= 0:
        return 1.0
    return float(min(1.0, step / max(1.0, anneal_frac * total_steps)))


__all__ = ["dspark_v41_loss", "beta_at_step", "_tvp_term"]
