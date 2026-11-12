"""Minimal test of the q2=0 (4-bit) path of the dual fused kernel: no repack anywhere."""
import os, sys
os.environ["VLLM_MOET_B12X_MOE_FUSED"] = "0"
sys.path.insert(0, "/opt/vllm-moet/dsv41_sm120/b12x_moe_fused")
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
sys.path.insert(0, "/data/patches")
import torch
import test_b12x_moe_fused_sm120 as T
import b12x.moe._shared.kernels.w4a8_compact_micro as cm
import b12x_moe_fused_sm120 as mf
import w4a8_compact_micro_q2 as q2

q2._moet_q2_w13 = lambda t: t          # no repack at all
E, H, N, TOPK, LIMIT, TOKENS = T.E, T.H, T.N, T.TOPK, T.LIMIT, T.TOKENS
dev = "cuda"

got = {}
orig = cm.launch_w4a8_compact_micro


def spy(**kw):
    got.update(kw)
    return orig(**kw)


cm.launch_w4a8_compact_micro = spy
served = T._Served(torch, T._weights(torch, dev), dev)   # ordinary 4-bit random weights
cases = T._all_cases(torch, dev)
proj_mine = torch.empty(TOKENS * TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
route_mine = torch.empty(TOKENS * TOPK, H, dtype=torch.bfloat16, device=dev)
inter_mine = None
rows_cap = TOKENS * TOPK
nok = 0
for i, (ids, w, x) in enumerate(cases[:4]):
    out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
    served.run(x, ids, w, out)
    torch.cuda.synchronize()
    lay = cm._layout(TOKENS, TOPK, H, N)
    st = got["scratch"].view(torch.uint8).reshape(-1)
    r = {n: st.narrow(0, *lay[n]) for n in ("a_values", "a_scales", "projections", "intermediate", "route_output")}
    if inter_mine is None:
        inter_mine = torch.zeros_like(r["intermediate"])
    route_ref = r["route_output"].view(torch.bfloat16).view(-1, H)[: TOKENS * TOPK].clone()
    flat = ids.reshape(-1)
    rows = torch.nonzero((flat >= 0) & (flat < E)).flatten()
    route_mine.fill_(float("nan"))
    mf.moe_fused(r["a_values"], r["a_scales"], got["w13"], got["w13_scales"], proj_mine, inter_mine, got["w2"],
                 got["w2_scales"], ids.reshape(-1), w.reshape(-1), got["alpha1"], got["input_scale"], got["alpha2"],
                 got["down_scale"], route_mine, TOKENS * TOPK, TOPK, rows_cap, H, N, LIMIT, 0)
    torch.cuda.synchronize()
    ok = torch.equal(route_mine[rows].view(torch.int16), route_ref[rows].view(torch.int16))
    nok += ok
    print(i, "q2=0", "ok" if ok else "DIFFER",
          float((route_mine[rows].float() - route_ref[rows].float()).abs().max()), flush=True)
print("Q20 OK", nok, "of 4")
