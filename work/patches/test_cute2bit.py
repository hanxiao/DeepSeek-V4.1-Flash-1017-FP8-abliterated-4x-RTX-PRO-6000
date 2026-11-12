"""Local (no server) correctness loop for the 2-bit CuTe projection path: run b12x's CuTe MoE on
2-bit-valued synthetic weights and dump projections/route_output; run once with the stock kernel
(reference) and once with the 2-bit patches, then compare bit for bit."""
import os, sys
mode = os.environ.get("Q2MODE", "ref")
os.environ["VLLM_MOET_B12X_MOE_FUSED"] = "0"
sys.path.insert(0, "/opt/vllm-moet/dsv41_sm120/b12x_moe_fused")
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import torch
import test_b12x_moe_fused_sm120 as T
import b12x.moe._shared.kernels.w4a8_compact_micro as cm

E, H, N, TOPK, LIMIT, TOKENS = T.E, T.H, T.N, T.TOPK, T.LIMIT, T.TOKENS
dev = "cuda"


def q2_weights():
    w1, w1s, w2, w2s = T._weights(torch, dev)

    def fix(p):
        lo = (p & 0x0F).to(torch.int32)
        hi = ((p >> 4) & 0x0F).to(torch.int32)
        def m(n):
            mag = torch.where((n & 0x7) < 2, 1, 3)
            return ((n & 0x8) | mag).to(torch.uint8)
        return (m(lo) | (m(hi) << 4)).to(torch.uint8)

    return fix(w1), w1s, fix(w2), w2s


got = {}
orig = cm.launch_w4a8_compact_micro


def spy(**kw):
    got.update(kw)
    return orig(**kw)


cm.launch_w4a8_compact_micro = spy
served = T._Served(torch, q2_weights(), dev)
cases = T._all_cases(torch, dev)
outs = []
for ids, w, x in cases[:6]:
    out = torch.empty(TOKENS, H, dtype=torch.bfloat16, device=dev)
    served.run(x, ids, w, out)
    torch.cuda.synchronize()
    lay = cm._layout(TOKENS, TOPK, H, N)
    st = got["scratch"].view(torch.uint8).reshape(-1)
    proj = st.narrow(0, *lay["projections"]).view(torch.bfloat16).clone()
    route = st.narrow(0, *lay["route_output"]).view(torch.bfloat16).clone()
    outs.append({"proj": proj.cpu(), "route": route.cpu(), "ids": ids.cpu()})
torch.save(outs, "/data/patches/out_%s.pt" % mode)
print("dumped", mode, flush=True)
