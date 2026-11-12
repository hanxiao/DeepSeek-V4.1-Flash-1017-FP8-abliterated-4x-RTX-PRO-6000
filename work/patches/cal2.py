"""Offline imatrix calibration v2: probe the MoE input of the served model."""
import torch

CAL = {"sum": None, "n": 0, "shown": False, "err": False}


def install():
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as MR
    print("MoERunner forward-ish attrs:", [x for x in dir(MR.MoERunner) if "forward" in x], flush=True)
    for name in ("forward", "_forward_impl"):
        if not hasattr(MR.MoERunner, name):
            continue
        orig = getattr(MR.MoERunner, name)

        def mk(orig=orig, name=name):
            def fwd(self, *a, **kw):
                x = a[0] if a else kw.get("hidden_states")
                try:
                    if x is not None and not torch.cuda.is_current_stream_capturing():
                        v = x.detach().float().pow(2).sum(0)
                        CAL["sum"] = v if CAL["sum"] is None else CAL["sum"] + v
                        CAL["n"] += int(x.shape[0])
                        if not CAL["shown"]:
                            CAL["shown"] = True
                            print("hit", name, "shape", tuple(x.shape), flush=True)
                except Exception as e:
                    if not CAL["err"]:
                        CAL["err"] = True
                        print("hook error", name, type(e).__name__, e, flush=True)
                return orig(self, *a, **kw)

            return fwd

        setattr(MR.MoERunner, name, mk())
        print("hooked", name, flush=True)


install()
from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/ds41-abl", tensor_parallel_size=4, enforce_eager=True, max_model_len=8192,
          gpu_memory_utilization=0.90, dtype="bfloat16", max_num_seqs=4, max_num_batched_tokens=8192)
print("loaded; n =", CAL["n"], flush=True)
PROMPTS = [
    "What is 17*23? Answer in one short sentence.",
    "Write a python function that reverses a string, then explain it.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Summarize the plot of Hamlet in about 100 words.",
    "Prove that the square root of 2 is irrational.",
    "Write a SQL query that finds the top 10 customers by revenue.",
    "What causes inflation? Give three reasons.",
    "Implement binary search in Rust and explain the borrow checker.",
]
llm.generate(PROMPTS, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256))
print("generated; n =", CAL["n"], flush=True)
if CAL["sum"] is not None:
    h = (CAL["sum"] / max(CAL["n"], 1)).float().cpu()
    torch.save({"h": h, "n": CAL["n"]}, "/data/patches/h.pt")
    print("H saved len", h.numel(), "mean", float(h.mean()), "max", float(h.max()), flush=True)
