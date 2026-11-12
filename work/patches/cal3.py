"""Offline imatrix calibration v3: per-rank dump (the hook runs in the TP worker processes)."""
import os
import torch

CAL = {"sum": None, "n": 0, "dev": None}


def install():
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as MR
    orig = MR.MoERunner.forward

    def fwd(self, *a, **kw):
        x = a[0] if a else kw.get("hidden_states")
        try:
            if x is not None and not torch.cuda.is_current_stream_capturing():
                if CAL["dev"] is None:
                    CAL["dev"] = int(torch.cuda.current_device())
                v = x.detach().float().pow(2).sum(0)
                CAL["sum"] = v if CAL["sum"] is None else CAL["sum"] + v
                CAL["n"] += int(x.shape[0])
                if CAL["n"] >= 200000:
                    torch.save({"h": (CAL["sum"] / CAL["n"]).float().cpu(), "n": CAL["n"]},
                               "/data/patches/h_rank%d.pt" % CAL["dev"])
        except Exception as e:
            print("hook err", type(e).__name__, e, flush=True)
        return orig(self, *a, **kw)

    MR.MoERunner.forward = fwd
    print("hooked", flush=True)


install()
from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/ds41-abl", tensor_parallel_size=4, enforce_eager=True, max_model_len=8192,
          gpu_memory_utilization=0.90, dtype="bfloat16", max_num_seqs=4, max_num_batched_tokens=8192)
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
print("done", flush=True)
