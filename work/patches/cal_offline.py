"""Offline imatrix calibration: run the served model on a few prompts and accumulate the diagonal of the
activation second moment at every MoE input (the per-channel importance llama.cpp's imatrix uses)."""
import json
import torch

CAL = {"sum": None, "n": 0}
INST = {"done": False}


def install():
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    name = "forward" if hasattr(MoERunner, "forward") else "_forward_impl"
    orig = getattr(MoERunner, name)

    def fwd(self, hidden_states, *a, **kw):
        try:
            if not torch.cuda.is_current_stream_capturing():
                v = hidden_states.detach().float().pow(2).sum(0)
                CAL["sum"] = v if CAL["sum"] is None else CAL["sum"] + v
                CAL["n"] += int(hidden_states.shape[0])
        except Exception:
            pass
        return orig(self, hidden_states, *a, **kw)

    setattr(MoERunner, name, fwd)
    INST["done"] = True
    print("hook installed on MoERunner." + name, flush=True)


install()

from vllm import LLM, SamplingParams

llm = LLM(model="/data/models/ds41-abl", tensor_parallel_size=4, enforce_eager=True,
          max_model_len=8192, gpu_memory_utilization=0.90, dtype="bfloat16", trust_remote_code=False,
          max_num_seqs=4, max_num_batched_tokens=8192)
print("model loaded, hook done:", INST["done"], flush=True)

PROMPTS = [
    "What is 17*23? Answer in one short sentence.",
    "Write a python function that reverses a string, then explain it.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Summarize the plot of Hamlet in about 100 words.",
    "Prove that the square root of 2 is irrational.",
    "Write a SQL query that finds the top 10 customers by revenue, and explain the query.",
    "What causes inflation? Give three reasons with one sentence each.",
    "Implement binary search in Rust and explain the borrow checker issues.",
]
sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
outs = llm.generate(PROMPTS, sp)
print("generated", len(outs), "hook done:", INST["done"], "n =", CAL["n"], flush=True)
if CAL["sum"] is not None:
    h = (CAL["sum"] / max(CAL["n"], 1)).float().cpu()
    torch.save({"h": h, "n": CAL["n"]}, "/data/patches/h.pt")
    print("H saved, len", h.numel(), "mean", float(h.mean()), "max", float(h.max()), flush=True)
