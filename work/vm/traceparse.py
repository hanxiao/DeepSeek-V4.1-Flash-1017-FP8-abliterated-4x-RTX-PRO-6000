#!/usr/bin/env python3
"""Summarise a vLLM torch-profiler Chrome trace: where does a decode step spend its time?

usage: traceparse.py TRACE.json[.gz] [--top N] [--by-thread]

Prints:
  * total kernel time and the number of launches, grouped by kernel name (top N), with % and mean us
  * a coarse grouping by name pattern (moe / mla / allreduce / gemm / drafter / sampler / other)
  * the wall time spanned by the trace and the GPU-busy fraction
  * if --by-thread: per-stream breakdown (which stream/thread owns the time)
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
topn = 40
bythread = "--by-thread" in sys.argv
for i, a in enumerate(sys.argv):
    if a == "--top":
        topn = int(sys.argv[i + 1])

opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as fh:
    d = json.load(fh)
ev = d["traceEvents"] if isinstance(d, dict) else d

kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and "dur" in e]
if not kern:
    # some versions tag kernels differently
    kern = [e for e in ev if e.get("ph") == "X" and "dur" in e and e.get("cat") in ("kernel", "Kernel", "gpu_op")]

agg = defaultdict(lambda: [0.0, 0])
for e in kern:
    a = agg[e["name"]]
    a[0] += e["dur"]
    a[1] += 1

total = sum(v[0] for v in agg.values())
t0 = min(e["ts"] for e in kern) if kern else 0
t1 = max(e["ts"] + e["dur"] for e in kern) if kern else 0
span = t1 - t0

print(f"trace {path}")
print(f"kernels {len(kern)}  total GPU kernel time {total/1e3:.1f} ms  wall span {span/1e3:.1f} ms  busy {100*total/span if span else 0:.1f}%")
print(f"launches {len(kern)}")
print()
print(f"{'kernel':<72} {'ms':>9} {'%':>6} {'n':>6} {'us/launch':>10}")
for name, (dur, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:topn]:
    print(f"{name[:72]:<72} {dur/1e3:9.3f} {100*dur/total:6.2f} {n:6d} {dur/n:10.2f}")

PAT = [
    ("moe", r"moe|expert|b12x|grouped"),
    ("mla_attn", r"mla|attention|flashinfer|sparse|indexer|topk|gather"),
    ("allreduce", r"all_?reduce|allreduce|nccl|p2p|oneshot|reduce_scatter|allgather"),
    ("dense_gemm", r"gemm|gemv|cutlass|mm_|linear|mxfp8|w8a8|w4a8"),
    ("drafter", r"draft|mtp|dspark|eagle"),
    ("sampler", r"sampl|softmax|top_p|top_k|reject|argmax|markov"),
    ("norm", r"rms|layernorm|norm"),
    ("other", r"."),
]
buck = defaultdict(lambda: [0.0, 0])
for name, (dur, n) in agg.items():
    low = name.lower()
    for tag, pat in PAT:
        if re.search(pat, low):
            buck[tag][0] += dur
            buck[tag][1] += n
            break
print()
print(f"{'bucket':<12} {'ms':>9} {'%':>6} {'n':>6}")
for tag, (dur, n) in sorted(buck.items(), key=lambda kv: -kv[1][0]):
    print(f"{tag:<12} {dur/1e3:9.3f} {100*dur/total:6.2f} {n:6d}")

if bythread:
    th = defaultdict(lambda: [0.0, 0])
    for e in kern:
        k = (e.get("tid"), e.get("pid"))
        th[k][0] += e["dur"]
        th[k][1] += 1
    print()
    for k, (dur, n) in sorted(th.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"tid={k[0]} pid={k[1]}  {dur/1e3:9.3f} ms  n={n}")
