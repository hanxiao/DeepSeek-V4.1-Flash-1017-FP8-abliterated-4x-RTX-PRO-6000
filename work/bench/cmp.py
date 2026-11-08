#!/usr/bin/env python3
"""Compare codebench arms per (size, task): the only fair way to A/B, because the task mix
changes with the plan and task 0 is much faster than tasks 1 and 2.

usage: cmp.py TAG [TAG ...]      reads work/bench/x-<tag>/raw.jsonl
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = sys.argv[1:]
data = {}
for t in TAGS:
    p = os.path.join(HERE, "x-%s" % t, "raw.jsonl")
    if not os.path.exists(p):
        print("missing", p)
        continue
    rows = [json.loads(l) for l in open(p)]
    for r in rows:
        if r.get("rate") and r.get("conc") == 1:
            data.setdefault((t, r["size"], r["task"]), []).append(r["rate"])

keys = sorted({(s, task) for (_, s, task) in data})
print("C1 per-task decode tok/s (mean of runs, n)")
print(f"{'size/task':<12}" + "".join(f"{t:>22}" for t in TAGS))
for s, task in keys:
    line = f"{s:>4}k/t{task:<5}"
    for t in TAGS:
        v = data.get((t, s, task))
        line += f"{(f'{statistics.mean(v):.1f} (n={len(v)})' if v else '-'):>22}"
    print(line)

print()
print("ratio vs first tag, per task")
for s, task in keys:
    base = data.get((TAGS[0], s, task))
    if not base:
        continue
    b = statistics.mean(base)
    line = f"{s:>4}k/t{task:<5}"
    for t in TAGS[1:]:
        v = data.get((t, s, task))
        line += f"{(f'{statistics.mean(v)/b:.3f}x' if v else '-'):>22}"
    print(line)

print()
for t in TAGS:
    allv = [x for (tt, _, _) in data for x in data[(tt, _, _)]] if False else []
    allv = [x for k, vs in data.items() if k[0] == t for x in vs]
    if allv:
        print(f"{t:<14} C1 all-task mean {statistics.mean(allv):7.1f}  median {statistics.median(allv):7.1f}  n={len(allv)}")
