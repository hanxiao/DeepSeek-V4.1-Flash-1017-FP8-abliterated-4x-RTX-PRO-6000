#!/usr/bin/env python3
"""Summarise a thinkbench run: accepted length and per-position conditional acceptance.

    python3 tools/report_think.py /var/lib/ds41/x/mtp-block9-k9
"""
import json
import os
import sys


def main() -> int:
    run = sys.argv[1]
    path = os.path.join(run, "think.jsonl")
    if not os.path.exists(path):
        print(f"no think.jsonl in {run}")
        return 0
    rows = [json.loads(line) for line in open(path) if line.strip()]
    gen = sum(r.get("generation_tokens", 0) for r in rows)
    acc = sum(r.get("accepted", 0) for r in rows)
    print(f"thinkbench: {len(rows)} runs, {gen} generated tokens, {acc} accepted draft tokens")
    if gen:
        print(f"accepted length = {1 + acc / gen:.3f}")
    for key in ("per_pos", "accept_per_pos", "accepted_per_pos", "drafts_per_pos"):
        if rows and key in rows[0]:
            tot = [sum(r[key][i] for r in rows if key in r) for i in range(len(rows[0][key]))]
            print(f"{key}: {tot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
