#!/usr/bin/env python3
"""Fixed-length AL probe. AL = 1 + accepted/drafts from metrics deltas over one forced-length
generation of a fixed thinking prompt (no task/length confound). VM side."""
import json, re, sys, time, urllib.request

URL = "http://127.0.0.1:8000"
KEY = [l.split("=", 1)[1].strip() for l in open("/etc/ds41/env") if l.startswith("VLLM_API_KEY=")][0]
MODEL = "DeepSeek-V4.1-Flash-1017-abliterated"

try:
    _CTX = open("/tmp/ds41cb/ctx_30.txt").read()[:110000]
except Exception:
    _CTX = ""
try:
    _CTX120 = open("/tmp/ds41cb/ctx_120.txt").read()[:470000]
except Exception:
    _CTX120 = ""

PROMPTS = {
    "long120": "Here are source files of the sglang repository.\n\n<repository>\n" + _CTX120 + "</repository>\n\n"
            "Task: implement a radix-tree prefix cache with copy-on-write nodes, then reason carefully about "
            "its concurrency. Think it through first.",
    "long": "Here are source files of the sglang repository.\n\n<repository>\n" + _CTX + "</repository>\n\n"
            "Task: implement a radix-tree prefix cache with copy-on-write nodes, then reason carefully about "
            "its concurrency. Think it through first.",
    "math": "Solve 1487 * 362, then verify by a different method, then compute 1487*362 - 3^11 and "
            "express the difference in base 7. Show every step.",
    "code": "Write a Python module implementing a lock-free MPSC ring buffer with a fixed capacity of "
            "1024 entries, including wraparound, memory-ordering notes and pytest tests. Think it through first.",
    "prose": "Explain, in careful detail, why PCIe peer-to-peer writes between four GPUs serialise when a "
             "single kernel writes to several peers, and what that implies for a 60 KB all-reduce.",
}


def metrics():
    r = urllib.request.Request(URL + "/metrics", headers={"Authorization": "Bearer " + KEY})
    text = urllib.request.urlopen(r, timeout=30).read().decode()
    got = lambda n: sum(float(v) for v in re.findall(r"^vllm:%s\{[^}]*\} ([0-9.e+]+)$" % n, text, re.M))
    return (got("generation_tokens_total"), got("spec_decode_num_drafts_total"),
            got("spec_decode_num_accepted_tokens_total"))


def gen(prompt, maxtok):
    body = {"model": MODEL, "max_tokens": maxtok, "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": maxtok - 1},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + KEY, "content-type": "application/json",
                                          "anthropic-version": "2023-06-01"})
    tf = None; tl = None; out = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except Exception:
                continue
            ty = ev.get("type"); now = time.time()
            if ty == "content_block_delta":
                if tf is None:
                    tf = now
                tl = now
            elif ty == "message_delta":
                out = (ev.get("usage", {}) or {}).get("output_tokens", out) or out
    return out, tf, tl


def run(prompt_key, maxtok, label):
    g0, d0, a0 = metrics()
    out, tf, tl = gen(PROMPTS[prompt_key], maxtok)
    g1, d1, a1 = metrics()
    al = 1 + (a1 - a0) / (d1 - d0) if d1 > d0 else float("nan")
    rate = (out - 1) / (tl - tf) if (tf and tl and tl > tf) else float("nan")
    # step from the verify-step count itself: independent of AL and of the text's content.
    step = 1000 * (tl - tf) / (d1 - d0) if d1 > d0 else float("nan")
    print("%-14s %-6s out=%5d AL=%.3f tok/s=%.1f step_ms=%.2f step_metrics=%.3f" % (label, prompt_key, out, al, rate, 1000 * al / rate, step), flush=True)
    return al, rate


if __name__ == "__main__":
    maxtok = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else ["math", "code"]
    label = sys.argv[3] if len(sys.argv) > 3 else "probe"
    for k in keys:
        run(k, maxtok, label)
