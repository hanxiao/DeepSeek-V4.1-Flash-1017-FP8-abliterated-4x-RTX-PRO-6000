#!/usr/bin/env python3
"""Profile DECODE iterations only: start a long streaming request, wait until the
server is decoding, then start the vLLM torch profiler for N iterations.

usage: profdecode.py OUTDIR [MAXTOK] [ITERS] [CTX]
Needs the container started with PROF=1 (--profiler-config ... --torch_profiler_dir /profiles).
"""
import json, os, re, sys, threading, time, urllib.request

OUT = sys.argv[1]
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CTX = int(sys.argv[4]) if len(sys.argv) > 4 else 30
BASE = "http://127.0.0.1:8000"
KEY = ""
for line in open("/etc/ds41/env"):
    if line.startswith("VLLM_API_KEY="):
        KEY = line.strip().split("=", 1)[1]
AUTH = {"Authorization": "Bearer " + KEY, "content-type": "application/json"}
MODEL = os.environ.get("SERVED_MODEL_NAME", "DeepSeek-V4.1-Flash-1017-abliterated")
ctx = open("/tmp/ds41cb/ctx_%d.txt" % CTX).read()


def gen():
    body = {"model": MODEL, "max_tokens": MAXTOK, "stream": True, "tools": [],
            "system": "session %s" % os.urandom(8).hex(),
            "messages": [{"role": "user", "content": ctx + "\n\nImplement a standalone Python function "
                          "`key_match_paged(key0, key1, page_size)` with radix-cache page semantics. Reply with code."}]}
    req = urllib.request.Request(BASE + "/v1/messages", data=json.dumps(body).encode(), headers=AUTH)
    n = 0
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if b'"content_block_delta"' in raw:
                n += 1
    print("request done: %d deltas in %.1fs" % (n, time.time() - t0), flush=True)


def gentok():
    try:
        t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
        m = re.search(r"^vllm:generation_tokens_total\{[^}]*\} ([0-9.e+]+)$", t, re.M)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


th = threading.Thread(target=gen, daemon=True)
th.start()
# wait until it is decoding: generation tokens must be rising
a = gentok()
time.sleep(1.0)
b = gentok()
while b - a < 40:          # 40 tokens/s of progress => decoding
    time.sleep(1.0)
    a, b = b, gentok()
print("decoding, gen tokens %d -> starting profiler" % b, flush=True)
os.makedirs(OUT, exist_ok=True)
req = urllib.request.Request(BASE + "/start_profile",
                             data=json.dumps({"profile_prefix": "dec", "delay_iterations": 0,
                                              "max_iterations": ITERS}).encode(), headers=AUTH)
print("start_profile", urllib.request.urlopen(req, timeout=60).status, flush=True)
time.sleep(0.4)
th.join()
try:
    req = urllib.request.Request(BASE + "/stop_profile", data=b"{}", headers=AUTH)
    print("stop_profile", urllib.request.urlopen(req, timeout=120).status, flush=True)
except Exception as e:
    print("stop_profile", repr(e), flush=True)
time.sleep(2)
