#!/usr/bin/env python3
"""Coding-agent workload bench for the Qwen3.8 SGLang service.

/v1/messages streaming, tools, thinking enabled (template default effort = xhigh, the max),
model default sampling (no temperature/top_p/top_k sent), real code contexts of ~30k/60k/120k
tokens with a coding task at the end. Per phase (size, concurrency) it reports per-stream decode
tok/s ((out_tokens-1)/(t_last-t_first), thinking+answer+tool args), TTFT, and tokens per verify
from the server counters. C1 runs also send a follow-up turn (previous context + assistant turn +
short tool_result) and report its TTFT and cache_read_input_tokens.

  SGLANG_API_KEY=... python3 codebench.py URL OUTDIR [plan] [max_tokens]
  plan e.g. "30:1x3,30:2x2,30:3x1,60:1x3,..."  (size:concurrency x rounds)
"""
import json, os, re, sys, time, threading, uuid, subprocess, tempfile, statistics, urllib.request

URL = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
PLAN = sys.argv[3] if len(sys.argv) > 3 else "30:1x3,30:2x2,30:3x2,60:1x3,60:2x2,60:3x2,120:1x3,120:2x2,120:3x2"
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
KEY = os.environ["CB_API_KEY"]
MODEL = os.environ.get("CB_MODEL", "DeepSeek-V4.1-Flash-1017-abliterated")
os.makedirs(OUT, exist_ok=True)
CTX = {s: open("/tmp/ds41cb/ctx_%d.txt" % s).read() for s in (30, 60, 120)}

TASKS = [
    ("key_match_paged",
     "Implement a standalone Python function `key_match_paged(key0: list[int], key1: list[int], page_size: int) -> int` "
     "with the same semantics as the paged key matching used by RadixCache in radix_cache.py above: it returns the "
     "number of leading tokens on which the two keys agree, counted in whole pages of `page_size` tokens only (a "
     "partially matching or incomplete trailing page does not count). page_size >= 1.",
     "from solution import key_match_paged as f\n"
     "import random\n"
     "def ref(a,b,p):\n    i=0\n    n=min(len(a),len(b))\n    while i+p<=n and a[i:i+p]==b[i:i+p]:\n        i+=p\n    return i\n"
     "random.seed(1)\n"
     "for _ in range(3000):\n    p=random.randint(1,5)\n    a=[random.randint(0,2) for _ in range(random.randint(0,20))]\n"
     "    b=a[:random.randint(0,len(a))]+[random.randint(0,2) for _ in range(random.randint(0,6))]\n"
     "    assert f(a,b,p)==ref(a,b,p),(a,b,p,f(a,b,p),ref(a,b,p))\n"
     "print('PASS')\n"),
    ("parse_tool_calls",
     "Implement a standalone Python function `parse_tool_calls(text: str) -> list[dict]` that extracts every tool call in "
     "the Qwen3-Coder XML format handled by qwen3_coder_detector.py above, i.e. blocks of the form "
     "`<tool_call>\\n<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>`, in order, "
     "returning `[{\"name\": NAME, \"arguments\": {P: VALUE, ...}}, ...]`. VALUE is a string with exactly one leading and "
     "one trailing newline removed if present (other whitespace kept); values may span several lines. Text outside "
     "tool_call blocks is ignored. Use only the standard library.",
     "from solution import parse_tool_calls as f\n"
     "t='hi\\n<tool_call>\\n<function=get_weather>\\n<parameter=city>\\nTokyo\\n</parameter>\\n<parameter=unit>\\nc\\n</parameter>\\n</function>\\n</tool_call>\\nmid <tool_call>\\n<function=run>\\n<parameter=cmd>\\nls -la\\n  echo  x\\n</parameter>\\n</function>\\n</tool_call>'\n"
     "r=f(t)\n"
     "assert r==[{'name':'get_weather','arguments':{'city':'Tokyo','unit':'c'}},{'name':'run','arguments':{'cmd':'ls -la\\n  echo  x'}}],r\n"
     "assert f('no calls here')==[]\n"
     "print('PASS')\n"),
    ("PrefixTree",
     "Implement a standalone Python class `PrefixTree` modelled on RadixCache in radix_cache.py above (page size 1, no "
     "device memory, no eviction): `insert(key: list[int]) -> int` inserts a token sequence and returns how many of its "
     "leading tokens were already present before the insert; `match_prefix(key: list[int]) -> int` returns the length "
     "of the longest prefix of `key` that is a prefix of some inserted sequence; `num_tokens() -> int` returns the total "
     "number of tokens stored on all edges (so shared prefixes are counted once). Edges must store token runs (a "
     "compressed radix tree with node splitting), like RadixCache.",
     "from solution import PrefixTree\n"
     "import random\n"
     "random.seed(2)\n"
     "for _ in range(200):\n    t=PrefixTree(); seqs=[]\n"
     "    for _ in range(random.randint(1,8)):\n        s=[random.randint(0,2) for _ in range(random.randint(0,10))]\n"
     "        exp=max([next((i for i,(x,y) in enumerate(zip(s,q)) if x!=y), min(len(s),len(q))) for q in seqs] or [0])\n"
     "        got=t.insert(s); assert got==exp,(seqs,s,got,exp); seqs.append(s)\n"
     "    pre={tuple(q[:i]) for q in seqs for i in range(len(q)+1)}\n"
     "    assert t.num_tokens()==len(pre)-1,(seqs,t.num_tokens(),len(pre)-1)\n"
     "    for _ in range(5):\n        k=[random.randint(0,2) for _ in range(random.randint(0,10))]\n"
     "        exp=max(i for i in range(len(k)+1) if tuple(k[:i]) in pre)\n"
     "        assert t.match_prefix(k)==exp\n"
     "print('PASS')\n"),
]

TOOLS = [
    {"name": "write_file", "description": "Write a file (overwrites).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "bash", "description": "Run a shell command in the repository root and return its output.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a file from the repository.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


def user_prompt(size, task):
    return ("Here are source files of the sglang repository.\n\n<repository>\n" + CTX[size] + "</repository>\n\n"
            "Task: " + TASKS[task][1] + "\n\nWrite the implementation, plus a few pytest tests, to `/work/solution.py` "
            "with ONE call of the write_file tool. Everything you need is in the files above, so do not read files or "
            "run commands before writing. After the write_file call, stop.")


def metrics():
    try:
        r = urllib.request.Request(URL + "/metrics", headers={"Authorization": "Bearer " + KEY})
        text = urllib.request.urlopen(r, timeout=30).read().decode()
        got = lambda n: sum(float(v) for v in re.findall(r"^vllm:%s\{[^}]*\} ([0-9.e+]+)$" % n, text, re.M))
        return got("generation_tokens_total"), got("spec_decode_num_drafts_total")
    except Exception:
        return float("nan"), float("nan")


def stream(messages, system, max_tokens):
    body = {"model": MODEL, "max_tokens": max_tokens, "stream": True, "system": system, "tools": TOOLS,
            **({"thinking": {"type": "enabled", "budget_tokens": max_tokens - 1}} if max_tokens > 1025 else {}), "messages": messages}
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + KEY, "content-type": "application/json",
                                          "anthropic-version": "2023-06-01"})
    res = {"t_send": time.time(), "t_first": None, "t_last": None, "out": 0, "in": 0, "cache_read": 0,
           "stop": None, "thinking": "", "text": "", "tools": [], "err": None}
    cur = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:])
                except Exception:
                    continue
                ty = ev.get("type")
                now = time.time()
                if ty == "message_start":
                    u = ev.get("message", {}).get("usage", {}) or {}
                    res["in"] = u.get("input_tokens", 0) or 0
                    res["cache_read"] = u.get("cache_read_input_tokens", 0) or 0
                elif ty == "content_block_start":
                    cb = ev.get("content_block", {})
                    cur = cb.get("type")
                    if cur == "tool_use":
                        res["tools"].append({"id": cb.get("id"), "name": cb.get("name"), "args": ""})
                elif ty == "content_block_delta":
                    d = ev.get("delta", {})
                    if res["t_first"] is None:
                        res["t_first"] = now
                    res["t_last"] = now
                    if d.get("type") == "thinking_delta":
                        res["thinking"] += d.get("thinking", "")
                    elif d.get("type") == "text_delta":
                        res["text"] += d.get("text", "")
                    elif d.get("type") == "input_json_delta" and res["tools"]:
                        res["tools"][-1]["args"] += d.get("partial_json", "")
                elif ty == "message_delta":
                    res["stop"] = ev.get("delta", {}).get("stop_reason")
                    u = ev.get("usage", {}) or {}
                    res["out"] = u.get("output_tokens", res["out"]) or res["out"]
                    if u.get("input_tokens"):
                        res["in"] = u["input_tokens"]
                    if u.get("cache_read_input_tokens") is not None:
                        res["cache_read"] = u["cache_read_input_tokens"]
                elif ty == "error":
                    res["err"] = json.dumps(ev)[:300]
    except Exception as e:
        res["err"] = repr(e)[:300]
    res["t_end"] = time.time()
    tf, tl = res["t_first"], res["t_last"]
    res["ttft"] = (tf - res["t_send"]) if tf else None
    res["rate"] = ((res["out"] - 1) / (tl - tf)) if tf and tl and tl > tf and res["out"] > 1 else None
    return res


def check_quality(task, res):
    code = None
    for t in res["tools"]:
        try:
            a = json.loads(t["args"])
        except Exception:
            continue
        if t["name"] == "write_file" and isinstance(a.get("content"), str):
            code = a["content"]
    if code is None:
        m = re.findall(r"```python\n(.*?)```", res["text"], re.S)
        code = m[-1] if m else None
    if code is None:
        return "nocode"
    d = tempfile.mkdtemp()
    open(d + "/solution.py", "w").write(code)
    open(d + "/hidden.py", "w").write(TASKS[task][2])
    try:
        p = subprocess.run([sys.executable, "hidden.py"], cwd=d, capture_output=True, text=True, timeout=60)
        return "pass" if "PASS" in p.stdout else "fail:" + (p.stderr.strip().splitlines() or ["?"])[-1][:120]
    except subprocess.TimeoutExpired:
        return "timeout"


def followup(size, task, system, first):
    content = []
    if first["thinking"]:
        content.append({"type": "thinking", "thinking": first["thinking"], "signature": ""})
    if first["text"]:
        content.append({"type": "text", "text": first["text"]})
    tus = []
    for t in first["tools"]:
        try:
            inp = json.loads(t["args"])
        except Exception:
            inp = {}
        tus.append({"type": "tool_use", "id": t["id"], "name": t["name"], "input": inp})
    if not tus:  # truncated turn: client-side tool call so the loop can continue
        tus = [{"type": "tool_use", "id": "call_" + uuid.uuid4().hex[:24], "name": "bash", "input": {"command": "ls /work"}}]
    content += tus
    results = [{"type": "tool_result", "tool_use_id": t["id"], "content": "ok (exit 0)"} for t in tus]
    msgs = [{"role": "user", "content": user_prompt(size, task)}, {"role": "assistant", "content": content},
            {"role": "user", "content": results}]
    return stream(msgs, system, 64)


def one(size, task, maxtok, out, idx, do_follow, barrier):
    system = "session %s\nYou are a coding agent working in the sglang repository." % uuid.uuid4().hex
    if barrier is not None:
        barrier.wait()
    r = stream([{"role": "user", "content": user_prompt(size, task)}], system, maxtok)
    r["task"] = task
    r["quality"] = check_quality(task, r) if r["stop"] != "max_tokens" else "truncated"
    if do_follow and not r["err"]:
        f = followup(size, task, system, r)
        r["fu_ttft"], r["fu_in"], r["fu_cache"], r["fu_err"] = f["ttft"], f["in"], f["cache_read"], f["err"]
    out[idx] = r


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, n=1):
    return "-" if x is None else ("%.*f" % (n, x))


def main():
    summ = []
    run_no = 0
    # warm-up (also checks auth)
    w = stream([{"role": "user", "content": "Say hi."}], "warm", 16)
    print("warm", w["err"], w["out"], flush=True)
    for item in PLAN.split(","):
        size, rest = item.split(":")
        size = int(size)
        conc, rounds = [int(x) for x in rest.split("x")]
        rows = []
        g0, v0 = metrics()
        for rd in range(rounds):
            out = [None] * conc
            bar = threading.Barrier(conc) if conc > 1 else None
            ths = []
            for i in range(conc):
                task = (run_no + i) % 3
                th = threading.Thread(target=one, args=(size, task, MAXTOK, out, i, conc == 1, bar))
                th.start()
                ths.append(th)
            for th in ths:
                th.join()
            run_no += conc
            for r in out:
                r.pop("thinking", None)
                r["size"], r["conc"], r["round"] = size, conc, rd
                rows.append(r)
                with open(OUT + "/raw.jsonl", "a") as fh:
                    fh.write(json.dumps({k: v for k, v in r.items() if k not in ("text",)}) + "\n")
        g1, v1 = metrics()
        acc = (g1 - g0) / (v1 - v0) if v1 == v1 and v1 > v0 else None
        s = {"size": size, "conc": conc, "n": len(rows), "in": med([r["in"] for r in rows]),
             "rate": med([r["rate"] for r in rows]), "rates": [round(r["rate"], 1) if r["rate"] else None for r in rows],
             "ttft": med([r["ttft"] for r in rows]), "out": med([r["out"] for r in rows]),
             "fu_ttft": med([r.get("fu_ttft") for r in rows]), "fu_cache": med([r.get("fu_cache") for r in rows]),
             "fu_in": med([r.get("fu_in") for r in rows]), "accept": acc,
             "quality": [r["quality"] for r in rows], "errs": [r["err"] for r in rows if r["err"]]}
        summ.append(s)
        print("ctx %3dk C%d n=%d in=%s  decode %s tok/s %s  ttft %s s  out %s  fu_ttft %s s fu_cache %s/%s  accept %s  q=%s %s" % (
            size, conc, s["n"], fmt(s["in"], 0), fmt(s["rate"]), s["rates"], fmt(s["ttft"], 2), fmt(s["out"], 0),
            fmt(s["fu_ttft"], 2), fmt(s["fu_cache"], 0), fmt(s["fu_in"], 0), fmt(acc, 2), s["quality"], s["errs"][:2]), flush=True)
        json.dump(summ, open(OUT + "/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
