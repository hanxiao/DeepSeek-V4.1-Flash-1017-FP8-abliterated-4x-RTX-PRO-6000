#!/usr/bin/env python3
"""C1 thinking-decode bench with per-request spec-decode deltas (run ON the VM, localhost:8000).
Polite: waits until the server is idle before each request; flags requests that overlapped other traffic.
usage: thinkbench.py OUT.jsonl TEMPS PROMPTS REPS MAXTOK
  TEMPS e.g. 1.0,0.8,0.6   PROMPTS e.g. all or math1,prose1
"""
import json, os, re, sys, time, threading, urllib.request

BASE = "http://127.0.0.1:8000"
OUT = sys.argv[1]
TEMPS = [float(x) for x in sys.argv[2].split(",")]
PSEL = sys.argv[3]
REPS = int(sys.argv[4])
MAXTOK = int(sys.argv[5])
KEY = ""
for line in open("/etc/ds41/env"):
    if line.startswith("VLLM_API_KEY="):
        KEY = line.strip().split("=", 1)[1]
AUTH = "Bear" + "er " + KEY
MODEL = "DeepSeek-V4.1-Flash-1017-abliterated"

CODECTX = ""
if os.path.exists("/tmp/ds41cb/ctx_30.txt"):
    CODECTX = open("/tmp/ds41cb/ctx_30.txt").read()

PROMPTS = {
    "math1": "Let S be the set of positive integers n <= 1000 such that n^2 + 3n + 5 is divisible by 121 or n^3 - n is divisible by 504. How many elements does S have? Show your reasoning and give the final answer in \\boxed{}.",
    "math2": "A bag contains 5 red, 4 blue and 3 green balls. Balls are drawn one at a time without replacement until two balls of the same colour have been drawn consecutively. What is the expected number of draws? Give an exact fraction in \\boxed{}.",
    "logic1": "Five people (Ana, Ben, Cai, Dee, Eli) each own a different pet (cat, dog, fish, bird, lizard) and live in houses 1-5 in a row. Clues: (1) The dog owner lives directly left of Cai. (2) Ana lives in an odd-numbered house but not house 1. (3) The fish owner lives at one end. (4) Ben owns the bird and does not live next to Ana. (5) Eli lives in house 2 and does not own the fish. (6) The lizard owner lives next to the cat owner. (7) Dee does not own the cat. Determine the full assignment and explain.",
    "prose1": "Write a thoughtful ~500-word essay arguing both sides of whether cities should replace car lanes with bike lanes, then give your own conclusion. Plain prose, no headings.",
    "plan1": "I am moving from Berlin to Singapore with a 4-year-old and a dog in March. Make a careful, prioritized 8-week relocation plan covering pet import rules, school/kindergarten, housing, banking, taxes and health insurance. Be concrete and flag the things with long lead times.",
    "code1": "Implement in Python an LRU cache class with O(1) get/put that also supports per-key TTL expiry and a method `resize(new_capacity)` that evicts as needed. Explain the design briefly, then give the full code with a few asserts.",
    "zh1": "请仔细推理：一个水池有甲乙两个进水管和丙一个出水管。单开甲6小时注满，单开乙8小时注满，单开丙12小时放空。先同时开甲乙2小时，然后关甲开丙，问还需多少小时注满？之后再用一段话解释这类题的通用解法。",
}
if CODECTX:
    PROMPTS["cb30"] = CODECTX + "\n\nImplement a standalone Python function `key_match_paged(key0: list[int], key1: list[int], page_size: int) -> int` with the same semantics as the paged key matching used by RadixCache in radix_cache.py above: it returns the number of leading tokens on which the two keys agree, counted in whole pages of `page_size` tokens only. Reply with the code in a single python block."

SEL = list(PROMPTS) if PSEL == "all" else PSEL.split(",")


def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    m = {"pos": [0.0] * 8}
    for line in t.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            m["drafts"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            m["dtok"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            m["acc"] = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            p = int(re.search(r'position="(\d+)"', line).group(1))
            m["pos"][p] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_running"):
            m["run"] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_waiting{"):
            m["wait"] = float(line.split()[-1])
    return m


def delta(a, b):
    d = {k: b[k] - a[k] for k in ("drafts", "dtok", "acc")}
    d["pos"] = [b["pos"][i] - a["pos"][i] for i in range(8)]
    d["tps"] = (d["acc"] + d["drafts"]) / d["drafts"] if d["drafts"] else None
    return d


def tokenize(text):
    if not text:
        return 0
    body = json.dumps({"model": MODEL, "prompt": text, "add_special_tokens": False}).encode()
    rq = urllib.request.Request(BASE + "/tokenize", data=body, headers={"Content-Type": "application/json", "Authorization": AUTH})
    return json.loads(urllib.request.urlopen(rq, timeout=60).read())["count"]


def wait_idle():
    waited = 0
    while True:
        m = metrics()
        if m.get("run", 0) == 0 and m.get("wait", 0) == 0:
            return waited
        time.sleep(5)
        waited += 5


class Watch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.maxrun = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                self.maxrun = max(self.maxrun, metrics().get("run", 0))
            except Exception:
                pass
            time.sleep(1.5)


def one(pname, temp, rep):
    waited = wait_idle()
    msgs = [{"role": "user", "content": PROMPTS[pname]}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": MAXTOK, "temperature": temp, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True}}
    rq = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json", "Authorization": AUTH})
    w = Watch()
    m0 = metrics()
    w.start()
    t0 = time.time()
    tfirst = tans = tlast = None
    mans = None
    think, ans = [], []
    usage = None
    finish = None
    with urllib.request.urlopen(rq, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                rc = d.get("reasoning_content") or d.get("reasoning") or ""
                c = d.get("content") or ""
                now = time.time()
                if rc or c:
                    if tfirst is None:
                        tfirst = now
                    tlast = now
                if rc:
                    think.append(rc)
                if c:
                    if tans is None:
                        tans = now
                        mans = metrics()
                    ans.append(c)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    m1 = metrics()
    w.stop = True
    w.join()
    thinktxt, anstxt = "".join(think), "".join(ans)
    total = usage["completion_tokens"] if usage else None
    nthink = tokenize(thinktxt)
    rec = {"prompt": pname, "temp": temp, "rep": rep, "waited_s": waited, "finish": finish,
           "completion_tokens": total, "prompt_tokens": usage and usage.get("prompt_tokens"),
           "think_tokens": nthink, "answer_tokens": (total - nthink) if total is not None else None,
           "ttft": tfirst - t0 if tfirst else None, "maxrun_seen": w.maxrun,
           "overlap": w.maxrun > 1 or m1.get("run", 0) > 0}
    if tfirst and tlast and total:
        rec["decode_tps_all"] = (total - 1) / max(1e-6, tlast - tfirst)
    rec["spec_all"] = delta(m0, m1)
    if tans and mans:
        rec["spec_think"] = delta(m0, mans)
        rec["spec_answer"] = delta(mans, m1)
        rec["decode_tps_think"] = nthink / max(1e-6, tans - tfirst)
        if rec["answer_tokens"] and tlast > tans:
            rec["decode_tps_answer"] = rec["answer_tokens"] / (tlast - tans)
    else:
        rec["spec_think"] = rec["spec_all"]
        rec["decode_tps_think"] = rec.get("decode_tps_all")
    rec["answer_tail"] = anstxt[-300:]
    return rec


def main():
    order = []
    for rep in range(REPS):
        for p in SEL:
            for t in TEMPS:
                order.append((p, t, rep))
    with open(OUT, "a") as f:
        for p, t, rep in order:
            try:
                rec = one(p, t, rep)
            except Exception as e:
                rec = {"prompt": p, "temp": t, "rep": rep, "error": repr(e)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            st = rec.get("spec_all", {})
            print(p, t, rep, rec.get("completion_tokens"), "think", rec.get("think_tokens"),
                  "tps_all %.1f" % (rec.get("decode_tps_all") or 0),
                  "tok/step all %.2f think %.2f ans %s" % (st.get("tps") or 0, (rec.get("spec_think") or {}).get("tps") or 0,
                                                          (rec.get("spec_answer") or {}).get("tps")),
                  "overlap", rec.get("overlap"), rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
