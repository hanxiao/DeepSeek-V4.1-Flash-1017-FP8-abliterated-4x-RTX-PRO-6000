#!/usr/bin/env python3
"""Drive the served model so the patched speculator can dump in-domain hidden states.

The workload is the one the drafter is graded on: a coding agent with a repository context, thinking
at max effort, temperature 1.0 / top_p 0.95 (the served generation config), tools offered so the
model emits tool calls, and a follow-up turn after a tool result.

Everything the cache needs (token ids, hidden states) comes from the engine-side dump; this script
only keeps the engine busy and reports throughput.

    python3 dump/gen_traces.py --url http://127.0.0.1:8000 --key $KEY \
        --requests 2000 --concurrency 8 --max-tokens 16384 --context-file /data/mtp/context.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.request

TASKS = [
    "Implement `PrefixTree.insert(seq)` returning the length of the longest stored prefix of `seq`, and "
    "`match_prefix(seq)` returning the length of the longest stored prefix that is a stored sequence prefix.",
    "Add a `--kv-cache-dtype` flag to the server launcher, validate it against the supported set, and thread "
    "it into the engine config; cover the validation with tests.",
    "Find and fix the race in the scheduler that drops requests when `max_num_seqs` is reached, and add a "
    "regression test that fails before the fix.",
    "Implement chunked prefill accounting so a request spanning two chunks reports correct TTFT, with tests.",
    "Write a benchmark that measures decode tok/s for a single stream at 30k, 60k and 120k context and "
    "prints a markdown table.",
    "Add structured logging of per-step accepted draft tokens, then a script that turns the logs into a "
    "per-position acceptance table.",
    "Implement a radix-cache eviction policy with an LRU tie-break and unit tests for the eviction order.",
    "Refactor the attention backend selection into a registry, keeping the existing flags working, and add "
    "tests for the fallback path.",
    "Implement speculative-decoding rejection sampling with the standard and block variants and verify the "
    "output distribution is unchanged on a toy model.",
    "Write a pytest fixture that boots a tiny model on CPU and asserts that prefill and decode produce "
    "identical logits for the same tokens.",
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


def make_prompt(context: str, task: str) -> str:
    ctx = f"<repository>\n{context}\n</repository>\n\n" if context else ""
    return (f"Here are source files of the sglang repository.\n\n{ctx}Task: {task}\n\n"
            "Write the implementation, plus a few pytest tests, to `/work/solution.py` with ONE call of the "
            "write_file tool. Everything you need is in the files above, so do not read files or run commands "
            "before writing. After the write_file call, stop.")


class Runner:
    def __init__(self, args):
        self.args = args
        # real prompts: prompts from Han's own agent logs (dump/real_prompts.py), so the hidden states
        # are dumped on the distribution of his actual work, not on synthetic prompts
        self.real = []
        if getattr(args, "prompts_jsonl", None):
            self.real = [json.loads(l) for l in open(args.prompts_jsonl, encoding="utf-8") if l.strip()]
            print(f"loaded {len(self.real)} real prompts from {args.prompts_jsonl}", flush=True)
        self.lock = threading.Lock()
        self.tokens = 0
        self.requests = 0
        self.errors = 0
        self.t0 = time.time()

    def one(self, i: int):
        a = self.args
        if self.real:
            r = self.real[i % len(self.real)]
            system = f"session {i:06d}\nYou are a coding agent working in the user's repository."
            user = (f"<session context>\n{r['context']}\n</session context>\n\n{r['prompt']}"
                    if r.get("context") else r["prompt"])
        else:
            task = TASKS[i % len(TASKS)]
            system = f"session {i:06d}\nYou are a coding agent working in the sglang repository."
            user = make_prompt(a.context, task)
        body = {
            "model": a.model,
            "max_tokens": a.max_tokens,
            "stream": True,
            "system": system,
            "tools": TOOLS,
            "thinking": {"type": "enabled", "budget_tokens": a.max_tokens - 1},
            "messages": [{"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            a.url.rstrip("/") + "/v1/messages", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": "Bearer " + a.key, "content-type": "application/json",
                     "anthropic-version": "2023-06-01"})
        out = 0
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:])
                    except Exception:
                        continue
                    if ev.get("type") == "message_delta" and ev.get("usage"):
                        out = ev["usage"].get("output_tokens", out)
        except Exception as exc:                     # keep going: a failed request is 0 tokens
            with self.lock:
                self.errors += 1
            if a.verbose:
                print(f"  request {i} failed: {type(exc).__name__}: {exc}", flush=True)
            return
        with self.lock:
            self.tokens += out
            self.requests += 1
            if self.requests % 20 == 0:
                dt = time.time() - self.t0
                print(f"  {self.requests}/{a.requests} requests, {self.tokens} tokens, "
                      f"{self.tokens/dt:.0f} tok/s, {self.errors} errors", flush=True)

    def run(self):
        a = self.args
        pending = list(range(a.requests))
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    if not pending:
                        return
                    i = pending.pop(0)
                self.one(i)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - self.t0
        print(f"done: {self.requests} requests, {self.tokens} tokens in {dt/60:.1f} min "
              f"({self.tokens/max(dt,1):.0f} tok/s), {self.errors} errors")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    p.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "DeepSeek-V4.1-Flash-1017-abliterated"))
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--context-file", default=None, help="repo blob prepended to each prompt")
    p.add_argument("--context-chars", type=int, default=120_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    context = ""
    if args.context_file:
        context = open(args.context_file, encoding="utf-8", errors="replace").read()[: args.context_chars]
    args.context = context
    random.seed(args.seed)
    Runner(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
