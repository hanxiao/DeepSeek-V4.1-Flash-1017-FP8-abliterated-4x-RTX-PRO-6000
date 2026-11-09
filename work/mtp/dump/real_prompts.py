#!/usr/bin/env python3
"""Collect real prompts from Han's agent logs, for the engine dump.

The MTP drafter is fine-tuned on what the engine generates, so the prompts that seed that generation
should be Han's real requests rather than synthetic ones. This reads his real logs:

  ~/.claude/projects/**/*.jsonl                  Claude Code transcripts (every project)
  ~/.dsh/sessions/**/session.v4.jsonl[.zstd]     this harness's sessions
  ~/.openclaw/workspace/**/*.jsonl               the openclaw pi sessions

It keeps the user-role turns (the human asks) as `prompt`, and the assistant text that preceded them in
the same session as `context` (truncated), so the dump can build an agent-shaped request that has a
real long prefix followed by a genuine task.

Output: JSONL per line {"prompt": ..., "context": ..., "source": ...}:
    python3 dump/real_prompts.py --out /tmp/real_prompts.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

HOME = os.path.expanduser("~")
SRC = {
    "claude": (os.path.join(HOME, ".claude", "projects"), "*.jsonl"),
    "dsh": (os.path.join(HOME, ".dsh", "sessions"), "*.jsonl.zstd"),
    "openclaw": (os.path.join(HOME, ".openclaw", "workspace"), "*.jsonl"),
}

MAX_TRACE = re.compile(r"\[request id [0-9a-f]+\]")
XML_BLOB = re.compile(r"<[^>]{80,}>", re.S)
DROP = re.compile(r"<(system-reminder|local-command-caveat)>.*?</\1>?", re.S)


def clean(text: str) -> str:
    t = DROP.sub(" ", text or "")
    t = re.sub(r"<(?:command-name|command-message|command-args)>.*?</[^>]*>", " ", t, flags=re.S)
    t = XML_BLOB.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            out.append(part["text"])
        elif isinstance(part.get("content"), str) and part.get("type") == "tool_result":
            out.append(part["content"][:4000])
        elif isinstance(part.get("input"), dict):
            out.append(json.dumps(part["input"], ensure_ascii=False)[:6000])
    return "\n".join(out)


def read_lines(path: str):
    if path.endswith(".zstd"):
        raw = subprocess.run(["zstd", "-dc", path], capture_output=True, timeout=120).stdout
        for line in raw.split(b"\n"):
            yield line
    else:
        with open(path, "rb") as fh:
            for line in fh:
                yield line


def session(path: str, ctx_chars: int, min_prompt: int, max_prompt: int):
    """Two passes: collect the assistant text as context, and the user turns as prompts."""
    out = []
    ctx = ""
    try:
        for line in read_lines(path):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msg = rec.get("message")
            kind = None
            if isinstance(msg, dict):
                role = msg.get("role")
                body = text_of(msg.get("content"))
            else:
                # dsh / openclaw: {"type": "user/message", "data": {"role", "content", "source"}}
                t = str(rec.get("type") or "")
                d = rec.get("data") or {}
                role = (d.get("role") if isinstance(d, dict) else None) or (
                    t.split("/")[0] if "/" in t else None)
                body = text_of(d.get("content") if isinstance(d, dict) else None)
                if isinstance(d, dict):
                    src = d.get("source")
                    kind = src.split("/")[-1].split("x")[0] if isinstance(src, str) else None
                    kind = kind if isinstance(src, str) else (src or {}).get("kind") if isinstance(src, dict) else None
                else:
                    kind = None
            if role == "user" and isinstance(kind, str) and kind not in ("user", ""):
                continue
            if not body:
                continue
            if role == "assistant":
                if len(body) > 60:
                    ctx = (ctx + "\n" + body)[-ctx_chars:]
            elif role == "user":
                body = clean(body)
                if MAX_TRACE.search(body):
                    continue
                if min_prompt <= len(body) <= max_prompt:
                    out.append({"prompt": body, "context": ctx[-ctx_chars:] if ctx else ""})
    except Exception as e:  # noqa: BLE001
        print(f"  {path}: {e}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HOME, ".claude", "real_prompts.jsonl"))
    ap.add_argument("--ctx-chars", type=int, default=40000)
    ap.add_argument("--min-prompt", type=int, default=120)
    ap.add_argument("--max-prompt", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    seen = set()
    total = 0
    with open(a.out, "w") as fh:
        for name, (root, pat) in SRC.items():
            files = sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
            print(f"{name}: {len(files)} files")
            got = 0
            for f in files:
                for row in session(f, a.ctx_chars, a.min_prompt, a.max_prompt):
                    key = row["prompt"][:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    row["source"] = name
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    got += 1
                    total += 1
                    if a.limit and total >= a.limit:
                        break
                if a.limit and total >= a.limit:
                    break
            print(f"  {got} prompts")
    print(f"wrote {total} prompts -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
