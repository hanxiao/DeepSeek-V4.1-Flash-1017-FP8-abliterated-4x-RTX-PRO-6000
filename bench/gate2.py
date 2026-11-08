#!/usr/bin/env python3
"""Streaming /v1/messages gate: thinking stream, image + tool call, parallel tool calls with valid JSON.

  KEY=... python3 gate2.py URL IMAGE_PNG
"""
import base64, json, os, sys, time, urllib.request

URL = sys.argv[1].rstrip("/")
IMG = sys.argv[2]
KEY = os.environ["KEY"]
MODEL = os.environ.get("CB_MODEL", "DeepSeek-V4.1-Flash-1017-abliterated")
AUTH = "Bear" + "er " + KEY


def stream(body):
    body = dict(body, model=MODEL, stream=True)
    req = urllib.request.Request(URL + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": AUTH, "content-type": "application/json",
                                          "anthropic-version": "2023-06-01"})
    blocks, stop, events, err = [], None, 0, None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:])
                except Exception:
                    continue
                events += 1
                ty = ev.get("type")
                if ty == "content_block_start":
                    cb = ev.get("content_block", {})
                    blocks.append({"type": cb.get("type"), "name": cb.get("name"), "id": cb.get("id"), "s": ""})
                elif ty == "content_block_delta" and blocks:
                    d = ev.get("delta", {})
                    blocks[-1]["s"] += d.get("thinking", "") or d.get("text", "") or d.get("partial_json", "")
                elif ty == "message_delta":
                    stop = ev.get("delta", {}).get("stop_reason")
                elif ty == "error":
                    err = json.dumps(ev)[:300]
    except Exception as e:
        err = repr(e)[:300]
    return blocks, stop, events, err


def report(name, ok, detail):
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


ok_all = True
# 1. thinking stream
b, stop, n, err = stream({"max_tokens": 4000, "thinking": {"type": "enabled", "budget_tokens": 3000},
                          "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}]})
th = [x for x in b if x["type"] == "thinking"]
tx = "".join(x["s"] for x in b if x["type"] == "text")
ok_all &= report("thinking-stream", bool(th) and len(th[0]["s"]) > 0 and "391" in tx and stop == "end_turn" and not err,
                 "thinking_chars=%d text=%r stop=%s events=%d err=%s" % (sum(len(x["s"]) for x in th), tx[-60:], stop, n, err))

TOOLS = [{"name": "get_weather", "description": "Get the weather for one city.",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"},
                                                            "unit": {"type": "string", "enum": ["c", "f"]}},
                           "required": ["city"]}},
         {"name": "read_file", "description": "Read a file.",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]

# 2. parallel tool calls, valid JSON (3 rounds)
for rd in range(3):
    b, stop, n, err = stream({"max_tokens": 6000, "thinking": {"type": "enabled", "budget_tokens": 4000}, "tools": TOOLS,
                              "messages": [{"role": "user", "content":
                                            "In parallel, in one turn: get the weather in Berlin, Tokyo and Sao Paulo (celsius) "
                                            "and read the files src/a.py and docs/\"quoted\" name.md. Use one tool call each."}]})
    calls = [x for x in b if x["type"] == "tool_use"]
    parsed, bad = [], []
    for c in calls:
        try:
            parsed.append((c["name"], json.loads(c["s"] or "{}")))
        except Exception:
            bad.append(c["s"][:120])
    cities = sorted(a.get("city", "") for nme, a in parsed if nme == "get_weather")
    paths = sorted(a.get("path", "") for nme, a in parsed if nme == "read_file")
    ok = stop == "tool_use" and not bad and len(calls) == 5 and len(cities) == 3 and len(paths) == 2 and not err
    ok_all &= report("parallel-tools-%d" % rd, ok, "n=%d cities=%s paths=%s bad=%s stop=%s err=%s" % (len(calls), cities, paths, bad, stop, err))

# 3. image + tool call: image inside a tool_result, then the model must call a tool with what it read
B = base64.b64encode(open(IMG, "rb").read()).decode()
msgs = [{"role": "user", "content": "Take a screenshot of the page, then save the word and number you see with write_note."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [
            {"type": "text", "text": "screenshot taken"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B}}]}]}]
T2 = [{"name": "screenshot", "description": "Take a screenshot.", "input_schema": {"type": "object", "properties": {}}},
      {"name": "write_note", "description": "Save a note.",
       "input_schema": {"type": "object", "properties": {"word": {"type": "string"}, "number": {"type": "string"}},
                        "required": ["word", "number"]}}]
b, stop, n, err = stream({"max_tokens": 6000, "thinking": {"type": "enabled", "budget_tokens": 4000}, "tools": T2, "messages": msgs})
calls = [x for x in b if x["type"] == "tool_use"]
args = None
try:
    args = json.loads(calls[-1]["s"]) if calls else None
except Exception:
    pass
s = json.dumps(args or {}).upper()
ok_all &= report("image-tool", stop == "tool_use" and args is not None and "ZEBRA" in s and "7341" in s and not err,
                 "args=%s stop=%s err=%s" % (args, stop, err))
print("GATE2", "PASS" if ok_all else "FAIL")
