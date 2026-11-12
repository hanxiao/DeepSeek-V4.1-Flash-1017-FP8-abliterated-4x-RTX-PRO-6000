#!/bin/bash
# VM side, root: greedy determinism + bit-identity of the DET_TOPK switch on real prompts.
# At temperature 0 the whole pipeline is deterministic, so any change in the emitted tokens is a
# change in behaviour; identical tokens is the strongest available statement of "no quality change".
set -uo pipefail
R=/var/lib/ds41/x/greedy; mkdir -p $R
KEY=$(sed -n 's/^VLLM_API_KEY=//p' /etc/ds41/env); export KEY
run(){ local TAG=$1 E=$2
  echo "=========== $TAG $(date -u +%T)"
  EXTRA_ENV="$E" PROF=0 /usr/local/bin/ds41-start-r21.sh
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/metrics)" = 200 ] && break; sleep 5; done
  KEY=$KEY TAG=$TAG OUT=$R/$TAG.jsonl python3 - <<'PY'
import json, os, urllib.request
KEY = os.environ["KEY"]; OUT = os.environ["OUT"]
PROMPTS = [
 ("math", "Compute 47*89, then 47*89-3^7, then express the difference in base 5. Show every step."),
 ("code", "Write a Python function that merges two sorted lists without using sorted() or heapq, with pytest tests."),
 ("needle", "In one sentence: what is the capital of Burkina Faso, and since when?"),
 ("logic", "Three boxes: one has apples, one oranges, one both. All labels are wrong. You draw one fruit "
           "from one box. Which box do you draw from and why? Answer in two sentences."),
]
out = open(OUT, "w")
for name, p in PROMPTS:
    body = {"model": "DeepSeek-V4.1-Flash-1017-abliterated", "max_tokens": 1600, "temperature": 0.0,
            "top_p": 1.0, "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": p}]}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/messages", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "content-type": "application/json",
                 "anthropic-version": "2023-06-01"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    txt = "".join(b.get("text", "") or b.get("thinking", "") for b in r.get("content", []))
    out.write(json.dumps({"name": name, "out": txt, "usage": r.get("usage")}) + "\n")
    print(name, len(txt), flush=True)
out.close()
PY
}
run base ""
run det "-e VLLM_MOET_DET_TOPK=1"
echo "=== diff ==="
python3 - <<'PY'
import json
a = {json.loads(l)["name"]: json.loads(l)["out"] for l in open("/var/lib/ds41/x/greedy/base.jsonl")}
b = {json.loads(l)["name"]: json.loads(l)["out"] for l in open("/var/lib/ds41/x/greedy/det.jsonl")}
same = 0
for k in a:
    eq = a[k] == b[k]
    same += eq
    print("%-8s identical=%s len=%d/%d" % (k, eq, len(a[k]), len(b[k])))
print("IDENTICAL %d/%d" % (same, len(a)))
PY
echo "GREEDY DONE $(date -u +%T)"
