#!/usr/bin/env python3
"""Generate the engine-side dump patch: the image's DFlash speculator with the MTP recorder inlined.

The container's file is replaced through the launcher's existing ``OVERLAYS`` mechanism
(``OVERLAYS="host:container"``), so nothing in the image changes.

    python3 dump/make_patch.py --src vendor/vllm/.../dflash/speculator.py --out /data/mtp/engine/speculator.py
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp_dump_hook.py")


def patch(src: str, hook_path: str = HOOK) -> str:
    hook = open(hook_path).read()
    body = hook.split('"""', 2)[2]                     # drop the module docstring, keep imports + code

    anchor = "logger = init_logger(__name__)\n"
    assert src.count(anchor) == 1, "speculator.py changed: logger anchor not unique"
    src = src.replace(anchor, anchor + "\n# ---- [mtp-train] engine-side recorder ----\n" + body
                      + "\n# ---- [/mtp-train] ----\n")

    a2 = "        num_reqs = input_batch.num_reqs\n        if dummy_run and dp_sync_state is not None:"
    assert src.count(a2) == 1, "speculator.py changed: propose() anchor not unique"
    src = src.replace(a2, "        num_reqs = input_batch.num_reqs\n"
                          "        if not dummy_run and _DUMPER is None:\n            maybe_init_dumper()\n"
                          "        if dummy_run and dp_sync_state is not None:")

    a3 = "        is_full_cudagraph = batch_desc.cg_mode == CUDAGraphMode.FULL\n"
    assert src.count(a3) == 1, "speculator.py changed: cudagraph anchor not unique"
    src = src.replace(a3, "        dump_step(self, input_batch, last_hidden_states, aux_hidden_states, "
                          "dummy_run, is_profile, num_target_tokens)\n" + a3)
    ast.parse(src)
    return src


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = patch(open(a.src).read())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(out)
    print(f"wrote {a.out} ({len(out.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
