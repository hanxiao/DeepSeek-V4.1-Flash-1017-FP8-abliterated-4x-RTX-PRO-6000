#!/usr/bin/env python3
"""Turn a trained bf16 state dict into the checkpoint's native quantised ``mtp.*`` tensors.

    python tools/export_mtp.py --state /data/mtp/runs/block9/step_000100/model.safetensors \
        --model-dir /data/models/ds41-abl --out /data/mtp/export/mtp-v2.safetensors

Then ``quantize.py build --weights`` writes the overlay directory the engine serves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepspec_v41.modeling import V41DSparkArgs, V41DSparkModel  # noqa: E402
from tools.load_mtp import export_mtp  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, help="model.safetensors written by the trainer")
    p.add_argument("--model-dir", required=True, help="original checkpoint, for the tensor template")
    p.add_argument("--ckpt-config", default=None, help="defaults to <model-dir>/config.json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    meta = json.load(open(os.path.join(os.path.dirname(a.state), "train_meta.json")))
    args = V41DSparkArgs(**meta["args"])
    model = V41DSparkModel(args)
    missing, unexpected = model.load_state_dict(load_file(a.state), strict=False)
    assert not [m for m in missing if m not in ("embed_weight", "head_weight")], missing
    print(f"loaded {a.state}: missing {len(missing)}, unexpected {len(unexpected)}")
    export_mtp(model, a.model_dir, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
