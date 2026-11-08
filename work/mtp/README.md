# MTP drafter fine-tune

Read [PLAN.md](PLAN.md) first. This file is the map.

| file | what it is |
| --- | --- |
| `PLAN.md` | the plan: architecture facts, data pipeline, loss (TVP), hyperparameters, cost, eval gate, overlay loading, risks |
| `run.sh` | the operator script: `gate0 -> dump -> cache -> train -> export -> eval`, every stage idempotent |
| `quantize.py` | the checkpoint's native FP8/FP4/ue8m0 quantisation, plus `roundtrip` (Gate 0) and `build` (overlay checkpoint dir) |
| `train_mtp.py` | training entry (`torchrun`), config in `config/dspark_v41_block9.py` |
| `deepspec_v41/modeling.py` | the V4.1 DSpark draft head (mHC, MLA + DSpark mask, 128-expert MoE, Markov/confidence heads) |
| `deepspec_v41/loss.py` | DSpark/TVP objective on a top-K target support |
| `deepspec_v41/data.py` | reader for the flat memmap cache |
| `deepspec_v41/trainer.py` | FSDP + AdamW loop, TVP anneal, checkpointing |
| `tools/load_mtp.py` | checkpoint `<->` model tensor mapping + native dequant/requant |
| `tools/export_mtp.py` | trained bf16 state dict -> native `mtp.*` safetensors |
| `tools/gate0.py` | the cheap correctness gates (quantisation roundtrip, model load, finite forward) |
| `tools/report_think.py` | accepted length + per-position acceptance from a thinkbench run |
| `dump/speculator.py` | the image's DFlash speculator with the hidden-state recorder inlined (regenerate with `dump/make_patch.py`) |
| `dump/gen_traces.py` | generates the in-domain coding-agent traces |
| `dump/reassemble.py` | raw dump -> training cache (target top-K distribution) |
| `tests/test_local.py` | 9 CPU tests; `.venv/bin/python tests/test_local.py` |
| `vendor/` | DeepSpec (MIT), the checkpoint's own `inference/` reference, and the vLLM/transformers sources the plan cites |
