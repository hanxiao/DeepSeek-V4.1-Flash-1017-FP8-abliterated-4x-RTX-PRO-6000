"""Build a fake MXFP4 checkpoint with the real key names, to exercise ladder_on_checkpoint.py."""
import json, os
import numpy as np, torch
from safetensors.torch import save_file
from q2_core import E2M1, mx_block_scale, round_e2m1

out = "/tmp/fake-ds41"; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(3)
tensors, wmap = {}, {}
for layer in (6,):
    for e in (0,):
        for k in ("w1", "w3"):
            W = rng.standard_normal((256, 1024)).astype(np.float32)
            s0 = mx_block_scale(np.abs(W).reshape(-1, 32).max(1)).reshape(256, -1)
            q = round_e2m1(W.reshape(-1, 32) / s0.reshape(-1, 1)).reshape(256, 1024)
            code = np.where(q < 0, 8, 0) | np.abs(np.searchsorted(E2M1, np.abs(q)).clip(0, 7))
            code = code.astype(np.uint8).reshape(256, 512, 2)
            packed = (code[:, :, 1] << 4) | code[:, :, 0]
            expo = (np.log2(s0) + 127).astype(np.uint8)
            key = f"layers.{layer}.ffn.experts.{e}.{k}"
            tensors[key + ".weight"] = torch.from_numpy(packed)
            tensors[key + ".scale"] = torch.from_numpy(expo)
            wmap[key + ".weight"] = "m.safetensors"; wmap[key + ".scale"] = "m.safetensors"
save_file(tensors, f"{out}/m.safetensors")
json.dump({"weight_map": wmap}, open(f"{out}/model.safetensors.index.json", "w"))
print("fake checkpoint:", list(tensors), [tuple(v.shape) for v in tensors.values()])
