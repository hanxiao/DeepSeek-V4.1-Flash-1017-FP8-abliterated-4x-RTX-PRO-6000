#!/usr/bin/env python3
"""Turn the engine's raw hidden-state dump into the training cache.

Input  : ``$VLLM_MTP_DUMP`` records written by ``dump/speculator.py`` (see the header of
         ``dump/mtp_dump_hook.py`` for the layout).
Output : a directory of flat memmaps the trainer reads without any parsing:

           tokens.bin      int32  [N, T]        input token ids
           loss_mask.bin   uint8  [N, T]        1 = supervised
           aux.bin         bf16   [N, T, 15360] target hidden states entering layers 37/38/39
           topk_ids.bin    int32  [N, T, K]     target top-K vocabulary ids
           topk_logp.bin   float32[N, T, K]     their log-probabilities (temperature 1)
           residual.bin    float32[N, T]        target mass outside the top-K support
           meta.json                            shapes, K, source, token totals

The target's next-token distribution is recomputed here from the dumped final hidden state and the
checkpoint's LM head, which is exactly the tensor the served target samples from.

Sequences are reassembled per request id, ordered by position, and a position that the engine
processed more than once (rejected speculative rows are re-forwarded) keeps its last record.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

HIDDEN = 5120
NUM_AUX = 3
INT_BYTES = 3 * 4                                    # token, position, sequence
FLT_BYTES = (NUM_AUX * HIDDEN + HIDDEN) * 2          # bf16 aux then bf16 final hidden
ROW_BYTES = INT_BYTES + FLT_BYTES
INT_DTYPE = np.dtype([("token", "<i4"), ("pos", "<i4"), ("seq", "<i4")])
FLT_DTYPE = np.dtype([("aux", "<u2", (NUM_AUX * HIDDEN,)), ("last", "<u2", (HIDDEN,))])


def read_raw(paths: list[str]):
    """Yield the numpy structured array of one dump file."""
    for path in paths:
        with open(path, "rb") as f:
            while True:
                head = f.read(4)
                if len(head) < 4:
                    break
                (n,) = struct.unpack("<i", head)
                ibuf = f.read(n * INT_BYTES)
                fbuf = f.read(n * FLT_BYTES)
                assert len(ibuf) == n * INT_BYTES and len(fbuf) == n * FLT_BYTES, f"{path}: truncated record"
                ints = np.frombuffer(ibuf, dtype=INT_DTYPE, count=n)
                flts = np.frombuffer(fbuf, dtype=FLT_DTYPE, count=n)
                yield path, ints, flts


def load_head_weight(model_dir: str, dtype=torch.bfloat16) -> torch.Tensor:
    """``head.weight`` [vocab, hidden] from the checkpoint's shards, without touching the rest."""
    from safetensors import safe_open

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    shard = index["weight_map"]["head.weight"]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor("head.weight").to(dtype)


def topk_target(hidden: torch.Tensor, head: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[n, H] final hidden -> (ids [n,K], logprobs [n,K] fp32, residual mass [n] fp32)."""
    logits = torch.nn.functional.linear(hidden.float(), head.float())
    logp = torch.log_softmax(logits, dim=-1)
    vals, ids = logp.topk(k, dim=-1)
    residual = (1.0 - vals.exp().sum(-1)).clamp_min(0.0)
    return ids.to(torch.int32), vals, residual


def build(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    head = load_head_weight(args.model_dir).to(device)

    # ---- pass 1: group records by sequence, keep the last record per position ----
    seqs: dict[int, dict[int, tuple]] = {}
    n_raw = 0
    for path, ints, flts in read_raw(args.raw):
        for i in range(len(ints)):
            seqs.setdefault(int(ints["seq"][i]), {})[int(ints["pos"][i])] = (
                int(ints["token"][i]), flts["aux"][i], flts["last"][i])
            n_raw += 1
        print(f"  {path}: {len(ints)} rows, {len(seqs)} sequences so far", flush=True)
    print(f"{n_raw} raw rows -> {len(seqs)} sequences, {sum(len(v) for v in seqs.values())} unique positions")

    # ---- pass 2: slice into fixed-length samples and recompute the target distribution ----
    T = args.max_length
    samples = []
    for seq_id in sorted(seqs):
        pos_map = seqs[seq_id]
        positions = sorted(pos_map)
        # contiguous runs only: a hole means the engine skipped tokens (prefix cache hit)
        run = [positions[0]]
        for p in positions[1:]:
            if p == run[-1] + 1:
                run.append(p)
            else:
                samples.extend(_slice(run, pos_map, T, seq_id))
                run = [p]
        samples.extend(_slice(run, pos_map, T, seq_id))
    del seqs
    print(f"{len(samples)} training samples of up to {T} tokens")

    N = len(samples)
    K = args.topk
    shapes = {"tokens": (N, T), "loss_mask": (N, T), "lengths": (N,),
              "aux": (N, T, NUM_AUX * HIDDEN),
              "topk_ids": (N, T, K), "topk_logp": (N, T, K), "residual": (N, T)}
    dtypes = {"tokens": np.int32, "loss_mask": np.uint8, "lengths": np.int32, "aux": np.uint16,
              "topk_ids": np.int32, "topk_logp": np.float32, "residual": np.float32}
    for name, shape in shapes.items():
        path = os.path.join(args.out, f"{name}.bin")
        with open(path, "wb") as f:
            f.truncate(int(np.prod(shape)) * np.dtype(dtypes[name]).itemsize)

    mem = {name: np.memmap(os.path.join(args.out, f"{name}.bin"), dtype=dtypes[name], mode="r+", shape=shape)
           for name, shape in shapes.items()}
    total_tokens = 0
    for i, (seq_id, start, rows) in enumerate(samples):
        n = len(rows)
        total_tokens += n
        tokens = np.array([r[0] for r in rows], dtype=np.int32)
        mem["tokens"][i, :n] = tokens
        mem["loss_mask"][i, :n] = 1
        mem["lengths"][i] = n
        aux = np.stack([r[1] for r in rows])
        mem["aux"][i, :n] = aux
        last = _bf16_from_bits(np.stack([r[2] for r in rows]))
        ids, logp, res = topk_target(last.to(device), head, K)
        mem["topk_ids"][i, :n] = ids.cpu().numpy()
        mem["topk_logp"][i, :n] = logp.cpu().numpy()
        mem["residual"][i, :n] = res.cpu().numpy()
        if (i + 1) % 50 == 0 or i + 1 == N:
            print(f"  {i + 1}/{N} samples, {total_tokens} tokens", flush=True)
    for m in mem.values():
        m.flush()

    meta = {
        "num_samples": N, "max_length": T, "topk": K, "hidden_size": HIDDEN,
        "num_aux_layers": NUM_AUX, "total_tokens": int(total_tokens),
        "source": args.raw, "model_dir": args.model_dir,
        "aux_layout": "target hidden states entering layers 37/38/39, mean over hc copies, concatenated",
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"cache written to {args.out}: {N} samples, {total_tokens} tokens, "
          f"{sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 2**30:.1f} GiB")
    return 0


def _bf16_from_bits(bits: np.ndarray) -> torch.Tensor:
    """uint16 array holding bfloat16 bit patterns -> torch bfloat16."""
    return torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16)


def _slice(run: list[int], pos_map: dict, T: int, seq_id: int):
    """Chop a contiguous run of positions into samples of at most T rows."""
    out = []
    for start in range(0, len(run), T):
        chunk = run[start:start + T]
        if len(chunk) < 8:                      # tail fragments are not worth a sample
            continue
        out.append((seq_id, chunk[0], [pos_map[p] for p in chunk]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", nargs="+", required=True, help="dump files written by the patched speculator")
    p.add_argument("--model-dir", required=True, help="checkpoint dir, for head.weight")
    p.add_argument("--out", required=True)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return build(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
