#!/usr/bin/env python3
"""Turn the engine's raw hidden-state dump into the training cache.

Input  : ``$VLLM_MTP_DUMP`` records written by ``dump/speculator.py`` (layout in
         ``dump/mtp_dump_hook.py``):

             int32 n
             n * 3 int32                      token_id, position, seq_id
             n * 4*5120 int16                 bf16 aux[3*5120] then bf16 last[4*5120]

         ``aux`` is the target hidden state entering layers 37/38/39 (mean-pooled over the HC copies
         by the engine itself).  ``last`` is the model's *pre-collapse* hidden state: with MTP
         enabled ``DeepseekV4Model`` keeps the full HC stream, 4 copies of 5120, and only afterwards
         does ``hc_collapse_triton(hidden_states, pre_mix)`` + ``norm`` produce the 5120 that feeds
         ``head.weight``.  ``pre_mix`` is the last layer's per-token FFN mix and is not in the dump,
         so we stand in for the collapse with the **mean over the 4 HC copies**.  RMSNorm makes that
         exact whenever the four mix weights are equal (a uniform scale cancels), so the only error
         is the per-token non-uniformity of ``pre_mix`` -- a direction change, not a scale error.

Output : a directory of flat memmaps the trainer reads without any parsing:

           tokens.bin      int32  [N, T]        input token ids
           loss_mask.bin   uint8  [N, T]        1 = supervised
           lengths.bin     int32  [N]           valid rows per sample
           aux.bin         bf16   [N, T, 15360] target hidden states entering layers 37/38/39
           topk_ids.bin    int32  [N, T, K]     target top-K vocabulary ids
           topk_logp.bin   float32[N, T, K]     their log-probabilities (temperature 1)
           residual.bin    float32[N, T]        target mass outside the top-K support
           meta.json                            shapes, K, source, token totals, collapse rule

Two passes, bounded memory: the dump is 71.7 kB *per row* (729 GB for the 10.2 M rows of the first
run), so pass 1 reads only the 12-byte integer block of every record and remembers where each row's
float payload lives; pass 2 preads the rows a sample needs.  Nothing holds the file in RAM.

Sequences are reassembled per request id and ordered by position; a position the engine processed
more than once (rejected speculative rows are re-forwarded) keeps its **last** record.
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
NUM_AUX = 3                                          # target layers 37/38/39
NUM_HC = 4                                           # the model keeps the full HC stream when MTP is on
MAX_RECORD = 65536                                   # sanity bound on a record's row count
INT_BYTES = 3 * 4                                    # token, position, sequence
FLT_BYTES = (NUM_AUX + NUM_HC) * HIDDEN * 2          # bf16 aux then bf16 the 4 HC copies
ROW_BYTES = INT_BYTES + FLT_BYTES
INT_DTYPE = np.dtype([("token", "<i4"), ("pos", "<i4"), ("seq", "<i4")])


def row_bytes_for(hc_blocks: int) -> int:
    return INT_BYTES + (NUM_AUX + hc_blocks) * HIDDEN * 2


def _scan_headers(path: str, row_bytes: int) -> tuple[int, int]:
    """Walk record headers assuming ``row_bytes``; return (rows, bytes consumed).

    The int block is read (it is what tells us the record's row count), the float payload is only
    seeked over, so this costs two syscalls per record and no I/O.
    """
    size = os.path.getsize(path)
    rows = consumed = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = struct.unpack("<i", head)
            if n <= 0 or n > MAX_RECORD:
                break
            payload = f.tell()
            if payload + n * row_bytes > size:
                break
            f.seek(payload + n * INT_BYTES + n * (row_bytes - INT_BYTES))
            rows += n
            consumed = payload + n * row_bytes
    return rows, consumed


def detect_hc_blocks(paths: list[str]) -> int:
    """The dump's float payload is 3 aux blocks plus the model's HC stream; how many HC copies the
    engine kept is a property of the build, so identify it from the file itself: the right row size
    is the one that lets the header walk consume (almost) the whole file."""
    candidates = [NUM_HC, 3, 5, 2, 6, 1, 7, 0, 8]
    for hc in candidates:
        rb = row_bytes_for(hc)
        ok = True
        for path in paths:
            size = os.path.getsize(path)
            rows, consumed = _scan_headers(path, rb)
            if rows == 0 or consumed < 0.99 * size:
                ok = False
                break
        if ok:
            print(f"detected row layout: {INT_BYTES} int bytes + {row_bytes_for(hc) - INT_BYTES} float "
                  f"bytes = {rb} B ({NUM_AUX} aux + {hc} HC blocks of {HIDDEN})", flush=True)
            return hc
    sizes = ", ".join(f"{k}:{row_bytes_for(k)}B" for k in candidates)
    raise SystemExit(f"no row layout consumes these files (tried {sizes}); the dump is not what "
                     f"dump/speculator.py writes")


def scan(path: str, row_bytes: int = ROW_BYTES) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pass 1 for one file: every row's (seq, pos, token, offset of its float payload), file order."""
    seq_parts, pos_parts, off_parts, tok_parts = [], [], [], []
    size = os.path.getsize(path)
    n_rows = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (n,) = struct.unpack("<i", head)
            if n <= 0 or n > MAX_RECORD:
                print(f"  {path}: bad record length {n} at offset {f.tell() - 4}, stopping", flush=True)
                break
            payload = f.tell()
            if payload + n * row_bytes > size:
                print(f"  {path}: truncated tail record ({n} rows) at offset {payload - 4}, "
                      f"{size - payload} of {n * ROW_BYTES} bytes present, stopping", flush=True)
                break
            ibuf = f.read(n * INT_BYTES)
            if len(ibuf) != n * INT_BYTES:
                print(f"  {path}: short integer block at offset {payload}, stopping", flush=True)
                break
            ints = np.frombuffer(ibuf, dtype=INT_DTYPE, count=n)
            f.seek(payload + n * row_bytes)                 # past this record's float payload
            if n_rows == 0:
                # self-synchronising check: the next 4 bytes must be a plausible row count
                nxt = f.read(4)
                nxt_n = struct.unpack("<i", nxt)[0] if len(nxt) == 4 else 0
                if len(nxt) == 4 and not (0 < nxt_n <= MAX_RECORD):
                    raise SystemExit(
                        f"{path}: record 0 says the row size is {row_bytes} B, but the next header "
                        f"reads {nxt_n} -- the layout is wrong, not the data.")
                f.seek(payload + n * row_bytes)
            seq_parts.append(ints["seq"].astype(np.int32))
            pos_parts.append(ints["pos"].astype(np.int32))
            tok_parts.append(ints["token"].astype(np.int32))
            off_parts.append(payload + n * INT_BYTES + np.arange(n, dtype=np.int64) * row_bytes)
            n_rows += n
    print(f"  {path}: {size / 2**30:.1f} GiB, {n_rows} rows, "
          f"{size / row_bytes:.0f} rows if the file were exact", flush=True)
    return (np.concatenate(seq_parts), np.concatenate(pos_parts),
            np.concatenate(tok_parts), np.concatenate(off_parts))


def read_payload(fd: int, offsets: np.ndarray, hc_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: read one sample's rows -> (aux bits [n, 3*HIDDEN], last bits [n, hc*HIDDEN])."""
    n = len(offsets)
    row_flt = (NUM_AUX + hc_blocks) * HIDDEN * 2
    buf = np.empty((n, row_flt), dtype=np.uint8)
    for i, off in enumerate(offsets):
        got = os.preadv(fd, [buf[i]], int(off))
        if got != row_flt:
            raise SystemExit(f"short read at offset {off}: {got} of {row_flt} bytes")
    u16 = buf.view(np.uint16).reshape(n, -1)
    return u16[:, : NUM_AUX * HIDDEN], u16[:, NUM_AUX * HIDDEN:]


def load_head_weight(model_dir: str, dtype=torch.bfloat16) -> torch.Tensor:
    """``head.weight`` [vocab, hidden] from the checkpoint's shards, without touching the rest."""
    from safetensors import safe_open

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    shard = index["weight_map"]["head.weight"]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor("head.weight").to(dtype)


@torch.no_grad()
def topk_target(hidden: torch.Tensor, head: torch.Tensor, k: int,
                chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[n, H] final hidden -> (ids [n,K] int32, logprobs [n,K] fp32, residual mass [n] fp32)."""
    ids_out, logp_out, res_out = [], [], []
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start:start + chunk].float()
        logits = torch.nn.functional.linear(h, head.float())
        logp = torch.log_softmax(logits, dim=-1)
        vals, ids = logp.topk(k, dim=-1)
        ids_out.append(ids.to(torch.int32).cpu().numpy())
        logp_out.append(vals.cpu().numpy())
        res_out.append((1.0 - vals.exp().sum(-1)).clamp_min(0.0).cpu().numpy())
    return np.concatenate(ids_out), np.concatenate(logp_out), np.concatenate(res_out)


def plan_samples(seq: np.ndarray, pos: np.ndarray, tok: np.ndarray, off: np.ndarray, T: int):
    """Sort by (seq, pos), keep the last record per position, cut contiguous runs into <= T rows.

    Returns a list of (token ids [n], payload offsets [n]) in sample order.
    """
    order = np.lexsort((pos, seq))                      # stable: equal keys keep file order
    seq, pos, tok, off = seq[order], pos[order], tok[order], off[order]
    # keep the last of every run of equal (seq, pos): rejected speculative rows are re-forwarded
    keep = np.ones(len(seq), dtype=bool)
    if len(seq) > 1:
        same = (seq[1:] == seq[:-1]) & (pos[1:] == pos[:-1])
        keep[:-1] = ~same                               # drop a row when its successor is the same key
    seq, pos, tok, off = seq[keep], pos[keep], tok[keep], off[keep]

    samples = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i < len(seq) and seq[i] == seq[i - 1] and pos[i] == pos[i - 1] + 1:
            continue
        for s in range(start, i, T):
            if min(i, s + T) - s >= 8:                  # tail fragments are not worth a sample
                samples.append((tok[s:min(i, s + T)], off[s:min(i, s + T)]))
        start = i
    return samples


def build(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    head = load_head_weight(args.model_dir).to(device)

    hc_blocks = detect_hc_blocks(args.raw) if args.hc_blocks == "auto" else int(args.hc_blocks)
    row_bytes = row_bytes_for(hc_blocks)
    print(f"row layout: {INT_BYTES} int bytes + {row_bytes - INT_BYTES} float bytes = {row_bytes} B "
          f"({NUM_AUX} aux + {hc_blocks} HC blocks of {HIDDEN}, bf16)")
    T = args.max_length
    samples, files = [], []
    print("pass 1: scanning record headers")
    for path in args.raw:
        seq, pos, tok, off = scan(path, row_bytes)
        print(f"  {path}: {len(seq)} rows, {len(np.unique(seq))} sequences")
        for toks, offs in plan_samples(seq, pos, tok, off, T):
            samples.append((toks, offs))
            files.append(path)
        del seq, pos, tok, off
    total_tokens = sum(len(t) for t, _ in samples)
    print(f"{len(samples)} training samples of up to {T} tokens, {total_tokens} tokens")

    N, K = len(samples), args.topk
    shapes = {"tokens": (N, T), "loss_mask": (N, T), "lengths": (N,),
              "aux": (N, T, NUM_AUX * HIDDEN),
              "topk_ids": (N, T, K), "topk_logp": (N, T, K), "residual": (N, T)}
    dtypes = {"tokens": np.int32, "loss_mask": np.uint8, "lengths": np.int32, "aux": np.uint16,
              "topk_ids": np.int32, "topk_logp": np.float32, "residual": np.float32}
    for name, shape in shapes.items():
        with open(os.path.join(args.out, f"{name}.bin"), "wb") as f:
            f.truncate(int(np.prod(shape)) * np.dtype(dtypes[name]).itemsize)
    mem = {name: np.memmap(os.path.join(args.out, f"{name}.bin"), dtype=dtypes[name], mode="r+", shape=shape)
           for name, shape in shapes.items()}

    print("pass 2: reading payloads, collapsing the HC stream, recomputing the target top-K")
    done = 0
    fds = {path: os.open(path, os.O_RDONLY) for path in args.raw}
    try:
        for i, ((toks, offsets), path) in enumerate(zip(samples, files)):
            n = len(offsets)
            aux, last = read_payload(fds[path], offsets, hc_blocks)
            mem["tokens"][i, :n] = toks
            mem["loss_mask"][i, :n] = 1
            mem["lengths"][i] = n
            mem["aux"][i, :n] = aux
            last_t = torch.from_numpy(np.ascontiguousarray(last)).view(torch.bfloat16)
            last_t = last_t.reshape(n, hc_blocks, HIDDEN).float().mean(dim=1).to(torch.bfloat16).to(device)
            ids, logp, res = topk_target(last_t, head, K)
            if not np.isfinite(logp).all():
                bad = int((~np.isfinite(logp)).any(axis=1).sum())
                print(f"  WARNING sample {i}: {bad}/{n} rows have non-finite target log-probs "
                      f"(non-finite bf16 in the dumped HC stream?)", flush=True)
            mem["topk_ids"][i, :n] = ids
            mem["topk_logp"][i, :n] = logp
            mem["residual"][i, :n] = res
            done += n
            if (i + 1) % 50 == 0 or i + 1 == N:
                print(f"  {i + 1}/{N} samples, {done} tokens", flush=True)
    finally:
        for fd in fds.values():
            os.close(fd)
    for m in mem.values():
        m.flush()

    meta = {
        "num_samples": N, "max_length": T, "topk": K, "hidden_size": HIDDEN,
        "num_aux_layers": NUM_AUX, "num_hc": hc_blocks, "row_bytes": row_bytes,
        "total_tokens": int(total_tokens), "source": args.raw, "model_dir": args.model_dir,
        "aux_layout": "target hidden states entering layers 37/38/39, mean over hc copies (as served)",
        "last_layout": f"mean over the {hc_blocks} HC copies; the exact collapse needs pre_mix, "
                       "which is not dumped",
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"cache written to {args.out}: {N} samples, {total_tokens} tokens, {size / 2**30:.1f} GiB")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", nargs="+", required=True, help="dump files written by the patched speculator")
    p.add_argument("--model-dir", required=True, help="checkpoint dir, for head.weight")
    p.add_argument("--out", required=True)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hc-blocks", default="auto",
                   help="HC copies in the dump's float payload (default: detect from the file)")
    return build(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
