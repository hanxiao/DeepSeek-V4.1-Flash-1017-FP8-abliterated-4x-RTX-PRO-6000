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
MIN_SAMPLE = 8                                       # a sample shorter than this is not worth a step
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
            # the dump's last record is clipped by `docker stop` (a partial flush); pad it with value
            # bits instead of aborting an hour-long cache build
            print(f"  short read at offset {off}: {got} of {row_flt} bytes, padding", flush=True)
            buf[i, got:] = 0
    u16 = buf.view(np.uint16).reshape(n, -1)
    return u16[:, : NUM_AUX * HIDDEN], u16[:, NUM_AUX * HIDDEN:]


def load_head_weight(model_dir: str, dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """``head.weight`` [vocab, hidden] and the **backbone's** final ``norm.weight`` [hidden].

    The target's next-token distribution is ``head(norm(h))`` with the backbone's norm, not the
    drafter's ``mtp.2.norm``: they are different tensors and using the wrong one gives a plausible
    looking but wrong distribution.
    """
    from safetensors import safe_open

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    out = {}
    for name in ("head.weight", "norm.weight"):
        with safe_open(os.path.join(model_dir, index["weight_map"][name]), framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(name).to(dtype)
    return out["head.weight"], out["norm.weight"]


@torch.no_grad()
def topk_target(hidden: torch.Tensor, head: torch.Tensor, norm_w: torch.Tensor, k: int, eps: float = 1e-20,
                chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[n, H] collapsed hidden -> (ids [n,K] int32, logprobs [n,K] fp32, residual mass [n] fp32).

    Applies the backbone's final RMSNorm first: the dumped hidden state is the *pre-norm* HC
    stream, and skipping the norm scales the logits by the stream's RMS (~45x) which turns every
    target distribution into a near one-hot.
    """
    ids_out, logp_out, res_out = [], [], []
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start:start + chunk].float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w.float()
        logits = torch.nn.functional.linear(h, head.float())
        logp = torch.log_softmax(logits, dim=-1)
        vals, ids = logp.topk(k, dim=-1)
        ids_out.append(ids.to(torch.int32).cpu().numpy())
        logp_out.append(vals.cpu().numpy())
        res_out.append((1.0 - vals.exp().sum(-1)).clamp_min(0.0).cpu().numpy())
    return np.concatenate(ids_out), np.concatenate(logp_out), np.concatenate(res_out)


@torch.no_grad()
def _bf16_from_bits(bits: np.ndarray) -> torch.Tensor:
    """uint16 bf16 bit pattern -> torch.Tensor of the same shape (bf16)."""
    return torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16)


def probe_collapse(path: str, row_bytes: int, hc_blocks: int, head: torch.Tensor, norm_w: torch.Tensor,
                   device, limit: int = 40000) -> dict[str, float]:
    """Which stand-in for the missing HC collapse best reproduces the target's next token?

    For every pair of genuinely consecutive rows (same sequence, position + 1) the target's
    distribution from row i must put its argmax on the token at row i+1.  A correct (norm, head,
    collapse) pipeline scores 0.4-0.6; a wrong one scores ~0.  This is the check that catches a
    dump of the wrong tokens or a missing norm *before* an hour of training.
    """
    ints_all, flt_all = [], []
    with open(path, "rb") as f:
        while len(ints_all) < limit:
            head4 = f.read(4)
            if len(head4) < 4:
                break
            (n,) = struct.unpack("<i", head4)
            if n <= 0 or n > MAX_RECORD:
                break
            ibuf = f.read(n * INT_BYTES)
            fbuf = f.read(n * (row_bytes - INT_BYTES))   # n rows of float payload, not n*row - int
            if len(fbuf) < n * (row_bytes - INT_BYTES):
                break
            ints_all.append(np.frombuffer(ibuf, dtype=INT_DTYPE, count=n))
            flt_all.append(np.frombuffer(fbuf, dtype=np.uint16, count=n * (NUM_AUX + hc_blocks) * HIDDEN))
    ints = np.concatenate(ints_all)
    u16 = np.concatenate(flt_all).reshape(-1, (NUM_AUX + hc_blocks) * HIDDEN)
    tok = ints["token"].astype(np.int64)
    pos = ints["pos"]
    seq = ints["seq"]
    last = _bf16_from_bits(u16[:, NUM_AUX * HIDDEN:]).float().reshape(-1, hc_blocks, HIDDEN).to(device)
    consec = (pos[1:] == pos[:-1] + 1) & (seq[1:] == seq[:-1])
    if consec.sum() < 100:
        return {"none": 0.0}
    ok_cpu = torch.from_numpy(consec)
    ok = ok_cpu.to(device)
    nxt = torch.from_numpy(tok[1:])[ok_cpu].to(device)      # shift 1: predicts the NEXT token
    prev = torch.from_numpy(tok[:-1])[ok_cpu].to(device)    # shift 0: predicts its own token
    proxies = {"mean": last.mean(1)} | {f"s{j}": last[:, j] for j in range(hc_blocks)}
    out = {}
    chunk = 2048  # the 129M-row logits do not fit for a whole file; the argmax is per row
    for name, x in proxies.items():
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-20) * norm_w.float()
        am = torch.cat([torch.nn.functional.linear(h[i:i + chunk], head.float()).argmax(-1)
                        for i in range(0, h.shape[0], chunk)])
        am = am[:-1][ok]
        out[name] = float((am == nxt).float().mean())
        out[name + "_shift0"] = float((am == prev).float().mean())
    return out


def plan_samples(seq: np.ndarray, pos: np.ndarray, tok: np.ndarray, off: np.ndarray, T: int):
    """Sort by (seq, position), keep the last record per position, then pack **each sequence's whole
    span into samples of up to T rows, allowing position gaps**.

    The dump is not position-contiguous: a decode record holds one verify step's rows (bonus + k
    drafts) and the next record starts at the next accepted position, so a sequence arrives as many
    short runs separated by holes.  One sample per run would give a ~19-token sample per record
    (217 k samples, 99.8 % padding); packing across the holes gives ~T-token samples.

    Returns (samples, stats) with samples = [(token ids, payload offsets, true positions, valid)].
    ``valid`` is 0 for a row that follows a hole: its draft window would span the discontinuity, so
    ``loss_mask`` is zeroed there and no anchor is sampled across it.
    """
    order = np.lexsort((pos, seq))                      # stable: equal keys keep file order
    seq, pos, tok, off = seq[order], pos[order], tok[order], off[order]
    # keep the last of every run of equal (seq, pos): rejected speculative rows are re-forwarded
    keep = np.ones(len(seq), dtype=bool)
    if len(seq) > 1:
        same = (seq[1:] == seq[:-1]) & (pos[1:] == pos[:-1])
        keep[:-1] = ~same
    seq, pos, tok, off = seq[keep], pos[keep], tok[keep], off[keep]
    if len(seq) == 0:
        return [], {"runs": 0, "gaps": 0, "rows": 0, "seqs": 0}

    # a row starts a run when its sequence changes or its position does not continue the previous one
    run_start = np.ones(len(seq), dtype=bool)
    run_start[1:] = (seq[1:] != seq[:-1]) | (pos[1:] != pos[:-1] + 1)
    seq_change = np.ones(len(seq), dtype=bool)
    seq_change[1:] = seq[1:] != seq[:-1]

    samples = []
    for sq in np.unique(seq):
        idx = np.flatnonzero(seq == sq)
        valid_all = (~run_start[idx]).astype(np.uint8)
        valid_all[0] = 1                                # the first row of a sequence has no hole before it
        for start in range(0, len(idx), T):
            chunk = idx[start:start + T]
            if len(chunk) < MIN_SAMPLE:
                continue
            samples.append((tok[chunk], off[chunk], pos[chunk], valid_all[start:start + T]))
    stats = {
        "rows": int(len(seq)),
        "seqs": int(len(np.unique(seq))),
        "runs": int(run_start.sum()),
        "gaps": int(run_start.sum() - len(np.unique(seq))),   # holes = run starts that are not a new seq
        "mean_run": float(len(seq) / max(int(run_start.sum()), 1)),
    }
    return samples, stats


def build(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    head, norm_w = load_head_weight(args.model_dir)
    head, norm_w = head.to(device), norm_w.to(device)

    hc_blocks = detect_hc_blocks(args.raw) if args.hc_blocks == "auto" else int(args.hc_blocks)
    row_bytes = row_bytes_for(hc_blocks)
    print(f"row layout: {INT_BYTES} int bytes + {row_bytes - INT_BYTES} float bytes = {row_bytes} B "
          f"({NUM_AUX} aux + {hc_blocks} HC blocks of {HIDDEN}, bf16)")
    T = args.max_length
    samples, files, all_stats = [], [], []
    print("pass 1: scanning record headers")
    for path in args.raw:
        seq, pos, tok, off = scan(path, row_bytes)
        planned, stats = plan_samples(seq, pos, tok, off, T)
        all_stats.append(stats)
        print(f"  {path}: {stats['rows']} rows, {stats['seqs']} sequences, {stats['runs']} runs, "
              f"mean run {stats['mean_run']:.1f} rows", flush=True)
        for toks, offs, poss, valid in planned:
            samples.append((toks, offs, poss, valid))
            files.append(path)
        del seq, pos, tok, off
    total_tokens = sum(len(t[0]) for t in samples)
    assert samples, "no training sample survived: every run was shorter than MIN_SAMPLE"
    assert min(len(t[0]) for t in samples) > 0, "empty sample"
    print(f"{len(samples)} training samples of up to {T} tokens, {total_tokens} tokens "
          f"({total_tokens / len(samples):.0f} tokens/sample)")

    N, K = len(samples), args.topk
    shapes = {"tokens": (N, T), "loss_mask": (N, T), "lengths": (N,), "positions": (N, T),
              "aux": (N, T, NUM_AUX * HIDDEN),
              "topk_ids": (N, T, K), "topk_logp": (N, T, K), "residual": (N, T)}
    dtypes = {"tokens": np.int32, "loss_mask": np.uint8, "lengths": np.int32, "positions": np.int32,
              "aux": np.uint16,
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
        for i, ((toks, offsets, poss, valid), path) in enumerate(zip(samples, files)):
            n = len(offsets)
            assert n > 0, f"sample {i} is empty"
            aux, last = read_payload(fds[path], offsets, hc_blocks)
            mem["tokens"][i, :n] = toks
            mem["positions"][i, :n] = poss                # true token positions, not array indices
            mem["loss_mask"][i, :n] = valid               # 0 at a row that follows a hole
            mem["lengths"][i] = n
            mem["aux"][i, :n] = aux
            last_t = torch.from_numpy(np.ascontiguousarray(last)).view(torch.bfloat16)
            last_t = last_t.reshape(n, hc_blocks, HIDDEN).float()
            last_t = (last_t.mean(dim=1) if args.collapse == "mean"
                      else last_t[:, int(args.collapse[1:])]).to(torch.bfloat16).to(device)
            ids, logp, res = topk_target(last_t, head, norm_w, K)
            if not np.isfinite(logp).all():
                bad = int((~np.isfinite(logp)).any(axis=1).sum())
                print(f"  WARNING sample {i}: {bad}/{n} rows have non-finite target log-probs "
                      f"(non-finite bf16 in the dumped HC stream?)", flush=True)
                # a NaN: row is a row with no (the sanitised) target; give it so the loss cannot NaN
                logp = np.nan_to_num(logp, nan=0.0, posinf=0.0, neginf=0.0)
                res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
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
        "sample_formation": "runs of one sequence packed into <= max_length samples, gaps allowed; "
                            "loss_mask=0 on the first row after a hole",
        "run_stats": all_stats,
        "last_layout": f"mean over the {hc_blocks} HC copies; the exact collapse needs pre_mix, "
                       "which is not dumped",
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"cache written to {args.out}: {N} samples, {total_tokens} tokens, {size / 2**30:.1f} GiB")
    return 0


def cmd_validate(args) -> int:
    device = torch.device(args.device)
    head, norm_w = load_head_weight(args.model_dir)
    head, norm_w = head.to(device), norm_w.to(device)
    hc = detect_hc_blocks(args.raw)
    row_bytes = row_bytes_for(hc)
    print("collapse proxy -> P(argmax == next token) on consecutive rows:")
    scores = {}
    for path in args.raw:
        scores.update(probe_collapse(path, row_bytes, hc, head, norm_w, device, args.limit))
    for name, hit in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:6s} {hit:.3f}")
    best = max(scores.values()) if scores else 0.0
    if best < 0.25:
        raise SystemExit(
            f"best collapse proxy scores {best:.3f} (expected 0.4-0.6). The dump's tokens, the norm "
            f"or the head are wrong; check the dump's noise-token fraction first.")
    print(f"validate ok: best proxy {max(scores, key=scores.get)} at {best:.3f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", default="build", choices=["build", "validate"])
    p.add_argument("--raw", nargs="+", required=True, help="dump files written by the patched speculator")
    p.add_argument("--model-dir", required=True, help="checkpoint dir, for head.weight")
    p.add_argument("--out", required=True)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--collapse", default="mean", help="HC stand-in: mean | s0..s3")
    p.add_argument("--limit", type=int, default=40000, help="rows to probe in --validate")
    p.add_argument("--hc-blocks", default="auto",
                   help="HC copies in the dump's float payload (default: detect from the file)")
    a = p.parse_args(argv)
    return cmd_validate(a) if a.cmd == "validate" else build(a)


if __name__ == "__main__":
    sys.exit(main())
