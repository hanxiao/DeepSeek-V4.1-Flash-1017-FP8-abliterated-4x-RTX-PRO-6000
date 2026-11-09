"""Engine-side recorder for the MTP training data. Appended to a copy of
``vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`` by ``dump/make_patch.py``.

One record per token the target model forwards (prefill chunks and decode steps alike):

    int32  n_tokens
    then n_tokens * 3 int32      token_id, position, seq_id
    then n_tokens * (3 + hc) * 5120 int16   bf16 aux[3*5120] then the model's hidden state

``token_id`` / ``position`` are the **target's** rows: the token at that absolute position, read out
of ``input_batch.token_ids_cpu``, with the position derived from the drafter's query positions
(query slot 0 of a request sits at ``p+1``, so the target's row ``j`` is at ``p+j``).  The
speculator's own ``input_buffers.input_ids`` must NOT be used: they are the *drafter's* query ids,
i.e. the anchor token followed by ``parallel_drafting_token_id`` (128799) for every other slot.

``hc`` is the number of HC copies the model keeps: with MTP/dspark enabled it returns the full HC
stream (hc = 4 for V4.1, i.e. 4*5120), not the collapsed 5120.  ``dump/reassemble.py`` detects the
row size from the file itself rather than assuming either.

``aux`` is the target hidden state entering layers 37/38/39 (mean-pooled over the hc copies, the
same tensor the served drafter consumes).  The trailing block is the model's hidden state *before*
the final hc collapse + norm, which is what feeds ``head.weight``; the offline pass mean-collapses
the HC copies (the exact collapse needs the last layer's per-token pre-mix, which is not dumped) and
recomputes the target's top-K next-token distribution.

Only rank 0 writes: with TP4 and no sequence parallelism the target hidden states are replicated.
"""
import os
import struct

import numpy as np
import torch

class MTPDumper:
    def __init__(self, path: str, hidden: int = 5120, num_aux: int = 3, noise_token_id: int = 128799):
        self.hidden = hidden
        self.num_aux = num_aux
        self.noise_token_id = noise_token_id
        self.noise_hits = 0
        self.tok_total = 0
        self.bad_rows = 0
        self._flushed = 0
        self.flush_every = 20000            # flush every ~20000 rows so a crash keeps the dump
        self.path = path
        self.fh = open(path, "wb", buffering=1024 * 1024 * 8)
        self.staging = None
        self.pending = 0
        self.seq_of_req = {}
        self.next_seq = 0
        self.records = 0

    def seq_ids(self, req_ids):
        out = []
        for rid in req_ids:
            if rid not in self.seq_of_req:
                self.seq_of_req[rid] = self.next_seq
                self.next_seq += 1
            out.append(self.seq_of_req[rid])
        return torch.tensor(out, dtype=torch.int32)

    @torch.no_grad()
    def record(self, input_batch, query_positions, num_query_per_req, aux_hidden_states,
               last_hidden_states, num_tokens):
        if not aux_hidden_states:
            return
        dev = query_positions.device
        n = int(num_tokens)
        # request id per row, from the batch's per-request query boundaries
        qsl = input_batch.query_start_loc
        if torch.is_tensor(qsl):
            qsl = qsl[: input_batch.num_reqs + 1].tolist()
        else:
            qsl = list(qsl)[: input_batch.num_reqs + 1]
        req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
        seq = torch.full((n,), -1, dtype=torch.int32, device=dev)
        sid = self.seq_ids(req_ids).to(dev)
        # the target's tokens: row j of request r sits at absolute position p + j, where p is one
        # before the request's first query position
        tok = torch.full((n,), -1, dtype=torch.int32, device=dev)
        posn = torch.full((n,), -1, dtype=torch.int32, device=dev)
        qpos = query_positions[: len(req_ids) * num_query_per_req].reshape(len(req_ids), num_query_per_req)
        tok_ids_cpu = input_batch.token_ids_cpu
        for r in range(len(req_ids)):
            start, end = qsl[r], qsl[r + 1]
            p0 = int(qpos[r, 0]) - 1
            for j in range(end - start):
                pos = p0 + j
                if 0 <= pos < tok_ids_cpu.shape[1]:
                    tok[start + j] = int(tok_ids_cpu[r, pos])
                    posn[start + j] = pos
            seq[start:end] = sid[r]
        self.noise_hits += int((tok == int(self.noise_token_id)).sum())
        self.tok_total += n
        aux = torch.stack(aux_hidden_states, dim=1)[:n].reshape(n, -1)      # [n, 3*5120]
        ints = torch.stack([tok, posn, seq], dim=1).cpu().contiguous()      # [n, 3] int32
        floats = torch.cat([
            aux.to(torch.bfloat16).view(torch.int16),
            last_hidden_states[:n].to(torch.bfloat16).view(torch.int16),
        ], dim=1).cpu().contiguous()                                        # [n, 4*5120] int16
        self.fh.write(struct.pack("<i", n))
        self.fh.write(ints.numpy().tobytes())
        self.fh.write(floats.numpy().tobytes())
        self.records += n

    def close(self):
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        frac = self.noise_hits / max(self.tok_total, 1)
        print(f"[mtp-dump] {self.records} token rows -> {self.path}; "
              f"noise-token fraction {frac:.4f} (must be ~0: >0.2 means the drafter's query ids "
              f"were recorded instead of the target's tokens); {self.bad_rows} out-of-range rows",
              flush=True)


_DUMPER = None


def maybe_init_dumper():
    """Called once from the patched speculator's __init__."""
    global _DUMPER
    path = os.environ.get("VLLM_MTP_DUMP")
    if not path or _DUMPER is not None:
        return _DUMPER
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        _DUMPER = False
        return _DUMPER
    _DUMPER = MTPDumper(path)
    return _DUMPER


def dump_step(speculator, input_batch, last_hidden_states, aux_hidden_states, dummy_run, is_profile, num_tokens):
    d = _DUMPER
    if d is None or d is False or dummy_run or is_profile:
        return
    d.record(
        input_batch,
        speculator.input_buffers.positions,      # the DRAFTER's query positions, used as an index
        int(speculator.num_query_per_req),
        aux_hidden_states,
        last_hidden_states,
        num_tokens,
    )
