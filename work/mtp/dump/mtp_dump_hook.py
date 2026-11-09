"""Engine-side recorder for the MTP training data. Appended to a copy of
``vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`` by ``dump/make_patch.py``.

One record per token the target model forwards (prefill chunks and decode steps alike):

    int32  n_tokens
    then n_tokens * 3 int32      token_id, position, seq_id
    then n_tokens * (3 + hc) * 5120 int16   bf16 aux[3*5120] then the model's hidden state

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

import torch

class MTPDumper:
    def __init__(self, path: str, hidden: int = 5120, num_aux: int = 3):
        self.hidden = hidden
        self.num_aux = num_aux
        self.path = path
        self.fh = open(path, "wb", buffering=1024 * 1024 * 64)
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
    def record(self, input_batch, input_ids, positions, aux_hidden_states, last_hidden_states, num_tokens):
        if not aux_hidden_states:
            return
        dev = input_ids.device
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
        for r in range(len(req_ids)):
            seq[qsl[r]: qsl[r + 1]] = sid[r]
        aux = torch.stack(aux_hidden_states, dim=1)[:n].reshape(n, -1)      # [n, 3*5120]
        ints = torch.stack([
            input_ids[:n].to(torch.int32),
            positions[:n].to(torch.int32),
            seq,
        ], dim=1).cpu().contiguous()                                        # [n, 3] int32
        floats = torch.cat([
            aux.to(torch.bfloat16).view(torch.int16),
            last_hidden_states[:n].to(torch.bfloat16).view(torch.int16),
        ], dim=1).cpu().contiguous()                                        # [n, 4*5120] int16
        self.fh.write(struct.pack("<i", n))
        self.fh.write(ints.numpy().tobytes())
        self.fh.write(floats.numpy().tobytes())
        self.records += n

    def close(self):
        self.fh.close()
        print(f"[mtp-dump] {self.records} token records -> {self.path}", flush=True)


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
        speculator.input_buffers.input_ids,
        speculator.input_buffers.positions,
        aux_hidden_states,
        last_hidden_states,
        num_tokens,
    )
