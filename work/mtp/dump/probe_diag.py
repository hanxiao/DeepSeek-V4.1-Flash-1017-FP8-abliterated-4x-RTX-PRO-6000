import os, struct
import numpy as np
p = "/data/mtp/raw/dump-0.bin"
HIDDEN = 5120; NUM_AUX = 3; NUM_HC = 4
ROW = 12 + (NUM_AUX + NUM_HC) * HIDDEN * 2
size = os.path.getsize(p)
INT = np.dtype([("token","<i4"),("pos","<i4"),("seq","<i4")])
ints = []; off = 0; rows = 0
with open(p, "rb") as f:
    while rows < 40000:
        f.seek(off); h = f.read(4)
        if len(h) < 4: break
        n = struct.unpack("<i", h)[0]
        if not (1 <= n <= 65536): break
        if off + 4 + n*ROW > size: break
        d = f.read(n*12)
        ints.append(np.frombuffer(d, dtype=INT, count=n))
        off += 4 + n*ROW; rows += n
a = np.concatenate(ints)
pos = a["pos"].astype(np.int64); seq = a["seq"]
print("rows read", len(a), "first 12 (pos,seq):", list(zip(pos[:12].tolist(), seq[:12].tolist())))
print("pos range", pos.min(), pos.max(), "seq range", seq.min(), seq.max())
consec = (pos[1:] == pos[:-1] + 1) & (seq[1:] == seq[:-1])
print("consec", int(consec.sum()), "of", len(pos)-1)
print("distinct seqs", len(np.unique(seq)))
