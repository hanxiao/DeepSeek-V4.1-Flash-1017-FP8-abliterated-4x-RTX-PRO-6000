#!/bin/bash
# Build the encoding file that ds41-start.sh mounts over vllm/tokenizers/deepseek_v41_encoding.py:
# the checkpoint's own 1017 encoding/encoding.py plus encoding-1017-vllm.patch (two lines).
# Keeps a copy on the boot disk, since a preemption wipes /data.
set -e
SRC=/data/models/ds41-abl/encoding/encoding.py
OUT=/data/models/encoding-1017-vllm.py
patch -s -o "$OUT" "$SRC" < /opt/ds41/encoding-1017-vllm.patch
mkdir -p /var/lib/ds41
cp "$OUT" /var/lib/ds41/encoding-1017-vllm.py
sha256sum "$SRC" "$OUT"
echo ENCODING_DONE
