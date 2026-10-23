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
# Second patch: vLLM tokenizers/deepseek_v41.py turns literal placeholder text into plain ASCII,
# so a web page or file that contains the placeholder cannot become a phantom image slot.
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
TOK=/data/models/deepseek_v41-vllm.py
docker run --rm --entrypoint cat "$IMAGE" /opt/venv/lib/python3.12/site-packages/vllm/tokenizers/deepseek_v41.py > /tmp/deepseek_v41.orig.py
patch -s -o "$TOK" /tmp/deepseek_v41.orig.py < /opt/ds41/deepseek_v41-placeholder.patch
cp "$TOK" /var/lib/ds41/deepseek_v41-vllm.py
sha256sum "$SRC" "$OUT" /tmp/deepseek_v41.orig.py "$TOK"
echo ENCODING_DONE
