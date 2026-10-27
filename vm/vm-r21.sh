#!/bin/bash
# Prepare the vLLM-Moet r21 engine on the VM (run as root after vm-download.sh and vm-encoding.sh).
# 1. The image: vLLM nightly df3bd094 + vLLM-Moet b0dc737, built from Dockerfile.sm120-dsv41-nightly
#    in about 12 min. It is in no registry, so a zstd copy (9.4 GB) stays on the boot disk, where
#    ds41-boot.sh restores it after a stop.
# 2. The tokenizer overlay: the image's vllm/tokenizers/deepseek_v41.py + deepseek_v41-moet.patch.
# 3. /etc/ds41/r21.env (the serving knobs) and /etc/ds41/engine=r21.
set -e
MOET_COMMIT=${MOET_COMMIT:-b0dc737}
R21_IMAGE=${R21_IMAGE:-vllm-moet-sm120:dsv41-nightly-df3bd094-r21}
SP=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers
mkdir -p /var/lib/ds41 /data/logs /etc/ds41
command -v zstd >/dev/null || apt-get install -y -q zstd >/dev/null
if ! docker image inspect "$R21_IMAGE" >/dev/null 2>&1; then
  if [ -s /var/lib/ds41/r21-image.tar.zst ]; then
    zstd -dc /var/lib/ds41/r21-image.tar.zst | docker load
  else
    mkdir -p /data/build
    [ -d /data/build/moet ] || git clone -q https://github.com/kacper-daftcode/vLLM-Moet /data/build/moet
    git -C /data/build/moet fetch -q
    git -C /data/build/moet checkout -q "$MOET_COMMIT"
    DOCKER_BUILDKIT=1 docker build -f /data/build/moet/Dockerfile.sm120-dsv41-nightly \
      -t "$R21_IMAGE" /data/build/moet > /data/logs/r21-build.log 2>&1
  fi
fi
[ -s /var/lib/ds41/r21-image.tar.zst ] || docker save "$R21_IMAGE" | zstd -T0 -3 -q -o /var/lib/ds41/r21-image.tar.zst
ORIG=/tmp/deepseek_v41.r21.orig.py
TOK=/data/models/deepseek_v41-r21.py
docker run --rm --entrypoint cat "$R21_IMAGE" "$SP/deepseek_v41.py" > "$ORIG"
patch -s -o "$TOK" "$ORIG" < /opt/ds41/deepseek_v41-moet.patch
cp "$TOK" /var/lib/ds41/deepseek_v41-r21.py
install -m 644 /opt/ds41/r21.env /etc/ds41/r21.env
echo r21 > /etc/ds41/engine
sha256sum "$ORIG" "$TOK"
echo R21_DONE
