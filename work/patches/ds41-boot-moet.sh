#!/bin/bash
# Boot-time recovery for the vLLM-Moet engine (replaces vm/ds41-boot.sh after the cutover).
# A spot preemption discards the local-SSD RAID: weights, docker images and caches are gone.
# Same as ds41-boot.sh, plus: the Moet image is not in a registry, so it is restored from the
# boot-disk copy /var/lib/ds41/moet-image.tar.zst (fallback: gs://PROJECT-ds41-cache), and
# the Moet kernel cache comes back from /var/lib/ds41/moet-cache.
exec >> /var/log/ds41-boot.log 2>&1
echo "=== ds41-boot (moet) $(date -u +%FT%TZ)"
MOET_IMAGE_ID=sha256:a576e4e797d7a787570d80d3f901472529e7a510b6ddbf32f51249c15a9d3d73
if mountpoint -q /data && [ -s /data/models/ds41-abl/model.safetensors.index.json ] \
   && docker ps --format '{{.Names}}' | grep -qx ds41; then
  echo "already serving"; exit 0
fi
systemctl stop docker docker.socket 2>/dev/null
bash /opt/ds41/vm-setup.sh
systemctl start docker
# weights (vm-download.sh also pulls the LiL image, kept for rollback)
bash /opt/ds41/vm-download.sh
if ! docker image inspect "$MOET_IMAGE_ID" >/dev/null 2>&1; then
  if [ -s /var/lib/ds41/moet-image.tar.zst ]; then
    zstd -dc /var/lib/ds41/moet-image.tar.zst | docker load
  else
    gcloud storage cat gs://PROJECT-ds41-cache/moet-image.tar.zst | zstd -dc | docker load
  fi
fi
docker image inspect "$MOET_IMAGE_ID" >/dev/null 2>&1 || { echo "Moet image missing, starting LiL"; MOET_FAIL=1; }
mkdir -p /data/cache/lil-cache /data/cache/moet
[ -d /var/lib/ds41/lil-cache ] && cp -a /var/lib/ds41/lil-cache/. /data/cache/lil-cache/
[ -d /var/lib/ds41/moet-cache ] && cp -a /var/lib/ds41/moet-cache/. /data/cache/moet/
cp /var/lib/ds41/encoding-1017-vllm.py /data/models/encoding-1017-vllm.py
cp /var/lib/ds41/deepseek_v41-vllm.py /data/models/deepseek_v41-vllm.py
cp /var/lib/ds41/deepseek_v41-moet.py /data/models/deepseek_v41-moet.py
[ -s /var/lib/ds41/deepseek_v41-r21.py ] && cp /var/lib/ds41/deepseek_v41-r21.py /data/models/deepseek_v41-r21.py
ENGINE=$(cat /etc/ds41/engine 2>/dev/null || echo moet)
R21=vllm-moet-sm120:dsv41-nightly-df3bd094-r21
if [ "$ENGINE" = r21 ] && ! docker image inspect $R21 >/dev/null 2>&1 && [ -s /var/lib/ds41/r21-image.tar.zst ]; then
  zstd -dc /var/lib/ds41/r21-image.tar.zst | docker load
fi
if [ "$ENGINE" = r21 ] && docker image inspect $R21 >/dev/null 2>&1; then
  mkdir -p /data/cache/r21; [ -d /var/lib/ds41/r21-cache ] && cp -a /var/lib/ds41/r21-cache/. /data/cache/r21/
  /usr/local/bin/ds41-start-r21.sh
elif [ -z "${MOET_FAIL:-}" ] && [ "$ENGINE" != lil ]; then
  /usr/local/bin/ds41-start-moet.sh
else
  /usr/local/bin/ds41-start.sh
fi
echo "=== ds41-boot done $(date -u +%FT%TZ)"
