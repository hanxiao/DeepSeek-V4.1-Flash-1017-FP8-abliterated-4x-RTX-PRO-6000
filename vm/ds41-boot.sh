#!/bin/bash
# Boot-time recovery. A stopped Flex-start or spot VM loses its local-SSD RAID: weights, Docker
# images and kernel caches. This rebuilds /data, downloads the weights again, restores the engine
# image, the tokenizer files and the kernel cache from the boot disk (/var/lib/ds41), and starts the
# engine named in /etc/ds41/engine: r21 (default), moet (r13) or lil. When that engine cannot be
# restored, it falls back to the next one. A reboot with /data intact just exits.
exec >> /var/log/ds41-boot.log 2>&1
echo "=== ds41-boot $(date -u +%FT%TZ)"
if mountpoint -q /data && [ -s /data/models/ds41-abl/model.safetensors.index.json ] \
   && docker ps --format '{{.Names}}' | grep -qx ds41; then
  echo "already serving"; exit 0
fi
ENGINE=$(cat /etc/ds41/engine 2>/dev/null || echo r21)
R21_IMAGE=vllm-moet-sm120:dsv41-nightly-df3bd094-r21
MOET_IMAGE=sha256:a576e4e797d7a787570d80d3f901472529e7a510b6ddbf32f51249c15a9d3d73
systemctl stop docker docker.socket 2>/dev/null
bash /opt/ds41/vm-setup.sh
systemctl start docker
bash /opt/ds41/vm-download.sh     # weights; also pulls the LiL image, the last fallback

restore() {   # restore IMAGE TARBALL: load the image from the boot disk unless Docker has it
  docker image inspect "$1" >/dev/null 2>&1 && return 0
  [ -s "$2" ] && zstd -dc "$2" | docker load
  docker image inspect "$1" >/dev/null 2>&1
}
cache() {     # cache BACKUP DIR: copy the kernel cache of an engine from the boot disk onto /data
  mkdir -p "/data/cache/$2"
  [ -d "/var/lib/ds41/$1" ] && cp -a "/var/lib/ds41/$1/." "/data/cache/$2/"
  return 0
}

for f in encoding-1017-vllm.py deepseek_v41-vllm.py deepseek_v41-moet.py deepseek_v41-r21.py; do
  [ -s "/var/lib/ds41/$f" ] && cp "/var/lib/ds41/$f" "/data/models/$f"
done
# The r13 and r21 images ship the same tokenizer file, so their patched overlays are identical.
[ -s /data/models/deepseek_v41-r21.py ] || cp /data/models/deepseek_v41-moet.py /data/models/deepseek_v41-r21.py 2>/dev/null

if [ "$ENGINE" = r21 ] && restore "$R21_IMAGE" /var/lib/ds41/r21-image.tar.zst; then
  cache r21-cache r21
  /usr/local/bin/ds41-start-r21.sh
elif [ "$ENGINE" != lil ] && restore "$MOET_IMAGE" /var/lib/ds41/moet-image.tar.zst; then
  cache moet-cache moet
  /usr/local/bin/ds41-start-moet.sh
else
  cache lil-cache lil-cache
  /usr/local/bin/ds41-start.sh
fi
echo "=== ds41-boot done ($ENGINE) $(date -u +%FT%TZ)"
