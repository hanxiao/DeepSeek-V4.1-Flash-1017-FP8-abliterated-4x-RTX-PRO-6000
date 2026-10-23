#!/bin/bash
# Boot-time recovery. A spot preemption (or any stop) discards the local-SSD RAID: weights, docker
# image and the b12x/JIT cache are gone. This rebuilds /data, re-downloads (~4.5 min at ~1.9 GB/s),
# re-pulls the image, restores the JIT cache backup from the boot disk and starts the server.
# A plain reboot just remounts and restarts.
exec >> /var/log/ds41-boot.log 2>&1
echo "=== ds41-boot $(date -u +%FT%TZ)"
if mountpoint -q /data && [ -s /data/models/ds41-abl/model.safetensors.index.json ] \
   && docker ps --format '{{.Names}}' | grep -qx ds41; then
  echo "already serving"; exit 0
fi
systemctl stop docker docker.socket 2>/dev/null
bash /opt/ds41/vm-setup.sh
systemctl start docker
bash /opt/ds41/vm-download.sh
mkdir -p /data/cache/lil-cache
[ -d /var/lib/ds41/lil-cache ] && cp -a /var/lib/ds41/lil-cache/. /data/cache/lil-cache/
cp /var/lib/ds41/encoding-1017-vllm.py /data/models/encoding-1017-vllm.py
cp /var/lib/ds41/deepseek_v41-vllm.py /data/models/deepseek_v41-vllm.py
/usr/local/bin/ds41-start.sh
echo "=== ds41-boot done $(date -u +%FT%TZ)"
