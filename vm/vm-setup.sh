#!/bin/bash
# Run on the VM as root. Idempotent: reuses an existing RAID, creates one on fresh local SSDs.
set -x
# 1) RAID0 over the 16 local NVMe (Titanium SSD) -> /data
if ! mountpoint -q /data; then
  apt-get install -y mdadm xfsprogs >/dev/null 2>&1 || true
  mdadm --assemble --scan 2>/dev/null || true
  MD=$(awk '/^md[0-9]+ : active/{print "/dev/"$1; exit}' /proc/mdstat)
  if [ -z "$MD" ]; then
    # shellcheck disable=SC2010
    DEVS=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | grep -v part | sort)
    N=$(echo "$DEVS" | wc -l)
    # shellcheck disable=SC2086
    mdadm --create /dev/md0 --level=0 --raid-devices="$N" $DEVS --force --run
    mkfs.xfs -f /dev/md0
    MD=/dev/md0
  fi
  mkdir -p /data && mount -o noatime,discard "$MD" /data
  chmod 777 /data
fi
df -h /data
# 2) docker data-root on /data (the image is 44 GB, the boot disk stays clean)
if ! grep -q '"/data/docker"' /etc/docker/daemon.json 2>/dev/null; then
  systemctl stop docker
  python3 -c "import json,os;p='/etc/docker/daemon.json';d=json.load(open(p)) if os.path.exists(p) else {};d['data-root']='/data/docker';json.dump(d,open(p,'w'),indent=2)"
  systemctl start docker
fi
docker info 2>/dev/null | grep -E 'Docker Root|Runtimes'
# 3) hf cli
apt-get install -y python3-venv >/dev/null 2>&1 || true
[ -x /opt/hf/bin/hf ] || { python3 -m venv /opt/hf && /opt/hf/bin/pip install -q "huggingface_hub[hf_xet]==2.2.0" "hf-xet==1.7.0"; }
echo SETUP_DONE
