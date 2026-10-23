#!/bin/bash
# Install docker + nvidia-container-toolkit if missing. The common-cu129 Ubuntu 24.04 image ships
# the NVIDIA driver but neither of these.
set -x
export DEBIAN_FRONTEND=noninteractive
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q && apt-get install -y -q nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker --version; nvidia-ctk --version | head -1
echo DOCKER_DONE
