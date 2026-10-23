# shellcheck shell=bash
# Every non-secret value of the deployment. Sourced by the scripts in gcp/ and scripts/.
# A value already set in the environment, or in .env next to this file, wins.
# The two secrets, DS41_API_KEY and HF_TOKEN, are never set here.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_here/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$_here/.env"
  set +a
fi
export REPO_DIR="$_here"

# GCP
: "${PROJECT:=jinaai-dev}"
: "${VM_NAME:=ds41-flash-rtx-spot}"
: "${MACHINE:=g4-standard-192}"                  # 4x RTX PRO 6000 Blackwell, 192 vCPU, 720 GB RAM
: "${PROVISIONING:=SPOT}"                        # SPOT or STANDARD (on-demand)
: "${LOCAL_SSDS:=16}"                            # g4-standard-192 takes exactly 0 or 16 x 375 GB
: "${BOOT_DISK_SIZE:=400GB}"
: "${BOOT_DISK_TYPE:=hyperdisk-balanced}"
: "${GCE_IMAGE:=common-cu129-ubuntu-2404-nvidia-580-v20261019}"   # driver 580.178.04
: "${GCE_IMAGE_PROJECT:=deeplearning-platform-release}"
: "${NETWORK_TAG:=ds41-serve}"
: "${FIREWALL_RULE:=allow-ds41-https}"
: "${STATIC_IP_NAME:=ds41-flash-ip}"
: "${LABELS:=owner=hanxiao,purpose=ds41-flash-abliterated}"

# Zones tried in order, one round at a time. Spot quota for RTX PRO 6000 is 32 GPUs in every
# region; on-demand quota exists only in us-west1 (10) and us-central1.
: "${SPOT_ZONES:=us-west3-a us-east4-b us-east4-c us-east5-a us-east5-b us-east5-c us-south1-a us-south1-b us-west4-a us-west4-b us-west4-c us-east1-b us-east1-d us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b us-west1-c}"
: "${STANDARD_ZONES:=us-west1-a us-west1-b us-west1-c us-central1-b us-central1-c us-central1-f}"
: "${GRAB_MAX_SECS:=5400}"                       # give up after 90 min
: "${GRAB_SLEEP:=45}"                            # pause between rounds

# Model and server
: "${MODEL_REPO:=mrfakename/DeepSeek-V4.1-Flash-1017-abliterated}"
: "${MODEL_REV:=517efab9ccfef7f714dbd7d679f181a86f43ccf9}"
: "${SERVED_MODEL_NAME:=DeepSeek-V4.1-Flash-1017-abliterated}"
: "${IMAGE:=ghcr.io/local-inference-lab/vllm@sha256:fd74019fc217a541060093d18427c36330df6ac5700651b55593dafdbb936920}"

# Endpoint. Empty DOMAIN means <static-ip-with-dashes>.sslip.io, computed from the static IP.
: "${DOMAIN:=}"
: "${ACME_EMAIL:=}"                              # optional contact for Let's Encrypt

zone_of_vm() {
  gcloud compute instances list --project="$PROJECT" --filter="name=$VM_NAME" --format='value(zone.basename())' | head -1
}

domain_of_vm() {
  if [ -n "$DOMAIN" ]; then echo "$DOMAIN"; return; fi
  local z ip
  z="$(zone_of_vm)"
  ip="$(gcloud compute instances describe "$VM_NAME" --project="$PROJECT" --zone="$z" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
  echo "${ip//./-}.sslip.io"
}

vm_ssh() {
  local z
  z="${ZONE:-$(zone_of_vm)}"
  gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$z" --quiet "$@"
}
