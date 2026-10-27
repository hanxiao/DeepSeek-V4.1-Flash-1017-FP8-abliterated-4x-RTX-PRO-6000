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
: "${VM_NAME:=ds41-fraser-asia-southeast2-c}"     # the serving VM; Flex-start names it <FLEX_PREFIX>-<zone>
: "${MACHINE:=g4-standard-192}"                  # 4x RTX PRO 6000 Blackwell, 192 vCPU, 720 GB RAM
: "${PROVISIONING:=FLEX_START}"                  # FLEX_START (gcp/flex.sh), SPOT or STANDARD (gcp/grab.sh)
: "${LOCAL_SSDS:=16}"                            # g4-standard-192 takes exactly 0 or 16 x 375 GB
: "${BOOT_DISK_SIZE:=400GB}"
: "${BOOT_DISK_TYPE:=hyperdisk-balanced}"
: "${GCE_IMAGE:=common-cu129-ubuntu-2404-nvidia-580-v20261019}"   # driver 580.178.04
: "${GCE_IMAGE_PROJECT:=deeplearning-platform-release}"
: "${BOOT_SNAPSHOT:=}"                            # Flex-start: boot disk from this snapshot instead of GCE_IMAGE
: "${NETWORK_TAG:=ds41-serve}"
: "${FIREWALL_RULE:=allow-ds41-https}"
: "${STATIC_IP_NAME:=ds41-flash-ip}"
: "${LABELS:=owner=hanxiao,purpose=ds41-flash-abliterated}"

# Flex-start runs up to FLEX_MAX_RUN without preemption and then stops. It draws on the preemptible
# GPU quota (32 RTX PRO 6000 in every region of the project). These 46 zones offer g4-standard-192.
: "${FLEX_PREFIX:=ds41-flex}"
: "${FLEX_MAX_RUN:=7d}"
: "${FLEX_REQUEST_VALID:=1h}"
: "${FLEX_ZONES:=asia-east1-a asia-northeast1-b asia-south1-c asia-south2-a asia-south2-c asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-southeast2-b asia-southeast2-c australia-southeast2-c europe-north1-a europe-north1-b europe-north1-c europe-west1-b europe-west1-c europe-west10-b europe-west2-b europe-west2-c europe-west4-a europe-west4-b europe-west4-c europe-west8-b europe-west8-c me-central2-a southamerica-east1-a southamerica-east1-b us-central1-b us-central1-c us-central1-f us-east1-b us-east1-d us-east4-b us-east4-c us-east5-a us-east5-b us-east5-c us-south1-a us-south1-b us-west1-a us-west1-b us-west1-c us-west3-a us-west4-a us-west4-b us-west4-c}"
# Spot and on-demand zones for gcp/grab.sh. On-demand quota exists only in us-west1 (10) and us-central1.
: "${SPOT_ZONES:=us-west3-a us-east4-b us-east4-c us-east5-a us-east5-b us-east5-c us-south1-a us-south1-b us-west4-a us-west4-b us-west4-c us-east1-b us-east1-d us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b us-west1-c}"
: "${STANDARD_ZONES:=us-west1-a us-west1-b us-west1-c us-central1-b us-central1-c us-central1-f}"
: "${GRAB_MAX_SECS:=5400}"                       # give up after 90 min
: "${GRAB_SLEEP:=45}"                            # pause between rounds (grab.sh)

# Model and server. ENGINE: r21 (vLLM-Moet r21, production) or lil (Local Inference Lab vLLM).
: "${ENGINE:=r21}"
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
