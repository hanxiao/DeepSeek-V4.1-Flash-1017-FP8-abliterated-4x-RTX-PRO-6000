#!/bin/bash
# Create everything from nothing: firewall rule, VM (gcp/grab.sh), static IP, then gcp/bootstrap.sh.
# Needs DS41_API_KEY in the environment or in .env. Takes ~25 min on a cold VM:
# create ~1 min, driver + docker ~3 min, download 4.5 min, first server start 16.5 min.
set -euo pipefail
# shellcheck source=../config.sh
. "$(dirname "$0")/../config.sh"
: "${DS41_API_KEY:?set DS41_API_KEY (see .env.example)}"

say(){ printf '\n\033[1;36m> %s\033[0m\n' "$*"; }

if ! gcloud compute firewall-rules describe "$FIREWALL_RULE" --project="$PROJECT" >/dev/null 2>&1; then
  say "Firewall: tcp 443 and 80 to tag $NETWORK_TAG (port 80 serves only ACME and the redirect)"
  gcloud compute firewall-rules create "$FIREWALL_RULE" --project="$PROJECT" \
    --allow=tcp:443,tcp:80 --target-tags="$NETWORK_TAG" --source-ranges=0.0.0.0/0 \
    --description="Caddy TLS + Bearer key in front of vLLM (DeepSeek-V4.1-Flash)"
fi

if [ -n "$(zone_of_vm)" ]; then
  ZONE="$(zone_of_vm)"
  say "$VM_NAME already exists in $ZONE"
else
  say "Creating $PROVISIONING $MACHINE ($LOCAL_SSDS local SSDs)"
  ZONE="$(bash "$REPO_DIR/gcp/grab.sh")"
fi
REGION="${ZONE%-*}"
export ZONE

# Static IP: promote the ephemeral IP the VM got, or attach an existing unused reservation.
IP="$(gcloud compute instances describe "$VM_NAME" --project="$PROJECT" --zone="$ZONE" \
      --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
STATUS="$(gcloud compute addresses describe "$STATIC_IP_NAME" --project="$PROJECT" --region="$REGION" \
          --format='value(status)' 2>/dev/null || true)"
if [ -z "$STATUS" ]; then
  say "Promoting $IP to static address $STATIC_IP_NAME"
  gcloud compute addresses create "$STATIC_IP_NAME" --project="$PROJECT" --region="$REGION" --addresses="$IP"
elif [ "$STATUS" = RESERVED ]; then
  STATIC="$(gcloud compute addresses describe "$STATIC_IP_NAME" --project="$PROJECT" --region="$REGION" --format='value(address)')"
  say "Attaching reserved $STATIC_IP_NAME ($STATIC)"
  gcloud compute instances delete-access-config "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --access-config-name="external-nat"
  gcloud compute instances add-access-config "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --access-config-name="external-nat" --address="$STATIC"
fi

say "Waiting for SSH"
until vm_ssh --command true >/dev/null 2>&1; do sleep 10; done

bash "$REPO_DIR/gcp/bootstrap.sh"
