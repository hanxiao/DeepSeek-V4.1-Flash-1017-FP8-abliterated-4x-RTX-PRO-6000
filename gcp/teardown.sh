#!/bin/bash
# bash gcp/teardown.sh stop     stop the VM: GPUs and vCPUs stop billing, the local SSDs are wiped,
#                               the boot disk and static IP stay (~$36/month)
# bash gcp/teardown.sh start    start it again; ds41-boot.service rebuilds /data and the server
# bash gcp/teardown.sh delete   delete the VM, release the static IP, remove the firewall rule
set -euo pipefail
# shellcheck source=../config.sh
. "$(dirname "$0")/../config.sh"
ZONE="$(zone_of_vm)"
REGION="${ZONE%-*}"
case "${1:-}" in
  stop)
    gcloud compute instances stop "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --discard-local-ssd=true ;;
  start)
    gcloud compute instances start "$VM_NAME" --project="$PROJECT" --zone="$ZONE"
    echo "allow ~15 min to the first token; log: sudo tail -f /var/log/ds41-boot.log" ;;
  delete)
    read -r -p "Delete $VM_NAME ($ZONE), $STATIC_IP_NAME and $FIREWALL_RULE in $PROJECT? [y/N] " a
    [ "$a" = y ] || exit 1
    [ -n "$ZONE" ] && gcloud compute instances delete "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --quiet
    if [ -n "$ZONE" ]; then
      gcloud compute addresses delete "$STATIC_IP_NAME" --project="$PROJECT" --region="$REGION" --quiet || true
    else
      gcloud compute addresses list --project="$PROJECT" --filter="name=$STATIC_IP_NAME" --format='value(region.basename())' \
        | while read -r r; do gcloud compute addresses delete "$STATIC_IP_NAME" --project="$PROJECT" --region="$r" --quiet; done
    fi
    gcloud compute firewall-rules delete "$FIREWALL_RULE" --project="$PROJECT" --quiet || true ;;
  *)
    sed -n '2,5p' "$0"; exit 1 ;;
esac
