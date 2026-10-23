#!/bin/bash
# Create the g4-standard-192 VM in the first zone with capacity. Tries every zone of the list once
# per round, sleeps GRAB_SLEEP seconds, and gives up after GRAB_MAX_SECS. A failed create that
# leaves a half-created instance behind is deleted before the next zone.
#   PROVISIONING=SPOT      (default) zones from SPOT_ZONES, termination action STOP
#   PROVISIONING=STANDARD  on-demand, zones from STANDARD_ZONES
#   ZONES="a b c"          overrides both lists
# Prints the zone on success. Log: grab.log.
set -uo pipefail
# shellcheck source=../config.sh
. "$(dirname "$0")/../config.sh"

if [ "$PROVISIONING" = SPOT ]; then
  ZONES="${ZONES:-$SPOT_ZONES}"
  PROV=(--provisioning-model=SPOT --instance-termination-action=STOP)
else
  ZONES="${ZONES:-$STANDARD_ZONES}"
  PROV=(--provisioning-model=STANDARD)
fi
SSD=()
for _ in $(seq 1 "$LOCAL_SSDS"); do SSD+=(--local-ssd=interface=NVME); done

LOG="$REPO_DIR/grab.log"
END=$(( $(date +%s) + GRAB_MAX_SECS ))
n=0
while [ "$(date +%s)" -lt "$END" ]; do
  n=$((n+1))
  for z in $ZONES; do
    if out=$(gcloud compute instances create "$VM_NAME" --project="$PROJECT" --zone="$z" \
      --machine-type="$MACHINE" "${PROV[@]}" --maintenance-policy=TERMINATE \
      --image-project="$GCE_IMAGE_PROJECT" --image="$GCE_IMAGE" \
      --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type="$BOOT_DISK_TYPE" "${SSD[@]}" \
      --tags="$NETWORK_TAG" --metadata=install-nvidia-driver=True --scopes=cloud-platform \
      --labels="$LABELS" 2>&1); then
      echo "$(date -u +%FT%TZ) round=$n OK zone=$z" >> "$LOG"
      echo "$out" >> "$LOG"
      echo "$z"
      exit 0
    fi
    r=$(echo "$out" | grep -E -o "reason: [a-z_]+|Quota '[^']*' exceeded" | sort -u | tr '\n' ' ')
    echo "$(date -u +%FT%TZ) round=$n $z FAIL $r" | tee -a "$LOG" >&2
    if gcloud compute instances describe "$VM_NAME" --zone="$z" --project="$PROJECT" >/dev/null 2>&1; then
      echo "$(date -u +%FT%TZ) cleaning half-created instance in $z" >> "$LOG"
      gcloud compute instances delete "$VM_NAME" --zone="$z" --project="$PROJECT" --quiet >> "$LOG" 2>&1
    fi
  done
  sleep "$GRAB_SLEEP"
done
echo "$(date -u +%FT%TZ) GAVE UP after $n rounds" | tee -a "$LOG" >&2
exit 1
