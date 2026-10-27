#!/bin/bash
# Create the g4-standard-192 VM as a Flex-start VM in the first zone with capacity.
# One retry loop per zone of FLEX_ZONES submits a Flex-start request (valid FLEX_REQUEST_VALID,
# resubmitted when it expires) for a candidate named $FLEX_PREFIX-<zone>. A coordinator keeps the
# first candidate that starts, stops the loops and deletes every other candidate of this run.
# Candidates whose name already exists are skipped, so running VMs are never touched.
# Boot disks are created with auto-delete, so a deleted candidate leaves nothing behind; the
# winner's boot disk is switched to keep. Prints "NAME ZONE" of the winner. Logs: flex/<zone>.log.
set -uo pipefail
# shellcheck source=../config.sh
. "$(dirname "$0")/../config.sh"
ZONES="${ZONES:-$FLEX_ZONES}"
DIR="$REPO_DIR/flex"
mkdir -p "$DIR"; rm -f "$DIR/stop"
END=$(( $(date +%s) + GRAB_MAX_SECS ))
SSD=(); for _ in $(seq 1 "$LOCAL_SSDS"); do SSD+=(--local-ssd=interface=NVME); done
if [ -n "$BOOT_SNAPSHOT" ]; then SRC="source-snapshot=$BOOT_SNAPSHOT"
else SRC="image=$GCE_IMAGE,image-project=$GCE_IMAGE_PROJECT"; fi

zone_loop() {
  local z=$1 n="$FLEX_PREFIX-$1" log="$DIR/$1.log" s out rc
  while [ "$(date +%s)" -lt "$END" ] && [ ! -f "$DIR/stop" ]; do
    s=$(gcloud compute instances describe "$n" --project="$PROJECT" --zone="$z" --format='value(status)' 2>/dev/null)
    case "$s" in
      RUNNING|STAGING|PROVISIONING) return 0 ;;
      PENDING) sleep 30; continue ;;
      "") ;;
      *) gcloud compute instances delete "$n" --project="$PROJECT" --zone="$z" --quiet >> "$log" 2>&1 ;;
    esac
    out=$(gcloud compute instances create "$n" --project="$PROJECT" --zone="$z" \
      --machine-type="$MACHINE" --provisioning-model=FLEX_START \
      --max-run-duration="$FLEX_MAX_RUN" --request-valid-for-duration="$FLEX_REQUEST_VALID" \
      --instance-termination-action=STOP --maintenance-policy=TERMINATE --no-restart-on-failure \
      --reservation-affinity=none --discard-local-ssds-at-termination-timestamp=true \
      --create-disk="name=$n-boot,boot=yes,auto-delete=yes,device-name=persistent-disk-0,$SRC,size=$BOOT_DISK_SIZE,type=$BOOT_DISK_TYPE" \
      "${SSD[@]}" --tags="$NETWORK_TAG" --metadata=install-nvidia-driver=True --scopes=cloud-platform \
      --labels="$LABELS" 2>&1); rc=$?
    echo "$(date -u +%FT%TZ) rc=$rc $(grep -E -o "reason: [a-z_]+|STOCKOUT|Quota '[A-Z_0-9]+'" <<< "$out" | sort -u | tr '\n' ' ')" >> "$log"
    [ "$rc" = 0 ] && continue
    if grep -qi quota <<< "$out"; then sleep 600; else sleep $(( 50 + RANDOM % 30 )); fi
  done
}

CANDS=()
for z in $ZONES; do
  if gcloud compute instances describe "$FLEX_PREFIX-$z" --project="$PROJECT" --zone="$z" >/dev/null 2>&1; then
    echo "skip $z: $FLEX_PREFIX-$z already exists" >&2; continue
  fi
  CANDS+=("$FLEX_PREFIX-$z")
  zone_loop "$z" &
done
[ "${#CANDS[@]}" -gt 0 ] || { echo "no zone to try" >&2; exit 1; }
FILT="$(printf 'name=%s OR ' "${CANDS[@]}")"; FILT="${FILT% OR }"

WIN=""
while [ -z "$WIN" ] && [ "$(date +%s)" -lt "$END" ]; do
  sleep 20
  WIN=$(gcloud compute instances list --project="$PROJECT" --filter="$FILT" \
        --format='value(name,zone.basename(),status)' 2>/dev/null \
        | awk '$3=="RUNNING"||$3=="STAGING"||$3=="PROVISIONING"{print $1" "$2; exit}')
done
touch "$DIR/stop"
for p in $(jobs -p); do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
wait 2>/dev/null

# Requests already submitted can still start after the loops are gone, so sweep a few times.
for _ in 1 2 3 4 5; do
  gcloud compute instances list --project="$PROJECT" --filter="$FILT" --format='value(name,zone.basename())' 2>/dev/null \
    | while read -r n z; do
        [ -n "$n" ] && [ "$n $z" != "$WIN" ] && gcloud compute instances delete "$n" --project="$PROJECT" --zone="$z" --quiet >> "$DIR/sweep.log" 2>&1
      done
  sleep 30
done

[ -n "$WIN" ] || { echo "no Flex-start capacity within $GRAB_MAX_SECS s" >&2; exit 1; }
gcloud compute instances set-disk-auto-delete "${WIN% *}" --project="$PROJECT" --zone="${WIN#* }" \
  --device-name=persistent-disk-0 --no-auto-delete >> "$DIR/sweep.log" 2>&1
echo "$WIN"
