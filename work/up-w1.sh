#!/bin/bash
# Bring up ds41-test-w (4x RTX PRO 6000, flex-start, 7d) from the leftover ds41-test-x boot disk.
# Fresh ADC token per gcloud call; retries on stockout; never touches production.
set -uo pipefail
. /tmp/ds41/gtok.sh
NAME=${NAME:-ds41-test-w}; ZONE=${ZONE:-asia-southeast2-c}; P=${P:-PROJECT}; DISK=${DISK:-ds41-test-x-boot}
SSD=(); for _ in $(seq 1 16); do SSD+=(--local-ssd=interface=NVME); done
for i in $(seq 1 240); do
  . /tmp/ds41/gtok.sh
  S=$(gcloud compute instances describe "$NAME" --project=$P --zone=$ZONE --format='value(status)' 2>/dev/null)
  case "$S" in
    RUNNING) echo "$(date -u +%T) $NAME RUNNING"; exit 0;;
    STAGING|PROVISIONING|PENDING) echo "$(date -u +%T) $NAME status=$S"; sleep 20; continue;;
    TERMINATED) echo "$(date -u +%T) exists TERMINATED"; exit 0;;
  esac
  OUT=$(gcloud compute instances create "$NAME" --project=$P --zone=$ZONE \
    --machine-type=g4-standard-192 --provisioning-model=FLEX_START \
    --max-run-duration=7d --request-valid-for-duration=1h \
    --instance-termination-action=STOP --maintenance-policy=TERMINATE --no-restart-on-failure \
    --reservation-affinity=none --discard-local-ssds-at-termination-timestamp=true \
    --disk="name=$DISK,boot=yes,device-name=persistent-disk-0,auto-delete=no" \
    "${SSD[@]}" --tags=dsv41-https --metadata=install-nvidia-driver=True --scopes=cloud-platform \
    --labels=purpose=dsv41-test 2>&1)
  RC=$?
  echo "$(date -u +%T) create rc=$RC $(grep -Eo 'reason: [a-z_]+|STOCKOUT|ZONE_RESOURCE_POOL_EXHAUSTED|Quota .[A-Z_0-9]+' <<< "$OUT" | sort -u | tr '\n' ' ')"
  if [ "$RC" = 0 ]; then
    for j in $(seq 1 60); do
      . /tmp/ds41/gtok.sh
      S=$(gcloud compute instances describe "$NAME" --project=$P --zone=$ZONE --format='value(status)' 2>/dev/null)
      echo "$(date -u +%T) $NAME status=$S"
      [ "$S" = RUNNING ] && exit 0
      sleep 20
    done
  fi
  sleep 45
done
echo "FAILED to get capacity"
exit 1
