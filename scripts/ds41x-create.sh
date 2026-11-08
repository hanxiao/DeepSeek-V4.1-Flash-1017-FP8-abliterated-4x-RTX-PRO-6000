#!/bin/bash
# Create ONE Flex-start g4-standard-192 test VM from the prepared boot snapshot.
# usage: ds41x-create.sh ZONE NAME
set -uo pipefail
ZONE=${1:?zone}; NAME=${2:?name}
PROJECT=PROJECT
SNAP=${SNAP:-ds41-boot-snap}
SSD=(); for _ in $(seq 1 16); do SSD+=(--local-ssd=interface=NVME); done
gcloud compute instances create "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --machine-type=g4-standard-192 --provisioning-model=FLEX_START \
  --max-run-duration=7d --request-valid-for-duration=1h \
  --instance-termination-action=STOP --maintenance-policy=TERMINATE --no-restart-on-failure \
  --reservation-affinity=none --discard-local-ssds-at-termination-timestamp=true \
  --create-disk="name=$NAME-boot,boot=yes,auto-delete=no,device-name=persistent-disk-0,source-snapshot=$SNAP,size=400GB,type=hyperdisk-balanced" \
  "${SSD[@]}" --tags=dsv41-https --metadata=install-nvidia-driver=True --scopes=cloud-platform \
  --labels=purpose=dsv41-test 2>&1 | tail -5
