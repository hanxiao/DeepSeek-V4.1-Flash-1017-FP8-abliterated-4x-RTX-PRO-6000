#!/bin/bash
# Delete every ds41x-* flex candidate except $1, repeatedly, until the deadline.
# Flex-start requests already submitted can still start after the loops are gone, so one pass is not enough.
WIN=${1:?winner name}
PROJECT=PROJECT
END=$(( $(date +%s) + ${2:-1800} ))
while [ "$(date +%s)" -lt "$END" ]; do
  gcloud compute instances list --project="$PROJECT" --filter="name~^ds41x" --format='value(name,zone.basename())' 2>/dev/null \
  | while IFS=$'\t' read -r n z; do
      [ -z "$n" ] && continue
      [ "$n" = "$WIN" ] && continue
      echo "$(date -u +%T) delete $n $z"
      gcloud compute instances delete "$n" --project="$PROJECT" --zone="$z" --quiet 2>&1 | tail -2
    done
  sleep 25
done
echo KILLER_DONE
