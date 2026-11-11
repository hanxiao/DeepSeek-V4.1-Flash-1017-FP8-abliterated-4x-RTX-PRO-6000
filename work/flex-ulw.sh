#!/bin/bash
# Robust Flex-start grab: unique prefix, fresh ADC token per gcloud call, never deletes the winner.
# Prints "NAME ZONE" on success. Usage: bash work/flex-ulw.sh <prefix> [max_secs]
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=../config.sh
. ./config.sh
PREFIX="${1:-ds41w}"
MAX="${2:-5400}"
ZONES="${ZONES:-us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b us-west1-c us-east1-b us-east1-d us-east4-b us-east4-c europe-west4-a europe-west4-b europe-west4-c}"
ADC="$HOME/.config/gcloud/application_default_credentials.json"
TOK_AT=0
refresh() {
  [ $(( $(date +%s) - TOK_AT )) -lt 1500 ] && [ -n "${CLOUDSDK_AUTH_ACCESS_TOKEN:-}" ] && return 0
  CLOUDSDK_AUTH_ACCESS_TOKEN=$(python3 - "$ADC" <<'PY'
import json,sys,urllib.parse,urllib.request
d=json.load(open(sys.argv[1]))
body=urllib.parse.urlencode({"grant_type":"refresh_token","client_id":d["client_id"],"client_secret":d["client_secret"],"refresh_token":d["refresh_token"]}).encode()
print(json.load(urllib.request.urlopen("https://oauth2.googleapis.com/token",body))["access_token"])
PY
)
  export CLOUDSDK_AUTH_ACCESS_TOKEN
  TOK_AT=$(date +%s)
}
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
SSD=(); for _ in $(seq 1 "$LOCAL_SSDS"); do SSD+=(--local-ssd=interface=NVME); done
if [ -n "$BOOT_SNAPSHOT" ]; then SRC="source-snapshot=$BOOT_SNAPSHOT"; else SRC="image=$GCE_IMAGE,image-project=$GCE_IMAGE_PROJECT"; fi
END=$(( $(date +%s) + MAX ))

create_one() {
  local z=$1 n="$PREFIX-$1" s out
  refresh
  s=$(gcloud compute instances describe "$n" --project="$PROJECT" --zone="$z" --format='value(status)' 2>/dev/null)
  case "$s" in
    RUNNING|STAGING|PROVISIONING|PENDING) echo "exists $n $s" >&2; return 0 ;;
    "") ;;
    *) refresh; gcloud compute instances delete "$n" --project="$PROJECT" --zone="$z" --quiet >/dev/null 2>&1 ;;
  esac
  refresh
  out=$(gcloud compute instances create "$n" --project="$PROJECT" --zone="$z" \
    --machine-type="$MACHINE" --provisioning-model=FLEX_START \
    --max-run-duration="$FLEX_MAX_RUN" --request-valid-for-duration="$FLEX_REQUEST_VALID" \
    --instance-termination-action=STOP --maintenance-policy=TERMINATE --no-restart-on-failure \
    --reservation-affinity=none --discard-local-ssds-at-termination-timestamp=true \
    --create-disk="name=$n-boot,boot=yes,auto-delete=yes,device-name=persistent-disk-0,$SRC,size=$BOOT_DISK_SIZE,type=$BOOT_DISK_TYPE" \
    "${SSD[@]}" --tags="$NETWORK_TAG" --metadata=install-nvidia-driver=True --scopes=cloud-platform \
    --labels="$LABELS" 2>&1)
  echo "$(date -u +%FT%TZ) create $n rc=$? $(grep -E -o 'reason: [a-z_]+|STOCKOUT|Quota .[A-Z_0-9]+' <<< "$out" | sort -u | tr '\n' ' ')" >&2
}

for z in $ZONES; do create_one "$z" & done
wait

WIN=""
while [ -z "$WIN" ] && [ "$(date +%s)" -lt "$END" ]; do
  refresh
  WIN=$(gcloud compute instances list --project="$PROJECT" --filter="name~^$PREFIX-" \
        --format='value(name,zone.basename(),status)' 2>/dev/null \
        | awk '$3=="RUNNING"{print $1" "$2; exit}')
  [ -n "$WIN" ] && break
  sleep 20
done
[ -n "$WIN" ] || { echo "no capacity in $MAX s" >&2; exit 1; }
WIN_NAME=${WIN%% *}; WIN_ZONE=${WIN##* }
echo "winner $WIN_NAME $WIN_ZONE" >&2
# Stop the other candidates only, never the winner.
for _ in 1 2 3 4 5; do
  refresh
  gcloud compute instances list --project="$PROJECT" --filter="name~^$PREFIX-" --format='value(name,zone.basename())' 2>/dev/null \
    | while read -r n z; do
        [ -z "$n" ] && continue
        if [ "$n" = "$WIN_NAME" ] && [ "$z" = "$WIN_ZONE" ]; then continue; fi
        refresh; gcloud compute instances delete "$n" --project="$PROJECT" --zone="$z" --quiet >/dev/null 2>&1
      done
  sleep 20
done
refresh
gcloud compute instances set-disk-auto-delete "$WIN_NAME" --project="$PROJECT" --zone="$WIN_ZONE" \
  --device-name=persistent-disk-0 --no-auto-delete >/dev/null 2>&1
echo "$WIN"
