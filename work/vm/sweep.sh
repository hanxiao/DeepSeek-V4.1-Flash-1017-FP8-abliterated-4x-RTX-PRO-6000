#!/bin/bash
# VM side, root. Run a list of experiments unattended.
# usage: sweep.sh MANIFEST
#   MANIFEST lines: TAG|ENVFILE|PLAN|EXTRA|XENV
#     TAG     results in /var/lib/ds41/x/TAG
#     ENVFILE r21.env-style file
#     PLAN    codebench plan
#     EXTRA   extra docker args (e.g. "-e VLLM_MOET_P2P_AG=1")
#     XENV    env for exp.sh itself (e.g. "THINK=1 PROF=1")
#   '#' starts a comment line.
set -uo pipefail
M=${1:?manifest}
while IFS='|' read -r TAG ENVF PLAN EXTRA XENV; do
  [ -z "${TAG:-}" ] && continue
  [ "${TAG:0:1}" = "#" ] && continue
  echo "=== $(date -u +%FT%TZ) $TAG" | tee -a /var/lib/ds41/x/orch.log
  env ${XENV:-} bash /usr/local/bin/exp.sh "$TAG" "${ENVF:-/etc/ds41/r21.env}" "${PLAN:-30:1x3,120:1x2}" "${EXTRA:-}"
done < "$M"
echo SWEEP_DONE | tee -a /var/lib/ds41/x/orch.log
