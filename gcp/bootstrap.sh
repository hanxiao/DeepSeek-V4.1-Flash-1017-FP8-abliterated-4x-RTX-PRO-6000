#!/bin/bash
# Turn a fresh VM into the server, or bring an existing one in line with this repo.
# Copies vm/ to the VM, writes the secrets into root-only files, runs the vm-*.sh steps, starts
# the server, waits until a real /v1/messages call returns 200, then backs up the JIT cache to the
# boot disk and enables ds41-boot.service. A running container is left alone unless RESTART=1.
set -euo pipefail
# shellcheck source=../config.sh
. "$(dirname "$0")/../config.sh"
: "${DS41_API_KEY:?set DS41_API_KEY (see .env.example)}"
ZONE="${ZONE:-$(zone_of_vm)}"; export ZONE
[ -n "$ZONE" ] || { echo "no VM named $VM_NAME in $PROJECT"; exit 1; }
DOMAIN="$(domain_of_vm)"

say(){ printf '\n\033[1;36m> %s\033[0m\n' "$*"; }

say "Copying vm/ and the encoding patch to $VM_NAME ($ZONE)"
gcloud compute scp --project="$PROJECT" --zone="$ZONE" --quiet \
  "$REPO_DIR"/vm/* "$REPO_DIR/patches/encoding-1017-vllm.patch" "$VM_NAME":/tmp/
vm_ssh --command 'sudo bash -c "
  mkdir -p /opt/ds41 /etc/ds41 /var/lib/ds41 && chmod 755 /etc/ds41
  cd /tmp && install -m 755 vm-docker.sh vm-setup.sh vm-download.sh vm-encoding.sh vm-caddy.sh /opt/ds41/
  install -m 644 Caddyfile encoding-1017-vllm.patch /opt/ds41/
  install -m 755 ds41-start.sh ds41-boot.sh ds41-ready.sh /usr/local/bin/
  install -m 644 ds41-boot.service /etc/systemd/system/ds41-boot.service
  systemctl daemon-reload"'

say "Writing /etc/ds41/deploy.env and the root-only secret files"
printf "IMAGE='%s'\nMODEL_REPO='%s'\nMODEL_REV='%s'\nSERVED_MODEL_NAME='%s'\nDOMAIN='%s'\nACME_EMAIL='%s'\n" \
  "$IMAGE" "$MODEL_REPO" "$MODEL_REV" "$SERVED_MODEL_NAME" "$DOMAIN" "$ACME_EMAIL" \
  | vm_ssh --command 'sudo sh -c "cat > /etc/ds41/deploy.env && chmod 644 /etc/ds41/deploy.env"'
printf 'VLLM_API_KEY=%s\n' "$DS41_API_KEY" \
  | vm_ssh --command 'sudo sh -c "umask 077; cat > /etc/ds41/env"'
if [ -n "${HF_TOKEN:-}" ]; then
  printf '%s\n' "$HF_TOKEN" | vm_ssh --command 'sudo sh -c "umask 077; cat > /root/.hf_token"'
fi

say "Waiting for the NVIDIA driver (install-nvidia-driver=True runs on first boot)"
until vm_ssh --command 'nvidia-smi -L' 2>/dev/null | grep -q 'RTX PRO 6000'; do sleep 15; done

say "docker + nvidia-container-toolkit, RAID0 on /data, hf cli"
vm_ssh --command 'sudo bash /opt/ds41/vm-docker.sh && sudo bash /opt/ds41/vm-setup.sh' 2>&1 | { grep -E '_DONE|/data|Docker Root|RTX' || true; }

say "Weights ($MODEL_REPO @ ${MODEL_REV:0:8}) and image, in parallel"
vm_ssh --command 'sudo bash /opt/ds41/vm-download.sh; sudo tail -2 /data/logs/dl.log; sudo tail -1 /data/logs/pull.log'

say "Encoding patch and Caddy ($DOMAIN)"
vm_ssh --command 'sudo bash /opt/ds41/vm-encoding.sh && sudo bash /opt/ds41/vm-caddy.sh' 2>&1 | { grep -E '_DONE|active|Valid|error' || true; }

if vm_ssh --command 'sudo docker ps --format "{{.Names}}"' | grep -qx ds41 && [ "${RESTART:-0}" != 1 ]; then
  say "Container ds41 already running; RESTART=1 recreates it"
else
  say "Starting the server (first start ~16.5 min of kernel compilation, ~3 min with the cache)"
  vm_ssh --command 'sudo mkdir -p /data/cache/lil-cache && { [ -d /var/lib/ds41/lil-cache ] && sudo cp -a /var/lib/ds41/lil-cache/. /data/cache/lil-cache/ || true; } && sudo /usr/local/bin/ds41-start.sh'
fi

# /v1/models answers before warm-up ends, so readiness is a real completion.
say "Waiting for a real /v1/messages to return 200"
until [ "$(vm_ssh --command 'sudo /usr/local/bin/ds41-ready.sh' 2>/dev/null | tr -d '\r')" = 200 ]; do printf '.'; sleep 20; done; echo

say "Backing up the JIT cache to the boot disk and enabling ds41-boot.service"
vm_ssh --command 'sudo bash -c "mkdir -p /var/lib/ds41 && cp -a /data/cache/lil-cache /var/lib/ds41/ && du -sh /var/lib/ds41/lil-cache && systemctl enable ds41-boot.service 2>&1 | tail -1"'

cat <<EOF

READY
  Endpoint:  https://$DOMAIN   (Anthropic /v1/messages, OpenAI /v1/chat/completions)
  Model:     $SERVED_MODEL_NAME
  Test:      bash scripts/smoke.sh
  Logs:      gcloud compute ssh $VM_NAME --zone $ZONE --project $PROJECT --command 'sudo docker logs -f ds41'
EOF
