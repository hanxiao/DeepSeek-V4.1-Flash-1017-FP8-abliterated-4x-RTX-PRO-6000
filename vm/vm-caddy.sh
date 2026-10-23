#!/bin/bash
# Install Caddy, which terminates TLS on :443 with a Let's Encrypt certificate and forwards a request
# to vLLM only when it carries the key. The key and the domain come from root-only env files.
set -x
export DEBIAN_FRONTEND=noninteractive
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
if ! command -v caddy >/dev/null; then
  apt-get install -y -q debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy
fi
{
  if [ -n "$ACME_EMAIL" ]; then printf '{\n\temail %s\n}\n\n' "$ACME_EMAIL"; fi
  cat /opt/ds41/Caddyfile
} > /etc/caddy/Caddyfile
chmod 644 /etc/caddy/Caddyfile
mkdir -p /var/log/caddy && chown caddy:caddy /var/log/caddy
# key for Caddy comes from the same root-only env file vLLM uses
( umask 077; { sed 's/^VLLM_API_KEY=/DS41_KEY=/' /etc/ds41/env; echo "DS41_DOMAIN=$DOMAIN"; } > /etc/ds41/caddy.env )
chown root:root /etc/ds41/caddy.env
mkdir -p /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironmentFile=/etc/ds41/caddy.env\n' > /etc/systemd/system/caddy.service.d/ds41.conf
systemctl daemon-reload
set +x   # do not trace the key
set -a
# shellcheck disable=SC1091
. /etc/ds41/caddy.env
set +a
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | tail -2
set -x
systemctl restart caddy
sleep 3
systemctl is-active caddy
echo CADDY_DONE
