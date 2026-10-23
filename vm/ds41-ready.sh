#!/bin/bash
# Print the HTTP status of a real /v1/messages call to vLLM on 127.0.0.1:8000.
# /v1/models answers before warm-up ends, so only this counts as ready.
# shellcheck disable=SC1091
. /etc/ds41/env
# shellcheck disable=SC1091
. /etc/ds41/deploy.env
curl -s -o /dev/null -w '%{http_code}\n' -m 120 http://127.0.0.1:8000/v1/messages \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'content-type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}]}"
