#!/usr/bin/env bash
set -euo pipefail

# Add nvm node to PATH for pi (a Node.js binary) in non-interactive subshells
# Prefer the explicit nvm path over asdf shims which require asdf runtime state
_NVM_NODE_BIN=""
for _v in v24.13.0 v22.17.0 v20.18.0; do
  if [[ -x "$HOME/.nvm/versions/node/$_v/bin/node" ]]; then
    _NVM_NODE_BIN="$HOME/.nvm/versions/node/$_v/bin"
    break
  fi
done
[[ -n "$_NVM_NODE_BIN" ]] && export PATH="$_NVM_NODE_BIN:$PATH"
unset _NVM_NODE_BIN _v

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=65536}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${PI_TOOLS:=read,grep,find,ls}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/pi.out"
PI_BIN="${PI_BIN:-$CLIENT_WORK_DIR/clients/pi/npm/bin/pi}"
HOME_DIR="$LOG_DIR/pi-home"
AGENT_DIR="$HOME_DIR/agent"
PROVIDER_API="${PROVIDER_API:-openai-responses}"
mkdir -p "$AGENT_DIR" "$HOME_DIR/sessions"

cat > "$AGENT_DIR/settings.json" <<JSON
{
  "compaction": {
    "enabled": false
  }
}
JSON

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

# Session-inject proxy: inject extra_body.session_id for bandit experiments.
PROXY_PID=""
CLIENT_BASE_URL="$BASE_URL"
if [[ -n "${PFLASH_SESSION_ID:-}" ]]; then
  PROXY_PORT="${PFLASH_PROXY_PORT:-18084}"
  python3 "$SCRIPT_DIR/session_inject_proxy.py" \
    --host "$HOST" \
    --port "$PROXY_PORT" \
    --upstream "$BASE_URL" \
    --session-id "$PFLASH_SESSION_ID" \
    >> "$LOG_DIR/proxy.log" 2>&1 &
  PROXY_PID=$!
  trap 'kill "$PROXY_PID" 2>/dev/null || true; wait "$PROXY_PID" 2>/dev/null || true; stop_lucebox_server' EXIT
  _proxy_ready=0
  for _i in $(seq 1 10); do
    if curl -fsS "http://$HOST:$PROXY_PORT/health" >/dev/null 2>&1; then _proxy_ready=1; break; fi
    sleep 1
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      echo "session-inject proxy exited early; log: $LOG_DIR/proxy.log" >&2
      cat "$LOG_DIR/proxy.log" >&2 || true
      exit 1
    fi
  done
  if [[ "$_proxy_ready" -eq 0 ]]; then
    echo "session-inject proxy did not become ready after 10s; log: $LOG_DIR/proxy.log" >&2
    cat "$LOG_DIR/proxy.log" >&2 || true
    kill "$PROXY_PID" 2>/dev/null || true
    exit 1
  fi
  CLIENT_BASE_URL="http://$HOST:$PROXY_PORT"
  echo "[run_pi] session-inject proxy up on $CLIENT_BASE_URL (session=$PFLASH_SESSION_ID)"
fi

# Write models.json after proxy resolution so baseUrl points at proxy when active.
cat > "$AGENT_DIR/models.json" <<JSON
{
  "providers": {
    "lucebox": {
      "baseUrl": "$CLIENT_BASE_URL/v1",
      "api": "$PROVIDER_API",
      "apiKey": "$API_KEY",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "$MODEL_ID",
          "name": "Lucebox DFlash",
          "api": "$PROVIDER_API",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": $MAX_CTX,
          "maxTokens": $MAX_TOKENS,
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        }
      ]
    }
  }
}
JSON

set +e
HOME="$HOME_DIR" \
PI_CODING_AGENT_DIR="$AGENT_DIR" \
PI_CODING_AGENT_SESSION_DIR="$HOME_DIR/sessions" \
PI_OFFLINE=1 \
timeout 300s "$PI_BIN" \
  --provider lucebox \
  --model "$MODEL_ID" \
  --print \
  --mode json \
  --tools "$PI_TOOLS" \
  --no-session \
  --offline \
  "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
