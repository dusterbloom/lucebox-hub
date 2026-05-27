#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=98304}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${HERMES_MAX_TURNS:=40}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/hermes.out"
HERMES_BIN="${HERMES_BIN:-$CLIENT_WORK_DIR/clients/hermes/home/.local/bin/hermes}"
HOME_DIR="$LOG_DIR/hermes-home"
mkdir -p "$HOME_DIR"

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

# Session-inject proxy: inject extra_body.session_id for bandit experiments.
PROXY_PID=""
CLIENT_BASE_URL="$BASE_URL"
if [[ -n "${PFLASH_SESSION_ID:-}" ]]; then
  PROXY_PORT="${PFLASH_PROXY_PORT:-18085}"
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
  echo "[run_hermes] session-inject proxy up on $CLIENT_BASE_URL (session=$PFLASH_SESSION_ID)"
fi

# Write config after proxy resolution so base_url points at proxy when active.
cat > "$HOME_DIR/config.yaml" <<YAML
model:
  default: "$MODEL_ID"
  provider: "lucebox"
  base_url: "$CLIENT_BASE_URL/v1"
  api_key: "$API_KEY"
  api_mode: "chat_completions"
  context_length: $MAX_CTX
  max_tokens: $MAX_TOKENS

custom_providers:
  - name: "lucebox"
    base_url: "$CLIENT_BASE_URL/v1"
    api_key: "$API_KEY"
    api_mode: "chat_completions"
    models:
      "$MODEL_ID":
        context_length: $MAX_CTX
        max_tokens: $MAX_TOKENS

terminal:
  backend: "local"
  cwd: "$REPO_DIR"
  timeout: 180
  lifetime_seconds: 300
YAML

cat > "$HOME_DIR/.env" <<ENV
OPENAI_API_KEY=$API_KEY
OPENAI_BASE_URL=$CLIENT_BASE_URL/v1
HERMES_INFERENCE_PROVIDER=lucebox
HERMES_INFERENCE_MODEL=$MODEL_ID
HERMES_ACCEPT_HOOKS=1
HERMES_API_TIMEOUT=600
HERMES_API_CALL_STALE_TIMEOUT=600
ENV

set +e
cd "$REPO_DIR"
HOME="$HOME_DIR" \
HERMES_HOME="$HOME_DIR" \
OPENAI_API_KEY="$API_KEY" \
OPENAI_BASE_URL="$CLIENT_BASE_URL/v1" \
HERMES_INFERENCE_PROVIDER=lucebox \
HERMES_INFERENCE_MODEL="$MODEL_ID" \
HERMES_ACCEPT_HOOKS=1 \
NO_COLOR=1 \
timeout 420s "$HERMES_BIN" chat \
  --quiet \
  --provider lucebox \
  --model "$MODEL_ID" \
  --accept-hooks \
  --yolo \
  --max-turns "$HERMES_MAX_TURNS" \
  --source lucebox-harness \
  --query "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
