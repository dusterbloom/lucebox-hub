#!/usr/bin/env bash
set -euo pipefail

# Add nvm node to PATH for codex (a Node.js binary) in non-interactive subshells
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
: "${MAX_CTX:=32768}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
if [[ "${MODEL_SERVER:-}" == "llamacpp" ]]; then
  : "${LLAMA_COMPAT_PROXY:=responses}"
fi
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/codex.out"
LAST_MSG="$LOG_DIR/codex-last-message.txt"
CODEX_BIN="${CODEX_BIN:-$CLIENT_WORK_DIR/clients/codex/npm/bin/codex}"
CODEX_HOME_DIR="$LOG_DIR/codex-home"
CODEX_SANDBOX="${CODEX_SANDBOX:-danger-full-access}"
CODEX_WIRE_API="${CODEX_WIRE_API:-responses}"
mkdir -p "$CODEX_HOME_DIR"

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

# Session-inject proxy: inject extra_body.session_id for bandit experiments.
PROXY_PID=""
CLIENT_BASE_URL="$BASE_URL"
if [[ -n "${PFLASH_SESSION_ID:-}" ]]; then
  PROXY_PORT="${PFLASH_PROXY_PORT:-18083}"
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
  echo "[run_codex] session-inject proxy up on $CLIENT_BASE_URL (session=$PFLASH_SESSION_ID)"
fi

cat > "$CODEX_HOME_DIR/config.toml" <<TOML
model = "$MODEL_ID"
model_provider = "luce"
approval_policy = "never"
sandbox_mode = "$CODEX_SANDBOX"

[model_providers.luce]
name = "Lucebox"
base_url = "$CLIENT_BASE_URL/v1"
env_key = "OPENAI_API_KEY"
wire_api = "$CODEX_WIRE_API"
TOML

set +e
HOME="$CODEX_HOME_DIR" \
CODEX_HOME="$CODEX_HOME_DIR" \
OPENAI_API_KEY="$API_KEY" \
timeout 420s "$CODEX_BIN" exec \
  --skip-git-repo-check \
  --sandbox "$CODEX_SANDBOX" \
  --model "$MODEL_ID" \
  --json \
  --output-last-message "$LAST_MSG" \
  "$PROMPT" \
  < /dev/null > "$CLIENT_OUT" 2>&1
RC=$?
set -e

cat "$LAST_MSG" >> "$CLIENT_OUT" 2>/dev/null || true
finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
