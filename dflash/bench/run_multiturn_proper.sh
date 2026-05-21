#!/usr/bin/env bash
# Multi-turn regression harness for pflash proper bench.
# Runs claude_code, opencode, and hermes against lucebox dflash_server
# in two modes: pflash off and pflash always at keep=0.05.
# Compares MTP accept rates against the pr-232 baseline (0.71-0.88).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/2026-05-21_envelope/multiturn_runs"
mkdir -p "$RESULTS_DIR"

TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
MTP_GGUF="/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf"
DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
DFLASH_SERVER="$REPO_DIR/dflash/build/dflash_server"
PORT=18080
HOST=127.0.0.1
BASE_URL="http://$HOST:$PORT"
MODEL_ID="luce-dflash"
API_KEY="sk-lucebox"
MAX_CTX=16384
MAX_TOKENS=2048
PROMPT_FILE="$REPO_DIR/harness/clients/prompts/decode_check.txt"
PROMPT="$(<"$PROMPT_FILE")"
MARKER="OK_DONE"

CLAUDE_BIN="${CLAUDE_BIN:-/home/peppi/.local/bin/claude}"
OPENCODE_BIN="${OPENCODE_BIN:-$(which opencode 2>/dev/null || echo "")}"
HERMES_BIN="${HERMES_BIN:-/home/peppi/.local/bin/hermes}"

SERVER_PID=""

start_server() {
    local mode="$1"
    local keep="$2"
    local stamp="$3"
    local server_log="$RESULTS_DIR/${stamp}/server.log"
    mkdir -p "$RESULTS_DIR/${stamp}"

    # Kill any old server on this port
    fuser -k "$PORT/tcp" 2>/dev/null || true
    sleep 1

    export GGML_CUDA_NO_VMM=1
    export DFLASH27B_KV_K=tq3_0
    export DFLASH27B_KV_V=tq3_0

    "$DFLASH_SERVER" "$TARGET" \
        --host "$HOST" --port "$PORT" \
        --max-ctx "$MAX_CTX" \
        --model-name "$MODEL_ID" \
        --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
        --prefill-compression "$mode" \
        --prefill-keep-ratio "$keep" \
        --prefill-skip-park \
        --prefill-drafter "$DRAFTER" \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    # Wait for healthy
    for i in $(seq 1 120); do
        if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
            echo "[harness] server ready (pid=$SERVER_PID) mode=$mode keep=$keep after ${i}s"
            return 0
        fi
        sleep 1
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[harness] server exited early; log: $server_log" >&2
            tail -n 40 "$server_log" >&2 || true
            return 1
        fi
    done
    echo "[harness] server timeout; log: $server_log" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

run_client_claude_code() {
    local stamp="$1"
    local out="$RESULTS_DIR/${stamp}/claude_code.out"
    local home_dir="$RESULTS_DIR/${stamp}/claude-home"
    mkdir -p "$home_dir"

    echo "[harness] running claude_code for $stamp"
    set +e
    HOME="$home_dir" \
    ANTHROPIC_API_KEY="$API_KEY" \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    CLAUDE_CODE_API_BASE_URL="$BASE_URL" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    CLAUDE_CODE_DISABLE_TELEMETRY=1 \
    CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1 \
    timeout 300s "$CLAUDE_BIN" \
        --print \
        --output-format json \
        --model "$MODEL_ID" \
        --tools none \
        --permission-mode dontAsk \
        --no-session-persistence \
        "$PROMPT" \
        < /dev/null > "$out" 2>&1
    local rc=$?
    set -e
    echo "[harness] claude_code rc=$rc"
    echo "$out"
    return $rc
}

run_client_opencode() {
    local stamp="$1"
    local out="$RESULTS_DIR/${stamp}/opencode.out"
    local home_dir="$RESULTS_DIR/${stamp}/opencode-home"
    local project_dir="$RESULTS_DIR/${stamp}/opencode-project"
    mkdir -p "$home_dir/.config" "$home_dir/.local/share" "$project_dir"

    if [[ -z "$OPENCODE_BIN" ]] || ! command -v "$OPENCODE_BIN" >/dev/null 2>&1; then
        echo "[harness] opencode not found, skipping"
        echo "SKIP: opencode binary not found" > "$out"
        return 0
    fi

    cat > "$project_dir/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lucebox/$MODEL_ID",
  "small_model": "lucebox/$MODEL_ID",
  "provider": {
    "lucebox": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lucebox",
      "options": {
        "baseURL": "$BASE_URL/v1",
        "apiKey": "$API_KEY",
        "timeout": 600000,
        "chunkTimeout": 60000
      },
      "models": {
        "$MODEL_ID": {
          "name": "Lucebox DFlash",
          "limit": {
            "context": $MAX_CTX,
            "output": $MAX_TOKENS
          }
        }
      }
    }
  },
  "tools": {"write": false, "bash": false}
}
JSON

    echo "[harness] running opencode for $stamp"
    set +e
    cd "$project_dir"
    HOME="$home_dir" \
    XDG_CONFIG_HOME="$home_dir/.config" \
    XDG_DATA_HOME="$home_dir/.local/share" \
    OPENAI_API_KEY="$API_KEY" \
    timeout 300s "$OPENCODE_BIN" run \
        --pure \
        --model "lucebox/$MODEL_ID" \
        --format json \
        "$PROMPT" \
        < /dev/null > "$out" 2>&1
    local rc=$?
    set -e
    cd "$SCRIPT_DIR"
    echo "[harness] opencode rc=$rc"
    echo "$out"
    return $rc
}

run_client_hermes() {
    local stamp="$1"
    local out="$RESULTS_DIR/${stamp}/hermes.out"
    local home_dir="$RESULTS_DIR/${stamp}/hermes-home"
    mkdir -p "$home_dir"

    cat > "$home_dir/config.yaml" <<YAML
model:
  default: "$MODEL_ID"
  provider: "lucebox"
  base_url: "$BASE_URL/v1"
  api_key: "$API_KEY"
  api_mode: "chat_completions"
  context_length: $MAX_CTX
  max_tokens: $MAX_TOKENS

custom_providers:
  - name: "lucebox"
    base_url: "$BASE_URL/v1"
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

    echo "[harness] running hermes for $stamp"
    set +e
    HOME="$home_dir" \
    HERMES_HOME="$home_dir" \
    OPENAI_API_KEY="$API_KEY" \
    OPENAI_BASE_URL="$BASE_URL/v1" \
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
        --max-turns 40 \
        --query "$PROMPT" \
        < /dev/null > "$out" 2>&1
    local rc=$?
    set -e
    echo "[harness] hermes rc=$rc"
    echo "$out"
    return $rc
}

extract_accept_rates() {
    local server_log="$1"
    grep "mtp_decode" "$server_log" 2>/dev/null | \
        python3 -c "
import sys, re
rates=[]
for line in sys.stdin:
    m = re.search(r'accept_rate=([\d.]+)', line)
    if m:
        rates.append(float(m.group(1)))
if rates:
    print(f'n={len(rates)} min={min(rates):.2f} max={max(rates):.2f} mean={sum(rates)/len(rates):.2f} rates={rates}')
else:
    print('no accept_rate lines found')
" 2>/dev/null || echo "no mtp_decode lines"
}

check_marker() {
    local out_file="$1"
    grep -c "$MARKER" "$out_file" 2>/dev/null || echo "0"
}

run_all_clients() {
    local mode="$1"
    local keep="$2"

    for client in claude_code opencode hermes; do
        local stamp="proper-bench-${client}-${mode}-$(date +%H%M)"
        echo ""
        echo "=== $client mode=$mode keep=$keep stamp=$stamp ==="

        start_server "$mode" "$keep" "$stamp"
        local server_log="$RESULTS_DIR/${stamp}/server.log"

        local client_rc=0
        case "$client" in
            claude_code) run_client_claude_code "$stamp" || client_rc=$? ;;
            opencode)    run_client_opencode    "$stamp" || client_rc=$? ;;
            hermes)      run_client_hermes      "$stamp" || client_rc=$? ;;
        esac

        echo "[harness] $client rc=$client_rc"

        # Extract MTP accept rates from server log
        echo "[harness] MTP accept rates:"
        extract_accept_rates "$server_log"

        # Check marker presence in output
        local out_file="$RESULTS_DIR/${stamp}/${client}.out"
        local marker_count
        marker_count=$(check_marker "$out_file" 2>/dev/null || echo "0")
        echo "[harness] marker ($MARKER) count: $marker_count"

        # Save per-run metadata
        python3 - "$server_log" "$out_file" "$client" "$mode" "$keep" <<'PYEOF'
import sys, re, json, os
server_log, out_file, client, mode, keep = sys.argv[1:]
rates = []
if os.path.exists(server_log):
    for line in open(server_log):
        m = re.search(r'accept_rate=([\d.]+)', line)
        if m: rates.append(float(m.group(1)))
marker_count = 0
if os.path.exists(out_file):
    marker_count = open(out_file).read().count("OK_DONE")
meta = {
    "client": client, "pflash_mode": mode, "keep_ratio": float(keep),
    "mtp_accept_rates": rates,
    "mtp_accept_mean": round(sum(rates)/len(rates), 3) if rates else None,
    "mtp_accept_min": min(rates) if rates else None,
    "mtp_accept_max": max(rates) if rates else None,
    "marker_count": marker_count,
}
stamp_dir = os.path.dirname(server_log)
json.dump(meta, open(f"{stamp_dir}/meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
PYEOF

        stop_server
        sleep 2
    done
}

echo "[harness] starting proper multi-turn bench: off and always at keep=0.05"
echo "[harness] results dir: $RESULTS_DIR"

run_all_clients "off" "0.05"
run_all_clients "always" "0.05"

echo ""
echo "[harness] all runs complete"
echo "[harness] results in: $RESULTS_DIR"
