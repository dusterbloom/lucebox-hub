#!/usr/bin/env bash
# Retry the two failed cells from the composition bench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/2026-05-21_harness_compose"

TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
MTP_GGUF="/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf"
DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
DFLASH_SERVER="$REPO_DIR/dflash/build/dflash_server"
PORT=18080
HOST=127.0.0.1
BASE_URL="http://$HOST:$PORT"
MODEL_ID="luce-dflash"
MAX_CTX=33792
MAX_TOKENS=512
SERVER_PID=""

export GGML_CUDA_NO_VMM=1
export DFLASH27B_KV_K=tq3_0
export DFLASH27B_KV_V=tq3_0

start_server() {
    local stamp="$1"
    shift
    local server_log="$RESULTS_DIR/${stamp}/server.log"
    mkdir -p "$RESULTS_DIR/${stamp}"

    fuser -k "$PORT/tcp" 2>/dev/null || true
    sleep 2

    "$DFLASH_SERVER" "$TARGET" \
        --host "$HOST" --port "$PORT" \
        --max-ctx "$MAX_CTX" \
        --max-tokens "$MAX_TOKENS" \
        --model-name "$MODEL_ID" \
        --cache-type-k tq3_0 \
        --cache-type-v tq3_0 \
        "$@" \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    for i in $(seq 1 180); do
        if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
            echo "[harness] server ready (pid=$SERVER_PID) after ${i}s"
            return 0
        fi
        sleep 1
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[harness] server exited early; log: $server_log" >&2
            tail -n 60 "$server_log" >&2 || true
            return 1
        fi
    done
    echo "[harness] server timeout" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

run_one() {
    local label="$1"
    local stamp="run-retry-${label}-$(date +%H%M%S)"
    shift

    echo ""
    echo "=============================="
    echo "RETRY: $label  stamp=$stamp"
    echo "=============================="

    local rc=0
    start_server "$stamp" "$@" || { echo "[harness] server start FAILED"; rc=1; }

    if [[ "$rc" -eq 0 ]]; then
        python3 "$SCRIPT_DIR/multiturn_client.py" \
            "$BASE_URL" "$MODEL_ID" \
            "$RESULTS_DIR/${stamp}/client.out" \
            "$RESULTS_DIR/${stamp}/timing.json" || rc=$?
    fi

    python3 "$SCRIPT_DIR/extract_metrics.py" \
        "$RESULTS_DIR/${stamp}/server.log" \
        "$RESULTS_DIR/${stamp}/timing.json" \
        "$RESULTS_DIR/${stamp}/meta.json" \
        "$label" "$stamp" || true

    stop_server
    sleep 3

    local meta="$RESULTS_DIR/${stamp}/meta.json"
    if [[ -f "$meta" ]]; then
        python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(m))" "$meta" >> "$RESULTS_DIR/runs.jsonl"
    else
        echo "{\"label\":\"$label\",\"stamp\":\"$stamp\",\"error\":\"meta not found\"}" >> "$RESULTS_DIR/runs.jsonl"
    fi

    echo "[harness] retry $label done (rc=$rc)"
    return $rc
}

echo "[harness] Retrying 2 failed cells..."

run_one "mtp-always-k05" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode always --prefill-keep-ratio 0.05 \
    --prefill-drafter "$DRAFTER" \
    || true

run_one "mtp-always-k025" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode always --prefill-keep-ratio 0.025 \
    --prefill-drafter "$DRAFTER" \
    || true

echo "[harness] Retries complete. Generating updated summary..."
python3 "$SCRIPT_DIR/gen_summary.py" "$RESULTS_DIR"
