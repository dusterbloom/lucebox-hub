#!/usr/bin/env bash
# Multi-turn harness composition benchmark: PFlash × {MTP, DFlash}
# Runs 7 cells: OFF/ALWAYS × {MTP, DFlash} + keep_ratio sweep for MTP.
# Uses direct HTTP multi-turn driver (3 chained turns per run).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/2026-05-21_harness_compose}"
mkdir -p "$RESULTS_DIR"

TARGET="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
MTP_GGUF="/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf"
DFLASH_DRAFT="/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf"
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

run_multiturn() {
    local stamp="$1"
    local out="$RESULTS_DIR/${stamp}/client.out"
    local timing_out="$RESULTS_DIR/${stamp}/timing.json"
    python3 "$SCRIPT_DIR/multiturn_client.py" \
        "$BASE_URL" "$MODEL_ID" "$out" "$timing_out"
}

extract_metrics() {
    local stamp="$1"
    local label="$2"
    shift 2
    local server_log="$RESULTS_DIR/${stamp}/server.log"
    local timing="$RESULTS_DIR/${stamp}/timing.json"
    local meta="$RESULTS_DIR/${stamp}/meta.json"
    python3 "$SCRIPT_DIR/extract_metrics.py" \
        "$server_log" "$timing" "$meta" "$label" "$stamp"
}

run_one() {
    local label="$1"
    local stamp="run-${label}-$(date +%H%M%S)"
    shift
    # remaining args go to dflash_server

    echo ""
    echo "=============================="
    echo "RUN: $label  stamp=$stamp"
    echo "=============================="

    local rc=0
    start_server "$stamp" "$@" || { echo "[harness] server start FAILED for $label"; rc=1; }

    if [[ "$rc" -eq 0 ]]; then
        run_multiturn "$stamp" || rc=$?
    fi

    echo "[harness] extracting metrics for $stamp"
    extract_metrics "$stamp" "$label" "$@" || true

    stop_server
    sleep 3

    # Append to runs.jsonl
    local meta="$RESULTS_DIR/${stamp}/meta.json"
    if [[ -f "$meta" ]]; then
        python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps(m))" "$meta" >> "$RESULTS_DIR/runs.jsonl" || true
    else
        echo "{\"label\":\"$label\",\"stamp\":\"$stamp\",\"error\":\"meta not found\"}" >> "$RESULTS_DIR/runs.jsonl"
    fi

    echo "[harness] run $label done (rc=$rc)"
    return $rc
}

# Verify DFlash draft model
if [[ ! -f "$DFLASH_DRAFT" ]]; then
    DFLASH_DRAFT=""
    echo "[harness] WARNING: DFlash draft model not found at $DFLASH_DRAFT; runs 3/4 will be skipped"
fi

echo "[harness] === PFlash × {MTP, DFlash} composition bench ==="
echo "[harness] results: $RESULTS_DIR"
echo "[harness] TARGET=$TARGET"
echo "[harness] MTP_GGUF=$MTP_GGUF"
echo "[harness] DRAFTER=$DRAFTER"
echo "[harness] DFLASH_DRAFT=${DFLASH_DRAFT:-MISSING}"

: > "$RESULTS_DIR/runs.jsonl"

# --- Run 1: MTP γ=2 / PFlash OFF ---
run_one "mtp-off" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode off \
    || true

# --- Run 2: MTP γ=2 / PFlash ALWAYS keep=0.05 ---
run_one "mtp-always-k05" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode always --prefill-keep-ratio 0.05 \
    --prefill-drafter "$DRAFTER" \
    || true

# --- Runs 3/4: DFlash chain ---
if [[ -n "${DFLASH_DRAFT:-}" ]] && [[ -f "$DFLASH_DRAFT" ]]; then
    run_one "dflash-off" \
        --draft "$DFLASH_DRAFT" --ddtree --ddtree-budget 16 \
        --pflash-mode off \
        || true

    run_one "dflash-always-k05" \
        --draft "$DFLASH_DRAFT" --ddtree --ddtree-budget 16 \
        --pflash-mode always --prefill-keep-ratio 0.05 \
        --prefill-drafter "$DRAFTER" \
        || true
else
    echo "[harness] SKIP runs 3/4: DFlash draft model not available"
    echo '{"label":"dflash-off","error":"draft model not found","ok_done_seen":false}' >> "$RESULTS_DIR/runs.jsonl"
    echo '{"label":"dflash-always-k05","error":"draft model not found","ok_done_seen":false}' >> "$RESULTS_DIR/runs.jsonl"
fi

# --- Run 5: MTP γ=2 / PFlash ALWAYS keep=0.025 ---
run_one "mtp-always-k025" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode always --prefill-keep-ratio 0.025 \
    --prefill-drafter "$DRAFTER" \
    || true

# --- Run 6: MTP γ=2 / PFlash ALWAYS keep=0.10 ---
run_one "mtp-always-k10" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode always --prefill-keep-ratio 0.10 \
    --prefill-drafter "$DRAFTER" \
    || true

# --- Run 7: MTP γ=2 / PFlash ALWAYS keep=0.20 ---
run_one "mtp-always-k20" \
    --mtp-gguf "$MTP_GGUF" --mtp-gamma 2 \
    --pflash-mode always --prefill-keep-ratio 0.20 \
    --prefill-drafter "$DRAFTER" \
    || true

echo ""
echo "[harness] === All runs complete ==="
echo "[harness] runs.jsonl: $RESULTS_DIR/runs.jsonl"
echo "[harness] Generating SUMMARY.md..."

python3 "$SCRIPT_DIR/gen_summary.py" "$RESULTS_DIR"

echo "[harness] Done."
