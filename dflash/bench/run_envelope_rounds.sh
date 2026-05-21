#!/usr/bin/env bash
# Multi-round envelope sweep driver.
# Starts dflash_server once per (mode, keep_ratio) config, runs sweep_envelope.py
# with a single-mode/single-keep grid slice, then kills the server.
#
# Usage:
#   bash dflash/bench/run_envelope_rounds.sh [--out <dir>] [--dry-run]
#
# Environment:
#   GGML_CUDA_NO_VMM=1  (always exported in this script)
#   DFLASH27B_KV_K=tq3_0, DFLASH27B_KV_V=tq3_0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVER="${WORKTREE_ROOT}/dflash/build/dflash_server"
MODEL="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
SERVER_LOG="/tmp/sweep_server.log"
SERVER_URL="http://127.0.0.1:8080"
MAX_CTX=73728
PORT=8080

OUT_DIR="${WORKTREE_ROOT}/dflash/bench/results/2026-05-21_envelope"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export GGML_CUDA_NO_VMM=1
export DFLASH27B_KV_K=tq3_0
export DFLASH27B_KV_V=tq3_0

SWEEP="${SCRIPT_DIR}/sweep_envelope.py"
GRID_BASE="${OUT_DIR}/grid.yaml"

START_SECS=$SECONDS

wait_for_server() {
    local deadline=$((SECONDS + 120))
    while [[ $SECONDS -lt $deadline ]]; do
        if grep -q "listening on http://127.0.0.1:${PORT}" "${SERVER_LOG}" 2>/dev/null; then
            echo "[runner] server up after $((SECONDS - start_secs))s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[runner] server died, last log:"
            tail -20 "${SERVER_LOG}"
            return 1
        fi
        sleep 2
    done
    echo "[runner] server timeout after 120s"
    tail -20 "${SERVER_LOG}"
    return 1
}

run_round() {
    local mode="$1"
    local keep="$2"
    local start_secs=$SECONDS

    echo ""
    echo "===== ROUND: mode=${mode} keep=${keep} ====="

    # Build a single-slice grid yaml for this round
    local tmp_grid
    tmp_grid=$(mktemp /tmp/grid_XXXXXX.yaml)
    cat > "${tmp_grid}" <<YAML
ctx_tokens: [4096, 8192, 16384, 32768, 65536]
keep_ratio: [${keep}]
mode: ["${mode}"]
n_per_cell: 5
tasks: [niah_single, vt, fwe, mqa]
YAML

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[runner] DRY-RUN: would start server mode=${mode} keep=${keep}"
        python3 "${SWEEP}" --grid "${tmp_grid}" --out "${OUT_DIR}" \
            --server-url "${SERVER_URL}" --dry-run
        rm -f "${tmp_grid}"
        return 0
    fi

    # Kill any stale server
    pkill -f dflash_server 2>/dev/null || true
    sleep 1

    # Launch server
    local pflash_flags=""
    if [[ "${mode}" == "off" ]]; then
        pflash_flags="--prefill-compression off"
    else
        pflash_flags="--prefill-compression ${mode} --prefill-keep-ratio ${keep} --prefill-drafter ${DRAFTER} --prefill-skip-park"
    fi

    # shellcheck disable=SC2086
    GGML_CUDA_NO_VMM=1 \
    DFLASH27B_KV_K=tq3_0 DFLASH27B_KV_V=tq3_0 \
    "${SERVER}" "${MODEL}" \
        --host 127.0.0.1 --port ${PORT} \
        --max-ctx ${MAX_CTX} \
        --cache-type-k tq3_0 \
        --cache-type-v tq3_0 \
        ${pflash_flags} \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "[runner] server PID=${SERVER_PID} mode=${mode} keep=${keep}"

    if ! wait_for_server; then
        echo "[runner] FATAL: server failed to start for mode=${mode} keep=${keep}"
        rm -f "${tmp_grid}"
        return 1
    fi

    # Run sweep for this slice
    python3 "${SWEEP}" --grid "${tmp_grid}" --out "${OUT_DIR}" \
        --server-url "${SERVER_URL}"

    # Kill server cleanly
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 2

    rm -f "${tmp_grid}"
    echo "[runner] round done in $((SECONDS - start_secs))s"
}

echo "[runner] output: ${OUT_DIR}"
echo "[runner] model: ${MODEL}"
echo "[runner] drafter: ${DRAFTER}"
echo "[runner] start: $(date)"
mkdir -p "${OUT_DIR}"

# Round 1: mode=off (keep_ratio irrelevant for off, but sweep still creates keep-tagged dirs)
run_round "off" "0.1"

# Rounds 2-5: mode=always, each keep_ratio
for keep in 0.025 0.05 0.1 0.2; do
    run_round "always" "${keep}"
done

echo ""
echo "[runner] ALL ROUNDS COMPLETE in $((SECONDS - START_SECS))s total"
echo "[runner] collecting frontier..."

# Merge per-round frontier.json files won't work (sweep writes one frontier.json
# per run, overwriting each time). Re-collect from all summary.json files.
python3 - "${OUT_DIR}" <<'PYEOF'
import sys, json, pathlib

base = pathlib.Path(sys.argv[1])
rows = []
for summary_file in sorted(base.rglob("summary.json")):
    try:
        d = json.loads(summary_file.read_text())
        task = d.get("task")
        ctx  = d.get("ctx_tokens")
        keep = d.get("keep_ratio")
        mode = d.get("mode")
        acc  = d.get("accuracy")
        rows.append({
            "task": task, "ctx": ctx, "keep": keep, "mode": mode,
            "accuracy": acc,
            "wall_p50": d.get("wall_p50"),
            "wall_p95": d.get("wall_p95"),
            "n_cases": d.get("n_cases"),
            "source": str(summary_file.relative_to(base)),
        })
    except Exception as e:
        print(f"  [warn] {summary_file}: {e}")

frontier_path = base / "frontier.json"
frontier_path.write_text(json.dumps(rows, indent=2))
print(f"[frontier] {len(rows)} rows -> {frontier_path}")
PYEOF

echo "[runner] done $(date)"
