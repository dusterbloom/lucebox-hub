#!/usr/bin/env bash
# run_bandit_abc_seeds.sh — multi-seed A/B/C run for variance evidence.
#
# Usage:
#   bash server/bench/run_bandit_abc_seeds.sh <seed_label> <prompt_basename> [<session_suffix>]
#
# Arguments:
#   seed_label:       seed1 | seed2 | seed3
#   prompt_basename:  basename of prompt under harness/clients/prompts/
#                     e.g. decode_check.txt, repo_inspection.txt
#   session_suffix:   (optional) unique suffix for condition C session_id;
#                     defaults to seed_label + timestamp
#
# CLIENT env var selects which primary-5 client to use (default: claude_code).
# Valid: claude_code | codex | pi | hermes | opencode
#
# Examples:
#   CLIENT=claude_code bash server/bench/run_bandit_abc_seeds.sh seed1 decode_check.txt day5s1
#   CLIENT=codex       bash server/bench/run_bandit_abc_seeds.sh seed2 repo_inspection.txt day5s2
#   CLIENT=hermes      bash server/bench/run_bandit_abc_seeds.sh seed3 decode_check.txt day5s3
#
# Port: 19099 (alternate; never touches user's :18099).
# Lock: flock /tmp/lucebox-bench.lock held per condition.
set -euo pipefail

SEED_LABEL="${1:?Usage: $0 <seed_label> <prompt_basename> [session_suffix]}"
PROMPT_BASENAME="${2:?Usage: $0 <seed_label> <prompt_basename> [session_suffix]}"
SESSION_SUFFIX="${3:-${SEED_LABEL}_$(date +%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
HARNESS_DIR="$REPO_DIR/harness/clients"

CLIENT="${CLIENT:-claude_code}"
RESULTS_BASE="${RESULTS_BASE:-$REPO_DIR/server/bench/results/2026-05-27_full_harness}"
RESULTS_DIR="$RESULTS_BASE/$CLIENT/bandit_abc_seeds/$SEED_LABEL"

SERVER_BIN="${DFLASH_SERVER_BIN:-$REPO_DIR/server/build/dflash_server}"
TARGET="${TARGET:-/path/to/your/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/path/to/your/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf}"
PFLASH_DRAFTER="${PFLASH_DRAFTER:-/path/to/your/models/Qwen3-0.6B-BF16.gguf}"
PROMPT_FILE="$HARNESS_DIR/prompts/$PROMPT_BASENAME"
MARKER="${MARKER:-OK_DONE}"
CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-600}"

HOST=127.0.0.1
PORT=19099
PROXY_PORT=19082
MODEL_ID="luce-dflash"
API_KEY="sk-lucebox"
BASE_URL="http://$HOST:$PORT"

# Validate client
VALID_CLIENTS=(claude_code codex pi hermes opencode)
CLIENT_VALID=0
for v in "${VALID_CLIENTS[@]}"; do
    if [[ "$CLIENT" == "$v" ]]; then CLIENT_VALID=1; break; fi
done
if [[ "$CLIENT_VALID" -eq 0 ]]; then
    echo "ERROR: CLIENT must be one of: ${VALID_CLIENTS[*]}" >&2
    exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

CLIENT_HARNESS="$HARNESS_DIR/run_${CLIENT}.sh"
if [[ ! -x "$CLIENT_HARNESS" ]]; then
    echo "ERROR: client harness not found or not executable: $CLIENT_HARNESS" >&2
    exit 1
fi

trap 'kill ${_OUTER_SERVER_PID:-} ${_OUTER_PROXY_PID:-} 2>/dev/null; exit' EXIT INT TERM

mkdir -p "$RESULTS_DIR"
echo "=== bandit seeds A/B/C [$SEED_LABEL] client=$CLIENT prompt=$PROMPT_BASENAME start $(date -Is) ===" | tee "$RESULTS_DIR/run.log"

run_condition() {
    local label="$1"
    local keep="$2"
    local sid="$3"
    local cdir="$RESULTS_DIR/$label"
    mkdir -p "$cdir"

    local slog="$cdir/server.log"
    local plog="$cdir/proxy.log"
    local cout="$cdir/client.out"
    local mfile="$cdir/metrics.txt"

    echo "--- [$SEED_LABEL/$label] keep=$keep sid='$sid' $(date -Is) ---" | tee -a "$RESULTS_DIR/run.log"
    local t0; t0=$(date +%s)

    _SID="$sid" _KEEP="$keep" _SLOG="$slog" _PLOG="$plog" _COUT="$cout" \
    _CHOME="$cdir/claude_home" \
    _SERVER_BIN="$SERVER_BIN" _TARGET="$TARGET" _DRAFT="$DRAFT" \
    _PFLASH_DRAFTER="$PFLASH_DRAFTER" \
    _HARNESS_DIR="$HARNESS_DIR" _CLIENT="$CLIENT" \
    _PROMPT_FILE="$PROMPT_FILE" \
    _HOST="$HOST" _PORT="$PORT" _PROXY_PORT="$PROXY_PORT" \
    _MODEL_ID="$MODEL_ID" _API_KEY="$API_KEY" _BASE_URL="$BASE_URL" \
    _CLAUDE_TIMEOUT="$CLAUDE_TIMEOUT" _MARKER="$MARKER" \
    flock -x /tmp/lucebox-bench.lock bash <<'INNER'
set -eo pipefail
export DFLASH27B_KV_K=tq3_0
export DFLASH27B_KV_V=tq3_0
export GGML_CUDA_NO_VMM=1
export PFLASH_DRAFTER_EARLY_EXIT_N=7
export PFLASH_DRAFTER_SCORE_LAYERS=7

trap 'kill ${SPID:-} ${PPID_VAR:-} 2>/dev/null; wait ${SPID:-} ${PPID_VAR:-} 2>/dev/null; exit' EXIT INT TERM

"$_SERVER_BIN" "$_TARGET" \
    --draft "$_DRAFT" \
    --prefill-drafter "$_PFLASH_DRAFTER" \
    --host $_HOST --port $_PORT \
    --max-ctx 98304 --max-tokens 512 \
    --model-name "$_MODEL_ID" \
    --ddtree --ddtree-budget 16 \
    --prefill-compression always \
    --prefill-keep-ratio "$_KEEP" \
    > "$_SLOG" 2>&1 &
SPID=$!

for i in $(seq 1 120); do
    if curl -fsS "http://$_HOST:$_PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
    if ! kill -0 "$SPID" 2>/dev/null; then
        echo "server died" >&2; tail -n 40 "$_SLOG" >&2; exit 1
    fi
    if [[ $i -eq 120 ]]; then echo "server timeout" >&2; exit 1; fi
done
echo "server up (pid=$SPID)"

PPID_VAR=""
CLIENT_URL="http://$_HOST:$_PORT"
if [[ -n "$_SID" ]]; then
    python3 "$_HARNESS_DIR/session_inject_proxy.py" \
        --host $_HOST \
        --port $_PROXY_PORT \
        --upstream "$CLIENT_URL" \
        --session-id "$_SID" \
        >> "$_PLOG" 2>&1 &
    PPID_VAR=$!
    _proxy_ready=0
    for i in $(seq 1 10); do
        if curl -fsS "http://$_HOST:$_PROXY_PORT/health" >/dev/null 2>&1; then _proxy_ready=1; break; fi
        sleep 1
        if ! kill -0 "$PPID_VAR" 2>/dev/null; then
            echo "proxy died" >&2; cat "$_PLOG" >&2; exit 1
        fi
    done
    if [[ "$_proxy_ready" -eq 0 ]]; then
        echo "proxy not ready after 10s" >&2; kill "$PPID_VAR" 2>/dev/null; exit 1
    fi
    CLIENT_URL="http://$_HOST:$_PROXY_PORT"
    echo "proxy up on $CLIENT_URL (session=$_SID)"
fi

export MODEL_SERVER=lucebox
export LUCEBOX_SERVER_BACKEND=cpp
export DFLASH_SERVER_BIN="$_SERVER_BIN"
export TARGET="$_TARGET" DRAFT="$_DRAFT"
export HOST="$_HOST" PORT="$_PORT" MODEL_ID="$_MODEL_ID" API_KEY="$_API_KEY"
export MARKER="$_MARKER" PROMPT_FILE="$_PROMPT_FILE" CLAUDE_TIMEOUT="$_CLAUDE_TIMEOUT"
export BASE_URL="$CLIENT_URL"
export STAMP="bandit-seeds-$$"
export RUN_DIR="/tmp/lucebox-bench-runs"
export REPO_DIR="$(cd "$_HARNESS_DIR/../.." && pwd)"

bash "$_HARNESS_DIR/run_${_CLIENT}.sh" > "$_COUT" 2>&1 || true
INNER

    local t1; t1=$(date +%s)
    local wall=$((t1 - t0))

    local ok_done="NO"
    if grep -q "$MARKER" "$cout" 2>/dev/null; then ok_done="YES"; fi

    local ar; ar=$(grep 'spec-decode' "$slog" 2>/dev/null | \
        grep -oE '\(([0-9.]+)%\)' | tail -1 | tr -d '()%' || echo "N/A")
    [[ -z "$ar" ]] && ar="N/A"

    local dfwd; dfwd=$(grep '\[drafter\] forward+score in' "$slog" 2>/dev/null | \
        grep -oE 'in [0-9.]+s' | \
        awk '{s+=$2*1000; n++} END{if(n) printf "%.0f ms (n=%d)",s/n,n; else print "N/A"}' || echo "N/A")
    [[ -z "$dfwd" ]] && dfwd="N/A"

    local bandit; bandit=$(grep '\[pflash-bandit\]' "$slog" 2>/dev/null | tail -5 || echo "none")

    {
        echo "seed=$SEED_LABEL"
        echo "client=$CLIENT"
        echo "prompt=$PROMPT_BASENAME"
        echo "label=$label"
        echo "keep_ratio=$keep"
        echo "session_id=$sid"
        echo "wall_s=$wall"
        echo "ok_done=$ok_done"
        echo "accept_rate=$ar"
        echo "mean_drafter_fwd_ms=$dfwd"
        echo "bandit_log:"
        echo "$bandit"
    } | tee "$mfile" | tee -a "$RESULTS_DIR/run.log"

    echo "[$SEED_LABEL/$label] wall=${wall}s ok=$ok_done ar=$ar" | tee -a "$RESULTS_DIR/run.log"
}

SESSION_ID="${PFLASH_SESSION_ID:-${CLIENT}_${SESSION_SUFFIX}}"

run_condition "A_fixed_low"  "0.05" ""
run_condition "B_fixed_high" "0.20" ""
run_condition "C_bandit"     "0.10" "$SESSION_ID"

echo "=== bandit seeds [$SEED_LABEL] client=$CLIENT done $(date -Is) ===" | tee -a "$RESULTS_DIR/run.log"

echo ""
echo "=== SUMMARY [$SEED_LABEL] client=$CLIENT ==="
printf "%-18s %10s %8s %12s %8s  %s\n" "Condition" "wall_s" "ok_done" "accept_rate" "keep" "bandit"
for cond in A_fixed_low B_fixed_high C_bandit; do
    mf="$RESULTS_DIR/$cond/metrics.txt"
    if [[ -f "$mf" ]]; then
        wall=$(grep "^wall_s=" "$mf" | cut -d= -f2)
        ok=$(grep "^ok_done=" "$mf" | cut -d= -f2)
        ar=$(grep "^accept_rate=" "$mf" | cut -d= -f2)
        keep=$(grep "^keep_ratio=" "$mf" | cut -d= -f2)
        sid=$(grep "^session_id=" "$mf" | cut -d= -f2)
        bandit_note=""
        if [[ -n "$sid" ]]; then bandit_note="yes"; else bandit_note="-"; fi
        printf "%-18s %10s %8s %12s %8s  %s\n" "$cond" "$wall" "$ok" "$ar" "$keep" "$bandit_note"
    fi
done
