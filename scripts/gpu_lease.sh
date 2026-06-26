#!/usr/bin/env bash
# gpu_lease.sh — single-owner GPU lease for lucebox-hub sessions.
# Usage: gpu_lease.sh <acquire|heartbeat|release|status|selftest> [args...]
#
# Lease file  : /tmp/lucebox_gpu.lease          (key=value, one per line)
# Lock file   : /tmp/lucebox_gpu.lease.lock     (flock target, never read)
#
# PID in lease = PPID of the acquire invocation (the calling shell).
# Liveness is checked via /proc/<pid>/status (works across uid boundaries).
# Heartbeat TTL = 120 s.  A lease is STALE when pid is dead OR heartbeat
# is older than TTL.  A stale lease is auto-reclaimed by any subsequent
# acquire call.

set -euo pipefail

LEASE_FILE="/tmp/lucebox_gpu.lease"
LOCK_FILE="/tmp/lucebox_gpu.lease.lock"
HEARTBEAT_TTL=120   # seconds

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_now() { date +%s; }

# Liveness via /proc — works without signal permission to foreign UIDs.
_pid_alive() {
    local p="${1:-}"
    [[ -n "$p" ]] && [[ -e "/proc/${p}/status" ]]
}

# Returns 0 (true/stale) when pid is dead OR heartbeat is too old.
_lease_stale() {
    local l_pid="${1:-}" l_hb="${2:-0}"
    if ! _pid_alive "$l_pid"; then return 0; fi
    local age=$(( $(_now) - l_hb ))
    (( age >= HEARTBEAT_TTL )) && return 0
    return 1
}

_age_s() { echo $(( $(_now) - ${1:-0} )); }

# Source lease fields into the CALLING scope.  Returns 1 if file missing.
_read_lease() {
    [[ -f "$LEASE_FILE" ]] || return 1
    # Declare expected fields first so they don't bleed from env.
    # shellcheck disable=SC1090
    source "$LEASE_FILE"
}

# Write a new lease; l_pid = PPID of this script invocation (the caller shell).
_write_lease() {
    local s_id="$1" s_purpose="$2" s_port="${3:-}" now
    now=$(_now)
    printf 'session_id=%s\npid=%s\npurpose=%s\nport=%s\nacquired_epoch=%s\nheartbeat_epoch=%s\n' \
        "$s_id" "$PPID" "$s_purpose" "$s_port" "$now" "$now" > "$LEASE_FILE"
}

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

cmd_acquire() {
    local arg_session="${1:?acquire requires <session_id>}"
    local arg_purpose="${2:?acquire requires <purpose>}"
    local arg_port="${3:-}"

    (
        flock -x 9

        # Isolate lease vars so source doesn't pollute outer scope.
        local session_id="" pid="" purpose="" port="" acquired_epoch=0 heartbeat_epoch=0

        if _read_lease 2>/dev/null; then
            if ! _lease_stale "$pid" "$heartbeat_epoch"; then
                local hb_age
                hb_age=$(_age_s "$heartbeat_epoch")
                echo "HELD by session=${session_id} pid=${pid} purpose=${purpose} port=${port} age=${hb_age}s"
                exit 1
            fi
            # Stale — reclaim.
            local old_hb_age
            old_hb_age=$(( $(_now) - heartbeat_epoch ))
            echo "RECLAIMED stale lease from session=${session_id} (pid=${pid} hb_age=${old_hb_age}s)"
        fi

        _write_lease "$arg_session" "$arg_purpose" "$arg_port"
        echo "ACQUIRED session=${arg_session} pid=${PPID} purpose=${arg_purpose} port=${arg_port}"

    ) 9>"$LOCK_FILE"
}

cmd_heartbeat() {
    local arg_session="${1:?heartbeat requires <session_id>}"

    (
        flock -x 9

        local session_id="" pid="" purpose="" port="" acquired_epoch=0 heartbeat_epoch=0

        if ! _read_lease 2>/dev/null; then
            echo "ERROR: no lease file" >&2; exit 1
        fi

        if [[ "$session_id" != "$arg_session" ]]; then
            echo "ERROR: not the owner (owner=${session_id})" >&2; exit 1
        fi

        local now
        now=$(_now)
        printf 'session_id=%s\npid=%s\npurpose=%s\nport=%s\nacquired_epoch=%s\nheartbeat_epoch=%s\n' \
            "$session_id" "$pid" "$purpose" "$port" "$acquired_epoch" "$now" > "$LEASE_FILE"
        echo "HEARTBEAT session=${session_id} epoch=${now}"

    ) 9>"$LOCK_FILE"
}

cmd_release() {
    local arg_session="${1:?release requires <session_id>}"

    (
        flock -x 9

        local session_id="" pid="" purpose="" port="" acquired_epoch=0 heartbeat_epoch=0

        if ! _read_lease 2>/dev/null; then
            echo "RELEASED (no lease — idempotent)"; exit 0
        fi

        if [[ "$session_id" != "$arg_session" ]]; then
            echo "ERROR: not the owner (owner=${session_id})" >&2; exit 1
        fi

        rm -f "$LEASE_FILE"
        echo "RELEASED session=${arg_session}"

    ) 9>"$LOCK_FILE"
}

cmd_status() {
    echo "=== lucebox GPU lease status ==="

    local session_id="" pid="" purpose="" port="" acquired_epoch=0 heartbeat_epoch=0

    if ! _read_lease 2>/dev/null; then
        echo "FREE"
    else
        local hb_age acq_age alive_flag stale_flag
        hb_age=$(_age_s "$heartbeat_epoch")
        acq_age=$(_age_s "$acquired_epoch")

        if _pid_alive "$pid"; then alive_flag="ALIVE"; else alive_flag="DEAD"; fi
        if _lease_stale "$pid" "$heartbeat_epoch"; then
            stale_flag="STALE (reclaimable)"
        else
            stale_flag="LIVE"
        fi

        printf 'session  : %s\npid      : %s (%s)\npurpose  : %s\nport     : %s\nacquired : %ss ago\nlast_hb  : %ss ago\nlease    : %s\n' \
            "$session_id" "$pid" "$alive_flag" "$purpose" "$port" \
            "$acq_age" "$hb_age" "$stale_flag"
    fi

    echo ""
    echo "=== GPU compute procs ==="
    if command -v nvidia-smi &>/dev/null; then
        local gpu_pids
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        if [[ -z "$gpu_pids" ]]; then
            echo "(none)"
        else
            while IFS= read -r gpid; do
                gpid="${gpid// /}"
                [[ -z "$gpid" ]] && continue
                local env_session="(not set)"
                if [[ -r "/proc/${gpid}/environ" ]]; then
                    local s
                    s=$(tr '\0' '\n' < "/proc/${gpid}/environ" 2>/dev/null \
                        | grep '^CLAUDE_CODE_SESSION_ID=' | cut -d= -f2- || true)
                    [[ -n "$s" ]] && env_session="$s"
                fi
                printf 'gpu_pid=%-8s  CLAUDE_CODE_SESSION_ID=%s\n' "$gpid" "$env_session"
            done <<< "$gpu_pids"
        fi
    else
        echo "(nvidia-smi not found — lease enforcement only)"
    fi
}

cmd_selftest() {
    echo "=== gpu_lease.sh selftest ==="
    local SCRIPT fail=0
    SCRIPT="$(realpath "$0")"

    rm -f "$LEASE_FILE" "$LOCK_FILE"

    # ---- Step 1: Write a synthetic live lease (pid=self, fresh heartbeat) ----
    echo "[1] Injecting live lease (pid=$$, fresh heartbeat)..."
    local now
    now=$(_now)
    printf 'session_id=session-A\npid=%s\npurpose=selftest-A\nport=18099\nacquired_epoch=%s\nheartbeat_epoch=%s\n' \
        "$$" "$now" "$now" > "$LEASE_FILE"
    echo "    Lease written: pid=$$, hb=${now}"
    echo ""

    # ---- Step 2: Acquire as B must FAIL (HELD) ----
    echo "[2] Acquire as session-B — must FAIL with HELD..."
    local out rc
    out=$(bash "$SCRIPT" acquire "session-B" "selftest-B" 18100 2>&1) && rc=$? || rc=$?
    echo "    -> ${out}"
    if (( rc != 0 )) && echo "$out" | grep -q "HELD"; then
        echo "PASS: session-B correctly blocked"
    else
        echo "FAIL: expected HELD, got rc=${rc}: ${out}"; fail=1
    fi
    echo ""

    # ---- Step 3: Expire heartbeat → stale ----
    echo "[3] Backdating heartbeat_epoch to make lease stale..."
    local stale_ts=$(( now - HEARTBEAT_TTL - 10 ))
    printf 'session_id=session-A\npid=%s\npurpose=selftest-A\nport=18099\nacquired_epoch=%s\nheartbeat_epoch=%s\n' \
        "$$" "$now" "$stale_ts" > "$LEASE_FILE"
    echo "    heartbeat_epoch set to ${stale_ts} (age $((now - stale_ts))s >= TTL ${HEARTBEAT_TTL}s)"
    echo ""

    # ---- Step 4: Acquire B reclaims stale lease ----
    echo "[4] Acquire as session-B — must RECLAIM stale lease..."
    out=$(bash "$SCRIPT" acquire "session-B" "selftest-B" 18100 2>&1) && rc=$? || rc=$?
    echo "    -> ${out}"
    if (( rc == 0 )) && echo "$out" | grep -q "RECLAIMED"; then
        echo "PASS: stale reclaim succeeded"
    else
        echo "FAIL: expected RECLAIMED+ACQUIRED, got rc=${rc}: ${out}"; fail=1
    fi
    echo ""

    # ---- Step 5: Release B ----
    echo "[5] Release session-B..."
    out=$(bash "$SCRIPT" release "session-B" 2>&1) && rc=$? || rc=$?
    echo "    -> ${out}"
    if echo "$out" | grep -q "RELEASED"; then
        echo "PASS"
    else
        echo "FAIL: ${out}"; fail=1
    fi
    echo ""

    # ---- Step 6: Acquire B on FREE lease ----
    echo "[6] Acquire as session-B on FREE lease — must SUCCEED..."
    out=$(bash "$SCRIPT" acquire "session-B" "selftest-B" 18100 2>&1) && rc=$? || rc=$?
    echo "    -> ${out}"
    if (( rc == 0 )) && echo "$out" | grep -q "ACQUIRED"; then
        echo "PASS"
    else
        echo "FAIL: expected ACQUIRED, got rc=${rc}: ${out}"; fail=1
    fi
    echo ""

    # Cleanup
    bash "$SCRIPT" release "session-B" > /dev/null 2>&1 || true
    rm -f "$LEASE_FILE" "$LOCK_FILE"

    if (( fail == 0 )); then
        echo "=== selftest PASSED ==="
        return 0
    else
        echo "=== selftest FAILED ==="
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

CMD="${1:-}"
shift || true

case "$CMD" in
    acquire)   cmd_acquire   "$@" ;;
    heartbeat) cmd_heartbeat "$@" ;;
    release)   cmd_release   "$@" ;;
    status)    cmd_status    ;;
    selftest)  cmd_selftest  ;;
    *)
        echo "Usage: $0 <acquire|heartbeat|release|status|selftest> [args...]" >&2
        exit 1
        ;;
esac
