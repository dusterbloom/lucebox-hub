#!/usr/bin/env bash
set -euo pipefail

lock_file="${KIMI_GPU_LEASE_LOCK:-/tmp/kimi-k3-gpu.lock}"
owner_file="${lock_file}.owner"

usage() {
    echo "usage: $0 status | run OWNER -- COMMAND [ARG ...]" >&2
    exit 2
}

case "${1:-}" in
status)
    exec 9>"$lock_file"
    if flock -n 9; then
        echo "free"
    else
        printf 'busy'
        if [[ -s "$owner_file" ]]; then
            printf ': '
            tr '\n' ' ' < "$owner_file"
        fi
        printf '\n'
    fi
    ;;
run)
    [[ $# -ge 4 && "$3" == "--" ]] || usage
    owner="$2"
    shift 3
    exec 9>"$lock_file"
    if ! flock -n 9; then
        printf 'GPU lease busy'
        if [[ -s "$owner_file" ]]; then
            printf ': '
            tr '\n' ' ' < "$owner_file"
        fi
        printf '\n' >&2
        exit 75
    fi
    printf 'owner=%s\npid=%s\nstarted_utc=%s\ncommand=' \
        "$owner" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$owner_file"
    printf '%q ' "$@" >> "$owner_file"
    printf '\n' >> "$owner_file"
    cleanup() {
        rm -f "$owner_file"
    }
    trap cleanup EXIT INT TERM
    "$@"
    ;;
*)
    usage
    ;;
esac
