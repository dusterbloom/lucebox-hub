#!/usr/bin/env bash
set -euo pipefail

pair_script=$(dirname "$0")/run_kimi_terminal_pair.sh
expected_pair_sha256=6c13193762a22cb6c5438caac3ac3bf802df547bd08cf6043a24d6654350d015
minimum_available_kib=${KIMI_K3_MIN_AVAILABLE_KIB:-60000000}
stable_seconds=${KIMI_K3_CAPACITY_STABLE_SECONDS:-30}
block_pattern=${KIMI_K3_CAPACITY_BLOCK_PATTERN:-^$}
poll_seconds=5

actual_pair_sha256=$(sha256sum "$pair_script" | cut -d' ' -f1)
if [[ $actual_pair_sha256 != "$expected_pair_sha256" ]]; then
    echo "paired runner changed: expected $expected_pair_sha256, found $actual_pair_sha256" >&2
    exit 1
fi
if [[ ! $minimum_available_kib =~ ^[0-9]+$ || ! $stable_seconds =~ ^[0-9]+$ ]]; then
    echo "capacity thresholds must be non-negative integers" >&2
    exit 2
fi

clear_seconds=0
while (( clear_seconds < stable_seconds )); do
    available_kib=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
    blocked=0
    if pgrep -af -- "$block_pattern" >/dev/null; then
        blocked=1
    fi
    if (( available_kib >= minimum_available_kib && blocked == 0 )); then
        clear_seconds=$((clear_seconds + poll_seconds))
    else
        clear_seconds=0
    fi
    if (( clear_seconds < stable_seconds )); then
        sleep "$poll_seconds"
    fi
done

exec "$pair_script" "$@"
