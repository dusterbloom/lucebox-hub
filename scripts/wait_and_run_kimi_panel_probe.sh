#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${KIMI_PANEL_GPU:-0}"
poll_seconds="${KIMI_PANEL_POLL_SECONDS:-60}"

timestamp() {
    date --iso-8601=seconds
}

if [[ ! "$poll_seconds" =~ ^[0-9]+$ ]] || (( poll_seconds < 5 || poll_seconds > 60 )); then
    echo "KIMI_PANEL_POLL_SECONDS must be from 5 through 60" >&2
    exit 2
fi

echo "$(timestamp) waiting for the Kimi checkpoint transfer"
while pgrep -f '[h]f download unsloth/Kimi-K3-GGUF' >/dev/null; do
    sleep "$poll_seconds"
done

echo "$(timestamp) transfer process ended; waiting for an idle graphics card"
idle_samples=0
while (( idle_samples < 3 )); do
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | rg -q '[0-9]'; then
        idle_samples=0
    else
        idle_samples=$((idle_samples + 1))
    fi
    if (( idle_samples < 3 )); then
        sleep "$poll_seconds"
    fi
done

echo "$(timestamp) graphics card remained idle; starting the real panel probe"
exec "$repo_dir/scripts/run_kimi_panel_probe.sh"
