#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${KIMI_PANEL_GPU:-0}"
poll_seconds="${KIMI_PANEL_POLL_SECONDS:-60}"
revision="a0836360ce58dfec088d966a97f2ddc8a606279b"
model_root="${KIMI_PANEL_MODEL_ROOT:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF}"
complete_marker="${KIMI_PANEL_DOWNLOAD_MARKER:-$model_root/.ud-iq1s-$revision.complete}"

timestamp() {
    date --iso-8601=seconds
}

if [[ ! "$poll_seconds" =~ ^[0-9]+$ ]] || (( poll_seconds < 5 || poll_seconds > 60 )); then
    echo "KIMI_PANEL_POLL_SECONDS must be from 5 through 60" >&2
    exit 2
fi

echo "$(timestamp) waiting for the Kimi checkpoint transfer"
while [[ ! -f "$complete_marker" ]]; do
    if ! tmux has-session -t k3-download 2>/dev/null; then
        echo "$(timestamp) download session ended without a completion marker" >&2
        exit 1
    fi
    sleep "$poll_seconds"
done

echo "$(timestamp) pinned transfer completed; waiting for an idle graphics card"
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
