#!/usr/bin/env bash
set -euo pipefail

repo="unsloth/Kimi-K3-GGUF"
revision="a0836360ce58dfec088d966a97f2ddc8a606279b"
model_root="${KIMI_PANEL_MODEL_ROOT:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF}"
hf_home="${KIMI_PANEL_HF_HOME:-/mnt/kimi-k3/hf-cache}"
complete_marker="${KIMI_PANEL_DOWNLOAD_MARKER:-$model_root/.ud-iq1s-$revision.complete}"
retry_seconds="${KIMI_PANEL_DOWNLOAD_RETRY_SECONDS:-60}"

timestamp() {
    date --iso-8601=seconds
}

if [[ ! "$retry_seconds" =~ ^[0-9]+$ ]] || (( retry_seconds < 5 || retry_seconds > 3600 )); then
    echo "KIMI_PANEL_DOWNLOAD_RETRY_SECONDS must be from 5 through 3600" >&2
    exit 2
fi

mkdir -p "$model_root" "$hf_home"

if [[ -f "$complete_marker" ]]; then
    echo "$(timestamp) pinned Kimi checkpoint is already marked complete"
    exit 0
fi

attempt=0
while :; do
    attempt=$((attempt + 1))
    echo "$(timestamp) starting resumable Kimi checkpoint transfer attempt $attempt"

    set +e
    env -u HF_HUB_DISABLE_XET \
        HF_HOME="$hf_home" \
        HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}" \
        hf download "$repo" \
            --revision "$revision" \
            --include 'UD-IQ1_S/*' \
            --local-dir "$model_root" \
            --max-workers 2
    status=$?
    set -e

    if (( status == 0 )); then
        printf '%s %s\n' "$revision" "$(timestamp)" > "$complete_marker"
        echo "$(timestamp) transfer completed; wrote $complete_marker"
        exit 0
    fi

    echo "$(timestamp) transfer attempt $attempt exited with status $status; retrying in ${retry_seconds}s" >&2
    sleep "$retry_seconds"
done
