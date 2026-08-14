#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
corpus="${KIMI_H18_CAPTURE_CORPUS:-/mnt/kimi-k3/captures/kimi_panel_smoke.jsonl}"
total_tokens="${KIMI_H18_CAPTURE_TOKENS:-8}"
capture_root="${KIMI_H18_CAPTURE_ROOT:-/mnt/kimi-k3/captures/kimi-h18-all-layer-$total_tokens}"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
gpu="${KIMI_PANEL_GPU:-0}"
max_context="${KIMI_H18_CAPTURE_MAX_CONTEXT:-512}"
chunk_tokens="${KIMI_H18_CAPTURE_CHUNK_TOKENS:-8}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

for required in "$model_path" "$corpus"; do
    if [[ ! -f "$required" ]]; then
        echo "missing H18 capture input: $required" >&2
        exit 1
    fi
done
if [[ -e "$capture_root" ]]; then
    if [[ ! -d "$capture_root" ]]; then
        echo "H18 capture root exists and is not a directory: $capture_root" >&2
        exit 1
    fi
    first_existing="$(find "$capture_root" -mindepth 1 -maxdepth 1 -print -quit)"
    if [[ -n "$first_existing" ]]; then
        echo "refusing to reuse nonempty H18 capture root: $capture_root" >&2
        echo "first existing entry: $first_existing" >&2
        exit 1
    fi
fi

# 92 copies of the existing v1 token record. Payload per token/layer is:
# BF16 z (3584*2) + int32 IDs (16*4) + F32 weights (16*4).
payload_bytes=$((total_tokens * 92 * (3584 * 2 + 16 * 4 + 16 * 4)))
free_kib="$(df --output=avail /mnt/kimi-k3 | tail -n 1 | tr -d ' ')"
safety_kib=$((64 * 1024 * 1024))
if (( free_kib * 1024 < payload_bytes + safety_kib * 1024 )); then
    echo "insufficient NVMe space for H18 capture plus 64 GiB safety margin" >&2
    exit 1
fi

mkdir -p "$capture_root"
exec 9>"$gpu_lock"
if ! flock -n 9; then
    echo "another cooperating job holds the graphics-card lease" >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | rg -q '[0-9]'; then
    echo "the graphics card is already in use" >&2
    exit 1
fi

cmake --build "$build_dir" -j4 --target capture_kimi_k3_panel
python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$capture_root/telemetry.json" \
    --samples-csv "$capture_root/telemetry.csv" \
    --stdout "$capture_root/stdout.log" \
    --stderr "$capture_root/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
    env \
        DFLASH_KIMI_CPU_THREADS="${KIMI_H18_CPU_THREADS:-18}" \
        DFLASH_MOE_NVME_DIRECT=on \
        DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H18_DEVICE_CACHE_MB:-16384}" \
        DFLASH_KIMI_MMAP_DROP_PAGES=1 \
        "$build_dir/capture_kimi_k3_panel" \
            "$model_path" "$corpus" "$capture_root" \
            "$gpu" all "$total_tokens" "$max_context" "$chunk_tokens" cpu

(
    cd "$capture_root"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "[kimi-h18-capture] completed tokens=$total_tokens root=$capture_root"
