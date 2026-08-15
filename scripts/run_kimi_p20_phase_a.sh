#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_P20_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_P20_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
aux="${KIMI_P20_AUX_DIR:-/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime}"
sidecars="${KIMI_P20_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
output="${KIMI_P20_OUTPUT_DIR:-/mnt/kimi-k3/results/kimi-p20-phase-a}"
gpu="${KIMI_P20_GPU:-0}"
prompt="${KIMI_P20_PROMPT:-Hi}"
generated="${KIMI_P20_GENERATED_TOKENS:-1}"
lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

for required in "$model" "$aux/all_layers_calibrated96_manifest.json" \
        "$sidecars/all_layers_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "missing P20 input: $required" >&2
        exit 1
    fi
done
if [[ -e "$output" ]]; then
    echo "refusing to overwrite P20 output: $output" >&2
    exit 1
fi
mkdir -p "$output"

exec 9>"$lock"
flock -n 9 || { echo "graphics-card lease is held" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | rg -q '[0-9]'; then
    echo "graphics card is already in use" >&2
    exit 1
fi

CCACHE_DIR=/tmp/kimi-p20-ccache cmake --build "$build_dir" -j4 \
    --target smoke_kimi_k3_forward
python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output/telemetry.json" \
    --samples-csv "$output/telemetry.csv" \
    --stdout "$output/stdout.log" --stderr "$output/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
    env \
      DFLASH_MOE_NVME_DIRECT=on \
      DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_P20_DEVICE_CACHE_MB:-16384}" \
      DFLASH_KIMI_CPU_THREADS="${KIMI_P20_CPU_THREADS:-18}" \
      DFLASH_KIMI_MMAP_DROP_PAGES=1 \
      DFLASH_KIMI_SMOKE_MAX_CTX=128 \
      DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96 \
      DFLASH_KIMI_CALIBRATED96_AUX_DIR="$aux" \
      DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecars" \
      DFLASH_KIMI_CALIBRATED96_METRICS_OUT="$output/traffic.tsv" \
      DFLASH_KIMI_P20_PHYSICAL_LAYOUT=reference \
      DFLASH_KIMI_P20_SLAB_BUDGET=96 \
      DFLASH_KIMI_P20_IO_TRACE="$output/io_trace.tsv" \
      DFLASH_KIMI_P20_PROMPT_ID=phase-a-smoke \
      "$build_dir/smoke_kimi_k3_forward" \
        "$model" "$gpu" "$generated" "$prompt" 1 -1 "" "$gpu" cpu

python3 "$repo_dir/scripts/analyze_kimi_p20_io.py" \
    --trace "$output/io_trace.tsv" --traffic "$output/traffic.tsv" \
    --process "$output/traffic.tsv.process.tsv" \
    --stderr "$output/stderr.log" --telemetry "$output/telemetry.json" \
    --output "$repo_dir/results/k3_p20_io_audit.json"

python3 "$repo_dir/scripts/k3_progressive_io_replay.py" \
    "$output/io_trace.tsv" --mode current --queue-depth 1 --cold \
    --output "$output/replay-current-cold.json"
python3 "$repo_dir/scripts/k3_progressive_io_replay.py" \
    "$output/io_trace.tsv" --mode batched-pread --queue-depth 8 --cold \
    --output "$output/replay-batched-cold.json"
cp "$output/replay-batched-cold.json" \
    "$repo_dir/results/k3_p20_io_replay.json"

(
    cd "$output"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "P20 Phase A complete: $output"
