#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_H22_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_H22_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
aux="${KIMI_H22_AUX_DIR:-/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime}"
sidecars="${KIMI_H22_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
native="${KIMI_H22_NATIVE_DIR:-/mnt/kimi-k3/results/kimi-h17-quality/native}"
table="${KIMI_H22_BUDGET_TABLE:-/mnt/kimi-k3/artifacts/kimi-h22-layer-budgets-20260816/h22_behavioral_avg96.txt}"
output="${KIMI_H22_QUALITY_OUTPUT:-/mnt/kimi-k3/results/kimi-h22-adaptive96-registered-20260816}"
fixture="$repo_dir/server/test/fixtures/kimi_k3_h16_registered.jsonl"
gpu="${KIMI_H22_GPU:-0}"
lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

for required in "$model" "$fixture" "$table" \
        "$native/suite-manifest.json" \
        "$aux/all_layers_calibrated96_manifest.json" \
        "$sidecars/all_layers_manifest.json"; do
    [[ -f "$required" ]] || { echo "missing H22 registered-quality input: $required" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { echo "refusing existing H22 registered-quality output: $output" >&2; exit 1; }
mkdir -p "$output"

exec 9>"$lock"
flock -n 9 || { echo "graphics-card lease is held" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | rg -q '[0-9]'; then
    echo "graphics card is already in use" >&2
    exit 1
fi

CCACHE_DIR=/tmp/kimi-p20-ccache cmake --build "$build_dir" -j4 \
    --target run_kimi_k3_h16_suite
python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output/telemetry.json" \
    --samples-csv "$output/telemetry.csv" \
    --stdout "$output/stdout.log" --stderr "$output/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
    env \
      DFLASH_MOE_NVME_DIRECT=on \
      DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H22_DEVICE_CACHE_MB:-16384}" \
      DFLASH_KIMI_CPU_THREADS="${KIMI_H22_CPU_THREADS:-18}" \
      DFLASH_KIMI_MMAP_DROP_PAGES=1 \
      DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96 \
      DFLASH_KIMI_CALIBRATED96_AUX_DIR="$aux" \
      DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecars" \
      DFLASH_KIMI_CALIBRATED96_METRICS_OUT="$output/traffic.tsv" \
      DFLASH_KIMI_P20_PHYSICAL_LAYOUT=scratch \
      DFLASH_KIMI_P20_IO_BACKEND=direct-pread \
      DFLASH_KIMI_H22_LAYER_BUDGETS="$table" \
      DFLASH_KIMI_P20_PROMPT_ID=h22-adaptive-registered \
      "$build_dir/run_kimi_k3_h16_suite" \
        "$model" "$fixture" "$output/suite" "$gpu" 256 0 cpu 8

python3 "$repo_dir/scripts/analyze_kimi_h22_adaptive_quality.py" \
    --native "$native" --candidate "$output/suite" \
    --budget-table "$table" --traffic "$output/traffic.tsv" \
    --telemetry "$output/telemetry.json" --sidecars "$sidecars" \
    --output "$repo_dir/results/kimi_h22_adaptive_registered_quality.json" \
    | tee "$output/analysis.stdout.log"
(
    cd "$output"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "H22 adaptive registered quality complete: $output"
