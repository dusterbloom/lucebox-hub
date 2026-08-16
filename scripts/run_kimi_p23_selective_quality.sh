#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_P23_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_P23_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
aux="${KIMI_P23_AUX_DIR:-/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime}"
sidecars="${KIMI_P23_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
native="${KIMI_P23_NATIVE_DIR:-/mnt/kimi-k3/results/kimi-h22-template-sanity-native-20260816}"
table="${KIMI_P23_BUDGET_TABLE:-/mnt/kimi-k3/artifacts/kimi-h22-layer-budgets-20260816/h22_behavioral_avg96.txt}"
output="${KIMI_P23_QUALITY_OUTPUT:-/mnt/kimi-k3/results/kimi-p23-latent-shared-capital-chat-quality-20260816}"
fixture="${KIMI_P23_FIXTURE:-$repo_dir/server/test/fixtures/kimi_k3_p23_capital_quality.jsonl}"
offload="${KIMI_P23_MOE_CORE_OFFLOAD:-latent,shared}"
chat_template="${KIMI_P23_CHAT_TEMPLATE:-1}"
gpu="${KIMI_P23_GPU:-0}"
lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

for required in "$model" "$fixture" "$table" \
        "$native/suite-manifest.json" \
        "$aux/all_layers_calibrated96_manifest.json" \
        "$sidecars/all_layers_manifest.json"; do
    [[ -f "$required" ]] || { echo "missing P23 quality input: $required" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { echo "refusing existing P23 quality output: $output" >&2; exit 1; }
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
    --mount-path /mnt/kimi-k3 --gpu "$gpu" \
    --interval "${KIMI_P23_TELEMETRY_INTERVAL:-5}" -- \
    env \
      DFLASH_MOE_NVME_DIRECT=on \
      DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_P23_DEVICE_CACHE_MB:-2048}" \
      DFLASH_KIMI_CPU_THREADS="${KIMI_P23_CPU_THREADS:-18}" \
      DFLASH_KIMI_MMAP_DROP_PAGES=0 \
      DFLASH_KIMI_MOE_CORE_OFFLOAD="$offload" \
      DFLASH_KIMI_H16_CHAT_TEMPLATE="$chat_template" \
      DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96 \
      DFLASH_KIMI_CALIBRATED96_AUX_DIR="$aux" \
      DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecars" \
      DFLASH_KIMI_CALIBRATED96_METRICS_OUT="$output/traffic.tsv" \
      DFLASH_KIMI_P20_PHYSICAL_LAYOUT=scratch \
      DFLASH_KIMI_P20_IO_BACKEND=direct-pread \
      DFLASH_KIMI_H22_LAYER_BUDGETS="$table" \
      DFLASH_KIMI_P20_PROMPT_ID="p23-$offload-quality" \
      "$build_dir/run_kimi_k3_h16_suite" \
        "$model" "$fixture" "$output/suite" "$gpu" 256 0 cpu 8

python3 "$repo_dir/scripts/analyze_kimi_h22_adaptive_quality.py" \
    --native "$native" --candidate "$output/suite" \
    --budget-table "$table" --traffic "$output/traffic.tsv" \
    --telemetry "$output/telemetry.json" --sidecars "$sidecars" \
    --output "$repo_dir/results/k3_p23_selective_chat_quality.json" \
    | tee "$output/analysis.stdout.log"
(
    cd "$output"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "P23 selective quality complete: $output"
