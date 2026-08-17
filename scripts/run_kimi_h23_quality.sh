#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
[[ "$mode" == native || "$mode" == candidate ]] || {
    echo "usage: $0 native|candidate" >&2
    exit 2
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_H23_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_H23_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
fixture="${KIMI_H23_FIXTURE:-$repo_dir/server/test/fixtures/kimi_k3_h23_native_success.jsonl}"
native_root="${KIMI_H23_NATIVE_ROOT:-/mnt/kimi-k3/results/kimi-h23-native-success-20260816}"
candidate_root="${KIMI_H23_CANDIDATE_ROOT:-/mnt/kimi-k3/results/kimi-h23-safe4gib-v2-20260816}"
aux="${KIMI_H23_AUX_DIR:-/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime}"
sidecars="${KIMI_H23_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
table="${KIMI_H23_BUDGET_TABLE:-$repo_dir/results/h23_policies/h23_safe_4_0gib.txt}"
analysis_output="${KIMI_H23_ANALYSIS_OUTPUT:-$repo_dir/results/h23_safe4gib_quality.json}"
native_analysis_output="${KIMI_H23_NATIVE_ANALYSIS_OUTPUT:-$repo_dir/results/h23_native_success.json}"
prompt_id="${KIMI_H23_PROMPT_ID:-h23-safe4gib}"
gpu="${KIMI_H23_GPU:-0}"
n_gen="${KIMI_H23_N_GEN:-8}"
draft="${KIMI_H23_DRAFT:-}"
draft_gpu="${KIMI_H23_DRAFT_GPU:-$gpu}"
output="$native_root"
[[ "$mode" == candidate ]] && output="$candidate_root"

required=("$model" "$fixture")
if [[ -n "$draft" ]]; then
    required+=("$draft")
fi
if [[ "$mode" == candidate ]]; then
    required+=("$native_root/suite/suite-manifest.json" "$aux/all_layers_calibrated96_manifest.json"
               "$sidecars/all_layers_manifest.json" "$table")
fi
for path in "${required[@]}"; do
    [[ -f "$path" ]] || { echo "missing H23 input: $path" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { echo "refusing existing H23 output: $output" >&2; exit 1; }

CCACHE_DIR=/tmp/kimi-p20-ccache cmake --build "$build_dir" -j4 \
    --target test_kimi_k3_progressive_provider run_kimi_k3_h16_suite
"$build_dir/test_kimi_k3_progressive_provider"
mkdir -p "$output"

common_env=(
    DFLASH_MOE_NVME_DIRECT=on
    DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H23_DEVICE_CACHE_MB:-2048}"
    DFLASH_KIMI_CPU_THREADS="${KIMI_H23_CPU_THREADS:-12}"
    DFLASH_KIMI_MMAP_DROP_PAGES=0
    DFLASH_KIMI_MOE_CORE_OFFLOAD=latent,shared
    DFLASH_KIMI_H16_CHAT_TEMPLATE=1
    KIMI_H16_SUITE_SHA256="$(sha256sum "$fixture" | awk '{print $1}')"
    KIMI_H16_REPOSITORY_COMMIT="$(git -C "$repo_dir" rev-parse HEAD)"
    KIMI_H16_REPOSITORY_STATUS="$(git -C "$repo_dir" status --short | sha256sum | awk '{print $1}')"
)
if [[ "$mode" == native ]]; then
    mode_env=(DFLASH_KIMI_LAYER1_PROVIDER=exact)
else
    mode_env=(
        DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96
        DFLASH_KIMI_CALIBRATED96_AUX_DIR="$aux"
        DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecars"
        DFLASH_KIMI_CALIBRATED96_METRICS_OUT="$output/traffic.tsv"
        DFLASH_KIMI_P20_IO_TRACE="$output/io_trace.tsv"
        DFLASH_KIMI_P20_PHYSICAL_LAYOUT=scratch
        DFLASH_KIMI_P20_IO_BACKEND=direct-pread
        DFLASH_KIMI_P23_PERSISTENT_SCRATCH=1
        DFLASH_KIMI_P25_COMPACT_UPLOAD=1
        DFLASH_KIMI_P26_PINNED_COMPACT=1
        DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT=1
        DFLASH_KIMI_H22_LAYER_BUDGETS="$table"
        DFLASH_KIMI_P20_PROMPT_ID="$prompt_id"
    )
fi

python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output/telemetry.json" \
    --samples-csv "$output/telemetry.csv" \
    --stdout "$output/stdout.log" --stderr "$output/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 2 -- \
    env "${common_env[@]}" "${mode_env[@]}" \
      "$build_dir/run_kimi_k3_h16_suite" \
        "$model" "$fixture" "$output/suite" "$gpu" 256 0 cpu "$n_gen" \
        "$draft" "$draft_gpu"

if [[ "$mode" == native ]]; then
    python3 "$repo_dir/scripts/analyze_kimi_h23_quality.py" \
        --native "$output/suite" --output "$native_analysis_output" \
        | tee "$output/analysis.stdout.log"
    python3 - "$native_analysis_output" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if not result["native"]["all_tasks_succeeded"]:
    raise SystemExit("H23 native-success fixture failed; do not run candidate")
PY
else
    python3 "$repo_dir/scripts/analyze_kimi_h23_quality.py" \
        --native "$native_root/suite" --candidate "$output/suite" \
        --traffic "$output/traffic.tsv" \
        --traffic-process "$output/traffic.tsv.process.tsv" \
        --telemetry "$output/telemetry.json" \
        --budget-table "$table" \
        --calibration-manifest "$aux/all_layers_calibrated96_manifest.json" \
        --sidecar-manifest "$sidecars/all_layers_manifest.json" \
        --warning "This deterministic product-mode gate disables thinking and is not directly comparable to official K3 max-reasoning benchmark scores." \
        --output "$analysis_output" \
        | tee "$output/analysis.stdout.log"
fi

(
    cd "$output"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "H23 $mode complete: $output"
