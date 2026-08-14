#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:?usage: run_kimi_h16_suite.sh MODE RUN-NAME}"
run_name="${2:?usage: run_kimi_h16_suite.sh MODE RUN-NAME}"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
suite="${KIMI_H16_SUITE:-$repo_dir/server/test/fixtures/kimi_k3_h16_registered.jsonl}"
output_root="${KIMI_H16_OUTPUT_ROOT:-/mnt/kimi-k3/results/kimi-h16-registered}"
output_dir="$output_root/$run_name"
reference_suite="${KIMI_H16_REFERENCE_SUITE:-}"
gpu="${KIMI_PANEL_GPU:-0}"
max_context="${KIMI_H16_MAX_CONTEXT:-256}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"
aux="${KIMI_H16_SLAB_AUX:-/mnt/kimi-k3/artifacts/kimi_layer01_slab_runtime.k3aux}"
sidecar="${KIMI_H16_SLAB_SIDECAR:-/mnt/kimi-k3/artifacts/kimi_layer01_progressive_slabs.k3slab}"

paired=0
provider_environment=()
case "$mode" in
    exact) ;;
    slabs96) provider=slabs; budget=96; paired=1 ;;
    slabs144) provider=slabs; budget=144; paired=1 ;;
    whole8) provider=whole; budget=8; paired=1 ;;
    whole12) provider=whole; budget=12; paired=1 ;;
    *) echo "unknown H16 suite mode: $mode" >&2; exit 2 ;;
esac
if [[ "$paired" == 1 ]]; then
    if [[ -z "$reference_suite" ]]; then
        echo "KIMI_H16_REFERENCE_SUITE is required for paired modes" >&2
        exit 2
    fi
    provider_environment=(
        "DFLASH_KIMI_LAYER1_PROVIDER=$provider"
        "DFLASH_KIMI_LAYER1_BUDGET=$budget"
        "DFLASH_KIMI_PROVIDER_LAYER=${KIMI_H16_LAYER:-1}"
        "DFLASH_KIMI_SLAB_AUX=$aux"
    )
    if [[ "$provider" == slabs ]]; then
        provider_environment+=("DFLASH_KIMI_SLAB_SIDECAR=$sidecar")
    fi
fi

for required in "$model_path" "$suite"; do
    if [[ ! -f "$required" ]]; then
        echo "missing H16 suite input: $required" >&2
        exit 1
    fi
done
if [[ "$paired" == 1 && ! -f "$aux" ]]; then
    echo "missing H16 calibration artifact: $aux" >&2
    exit 1
fi
if [[ "$mode" == slabs* && ! -f "$sidecar" ]]; then
    echo "missing H16 progressive sidecar: $sidecar" >&2
    exit 1
fi
if [[ -e "$output_dir/suite-manifest.json" ]]; then
    echo "refusing to overwrite completed suite: $output_dir" >&2
    exit 1
fi

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

cmake --build "$build_dir" -j4 --target run_kimi_k3_h16_suite
repository_commit="$(git -C "$repo_dir" rev-parse HEAD)"
repository_status=clean
if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    repository_status=dirty
fi
suite_sha256="$(sha256sum "$suite" | cut -d' ' -f1)"

python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output_dir/telemetry.json" \
    --samples-csv "$output_dir/telemetry.csv" \
    --stdout "$output_dir/stdout.log" \
    --stderr "$output_dir/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
    env \
        DFLASH_MOE_NVME_DIRECT=on \
        DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H16_DEVICE_CACHE_MB:-16384}" \
        DFLASH_KIMI_CPU_THREADS="${KIMI_H16_CPU_THREADS:-18}" \
        DFLASH_KIMI_MMAP_DROP_PAGES="${KIMI_H16_MMAP_DROP_PAGES:-1}" \
        KIMI_H16_REPOSITORY_COMMIT="$repository_commit" \
        KIMI_H16_REPOSITORY_STATUS="$repository_status" \
        KIMI_H16_SUITE_SHA256="$suite_sha256" \
        "${provider_environment[@]}" \
        "$build_dir/run_kimi_k3_h16_suite" \
            "$model_path" "$suite" "$output_dir" \
            "$gpu" "$max_context" "$paired" cpu

if [[ -n "$reference_suite" ]]; then
    analysis_arguments=(
        "$repo_dir/scripts/analyze_kimi_h16_suite.py"
        "$output_dir"
        --reference-suite "$reference_suite"
        --output-json "$output_dir/analysis.json"
    )
    if [[ "$paired" == 1 ]]; then
        analysis_arguments+=(--output-csv "$output_dir/rows.csv")
    fi
    python3 "${analysis_arguments[@]}"
fi

(
    cd "$output_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "[kimi-h16-suite] completed mode=$mode output=$output_dir"
