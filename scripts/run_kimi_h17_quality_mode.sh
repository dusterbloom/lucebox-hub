#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:?usage: run_kimi_h17_quality_mode.sh MODE}"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
sidecar_dir="${KIMI_H17_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
suite="${KIMI_H17_QUALITY_SUITE:-$repo_dir/server/test/fixtures/kimi_k3_h16_registered.jsonl}"
output_root="${KIMI_H17_QUALITY_ROOT:-/mnt/kimi-k3/results/kimi-h17-quality}"
output_dir="$output_root/$mode"
gpu="${KIMI_PANEL_GPU:-0}"
max_context="${KIMI_H17_QUALITY_MAX_CONTEXT:-256}"
n_gen="${KIMI_H17_QUALITY_N_GEN:-8}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

provider_environment=()
quality_label="EXPLORATORY PRACTICAL QUALITY"
provider_scope="native exact routed-expert evaluation"
case "$mode" in
    native) ;;
    slab192)
        provider_scope="all 192 natural-order slabs; known non-identical arithmetic recomposition"
        provider_environment=(
            DFLASH_KIMI_LAYER1_PROVIDER=all-slabs
            "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=$sidecar_dir"
            "DFLASH_KIMI_ALL_SLAB_METRICS_OUT=$output_dir/local.tsv"
        )
        ;;
    static96)
        quality_label="EXPLORATORY — confounded by all-192 arithmetic divergence"
        provider_scope="natural contiguous six-of-twelve prefix per active expert with zero omitted tail; not the unavailable calibrated all-layer selector"
        provider_environment=(
            DFLASH_KIMI_LAYER1_PROVIDER=all-slabs-static96
            "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=$sidecar_dir"
            "DFLASH_KIMI_ALL_SLAB_METRICS_OUT=$output_dir/local.tsv"
        )
        ;;
    oracle96)
        quality_label="EXPLORATORY — confounded by all-192 arithmetic divergence"
        provider_scope="greedy natural-prefix oracle at 96/192 with zero omitted tail; all slabs evaluated before selection; no speed claim"
        provider_environment=(
            DFLASH_KIMI_LAYER1_PROVIDER=all-slabs-oracle96
            "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=$sidecar_dir"
            "DFLASH_KIMI_ALL_SLAB_METRICS_OUT=$output_dir/local.tsv"
        )
        ;;
    oracle144)
        quality_label="EXPLORATORY — confounded by all-192 arithmetic divergence"
        provider_scope="greedy natural-prefix oracle at 144/192 with zero omitted tail; all slabs evaluated before selection; no speed claim"
        provider_environment=(
            DFLASH_KIMI_LAYER1_PROVIDER=all-slabs-oracle144
            "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=$sidecar_dir"
            "DFLASH_KIMI_ALL_SLAB_METRICS_OUT=$output_dir/local.tsv"
        )
        ;;
    *) echo "unknown H17 quality mode: $mode" >&2; exit 2 ;;
esac

for required in "$model_path" "$suite"; do
    if [[ ! -f "$required" ]]; then
        echo "missing H17 quality input: $required" >&2
        exit 1
    fi
done
if [[ "$mode" != native && ! -f "$sidecar_dir/all_layers_manifest.json" ]]; then
    echo "missing all-layer slab sidecars: $sidecar_dir" >&2
    exit 1
fi
if [[ -e "$output_dir/suite-manifest.json" ]]; then
    echo "refusing to overwrite completed H17 quality mode: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir"

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
commit="$(git -C "$repo_dir" rev-parse HEAD)"
suite_sha256="$(sha256sum "$suite" | awk '{print $1}')"
runner_sha256="$(sha256sum "$build_dir/run_kimi_k3_h16_suite" | awk '{print $1}')"
status=clean
if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    status=dirty
fi

python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output_dir/telemetry.json" \
    --samples-csv "$output_dir/telemetry.csv" \
    --stdout "$output_dir/stdout.log" \
    --stderr "$output_dir/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
    env \
        DFLASH_MOE_NVME_DIRECT=on \
        DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H17_DEVICE_CACHE_MB:-16384}" \
        DFLASH_KIMI_CPU_THREADS="${KIMI_H17_CPU_THREADS:-18}" \
        DFLASH_KIMI_MMAP_DROP_PAGES=1 \
        KIMI_H16_REPOSITORY_COMMIT="$commit" \
        KIMI_H16_REPOSITORY_STATUS="$status" \
        KIMI_H16_SUITE_SHA256="$suite_sha256" \
        KIMI_H17_RUNNER_SHA256="$runner_sha256" \
        KIMI_H17_QUALITY_LABEL="$quality_label" \
        KIMI_H17_PROVIDER_SCOPE="$provider_scope" \
        "${provider_environment[@]}" \
        "$build_dir/run_kimi_k3_h16_suite" \
            "$model_path" "$suite" "$output_dir" \
            "$gpu" "$max_context" 0 cpu "$n_gen"

(
    cd "$output_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "[kimi-h17-quality] completed mode=$mode output=$output_dir"
