#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
sidecar_dir="${KIMI_H17_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
output_dir="${KIMI_H17_DIVERGENCE_OUTPUT_DIR:-/mnt/kimi-k3/results/kimi-h17-divergence-trace}"
locked_reference="${KIMI_H17_EXACT_REFERENCE:-/mnt/kimi-k3/results/kimi-exact-baseline-local-cpu/exact-run-1.logits.f32}"
gpu="${KIMI_PANEL_GPU:-0}"
prompt="${KIMI_H17_PROMPT:-According to all known laws}"
teacher_tokens="${KIMI_H17_TEACHER_TOKENS:-198,1587,57195,422}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

for required in \
    "$model_path" \
    "$locked_reference" \
    "$sidecar_dir/all_layers_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "missing H17 divergence input: $required" >&2
        exit 1
    fi
done
if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite H17 divergence output: $output_dir" >&2
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

cmake --build "$build_dir" -j4 --target smoke_kimi_k3_forward
commit="$(git -C "$repo_dir" rev-parse HEAD)"
status=clean
if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    status=dirty
fi

run_mode() {
    local mode="$1"
    shift
    python3 "$repo_dir/scripts/run_with_telemetry.py" \
        --output-json "$output_dir/$mode.telemetry.json" \
        --samples-csv "$output_dir/$mode.telemetry.csv" \
        --stdout "$output_dir/$mode.stdout.log" \
        --stderr "$output_dir/$mode.stderr.log" \
        --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
        env \
            DFLASH_MOE_NVME_DIRECT=on \
            DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H17_DEVICE_CACHE_MB:-16384}" \
            DFLASH_KIMI_CPU_THREADS="${KIMI_H17_CPU_THREADS:-18}" \
            DFLASH_KIMI_MMAP_DROP_PAGES=1 \
            DFLASH_KIMI_TEACHER_FORCED_TOKENS="$teacher_tokens" \
            DFLASH_KIMI_LOGITS_TRACE_OUT="$output_dir/$mode.logits.f32" \
            DFLASH_KIMI_DIVERGENCE_TRACE_OUT="$output_dir/$mode.trace.bin" \
            KIMI_H17_REPOSITORY_COMMIT="$commit" \
            KIMI_H17_REPOSITORY_STATUS="$status" \
            "$@" \
            "$build_dir/smoke_kimi_k3_forward" \
                "$model_path" "$gpu" 4 "$prompt" 1 -1 "" "$gpu" cpu
}

run_mode native \
    DFLASH_KIMI_LAYER1_PROVIDER=exact

python3 "$repo_dir/scripts/compare_kimi_logits.py" \
    "$locked_reference" "$output_dir/native.logits.f32" \
    --output "$output_dir/native-vs-locked.json"

run_mode slab192 \
    DFLASH_KIMI_LAYER1_PROVIDER=all-slabs \
    DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecar_dir" \
    DFLASH_KIMI_ALL_SLAB_METRICS_OUT="$output_dir/slab192-local.tsv"

python3 "$repo_dir/scripts/compare_kimi_logits.py" \
    "$output_dir/native.logits.f32" "$output_dir/slab192.logits.f32" \
    --output "$output_dir/native-vs-slab192.json"

python3 "$repo_dir/scripts/analyze_kimi_h17_divergence.py" \
    --exact "$output_dir/native.trace.bin" \
    --slab192 "$output_dir/slab192.trace.bin" \
    --terminal-comparison "$output_dir/native-vs-slab192.json" \
    --output-json "$output_dir/divergence.json" \
    --output-csv "$output_dir/divergence.csv" \
    >"$output_dir/divergence.stdout.json"

(
    cd "$output_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)

python3 - "$output_dir/divergence.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({
    "first_numerical_divergence": result["first_numerical_divergence"],
    "first_router_order_divergence": result["first_router_order_divergence"],
    "first_router_top16_membership_divergence": (
        result["first_router_top16_membership_divergence"]
    ),
    "terminal": result["terminal_comparison"]["teacher_to_candidate_divergence"],
}, indent=2))
PY
