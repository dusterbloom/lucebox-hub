#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
sidecar_dir="${KIMI_H17_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
output_dir="${KIMI_H17_OUTPUT_DIR:-/mnt/kimi-k3/results/kimi-h17-all-slabs-control}"
reference="${KIMI_H17_EXACT_REFERENCE:-/mnt/kimi-k3/results/kimi-exact-baseline-local-cpu/exact-run-1.logits.f32}"
gpu="${KIMI_PANEL_GPU:-0}"
prompt="${KIMI_H17_PROMPT:-According to all known laws}"
teacher_tokens="${KIMI_H17_TEACHER_TOKENS:-198,1587,57195,422}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

for required in "$model_path" "$reference" "$sidecar_dir/all_layers_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "missing H17 control input: $required" >&2
        exit 1
    fi
done
for layer in $(seq 1 92); do
    sidecar="$sidecar_dir/kimi_layer$(printf '%02d' "$layer")_natural_slabs.k3slab"
    if [[ ! -f "$sidecar" ]]; then
        echo "missing H17 layer sidecar: $sidecar" >&2
        exit 1
    fi
done
if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite H17 output: $output_dir" >&2
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
        DFLASH_KIMI_LAYER1_PROVIDER=all-slabs \
        DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecar_dir" \
        DFLASH_KIMI_ALL_SLAB_METRICS_OUT="$output_dir/layer-numerics.tsv" \
        DFLASH_KIMI_TEACHER_FORCED_TOKENS="$teacher_tokens" \
        DFLASH_KIMI_LOGITS_TRACE_OUT="$output_dir/aligned.logits.f32" \
        KIMI_H17_REPOSITORY_COMMIT="$commit" \
        KIMI_H17_REPOSITORY_STATUS="$status" \
        "$build_dir/smoke_kimi_k3_forward" \
            "$model_path" "$gpu" 4 "$prompt" 1 -1 "" "$gpu" cpu

python3 "$repo_dir/scripts/compare_kimi_logits.py" \
    "$reference" "$output_dir/aligned.logits.f32" \
    --output "$output_dir/comparison.json"

python3 - "$output_dir/comparison.json" "$output_dir/verdict.json" <<'PY'
import json
import sys
from pathlib import Path

comparison = json.loads(Path(sys.argv[1]).read_text())
divergence = comparison["teacher_to_candidate_divergence"]
agreement = comparison["top_choice_agreement"]
thresholds = {
    "mean_terminal_kl_max": 1.0e-6,
    "maximum_terminal_kl_max": 1.0e-5,
    "top_choice_agreement_min": 1.0,
}
passed = (
    divergence["mean"] <= thresholds["mean_terminal_kl_max"]
    and divergence["maximum"] <= thresholds["maximum_terminal_kl_max"]
    and agreement["rate"] >= thresholds["top_choice_agreement_min"]
)
verdict = {
    "schema": "kimi-k3-h17-all-slabs-control-verdict-v1",
    "status": "PASS" if passed else "STOP",
    "thresholds_locked_before_execution": thresholds,
    "mean_terminal_kl": divergence["mean"],
    "maximum_terminal_kl": divergence["maximum"],
    "top_choice_agreement": agreement,
    "next": (
        "build all-layer calibration and partial-budget providers"
        if passed
        else "fix slab numerical plumbing before any 96/144 approximation"
    ),
}
Path(sys.argv[2]).write_text(json.dumps(verdict, indent=2) + "\n")
print(json.dumps(verdict, indent=2))
PY

(
    cd "$output_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
