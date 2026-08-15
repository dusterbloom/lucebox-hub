#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
baseline_dir="${KIMI_H16_BASELINE_DIR:-/mnt/kimi-k3/results/kimi-exact-baseline-local-cpu}"
output_dir="${KIMI_H16_OUTPUT_DIR:-/mnt/kimi-k3/results/kimi-h16-frozen-local-cpu}"
aux="${KIMI_H16_SLAB_AUX:-/mnt/kimi-k3/artifacts/kimi_layer01_slab_runtime.k3aux}"
sidecar="${KIMI_H16_SLAB_SIDECAR:-/mnt/kimi-k3/artifacts/kimi_layer01_progressive_slabs.k3slab}"
gpu="${KIMI_PANEL_GPU:-0}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"
prompt="${KIMI_EXACT_PROMPT:-According to all known laws}"
generated_tokens="${KIMI_EXACT_GENERATED_TOKENS:-4}"
max_context="${KIMI_EXACT_MAX_CONTEXT:-512}"
modes="${KIMI_H16_MODES:-slabs96 slabs144 whole8 whole12}"

teacher_logits="$baseline_dir/exact-run-1.logits.f32"
teacher_stdout="$baseline_dir/exact-run-1.stdout.log"
for required in "$model_path" "$teacher_logits" "$teacher_stdout" "$aux" "$sidecar"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required H16 input: $required" >&2
        exit 1
    fi
done

teacher_tokens="$(
    sed -n 's/^\[kimi-k3-smoke\] output_ids: *//p' "$teacher_stdout" \
        | tr ' ' ','
)"
if [[ -z "$teacher_tokens" ]]; then
    echo "Could not read frozen teacher tokens from $teacher_stdout" >&2
    exit 1
fi

exec 9>"$gpu_lock"
if ! flock -n 9; then
    echo "Another cooperating job holds the graphics-card lease: $gpu_lock" >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | rg -q '[0-9]'; then
    echo "The graphics card is already in use; refusing a contended H16 run." >&2
    exit 1
fi

mkdir -p "$output_dir"
cmake --build "$build_dir" --target smoke_kimi_k3_forward -j4

for name in $modes; do
    case "$name" in
        slabs96)  provider=slabs; budget=96 ;;
        slabs144) provider=slabs; budget=144 ;;
        # The full-budget split-slab control is deliberately explicit.  It
        # measures the provider's arithmetic footprint at an isolated layer
        # before interpreting any partial-budget terminal KL.
        slabs192) provider=slabs; budget=192 ;;
        whole8)   provider=whole; budget=8 ;;
        whole12)  provider=whole; budget=12 ;;
        *) echo "Unknown KIMI_H16_MODES entry: $name" >&2; exit 2 ;;
    esac
    prefix="$output_dir/$name"
    provider_env=()
    candidate_logits="$prefix.logits.f32"
    if [[ -n "${KIMI_H16_ACTIVE_POSITION:-}" ]]; then
        provider_env+=(
            "DFLASH_KIMI_LAYER1_ACTIVE_POSITION=$KIMI_H16_ACTIVE_POSITION"
        )
    fi
    if [[ "${KIMI_H16_PAIRED:-0}" == 1 ]]; then
        candidate_logits="$prefix.candidate.logits.f32"
        provider_env+=(
            "DFLASH_KIMI_H16_CANDIDATE_LOGITS_OUT=$candidate_logits"
        )
    fi
    if [[ -f "$prefix.result.json" && "${KIMI_H16_FORCE:-0}" != 1 ]]; then
        echo "[kimi-h16] keeping completed $name"
        continue
    fi
    python3 "$repo_dir/scripts/run_with_telemetry.py" \
        --output-json "$prefix.telemetry.json" \
        --samples-csv "$prefix.telemetry.csv" \
        --stdout "$prefix.stdout.log" --stderr "$prefix.stderr.log" \
        --mount-path /mnt/kimi-k3 --gpu "$gpu" -- \
        env \
            DFLASH_KIMI_SMOKE_MAX_CTX="$max_context" \
            DFLASH_KIMI_LOGITS_TRACE_OUT="$prefix.logits.f32" \
            DFLASH_KIMI_TEACHER_FORCED_TOKENS="$teacher_tokens" \
            DFLASH_KIMI_LAYER1_PROVIDER="$provider" \
            DFLASH_KIMI_LAYER1_BUDGET="$budget" \
            DFLASH_KIMI_PROVIDER_LAYER="${KIMI_H16_LAYER:-1}" \
            DFLASH_KIMI_SLAB_AUX="$aux" \
            DFLASH_KIMI_SLAB_SIDECAR="$sidecar" \
            DFLASH_KIMI_LAYER1_TRACE_OUT="$prefix.intervention.f32" \
            DFLASH_MOE_NVME_DIRECT=on \
            DFLASH_MOE_NVME_DEVICE_CACHE_MB=16384 \
            DFLASH_KIMI_CPU_THREADS="${KIMI_H16_CPU_THREADS:-18}" \
            DFLASH_KIMI_MMAP_DROP_PAGES="${KIMI_H16_MMAP_DROP_PAGES:-1}" \
            "${provider_env[@]}" \
            "$build_dir/smoke_kimi_k3_forward" \
                "$model_path" "$gpu" "$generated_tokens" "$prompt" \
                1 -1 "" "$gpu" cpu
    python3 "$repo_dir/scripts/compare_kimi_logits.py" \
        "$teacher_logits" "$candidate_logits" \
        --output "$prefix.logit-comparison.json"
    if [[ "${KIMI_H16_PAIRED:-0}" == 1 ]]; then
        python3 "$repo_dir/scripts/compare_kimi_logits.py" \
            "$teacher_logits" "$prefix.logits.f32" \
            --output "$prefix.paired-exact-comparison.json"
    fi
    python3 "$repo_dir/scripts/analyze_kimi_h16_intervention.py" \
        "$teacher_logits" "$candidate_logits" \
        "$prefix.intervention.f32" "$teacher_stdout" \
        --output-json "$prefix.result.json" \
        --output-csv "$prefix.rows.csv"
done

(
    cd "$output_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "[kimi-h16] frozen interventions complete: $output_dir"
