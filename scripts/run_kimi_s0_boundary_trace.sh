#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_S0_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_S0_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
output="${1:?usage: run_kimi_s0_boundary_trace.sh OUTPUT_DIR}"
gpu="${KIMI_S0_GPU:-0}"

[[ -f "$model" ]] || { echo "missing K3 model: $model" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite: $output" >&2; exit 1; }
mkdir -p "$output"

env CCACHE_DISABLE=1 cmake --build "$build_dir" -j4 \
    --target run_kimi_k3_s0_oracle

# Exit 3 is the expected result until the width-four parity defect is fixed.
# The trace is still complete and is the input to the boundary analyzer.
set +e
python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output/telemetry.json" \
    --samples-csv "$output/telemetry.csv" \
    --stdout "$output/stdout.log" --stderr "$output/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 1 -- \
    env \
      DFLASH_MOE_NVME_DIRECT=on \
      DFLASH_MOE_NVME_DEVICE_CACHE_MB=2048 \
      DFLASH_KIMI_CPU_THREADS="${KIMI_S0_CPU_THREADS:-18}" \
      DFLASH_KIMI_MMAP_DROP_PAGES=0 \
      DFLASH_KIMI_MOE_CORE_OFFLOAD=latent,shared \
      DFLASH_KIMI_STAGE_PROFILE=0 \
      DFLASH_KIMI_DIVERGENCE_TRACE_OUT="$output/divergence.trace" \
      DFLASH_KIMI_P20_PROMPT_ID=s0-native-m4-boundary \
      "$build_dir/run_kimi_k3_s0_oracle" \
        "$model" 18699 11,374,4936,261,814,2742,316,374 \
        "$output/s0.json" "$gpu" cpu -1 4 4
run_status=$?
set -e
if [[ "$run_status" -ne 0 && "$run_status" -ne 3 ]]; then
    echo "S0 runner failed unexpectedly with status $run_status" >&2
    exit "$run_status"
fi

python3 "$repo_dir/scripts/analyze_kimi_s0_batch_divergence.py" \
    --trace "$output/divergence.trace" \
    --base-position 1 --width 4 \
    --output "$output/boundary_divergence.json"

(
    cd "$output"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)

echo "S0 native width-four boundary trace complete: $output"
echo "Run only through: scripts/gpu_lease.sh run S0 -- scripts/run_kimi_s0_boundary_trace.sh ..."
