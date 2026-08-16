#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_P28_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_P28_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
aux="${KIMI_P28_AUX_DIR:-/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime}"
sidecars="${KIMI_P28_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
oracle_trace="${KIMI_P28_ORACLE_TRACE:-/mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/io_trace.tsv}"
reference_dir="${KIMI_P28_REFERENCE_DIR:-/mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816}"
output="${1:?usage: run_kimi_p28_oracle.sh OUTPUT_DIR [N_ROWS]}"
rows="${2:-32}"
gpu="${KIMI_P28_GPU:-0}"

for required in "$model" "$oracle_trace" "$reference_dir/logits.f32" \
        "$aux/all_layers_calibrated96_manifest.json" \
        "$sidecars/all_layers_manifest.json"; do
    [[ -f "$required" ]] || { echo "missing P28 input: $required" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite P28 output: $output" >&2; exit 1; }
mkdir -p "$output"

CCACHE_TEMPDIR="${CCACHE_TEMPDIR:-/tmp/ccache-p28}" \
    cmake --build "$build_dir" --target smoke_kimi_k3_forward -j4

python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output/telemetry.json" \
    --samples-csv "$output/telemetry.csv" \
    --stdout "$output/stdout.log" --stderr "$output/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 1 -- \
    env \
      DFLASH_MOE_NVME_DIRECT=on \
      DFLASH_MOE_NVME_DEVICE_CACHE_MB=2048 \
      DFLASH_KIMI_CPU_THREADS="${KIMI_P28_CPU_THREADS:-18}" \
      DFLASH_KIMI_MMAP_DROP_PAGES=0 \
      DFLASH_KIMI_MOE_CORE_OFFLOAD=latent,shared \
      DFLASH_KIMI_STAGE_PROFILE=1 \
      DFLASH_KIMI_SMOKE_MAX_CTX=128 \
      DFLASH_KIMI_LOGITS_TRACE_OUT="$output/logits.f32" \
      DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96 \
      DFLASH_KIMI_CALIBRATED96_AUX_DIR="$aux" \
      DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecars" \
      DFLASH_KIMI_CALIBRATED96_METRICS_OUT="$output/traffic.tsv" \
      DFLASH_KIMI_P20_PHYSICAL_LAYOUT=scratch \
      DFLASH_KIMI_P20_IO_BACKEND=direct-pread \
      DFLASH_KIMI_P23_PERSISTENT_SCRATCH=1 \
      DFLASH_KIMI_P25_COMPACT_UPLOAD=1 \
      DFLASH_KIMI_P26_PINNED_COMPACT=1 \
      DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT=1 \
      DFLASH_KIMI_P28_ORACLE_TRACE="$oracle_trace" \
      DFLASH_KIMI_P20_SLAB_BUDGET=96 \
      DFLASH_KIMI_P20_IO_TRACE="$output/io_trace.tsv" \
      DFLASH_KIMI_P20_PROMPT_ID=p28-oracle \
      "$build_dir/smoke_kimi_k3_forward" \
        "$model" "$gpu" "$rows" Hi 1 -1 "" "$gpu" cpu

python3 "$repo_dir/scripts/analyze_kimi_p28_integrated.py" \
    --reference-dir "$reference_dir" --candidate-dir "$output" \
    --output "$output/p28_integrated.json"
(
    cd "$output"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "P28 oracle replay complete: $output"
echo "Run only through: scripts/gpu_lease.sh run P28 -- scripts/run_kimi_p28_oracle.sh ..."
