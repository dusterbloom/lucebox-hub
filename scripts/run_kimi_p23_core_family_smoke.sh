#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_P23_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_P23_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
aux="${KIMI_P23_AUX_DIR:-/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime}"
sidecars="${KIMI_P23_SIDECAR_DIR:-/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars}"
mode="${1:?usage: run_kimi_p23_core_family_smoke.sh MODE OUTPUT_DIR [N_GEN]}"
output="${2:?usage: run_kimi_p23_core_family_smoke.sh MODE OUTPUT_DIR [N_GEN]}"
generated="${3:-2}"
gpu="${KIMI_P23_GPU:-0}"
prompt="${KIMI_P23_PROMPT:-Hi}"
device_cache_mb="${KIMI_P23_DEVICE_CACHE_MB:-2048}"
lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

case ",$mode," in
    ,all,|,router,|,latent,|,shared,|,router,latent,|,router,shared,|,latent,shared,|,router,latent,shared,) ;;
    *) echo "unsupported MoE-core family set: $mode" >&2; exit 2 ;;
esac

for required in "$model" "$aux/all_layers_calibrated96_manifest.json" \
        "$sidecars/all_layers_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "missing P23 input: $required" >&2
        exit 1
    fi
done
if [[ -e "$output" ]]; then
    echo "refusing to overwrite P23 output: $output" >&2
    exit 1
fi
mkdir -p "$output"

exec 9>"$lock"
flock -n 9 || { echo "graphics-card lease is held" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | rg -q '[0-9]'; then
    echo "graphics card is already in use" >&2
    exit 1
fi

CCACHE_DIR=/tmp/kimi-p20-ccache cmake --build "$build_dir" -j4 \
    --target smoke_kimi_k3_forward
python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$output/telemetry.json" \
    --samples-csv "$output/telemetry.csv" \
    --stdout "$output/stdout.log" --stderr "$output/stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" \
    --interval "${KIMI_P23_TELEMETRY_INTERVAL:-5}" -- \
    env \
      DFLASH_MOE_NVME_DIRECT=on \
      DFLASH_MOE_NVME_DEVICE_CACHE_MB="$device_cache_mb" \
      DFLASH_KIMI_CPU_THREADS="${KIMI_P23_CPU_THREADS:-18}" \
      DFLASH_KIMI_MMAP_DROP_PAGES=0 \
      DFLASH_KIMI_MOE_CORE_OFFLOAD="$mode" \
      DFLASH_KIMI_STAGE_PROFILE="${KIMI_P23_STAGE_PROFILE:-0}" \
      DFLASH_KIMI_SMOKE_MAX_CTX=128 \
      DFLASH_KIMI_LOGITS_TRACE_OUT="$output/logits.f32" \
      DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96 \
      DFLASH_KIMI_CALIBRATED96_AUX_DIR="$aux" \
      DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR="$sidecars" \
      DFLASH_KIMI_CALIBRATED96_METRICS_OUT="$output/traffic.tsv" \
      DFLASH_KIMI_P20_PHYSICAL_LAYOUT=scratch \
      DFLASH_KIMI_P20_IO_BACKEND=direct-pread \
      DFLASH_KIMI_P23_PERSISTENT_SCRATCH="${KIMI_P23_PERSISTENT_SCRATCH:-0}" \
      DFLASH_KIMI_P25_COMPACT_UPLOAD="${KIMI_P25_COMPACT_UPLOAD:-0}" \
      DFLASH_KIMI_P20_SLAB_BUDGET=96 \
      DFLASH_KIMI_P20_IO_TRACE="$output/io_trace.tsv" \
      DFLASH_KIMI_P20_PROMPT_ID="p23-core-$mode" \
      "$build_dir/smoke_kimi_k3_forward" \
        "$model" "$gpu" "$generated" "$prompt" 1 -1 "" "$gpu" cpu

(
    cd "$output"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 sha256sum >SHA256SUMS
)
echo "P23 family smoke complete: mode=$mode output=$output"
