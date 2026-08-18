#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${KIMI_P35_MODEL:-/mnt/kimi-k3/models/kimi-k3-kda-q4k-p32/Kimi-K3-KDA-Q4_K-00001-of-00014.gguf}"
output="${KIMI_P35_OUTPUT:-/mnt/kimi-k3/results/kimi-k3-p35-complete-prep-late18-code-20260818}"
gpu="${KIMI_P35_GPU:-0}"
layers="${KIMI_P35_LAYERS:-68,69,70,72,73,74,76,77,78,80,81,82,84,85,86,88,89,90}"
fixture="${KIMI_P35_FIXTURE:-$repo_dir/server/test/fixtures/kimi_k3_p30_cache_smoke.jsonl}"
native_root="${KIMI_P35_NATIVE_ROOT:-/mnt/kimi-k3/results/kimi-k3-p31-stage-code-20260818}"
n_gen="${KIMI_P35_N_GEN:-24}"
resume="${KIMI_P35_RESUME:-0}"
quality_output="${KIMI_P35_ANALYSIS_OUTPUT:-$repo_dir/results/k3_p35_complete_prep_late18_quality.json}"
stage_output="${KIMI_P35_STAGE_OUTPUT:-$repo_dir/results/k3_p35_complete_prep_late18_stage.json}"
lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

[[ -f "$model" ]] || { echo "missing P35 model: $model" >&2; exit 1; }
if [[ -e "$output" && "$resume" != 1 ]]; then
  echo "refusing existing P35 output: $output" >&2
  exit 1
fi

exec 9>"$lock"
flock -n 9 || { echo "graphics-card lease is held" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | rg -q '[0-9]'; then
    echo "graphics card is already in use" >&2
    exit 1
fi

complete_env=()
if [[ "$layers" != none ]]; then
  complete_env=(DFLASH_KIMI_COMPLETE_PREP_LAYERS="$layers")
fi

env "${complete_env[@]}" \
  DFLASH_KIMI_STAGE_PROFILE=1 \
  KIMI_H23_MODEL="$model" \
  KIMI_H23_FIXTURE="$fixture" \
  KIMI_H23_NATIVE_ROOT="$native_root" \
  KIMI_H23_CANDIDATE_ROOT="$output" \
  KIMI_H23_AUX_DIR="/mnt/kimi-k3/artifacts/kimi-h23-calibrated96-runtime-10000" \
  KIMI_H23_SIDECAR_DIR="/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars" \
  KIMI_H23_BUDGET_TABLE="$repo_dir/results/h23_10k_policies/h23_moonshot_1_2gib.txt" \
  KIMI_H23_ANALYSIS_OUTPUT="$quality_output" \
  KIMI_H23_PROMPT_ID="p35-complete-prep-late18-code" \
  KIMI_H23_GPU="$gpu" \
  KIMI_H23_N_GEN="$n_gen" \
  KIMI_H23_RESUME="$resume" \
  KIMI_H23_DEVICE_CACHE_MB=2048 \
  KIMI_H23_CPU_THREADS=12 \
  "$repo_dir/scripts/run_kimi_h23_quality.sh" candidate

python3 "$repo_dir/scripts/analyze_kimi_k3_p31_stage.py" \
  --root "$output" \
  --output "$stage_output"

echo "P35 complete-preparation quality/profile complete: $output"
