#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${KIMI_P32_MODEL:-/mnt/kimi-k3/models/kimi-k3-kda-q4k-p32/Kimi-K3-KDA-Q4_K-00001-of-00014.gguf}"
output="${KIMI_P32_OUTPUT:-/mnt/kimi-k3/results/kimi-k3-p32-kda-q4k-code-r2-20260818}"
gpu="${KIMI_P32_GPU:-0}"
lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

[[ -f "$model" ]] || { echo "missing P32 candidate: $model" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "refusing existing P32 output: $output" >&2; exit 1; }

exec 9>"$lock"
flock -n 9 || { echo "graphics-card lease is held" >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | rg -q '[0-9]'; then
    echo "graphics card is already in use" >&2
    exit 1
fi

env \
  DFLASH_KIMI_STAGE_PROFILE=1 \
  KIMI_H23_MODEL="$model" \
  KIMI_H23_FIXTURE="$repo_dir/server/test/fixtures/kimi_k3_p30_cache_smoke.jsonl" \
  KIMI_H23_NATIVE_ROOT="/mnt/kimi-k3/results/kimi-k3-p31-stage-code-20260818" \
  KIMI_H23_CANDIDATE_ROOT="$output" \
  KIMI_H23_AUX_DIR="/mnt/kimi-k3/artifacts/kimi-h23-calibrated96-runtime-10000" \
  KIMI_H23_SIDECAR_DIR="/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars" \
  KIMI_H23_BUDGET_TABLE="$repo_dir/results/h23_10k_policies/h23_moonshot_1_2gib.txt" \
  KIMI_H23_ANALYSIS_OUTPUT="$repo_dir/results/k3_p32_kda_q4k_code_quality.json" \
  KIMI_H23_PROMPT_ID="p32-kda-q4k-code" \
  KIMI_H23_GPU="$gpu" \
  KIMI_H23_N_GEN=24 \
  KIMI_H23_DEVICE_CACHE_MB=2048 \
  KIMI_H23_CPU_THREADS=12 \
  "$repo_dir/scripts/run_kimi_h23_quality.sh" candidate

python3 "$repo_dir/scripts/analyze_kimi_k3_p31_stage.py" \
  --root "$output" \
  --output "$repo_dir/results/k3_p32_kda_q4k_stage_profile.json"

echo "P32 KDA Q4_K quality/profile complete: $output"
