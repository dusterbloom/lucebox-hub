#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
output_dir="${KIMI_PANEL_OUTPUT_DIR:-/mnt/kimi-k3/captures}"
corpus_path="${KIMI_PANEL_CORPUS:-$output_dir/kimi_panel_smoke.jsonl}"
capture_path="${KIMI_PANEL_CAPTURE:-$output_dir/kimi_layer01_2048.bin}"
fit_state_dir="${KIMI_PANEL_FIT_STATE:-/mnt/kimi-k3/fit-state/kimi_layer01}"
result_prefix="${KIMI_PANEL_RESULT_PREFIX:-/mnt/kimi-k3/results/kimi_layer01_panel}"
panel_artifact="${KIMI_PANEL_ARTIFACT:-/mnt/kimi-k3/results/kimi_layer01_panel.safetensors}"
total_tokens="${KIMI_PANEL_TOTAL_TOKENS:-2048}"
gpu="${KIMI_PANEL_GPU:-0}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"
revision="a0836360ce58dfec088d966a97f2ddc8a606279b"
model_root="${KIMI_PANEL_MODEL_ROOT:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF}"
complete_marker="${KIMI_PANEL_DOWNLOAD_MARKER:-$model_root/.ud-iq1s-$revision.complete}"

exec 9>"$gpu_lock"
if ! flock -n 9; then
    echo "Another cooperating job holds the graphics-card lease: $gpu_lock" >&2
    exit 1
fi

if pgrep -f '^(python3|/usr/bin/python3) .*/hf download unsloth/Kimi-K3-GGUF([[:space:]]|$)' >/dev/null; then
    echo "The Kimi checkpoint download is still using the model drive." >&2
    echo "Wait for it to finish before capture so storage measurements are uncontended." >&2
    exit 1
fi

if [[ ! -f "$complete_marker" ]]; then
    echo "The pinned Kimi checkpoint does not have its completion marker: $complete_marker" >&2
    exit 1
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | rg -q '[0-9]'; then
    echo "The graphics card is already in use; refusing an uncontended capture." >&2
    exit 1
fi

if [[ ! -f "$model_path" ]]; then
    echo "Missing first model shard: $model_path" >&2
    exit 1
fi

mkdir -p "$output_dir"

echo "Verifying all fourteen model shards byte-for-byte..."
(
    cd "$model_dir"
    sha256sum -c "$repo_dir/scripts/kimi_k3_ud_iq1s.sha256"
)

python3 "$repo_dir/scripts/prepare_kimi_panel_corpus.py" \
    --conversation "$repo_dir/server/eval/mt_bench/question.jsonl" \
    --code "$repo_dir/server/eval/humaneval_plus/humanevalplus.jsonl" \
    --output "$corpus_path"

cmake --build "$build_dir" --target \
    capture_kimi_k3_panel fit_kimi_k3_panel -j4

python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$capture_path.telemetry.json" \
    --samples-csv "$capture_path.telemetry.csv" \
    --stdout "$capture_path.stdout.log" \
    --stderr "$capture_path.stderr.log" \
    --mount-path "$model_dir" --gpu "$gpu" -- \
    "$build_dir/capture_kimi_k3_panel" \
        "$model_path" "$corpus_path" "$capture_path" \
        "$gpu" 1 "$total_tokens" 4096 128

python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$result_prefix.fit.telemetry.json" \
    --samples-csv "$result_prefix.fit.telemetry.csv" \
    --stdout "$result_prefix.fit.stdout.log" \
    --stderr "$result_prefix.fit.stderr.log" \
    --mount-path "$model_dir" --gpu "$gpu" -- \
    "$build_dir/fit_kimi_k3_panel" \
        "$model_path" "$capture_path" "$fit_state_dir" "$result_prefix" \
        "$gpu" 128

python3 "$repo_dir/scripts/export_kimi_panel_safetensors.py" \
    "$result_prefix.panel.f32" "$panel_artifact"

sha256sum \
    "$capture_path" "$capture_path.json" \
    "$capture_path.telemetry.json" "$capture_path.telemetry.csv" \
    "$capture_path.stdout.log" "$capture_path.stderr.log" \
    "$result_prefix.json" "$result_prefix.csv" \
    "$result_prefix.fit.telemetry.json" "$result_prefix.fit.telemetry.csv" \
    "$result_prefix.fit.stdout.log" "$result_prefix.fit.stderr.log" \
    "$result_prefix.panel.f32" "$panel_artifact"
