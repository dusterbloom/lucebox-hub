#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
output_dir="${KIMI_PANEL_OUTPUT_DIR:-/mnt/kimi-k3/captures}"
corpus_path="${KIMI_PANEL_CORPUS:-$output_dir/kimi_panel_smoke.jsonl}"
capture_path="${KIMI_PANEL_CAPTURE:-$output_dir/kimi_layer01_2048.bin}"
total_tokens="${KIMI_PANEL_TOTAL_TOKENS:-2048}"
gpu="${KIMI_PANEL_GPU:-0}"

if pgrep -f '[h]f download unsloth/Kimi-K3-GGUF' >/dev/null; then
    echo "The Kimi checkpoint download is still using the model drive." >&2
    echo "Wait for it to finish before capture so storage measurements are uncontended." >&2
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

cmake --build "$build_dir" --target capture_kimi_k3_panel -j4

"$build_dir/capture_kimi_k3_panel" \
    "$model_path" "$corpus_path" "$capture_path" \
    "$gpu" 1 "$total_tokens" 4096 128

sha256sum "$capture_path" "$capture_path.json"
