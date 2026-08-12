#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
build_dir="${KIMI_PANEL_BUILD_DIR:-$repo_dir/server/build-k3-panel-cuda126}"
output_dir="${KIMI_EXACT_OUTPUT_DIR:-/mnt/kimi-k3/results/kimi-exact-baseline}"
gpu="${KIMI_PANEL_GPU:-0}"
prompt="${KIMI_EXACT_PROMPT:-According to all known laws}"
generated_tokens="${KIMI_EXACT_GENERATED_TOKENS:-4}"
max_context="${KIMI_EXACT_MAX_CONTEXT:-512}"

if pgrep -f '[h]f download unsloth/Kimi-K3-GGUF' >/dev/null; then
    echo "The Kimi checkpoint download is still using the model drive." >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | rg -q '[0-9]'; then
    echo "The graphics card is already in use; refusing a contended baseline." >&2
    exit 1
fi
if [[ ! -f "$model_path" ]]; then
    echo "Missing first model shard: $model_path" >&2
    exit 1
fi

mkdir -p "$output_dir"
(
    cd "$model_dir"
    sha256sum -c "$repo_dir/scripts/kimi_k3_ud_iq1s.sha256"
)
cmake --build "$build_dir" --target smoke_kimi_k3_forward -j4

for run in 1 2; do
    prefix="$output_dir/exact-run-$run"
    python3 "$repo_dir/scripts/run_with_telemetry.py" \
        --output-json "$prefix.telemetry.json" \
        --samples-csv "$prefix.telemetry.csv" \
        --stdout "$prefix.stdout.log" --stderr "$prefix.stderr.log" \
        --mount-path "$model_dir" --gpu "$gpu" -- \
        env \
            DFLASH_KIMI_SMOKE_MAX_CTX="$max_context" \
            DFLASH_KIMI_LOGITS_TRACE_OUT="$prefix.logits.f32" \
            "$build_dir/smoke_kimi_k3_forward" \
                "$model_path" "$gpu" "$generated_tokens" "$prompt" \
                1 -1
    rg '^\[kimi-k3-smoke\] (prompt_ids|output_ids|text):' \
        "$prefix.stdout.log" >"$prefix.behavior.txt"
done

cmp --silent \
    "$output_dir/exact-run-1.behavior.txt" \
    "$output_dir/exact-run-2.behavior.txt"
python3 "$repo_dir/scripts/compare_kimi_logits.py" \
    "$output_dir/exact-run-1.logits.f32" \
    "$output_dir/exact-run-2.logits.f32" \
    --output "$output_dir/exact-repeat-comparison.json"

python3 - "$output_dir/exact-repeat-comparison.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1]))
if not result["byte_identical"]:
    raise SystemExit("exact repeated logit traces were not byte-identical")
if result["teacher_to_candidate_divergence"]["maximum"] != 0:
    raise SystemExit("exact repeated logit traces had non-zero divergence")
print("exact repeated logits are byte-identical")
PY

(
    cd "$output_dir"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum >SHA256SUMS
)
