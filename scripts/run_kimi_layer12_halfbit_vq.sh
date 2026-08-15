#!/usr/bin/env bash
# Non-overwriting matched-byte all-width VQ rate/distortion probe for K3 L12.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${KIMI_PANEL_GPU:-0}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"
model="${KIMI_PANEL_MODEL_SHARD:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
capture="${KIMI_LAYER12_CAPTURE:-/mnt/kimi-k3/captures/kimi_layer12_10000.bin}"
teacher="${KIMI_LAYER12_TEACHER:-/mnt/kimi-k3/results/kimi_layer12_panel_10000.teacher.f32}"
responses="${KIMI_LAYER12_RESPONSES:-/mnt/kimi-k3/responses/kimi_layer12_10000}"
baseline_json="${KIMI_LAYER12_TAIL_BASELINE_JSON:-/mnt/kimi-k3/results/kimi_layer12_omitted_tail_ceiling_retry03.json}"
baseline_npz="${KIMI_LAYER12_TAIL_BASELINE_NPZ:-/mnt/kimi-k3/results/kimi_layer12_omitted_tail_ceiling_retry03.npz}"
prefix="${KIMI_LAYER12_HALFBIT_PREFIX:-/mnt/kimi-k3/results/kimi_layer12_halfbit_vq_retry01}"

for required in "$model" "$capture" "$teacher" "$baseline_json" "$baseline_npz" \
    "$responses/expert_0000.responses.f32"; do
    [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
for output in "$prefix.json" "$prefix.csv" "$prefix.stdout.log" "$prefix.stderr.log" \
    "$prefix.telemetry.json" "$prefix.telemetry.csv"; do
    [[ ! -e "$output" ]] || { echo "refusing to overwrite: $output" >&2; exit 2; }
done

exec 9>"$gpu_lock"
flock -n 9 || { echo "GPU lease is busy: $gpu_lock" >&2; exit 3; }

exec python3 "$repo_dir/scripts/run_with_telemetry.py" \
    --output-json "$prefix.telemetry.json" \
    --samples-csv "$prefix.telemetry.csv" \
    --stdout "$prefix.stdout.log" \
    --stderr "$prefix.stderr.log" \
    --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 1 -- \
    python3 "$repo_dir/scripts/probe_kimi_layer12_halfbit_vq.py" \
        "$model" "$capture" "$teacher" "$responses" "$prefix.json" \
        --output-csv "$prefix.csv" \
        --baseline-json "$baseline_json" --baseline-npz "$baseline_npz" \
        --layer 12 --device cuda
