#!/usr/bin/env bash
# Restartable exact expert-response fit for the merged H23 10K capture.
set -euo pipefail

mode="${1:-preflight}"
[[ "$mode" == preflight || "$mode" == status || "$mode" == run ]] || {
    echo "usage: $0 preflight|status|run" >&2
    exit 2
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_H23_FIT_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_H23_FIT_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
capture_root="${KIMI_H23_FIT_CAPTURE_ROOT:-/mnt/kimi-k3/captures/kimi-h23-all-layer-10000-chunked-v1/merged}"
fit_root="${KIMI_H23_FIT_STATE_ROOT:-/mnt/kimi-k3/fit-state/kimi-h23-panel-fit-10000}"
response_root="${KIMI_H23_FIT_RESPONSE_ROOT:-/mnt/kimi-k3/responses/kimi-h23-all-layer-10000}"
result_root="${KIMI_H23_FIT_RESULT_ROOT:-/mnt/kimi-k3/results/kimi-h23-all-layer-10000}"
first_layer="${KIMI_H23_FIT_FIRST_LAYER:-1}"
last_layer="${KIMI_H23_FIT_LAST_LAYER:-92}"
gpu="${KIMI_H23_FIT_GPU:-0}"
batch="${KIMI_H23_FIT_BATCH:-128}"
validator="$repo_dir/scripts/validate_kimi_h23_fit_layer.py"

if (( first_layer < 1 || last_layer > 92 || first_layer > last_layer )); then
    echo "layer range must be within 1..92" >&2
    exit 1
fi
for required in "$model" "$capture_root/all_layers_capture_manifest.json" "$validator"; do
    [[ -f "$required" ]] || { echo "missing H23 fit input: $required" >&2; exit 1; }
done

mkdir -p "$fit_root/receipts" "$response_root" "$result_root"
completed=0
for layer in $(seq "$first_layer" "$last_layer"); do
    receipt="$fit_root/receipts/layer$(printf '%02d' "$layer").json"
    [[ -f "$receipt" ]] && completed=$((completed + 1))
done
echo "H23 10K fit: completed=$completed total=$((last_layer - first_layer + 1))"
if [[ "$mode" == status ]]; then
    exit 0
fi

free_kib="$(df --output=avail /mnt/kimi-k3 | tail -n 1 | tr -d ' ')"
minimum_free_kib=$((280 * 1024 * 1024))
if (( free_kib < minimum_free_kib )); then
    echo "insufficient space: H23 fit requires a 280-GiB free-space reserve" >&2
    exit 1
fi
if [[ "$mode" == preflight ]]; then
    echo "capture=$capture_root tokens=10000 free=$((free_kib / 1024 / 1024))GiB"
    exit 0
fi
[[ "${KIMI_H23_ALLOW_FIT_RUN:-0}" == 1 ]] || {
    echo "refusing fit without KIMI_H23_ALLOW_FIT_RUN=1" >&2
    exit 1
}

CCACHE_DIR=/tmp/kimi-p20-ccache cmake --build "$build_dir" -j4 \
    --target fit_kimi_k3_panel

for layer in $(seq "$first_layer" "$last_layer"); do
    padded="$(printf '%02d' "$layer")"
    capture="$capture_root/kimi_layer${padded}_10000.bin"
    state="$fit_root/layer${padded}"
    responses="$response_root/layer${padded}"
    receipt="$fit_root/receipts/layer${padded}.json"
    if (( layer <= 54 )); then
        core=cuda
        prefix="$result_root/kimi_h18_layer${padded}_native"
    else
        core=cpu
        prefix="$result_root/kimi_h18_layer${padded}_native_cpu_core"
    fi
    if [[ -f "$receipt" ]]; then
        echo "[h23-fit] layer=$layer receipt=complete skip=true"
        continue
    fi
    [[ -f "$capture" ]] || { echo "missing layer capture: $capture" >&2; exit 1; }
    mkdir -p "$state" "$responses"
    echo "[h23-fit] layer=$layer core=$core state=starting"
    python3 "$repo_dir/scripts/run_with_telemetry.py" \
        --output-json "$prefix.fit.telemetry.json" \
        --samples-csv "$prefix.fit.telemetry.csv" \
        --stdout "$prefix.fit.stdout.log" --stderr "$prefix.fit.stderr.log" \
        --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 1 -- \
        "$build_dir/fit_kimi_k3_panel" \
            "$model" "$capture" "$state" "$prefix" \
            "$gpu" "$batch" "$responses" "$core"
    python3 "$validator" \
        --layer "$layer" --capture "$capture" --responses "$responses" \
        --state "$state" --prefix "$prefix" --receipt "$receipt" \
        >"$receipt.stdout.json"
    echo "[h23-fit] layer=$layer receipt=complete"
done

python3 - "$fit_root" "$capture_root" "$first_layer" "$last_layer" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
capture = Path(sys.argv[2])
first, last = int(sys.argv[3]), int(sys.argv[4])
rows = []
for layer in range(first, last + 1):
    path = root / "receipts" / f"layer{layer:02d}.json"
    rows.append({
        "layer": layer,
        "path": str(path),
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    })
manifest_path = capture / "all_layers_capture_manifest.json"
manifest = {
    "schema": "kimi-k3-h23-all-layer-fit-v1",
    "capture_manifest": str(manifest_path),
    "capture_manifest_sha256": hashlib.file_digest(manifest_path.open("rb"), "sha256").hexdigest(),
    "first_layer": first,
    "last_layer": last,
    "layers": rows,
}
(root / "all_layers_fit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "[h23-fit] complete layers=$first_layer..$last_layer root=$fit_root"
