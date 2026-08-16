#!/usr/bin/env bash
# Build per-layer progressive-slab calibration states from one completed H18
# all-layer capture.  Each layer is an independent, atomic NPZ write, so a
# restart skips only validated completed states and never overwrites them.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${KIMI_PANEL_MODEL_DIR:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S}"
model_path="$model_dir/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"
capture_root="${KIMI_H18_CAPTURE_ROOT:-/mnt/kimi-k3/captures/kimi-h18-all-layer-2048-chunk8}"
response_root="${KIMI_H18_RESPONSE_ROOT:-/mnt/kimi-k3/responses/kimi-h18-all-layer-2048}"
result_root="${KIMI_H18_RESULT_ROOT:-/mnt/kimi-k3/results}"
state_root="${KIMI_H18_CALIBRATION_ROOT:-/mnt/kimi-k3/fit-state/kimi-h18-slab-calibration-2048}"
first_layer="${KIMI_H18_FIRST_LAYER:-1}"
last_layer="${KIMI_H18_LAST_LAYER:-92}"
capture_tokens="${KIMI_H18_CAPTURE_TOKENS:-2048}"
gpu="${KIMI_PANEL_GPU:-0}"
gpu_lock="${KIMI_GPU_LOCK_FILE:-/tmp/lucebox-gpu-$gpu.lock}"

if (( first_layer < 1 || last_layer > 92 || first_layer > last_layer )); then
    echo "layer range must be within 1..92" >&2
    exit 1
fi
for required in "$model_path" "$capture_root/all_layers_capture_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        echo "missing H18 calibration input: $required" >&2
        exit 1
    fi
done

mkdir -p "$state_root"
exec 9>"$gpu_lock"
if ! flock -n 9; then
    echo "another cooperating job holds the graphics-card lease" >&2
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | rg -q '[0-9]'; then
    echo "the graphics card is already in use" >&2
    exit 1
fi

valid_state() {
    local state="$1"
    python3 -c '
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as values:
    expected = {
        "slab_means": (896, 12, 3584),
        "slab_expected_norm": (896, 12),
        "slab_expected_residual_norm": (896, 12),
        "native_means": (896, 3584),
        "native_expected_norm": (896,),
        "calibrated_experts": (896,),
    }
    if set(values.files) != set(expected):
        raise SystemExit(1)
    if any(values[name].shape != shape for name, shape in expected.items()):
        raise SystemExit(1)
    if not np.isfinite(values["slab_expected_residual_norm"]).all():
        raise SystemExit(1)
' "$state"
}

for layer in $(seq "$first_layer" "$last_layer"); do
    padded="$(printf '%02d' "$layer")"
    capture="$capture_root/kimi_layer${padded}_${capture_tokens}.bin"
    responses="$response_root/layer${padded}"
    teacher="$result_root/kimi_h18_layer${padded}_native.teacher.f32"
    if [[ ! -f "$teacher" ]]; then
        teacher="$result_root/kimi_h18_layer${padded}_native_cpu_core.teacher.f32"
    fi
    state="$state_root/kimi_layer${padded}_neuron_slabs_calibration.npz"
    result_stub="$result_root/kimi_h18_layer${padded}_neuron_slabs_calibration"
    for required in "$capture" "$teacher" "$responses/expert_0000.responses.f32"; do
        if [[ ! -f "$required" ]]; then
            echo "layer $layer missing calibration input: $required" >&2
            exit 1
        fi
    done
    if [[ -f "$state" ]] && valid_state "$state"; then
        echo "[kimi-h18-calibration] layer=$layer state=complete skip=true"
        continue
    fi
    if [[ -e "$state" ]]; then
        echo "layer $layer has an invalid state; refusing to overwrite: $state" >&2
        exit 1
    fi
    echo "[kimi-h18-calibration] layer=$layer state=starting"
    python3 "$repo_dir/scripts/run_with_telemetry.py" \
        --output-json "$result_stub.telemetry.json" \
        --samples-csv "$result_stub.telemetry.csv" \
        --stdout "$result_stub.stdout.log" \
        --stderr "$result_stub.stderr.log" \
        --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 1 -- \
        python3 "$repo_dir/scripts/probe_kimi_neuron_slabs.py" \
            "$model_path" "$capture" "$teacher" "$responses" \
            "$result_stub.json" \
            --output-csv "$result_stub.csv" \
            --fit-state "$state" \
            --calibration-only --exact-fallback-uncalibrated \
            --layer "$layer" --device cuda
    if ! valid_state "$state"; then
        echo "layer $layer wrote an invalid calibration state" >&2
        exit 1
    fi
    sha256sum "$state" >"$state.sha256"
    echo "[kimi-h18-calibration] layer=$layer state=complete"
done

commit="$(git -C "$repo_dir" rev-parse HEAD)"
python3 - "$state_root" "$capture_root" "$commit" "$first_layer" "$last_layer" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
capture = Path(sys.argv[2])
commit = sys.argv[3]
first = int(sys.argv[4])
last = int(sys.argv[5])
states = []
for layer in range(first, last + 1):
    path = root / f"kimi_layer{layer:02d}_neuron_slabs_calibration.npz"
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    states.append({"layer": layer, "path": str(path), "sha256": digest, "bytes": path.stat().st_size})
record = {
    "schema": "kimi-h18-all-layer-calibration-v1",
    "repository_commit": commit,
    "capture_root": str(capture),
    "capture_manifest_sha256": hashlib.file_digest((capture / "all_layers_capture_manifest.json").open("rb"), "sha256").hexdigest(),
    "layers": states,
}
(root / "all_layers_calibration_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
PY

echo "[kimi-h18-calibration] complete layers=$first_layer..$last_layer root=$state_root"
