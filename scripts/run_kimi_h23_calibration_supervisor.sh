#!/usr/bin/env bash
# Crash-resumable supervisor for the H23 10K all-layer slab calibration.
# The per-layer runner publishes atomically and skips validated outputs, so
# systemd can safely restart this command after a transient failure or WSL boot.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
state_root="/mnt/kimi-k3/fit-state/kimi-h23-slab-calibration-10000"
result_root="/mnt/kimi-k3/results/kimi-h23-all-layer-10000"

mkdir -p "$state_root" "$result_root"
echo "[h23-supervisor] start=$(date --iso-8601=seconds)"

export KIMI_H18_CAPTURE_ROOT=/mnt/kimi-k3/captures/kimi-h23-all-layer-10000-chunked-v1/merged
export KIMI_H18_RESPONSE_ROOT=/mnt/kimi-k3/responses/kimi-h23-all-layer-10000
export KIMI_H18_RESULT_ROOT="$result_root"
export KIMI_H18_CALIBRATION_ROOT="$state_root"
export KIMI_H18_CAPTURE_TOKENS=10000
export KIMI_H18_FIRST_LAYER=1
export KIMI_H18_LAST_LAYER=92

bash "$repo_dir/scripts/run_kimi_h18_all_layer_calibration.sh"

python3 - "$state_root" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "all_layers_calibration_manifest.json"
if not manifest_path.is_file():
    raise SystemExit("missing completed all-layer calibration manifest")
manifest = json.loads(manifest_path.read_text())
rows = manifest.get("layers", [])
if [row.get("layer") for row in rows] != list(range(1, 93)):
    raise SystemExit("all-layer manifest does not cover layers 1..92")
for row in rows:
    path = Path(row["path"])
    if not path.is_file():
        raise SystemExit(f"missing calibration state: {path}")
    actual = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if actual != row["sha256"]:
        raise SystemExit(f"calibration checksum mismatch: {path}")

receipt = {
    "schema": "kimi-k3-h23-calibration-supervisor-v1",
    "status": "COMPLETE",
    "layers": len(rows),
    "manifest": str(manifest_path),
    "manifest_sha256": hashlib.file_digest(
        manifest_path.open("rb"), "sha256"
    ).hexdigest(),
}
temporary = root / ".calibration_supervisor_complete.json.tmp"
complete = root / "calibration_supervisor_complete.json"
temporary.write_text(json.dumps(receipt, indent=2) + "\n")
os.replace(temporary, complete)
print(json.dumps(receipt, sort_keys=True))
PY

echo "[h23-supervisor] complete=$(date --iso-8601=seconds) layers=92"
