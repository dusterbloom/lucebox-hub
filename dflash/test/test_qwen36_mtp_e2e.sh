#!/usr/bin/env bash
# test_qwen36_mtp_e2e.sh — T3 end-to-end bench for Qwen3.6 native MTP.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/dflash/build"
RESULTS_DIR="${REPO_ROOT}/dflash/bench/results"
BENCH_TMP="${TMPDIR:-/tmp}/dflash_bench"
PROMPT_META="${BENCH_TMP}/qwen36_mtp_prompts.json"
BIN="${BUILD_DIR}/test_dflash"

N_RUNS="${N_RUNS:-3}"
N_GEN="${N_GEN:-128}"
MAX_CTX="${MAX_CTX:-4096}"
KV_TYPE="${KV_TYPE:-q8_0}"

if [[ -z "${QWEN36_MTP_GGUF:-}" ]]; then
    echo "[qwen36_mtp_e2e] QWEN36_MTP_GGUF unset; bench skipped"
    exit 0
fi
if [[ ! -f "${QWEN36_MTP_GGUF}" ]]; then
    echo "[qwen36_mtp_e2e] QWEN36_MTP_GGUF not found: ${QWEN36_MTP_GGUF}" >&2
    exit 1
fi
if [[ ! -x "${BIN}" ]]; then
    echo "[qwen36_mtp_e2e] missing binary: ${BIN}" >&2
    exit 1
fi

CMAKE_CACHE="${BUILD_DIR}/CMakeCache.txt"
BUILD_TYPE="unknown"
if [[ -f "${CMAKE_CACHE}" ]]; then
    BUILD_TYPE=$(grep '^CMAKE_BUILD_TYPE' "${CMAKE_CACHE}" | cut -d= -f2 | tr -d '[:space:]')
fi
if [[ "${BUILD_TYPE}" != "Release" ]]; then
    echo "[qwen36_mtp_e2e] WARNING: build type is '${BUILD_TYPE}', expected Release." >&2
fi
NDEBUG_VERIFIED=false
if [[ "${BUILD_TYPE}" == "Release" ]]; then
    NDEBUG_VERIFIED=true
fi

COMMIT_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown")
GPU_INFO="unknown"
DRIVER_VERSION="unknown"
POWER_LIMIT_WATTS="null"
if command -v nvidia-smi &>/dev/null; then
    if gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) &&
       [[ "${gpu_name}" != *"failed"* && "${gpu_name}" != *"Failed"* && -n "${gpu_name}" ]]; then
        GPU_INFO=$(printf '%s' "${gpu_name}" | sed 's/[[:space:]]*$//')
    fi
    if driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) &&
       [[ "${driver}" != *"failed"* && "${driver}" != *"Failed"* && -n "${driver}" ]]; then
        DRIVER_VERSION=$(printf '%s' "${driver}" | tr -d '[:space:]')
    fi
    if power=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1) &&
       [[ "${power}" != *"failed"* && "${power}" != *"Failed"* && -n "${power}" ]]; then
        POWER_LIMIT_WATTS=$(printf '%s' "${power}" | tr -d '[:space:]')
    fi
fi
CUDA_VERSION="unknown"
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1 || echo "unknown")
fi

echo "[qwen36_mtp_e2e] GPU: ${GPU_INFO}"
echo "[qwen36_mtp_e2e] Driver: ${DRIVER_VERSION}"
echo "[qwen36_mtp_e2e] CUDA: ${CUDA_VERSION}"
echo "[qwen36_mtp_e2e] Commit: ${COMMIT_SHA}"
echo "[qwen36_mtp_e2e] KV type: ${KV_TYPE}"
echo "[qwen36_mtp_e2e] n_predict: ${N_GEN}"
echo "[qwen36_mtp_e2e] QWEN36_MTP_GGUF=${QWEN36_MTP_GGUF}"

mkdir -p "${BENCH_TMP}" "${RESULTS_DIR}"

# Prefer the system CUDA driver shim and the CUDA toolkit selected at
# configure time for the benchmark binary. Keep this after nvidia-smi
# fingerprinting because NVML may be provided by a different host shim.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

python3 - "${BENCH_TMP}" "${PROMPT_META}" <<'PY'
import json
import struct
import sys
from pathlib import Path

tmp = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
prompts = [
    ("has_close_elements",
     "from typing import List\n\n"
     "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
     "    \"\"\"Check whether any two numbers are closer than threshold.\n"
     "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
     "    False\n"
     "    \"\"\"\n"
     "    for"),
    ("separate_paren_groups",
     "from typing import List\n\n"
     "def separate_paren_groups(paren_string: str) -> List[str]:\n"
     "    \"\"\"Separate balanced parenthesis groups into strings.\n"
     "    >>> separate_paren_groups('( ) (( ))')\n"
     "    ['()', '(())']\n"
     "    \"\"\"\n"
     "    result = []\n"
     "    for"),
    ("truncate_number",
     "def truncate_number(number: float) -> float:\n"
     "    \"\"\"Return the decimal part of a positive floating point number.\n"
     "    >>> truncate_number(3.5)\n"
     "    0.5\n"
     "    \"\"\"\n"
     "    return"),
    ("below_zero",
     "from typing import List\n\n"
     "def below_zero(operations: List[int]) -> bool:\n"
     "    \"\"\"Return True if the running account balance falls below zero.\n"
     "    >>> below_zero([1, 2, -4, 5])\n"
     "    True\n"
     "    \"\"\"\n"
     "    balance = 0\n"
     "    for op in"),
    ("mean_absolute_deviation",
     "from typing import List\n\n"
     "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
     "    \"\"\"Calculate mean absolute deviation around the mean.\n"
     "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
     "    1.0\n"
     "    \"\"\"\n"
     "    mean ="),
]

def token_count(path: Path) -> int:
    return path.stat().st_size // 4

records = []
paths = [tmp / f"qwen36_mtp_prompt_{i:02d}.bin" for i in range(len(prompts))]
if not all(p.exists() for p in paths):
    alt = [tmp / f"he_prompt_Qwen_Qwen3.6-27B_{i:02d}.bin" for i in range(len(prompts))]
    if all(p.exists() for p in alt):
        paths = alt
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B",
                                            trust_remote_code=True,
                                            local_files_only=True)
        for i, (_, text) in enumerate(prompts):
            ids = tok.encode(text, add_special_tokens=False)
            with paths[i].open("wb") as f:
                for t in ids:
                    f.write(struct.pack("<i", int(t)))

for i, ((name, text), path) in enumerate(zip(prompts, paths), start=1):
    records.append({"id": i, "name": name, "text": text,
                    "tokens": token_count(path), "path": str(path)})
meta_path.write_text(json.dumps(records, indent=2))
PY

mapfile -t PROMPT_IDS < <(python3 - "${PROMPT_META}" <<'PY'
import json, sys
for p in json.load(open(sys.argv[1])):
    print(p["id"])
PY
)
mapfile -t PROMPT_FILES < <(python3 - "${PROMPT_META}" <<'PY'
import json, sys
for p in json.load(open(sys.argv[1])):
    print(p["path"])
PY
)

run_one() {
    local gamma="$1"
    local prompt_id="$2"
    local prompt_file="$3"
    local log_file
    log_file="$(mktemp "${BENCH_TMP}/qwen36_mtp_run.XXXXXX.log")"
    # WSL2 mitigation: libggml-cuda.so teardown exhausts cudaGetDeviceCount for
    # the next process in the same shell session.  Give the driver 15 s to
    # release the device, then launch with a clean environment so no stale CUDA
    # state from the parent shell leaks in.
    sleep 15
    set +e
    env -i HOME="${HOME}" \
        PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda/lib64" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "${BIN}" "${QWEN36_MTP_GGUF}" \
        --mtp-gguf "${QWEN36_MTP_GGUF}" \
        --prompt-bin "${prompt_file}" \
        --n-gen "${N_GEN}" \
        --gamma "${gamma}" \
        --prompt-id "${prompt_id}" \
        --max-ctx "${MAX_CTX}" \
        -ctk "${KV_TYPE}" -ctv "${KV_TYPE}" >"${log_file}" 2>&1
    local status=$?
    set -e
    cat "${log_file}" >&2
    if [[ ${status} -ne 0 ]]; then
        echo "[qwen36_mtp_e2e] binary failed with status ${status}" >&2
        return "${status}"
    fi
    awk '/RESULT tok_s=/{for(i=1;i<=NF;i++) if($i ~ /^tok_s=/){split($i,a,"="); v=a[2]}} END{if(v!="") print v; else exit 1}' "${log_file}"
}

mean_of() {
    python3 - "$@" <<'PY'
import sys
xs = [float(x) for x in sys.argv[1:]]
print(f"{sum(xs) / len(xs):.6f}")
PY
}

AR_RUNS_FILE="${BENCH_TMP}/qwen36_mtp_dense_ar_runs.txt"
MTP_RUNS_FILE="${BENCH_TMP}/qwen36_mtp_dense_mtp_runs.txt"
: >"${AR_RUNS_FILE}"
: >"${MTP_RUNS_FILE}"

echo ""
echo "=== Dense family (27B) ==="
for run in $(seq 1 "${N_RUNS}"); do
    echo "  [run ${run}/${N_RUNS}] A - 27B Dense AR baseline"
    vals=()
    for idx in "${!PROMPT_FILES[@]}"; do
        vals+=("$(run_one 0 "${PROMPT_IDS[$idx]}" "${PROMPT_FILES[$idx]}")")
    done
    mean_of "${vals[@]}" >>"${AR_RUNS_FILE}"

    echo "  [run ${run}/${N_RUNS}] B - 27B Dense + Qwen3.6 MTP gamma=1"
    vals=()
    for idx in "${!PROMPT_FILES[@]}"; do
        vals+=("$(run_one 1 "${PROMPT_IDS[$idx]}" "${PROMPT_FILES[$idx]}")")
    done
    mean_of "${vals[@]}" >>"${MTP_RUNS_FILE}"
done

MOE_SKIPPED=true
echo ""
echo "=== MoE family (35B-A3B) ==="
if [[ -n "${QWEN36_MTP_MOE_GGUF:-}" && -f "${QWEN36_MTP_MOE_GGUF}" ]]; then
    echo "[qwen36_mtp_e2e] MoE path provided but this bench only wires the dense qwen35 backbone; skipping MoE cells."
else
    echo "[qwen36_mtp_e2e] MoE GGUF not on disk; skipping 35B-A3B AR and MTP cells."
fi

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OUTPUT_FILE="${RESULTS_DIR}/$(date +%Y%m%d-%H%M)-qwen36-mtp.json"

python3 - "${OUTPUT_FILE}" "${PROMPT_META}" "${AR_RUNS_FILE}" "${MTP_RUNS_FILE}" <<PY
import json
import math
import statistics
import sys

out_path, prompt_meta, ar_file, mtp_file = sys.argv[1:5]
prompts = json.load(open(prompt_meta))

def read_runs(path):
    return [float(x.strip()) for x in open(path) if x.strip()]

def cell(name, runs, baseline=None):
    med = statistics.median(runs)
    avg = sum(runs) / len(runs)
    sigma = math.sqrt(sum((x - avg) ** 2 for x in runs) / len(runs))
    obj = {
        "name": name,
        "tok_s_median": round(med, 2),
        "tok_s_stddev": round(sigma, 2),
        "runs": [round(x, 2) for x in runs],
        "gpu_temp_max_celsius": None,
    }
    if baseline:
        obj["speedup_vs_baseline"] = round(med / baseline, 2)
    return obj

ar = read_runs(ar_file)
mtp = read_runs(mtp_file)
ar_med = statistics.median(ar)
data = {
    "timestamp": "${TIMESTAMP}",
    "commit": "${COMMIT_SHA}",
    "gpu": {"model": "${GPU_INFO}", "driver_version": "${DRIVER_VERSION}"},
    "cuda_version": "${CUDA_VERSION}",
    "power_limit_watts": None if "${POWER_LIMIT_WATTS}" == "null" else float("${POWER_LIMIT_WATTS}"),
    "kv_type": "${KV_TYPE}",
    "build_type": "${BUILD_TYPE}",
    "ndebug_verified": ${NDEBUG_VERIFIED},
    "n_predict": int("${N_GEN}"),
    "prompts": [{k: p[k] for k in ("id", "name", "text", "tokens")} for p in prompts],
    "cells": [
        cell("27B Dense AR baseline", ar),
        cell("27B Dense + Qwen3.6 MTP gamma=1", mtp, ar_med),
        {"name": "35B-A3B AR baseline", "skipped": True, "reason": "MoE GGUF not on disk"},
        {"name": "35B-A3B + Qwen3.6 MTP gamma=1", "skipped": True, "reason": "MoE GGUF not on disk"},
    ],
}
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\\n")

gpu = data["gpu"]["model"]
power = data["power_limit_watts"]
power_text = f" @ {power:.0f}W" if power else ""
commit = data["commit"][:7] if data["commit"] != "unknown" else "unknown"
print("")
print(f"{gpu}{power_text}, KV {data['kv_type']}, commit {commit}, NDEBUG verified, n_predict={data['n_predict']}, 5 HE prompts x 3 runs median:")
print("")
print("| Configuration | tok/s | sigma | Speedup |")
print("|---|---:|---:|---:|")
for c in data["cells"]:
    if c.get("skipped"):
        print(f"| {c['name']} | skip | - | - |")
    else:
        speed = f"{c['speedup_vs_baseline']:.2f}x" if "speedup_vs_baseline" in c else "-"
        print(f"| {c['name']} | {c['tok_s_median']:.2f} | {c['tok_s_stddev']:.2f} | {speed} |")
print("")
print(f"[qwen36_mtp_e2e] JSON written to: {out_path}")
PY
