#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 scalar|oracle ARTIFACT_DIR [ORACLE_IDS]" >&2
    exit 2
fi

mode=$1
root=$2
oracle_ids=${3:-}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
binary=${KIMI_K3_BINARY:-$repo/server/build-progressive-hip/smoke_kimi_k3_forward}
model=${KIMI_K3_MODEL:-/home/duster/kimi-k3-deploy/native-iq1s-slim-core/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}
prompt=${KIMI_K3_PROMPT_IDS:-/home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-prompt.ids}
budget=$repo/fixtures/k3_uniform_budget16.txt

case $mode in
scalar)
    [[ -z $oracle_ids ]] || { echo "scalar mode takes no oracle file" >&2; exit 2; }
    ;;
oracle)
    [[ -n $oracle_ids && -f $oracle_ids ]] || {
        echo "oracle mode requires an existing token file" >&2
        exit 2
    }
    ;;
*)
    echo "mode must be scalar or oracle" >&2
    exit 2
    ;;
esac

[[ ! -e $root ]] || { echo "refusing existing artifact directory: $root" >&2; exit 3; }
for path in "$binary" "$model" "$prompt" "$budget"; do
    [[ -f $path ]] || { echo "missing required file: $path" >&2; exit 4; }
done
[[ $(sha256sum "$prompt" | cut -d' ' -f1) == fd835f624d053f0f2da04114215461906430685463e99fb5e94cdf17115acbb0 ]] || {
    echo "frozen prompt hash mismatch" >&2
    exit 5
}
[[ $(sha256sum "$budget" | cut -d' ' -f1) == 4e40e7b834baf78152ff39a2f5cfec99fb5af52527633c123cb77cb0aa2a325e ]] || {
    echo "Budget16 policy hash mismatch" >&2
    exit 5
}
[[ $(wc -w < "$prompt") -eq 53 ]] || { echo "frozen prompt must have 53 IDs" >&2; exit 5; }
if [[ $mode == oracle ]]; then
    [[ $(wc -w < "$oracle_ids") -eq 18 ]] || { echo "oracle file must have 18 IDs" >&2; exit 5; }
fi

mkdir "$root"
git -C "$repo" rev-parse HEAD > "$root/source-commit.txt"
git -C "$repo" status --porcelain=v1 > "$root/source-status.txt"
[[ ! -s $root/source-status.txt ]] || { echo "source worktree is dirty" >&2; exit 6; }
sha256sum "$binary" > "$root/executable.sha256"
sha256sum "$model" "$prompt" "$budget" > "$root/inputs.sha256"
[[ $mode == scalar ]] || sha256sum "$oracle_ids" >> "$root/inputs.sha256"
uname -a > "$root/uname.txt"
free -h > "$root/memory-before.txt"
/opt/rocm/bin/rocm-smi -d 1 --showperflevel --showclocks > "$root/rocm-smi-before.txt" 2>&1 || true

export HIP_VISIBLE_DEVICES=1,0
export ROCBLAS_USE_HIPBLASLT=0
export DFLASH_KIMI_PRODUCTION_DEFAULTS=1
export DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96
export DFLASH_KIMI_CALIBRATED96_AUX_DIR=/home/duster/kimi-k3-deploy/aux
export DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=/home/duster/kimi-k3-deploy/streamed-bank/natural-sidecars
export DFLASH_KIMI_SIDECAR_AUTHORITATIVE=1
export DFLASH_KIMI_H22_LAYER_BUDGETS=$budget
export DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT=12
export DFLASH_KIMI_SMOKE_MAX_CTX=128
export DFLASH_KIMI_STAGE_PROFILE=1
export DFLASH_KIMI_CALIBRATED96_METRICS_OUT=$root/traffic.tsv
export DFLASH_KIMI_LOGITS_OUT=$root/final.f32
export DFLASH_KIMI_EXPERIMENT_STATE_OUT=$root/final.state.bin
unset DFLASH_KIMI_EXPERIMENT_CHUNKED_KDA || true
unset DFLASH_KIMI_EXPERIMENT_TOOL_REQUEST_B24 || true
unset DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE || true
if [[ $mode == oracle ]]; then
    export DFLASH_KIMI_EXPERIMENT_V16_ORACLE_TOKENS=$oracle_ids
else
    unset DFLASH_KIMI_EXPERIMENT_V16_ORACLE_TOKENS || true
fi

env -0 | sort -z > "$root/environment.nul"
command=(/usr/bin/time -v "$binary" "$model" 0 18 "@$prompt")
printf '%s\0' "${command[@]}" > "$root/command.nul"
"${command[@]}" > "$root/run.log" 2>&1

sed -n 's/^\[kimi-k3-smoke\] output_ids: //p' "$root/run.log" | tail -n 1 > "$root/output.ids"
[[ $(wc -w < "$root/output.ids") -eq 18 ]] || { echo "run did not emit 18 IDs" >&2; exit 7; }
if [[ $mode == oracle ]]; then
    cmp -s "$oracle_ids" "$root/output.ids" || { echo "oracle output IDs differ" >&2; exit 8; }
    grep -q '^\[kimi-k3-v16-oracle\] rows=16 ' "$root/run.log" || {
        echo "missing V16 oracle telemetry" >&2
        exit 8
    }
fi
for path in final.f32 final.state.bin traffic.tsv; do
    [[ -s $root/$path ]] || { echo "missing result file: $root/$path" >&2; exit 9; }
done

free -h > "$root/memory-after.txt"
/opt/rocm/bin/rocm-smi -d 1 --showperflevel --showclocks > "$root/rocm-smi-after.txt" 2>&1 || true
(
    cd "$root"
    sha256sum command.nul environment.nul executable.sha256 final.f32 \
        final.state.bin inputs.sha256 memory-after.txt memory-before.txt \
        output.ids rocm-smi-after.txt rocm-smi-before.txt run.log \
        source-commit.txt source-status.txt traffic.tsv uname.txt > SHA256SUMS
)
