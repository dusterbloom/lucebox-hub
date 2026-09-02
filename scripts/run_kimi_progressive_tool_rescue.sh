#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
    echo "usage: $0 ARTIFACT_ROOT SERVER_BINARY POSITION_BUDGETS PORT [REQUEST_JSON] [SCHEMA_RESCUE_0_OR_1] [LAYER_BUDGET_POLICY]" >&2
    exit 2
fi

root=$1
binary=$2
position_budgets=$3
port=$4
model=${KIMI_K3_MODEL:-/home/duster/kimi-k3-deploy/p32-core/Kimi-K3-KDA-HYBRID-Q2-MIDLATE-00001-of-00014.gguf}
aux=/home/duster/kimi-k3-deploy/aux
sidecars=/home/duster/kimi-k3-deploy/streamed-bank/natural-sidecars
fixture=${5:-$(dirname "$0")/../fixtures/k3_tool_weather_request.json}
schema_rescue=${6:-0}
policy=${7:-/home/duster/lucebox-k3-b7b74cc/results/h23_10k_policies/h23_moonshot_1_2gib.txt}
teacher_tokens=${KIMI_K3_TEACHER_TOKEN_IDS:-}

if [[ -e $root ]]; then
    echo "artifact root already exists: $root" >&2
    exit 3
fi
for path in "$binary" "$model" "$policy" "$fixture"; do
    if [[ ! -f $path ]]; then
        echo "missing required file: $path" >&2
        exit 4
    fi
done
if [[ -n $teacher_tokens && ! -f $teacher_tokens ]]; then
    echo "missing teacher token IDs: $teacher_tokens" >&2
    exit 4
fi
for path in "$aux" "$sidecars"; do
    if [[ ! -d $path ]]; then
        echo "missing required directory: $path" >&2
        exit 4
    fi
done
if [[ ! $port =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
    echo "invalid port: $port" >&2
    exit 5
fi
if [[ $schema_rescue != 0 && $schema_rescue != 1 ]]; then
    echo "SCHEMA_RESCUE_0_OR_1 must be 0 or 1" >&2
    exit 5
fi

mkdir "$root"
cp "$fixture" "$root/request.json"
if [[ -n $teacher_tokens ]]; then
    cp "$teacher_tokens" "$root/teacher-token-ids.txt"
fi
repo_root=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
git -C "$repo_root" rev-parse HEAD > "$root/source-commit.txt"
git -C "$repo_root" status --porcelain=v1 > "$root/source-status.txt"
if [[ -s $root/source-status.txt ]]; then
    echo "source worktree is dirty" >&2
    exit 6
fi
sha256sum "$binary" > "$root/executable.sha256"
sha256sum "$model" "$policy" "$fixture" > "$root/inputs.sha256"
if [[ -n $teacher_tokens ]]; then
    sha256sum "$root/teacher-token-ids.txt" >> "$root/inputs.sha256"
fi
uname -a > "$root/uname.txt"
free -h > "$root/memory-before.txt"

export HIP_VISIBLE_DEVICES=1,0
export DFLASH_KIMI_PRODUCTION_DEFAULTS=${KIMI_K3_PRODUCTION_DEFAULTS:-1}
export DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96
export DFLASH_KIMI_CALIBRATED96_AUX_DIR=$aux
export DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=$sidecars
export DFLASH_KIMI_H22_LAYER_BUDGETS=$policy
export DFLASH_KIMI_SIDECAR_AUTHORITATIVE=1
export DFLASH_KIMI_STAGE_PROFILE=1
export DFLASH_SERVER_COMMITTED_TOKEN_TRACE=1
export DFLASH_KIMI_V8_FACTOR_CAPTURE=0
export DFLASH_SINGLE_CHAIN_ROLLBACK_DIAG=1
export DFLASH_KIMI_EXPERIMENT_TOOL_REQUEST_B24=1
export DFLASH_KIMI_CALIBRATED96_METRICS_OUT=$root/traffic.tsv
export DFLASH_KIMI_LOGITS_OUT=$root/final.f32
if [[ -n $position_budgets ]]; then
    export DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS=$position_budgets
else
    unset DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS || true
fi
if [[ $schema_rescue == 1 ]]; then
    export DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE=1
else
    unset DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE || true
fi
if [[ -n $teacher_tokens ]]; then
    export DFLASH_KIMI_EXPERIMENT_TEACHER_TOKEN_IDS=$root/teacher-token-ids.txt
else
    unset DFLASH_KIMI_EXPERIMENT_TEACHER_TOKEN_IDS || true
fi
env -0 > "$root/environment.nul"

command=(
    "$binary" "$model"
    --host 127.0.0.1 --port "$port" --model-name dflash
    --max-ctx 512 --max-tokens 256 --target-device hip:0
    --cache-type-k q4_0 --cache-type-v q4_0
    --prefix-cache-slots 0 --prefill-cache-slots 0 --chunk 512
)
printf '%s\0' "${command[@]}" > "$root/command.nul"

server_pid=
cleanup() {
    if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"${command[@]}" > "$root/server.stdout" 2> "$root/server.stderr" &
server_pid=$!
ready=0
for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/health" > "$root/health.json"; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server exited before readiness" >&2
        exit 7
    fi
    sleep 1
done
if (( ready == 0 )); then
    echo "server readiness timeout" >&2
    exit 8
fi

/usr/bin/time -f '%e\t%U\t%S\t%M' -o "$root/client.time.tsv" \
    curl -fsS -H 'Content-Type: application/json' \
    --data-binary "@$root/request.json" \
    "http://127.0.0.1:$port/v1/chat/completions" \
    > "$root/response.json"

kill -TERM "$server_pid"
wait "$server_pid" || true
server_pid=
free -h > "$root/memory-after.txt"

for path in final.f32 response.json traffic.tsv; do
    if [[ ! -s $root/$path ]]; then
        echo "missing or empty result file: $root/$path" >&2
        exit 9
    fi
done

(
    cd "$root"
    sha256sum client.time.tsv command.nul environment.nul executable.sha256 \
        final.f32 health.json inputs.sha256 memory-after.txt memory-before.txt \
        request.json response.json server.stderr server.stdout source-commit.txt \
        source-status.txt traffic.tsv uname.txt \
        ${teacher_tokens:+teacher-token-ids.txt} > SHA256SUMS
)
trap - EXIT
