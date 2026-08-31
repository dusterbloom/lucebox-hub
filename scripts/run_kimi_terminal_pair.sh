#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
    echo "usage: $0 ARTIFACT_DIR BUDGET [force|drop TARGETS] [force|drop TARGETS]" >&2
    exit 2
fi

artifact_dir=$1
budget=$2
action=${3:-}
target=${4:-}
action2=${5:-}
target2=${6:-}
model=${KIMI_K3_MODEL:-/home/duster/kimi-k3-deploy/p32-core/Kimi-K3-KDA-HYBRID-Q2-MIDLATE-00001-of-00014.gguf}
binary=${KIMI_K3_BINARY:-/home/duster/k3-terminal-kl-bws-v2/server/build-terminal-kl-hip/smoke_kimi_k3_forward}
prompt=${KIMI_K3_PROMPT:-According to all known laws}

if [[ -e $artifact_dir ]]; then
    echo "refusing existing artifact directory: $artifact_dir" >&2
    exit 1
fi
for pair in "$action:$target" "$action2:$target2"; do
    pair_action=${pair%%:*}
    pair_target=${pair#*:}
    if [[ $pair_action != "" && $pair_action != force && $pair_action != drop ]]; then
        echo "action must be force or drop" >&2
        exit 2
    fi
    if [[ -n $pair_action && -z $pair_target ]]; then
        echo "an intervention target is required" >&2
        exit 2
    fi
done

mkdir -p "$artifact_dir"
export HIP_VISIBLE_DEVICES=1
export DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96
export DFLASH_KIMI_CALIBRATED96_AUX_DIR=/home/duster/kimi-k3-deploy/aux
export DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=/home/duster/kimi-k3-deploy/streamed-bank/natural-sidecars
export DFLASH_KIMI_SIDECAR_AUTHORITATIVE=1
export DFLASH_KIMI_P20_SLAB_BUDGET=$budget
export DFLASH_KIMI_EXPERIMENT_ACTIVE_LAYER=92
export DFLASH_KIMI_EXPERIMENT_PAIRED_LOGITS_OUT=$artifact_dir/candidate-terminal.f32
export DFLASH_KIMI_LOGITS_OUT=$artifact_dir/exact-terminal.f32
export DFLASH_KIMI_EXPERIMENT_PLAN_OUT=$artifact_dir/candidate-plan.tsv
export DFLASH_KIMI_CALIBRATED96_METRICS_OUT=$artifact_dir/candidate-traffic.tsv
for pair in "$action:$target" "$action2:$target2"; do
    pair_action=${pair%%:*}
    pair_target=${pair#*:}
    if [[ $pair_action == force ]]; then
        export DFLASH_KIMI_EXPERIMENT_SLAB_FORCE=$pair_target
    elif [[ $pair_action == drop ]]; then
        export DFLASH_KIMI_EXPERIMENT_SLAB_DROP=$pair_target
    fi
done

printf '%s\n' "$prompt" > "$artifact_dir/prompt.txt"
env -0 | sort -z > "$artifact_dir/environment.nul"
printf '%s\0' /usr/bin/time -v "$binary" "$model" 0 1 "$prompt" > "$artifact_dir/command.nul"
/usr/bin/time -v "$binary" "$model" 0 1 "$prompt" \
    > "$artifact_dir/run.stdout" 2> "$artifact_dir/run.stderr"
sha256sum "$artifact_dir"/* > "$artifact_dir/SHA256SUMS"
