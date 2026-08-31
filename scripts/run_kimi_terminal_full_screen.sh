#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 SCREEN_ROOT BUDGET24_BASELINE_ROOT ACTIVE_LAYER" >&2
    exit 2
fi

screen_root=$1
baseline_root=$2
active_layer=$3
pair_script=${KIMI_K3_PAIR_SCRIPT:-$(dirname "$0")/run_kimi_terminal_pair.sh}
binary=${KIMI_K3_BINARY:-/home/duster/k3-terminal-kl-bws-v2/server/build-terminal-kl-hip/smoke_kimi_k3_forward}

if [[ ! $active_layer =~ ^[0-9]+$ || $active_layer -lt 1 || $active_layer -gt 92 ]]; then
    echo "ACTIVE_LAYER must be an integer in [1, 92]" >&2
    exit 2
fi
for path in "$pair_script" "$binary" "$baseline_root/candidate-plan.tsv" "$baseline_root/SHA256SUMS"; do
    if [[ ! -f $path ]]; then
        echo "missing required file: $path" >&2
        exit 1
    fi
done
(cd "$baseline_root" && sha256sum -c SHA256SUMS >/dev/null)

mkdir -p "$screen_root/runs"
plan=$screen_root/screen-plan.tsv
plan_tmp=$(mktemp "$screen_root/.screen-plan.XXXXXX")
python3 - "$baseline_root/candidate-plan.tsv" "$active_layer" "$screen_root" > "$plan_tmp" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
layer = int(sys.argv[2])
root = Path(sys.argv[3])
with source.open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle, delimiter="\t")
            if int(row["model_layer"]) == layer]
if not rows:
    raise SystemExit(f"no rows for active layer {layer}: {source}")
terminal_pos = max(int(row["base_pos"]) for row in rows)
rows = [row for row in rows if int(row["base_pos"]) == terminal_pos]
if len(rows) != 16:
    raise SystemExit(f"expected 16 terminal routes, found {len(rows)}")
experts = [int(row["expert"]) for row in rows]
if len(set(experts)) != 16:
    raise SystemExit("terminal route experts are not unique")
selected = set()
for row in rows:
    expert = int(row["expert"])
    for slab in filter(None, row["selected_ranks"].split(",")):
        selected.add((expert, int(slab)))
if len(selected) != 24:
    raise SystemExit(f"expected Budget24 to select 24 records, found {len(selected)}")
print("action\ttarget\tartifact_dir")
for expert in experts:
    for slab in range(12):
        action = "drop" if (expert, slab) in selected else "force"
        target = f"{layer}:{expert}:{slab}"
        artifact = root / "runs" / f"{action}-l{layer}-e{expert}-s{slab}"
        print(f"{action}\t{target}\t{artifact}")
PY

if [[ -f $plan ]]; then
    if ! cmp -s "$plan_tmp" "$plan"; then
        echo "existing screen plan differs: $plan" >&2
        exit 1
    fi
    rm "$plan_tmp"
else
    mv "$plan_tmp" "$plan"
    sha256sum "$plan" > "$screen_root/screen-plan.sha256"
    sha256sum "$binary" > "$screen_root/binary.sha256"
    sha256sum "$pair_script" > "$screen_root/pair-script.sha256"
    git -C "$(dirname "$pair_script")/.." rev-parse HEAD > "$screen_root/source-commit.txt"
fi

sha256sum -c "$screen_root/screen-plan.sha256" >/dev/null
expected_binary=$(cut -d' ' -f1 "$screen_root/binary.sha256")
actual_binary=$(sha256sum "$binary" | cut -d' ' -f1)
if [[ $actual_binary != "$expected_binary" ]]; then
    echo "binary changed during resumable screen" >&2
    exit 1
fi
expected_pair_script=$(cut -d' ' -f1 "$screen_root/pair-script.sha256")
actual_pair_script=$(sha256sum "$pair_script" | cut -d' ' -f1)
if [[ $actual_pair_script != "$expected_pair_script" ]]; then
    echo "paired runner changed during resumable screen" >&2
    exit 1
fi

tail -n +2 "$plan" | while IFS=$'\t' read -r action target artifact_dir; do
    if [[ -d $artifact_dir ]]; then
        if [[ ! -f $artifact_dir/SHA256SUMS ]] ||
           ! (cd "$artifact_dir" && sha256sum -c SHA256SUMS >/dev/null); then
            echo "existing intervention is incomplete or corrupt: $artifact_dir" >&2
            exit 1
        fi
        continue
    fi
    KIMI_K3_ACTIVE_LAYER=$active_layer "$pair_script" "$artifact_dir" 24 "$action" "$target"
done

run_count=$(find "$screen_root/runs" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [[ $run_count -ne 192 ]]; then
    echo "expected 192 intervention directories, found $run_count" >&2
    exit 1
fi
if [[ -f $screen_root/COMPLETE ]]; then
    if [[ $(cat "$screen_root/COMPLETE") != $'complete\t192' ]]; then
        echo "invalid completion marker: $screen_root/COMPLETE" >&2
        exit 1
    fi
else
    printf 'complete\t192\n' > "$screen_root/COMPLETE"
fi
