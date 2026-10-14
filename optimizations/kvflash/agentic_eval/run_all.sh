#!/bin/bash
# Full run: FULL (all candidates) -> retained set from FULL alone -> POOL8K, POOL16K (retained probes + all decision
# points) -> FULLB control on the first 20 trajectories. Resumable (each arm skips finished ids).
# Usage: run_all.sh <out_dir>
set -u
D=/mnt/kimi-k3/datasets/agentic-eval-prepared
A=/mnt/kimi-k3/artifacts/agentic-eval
H=$(dirname "$(readlink -f "$0")")
OUT=${1:-$A/full}
mkdir -p "$OUT"
TASKS=$A/tasks_all60.jsonl
TRAJ=$D/trajectories.jsonl

gpu_empty() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
arm() {  # first arm of a session waits for the 2-poll idle check; later arms start at once if the GPU is empty
  local a=$1; shift
  if gpu_empty && [ -f "$OUT/.session_started" ]; then W=--no-wait; else W=""; fi
  touch "$OUT/.session_started"
  python3 -I "$H/run_arm.py" "$a" "$@" $W > "$OUT/$a.out" 2>&1
}

arm FULL "$TASKS" "$TRAJ" "$OUT"
python3 -I "$H/score.py" "$TASKS" "$OUT" "$OUT/score_full_only.json" --retained-out "$OUT/retained.json" --arms FULL > /dev/null
for a in POOL8K POOL16K; do
  arm $a "$TASKS" "$TRAJ" "$OUT" --retained "$OUT/retained.json"
done
head -20 "$TRAJ" > "$OUT/traj20.jsonl"
jq -r .trajectory_id "$OUT/traj20.jsonl" > "$OUT/traj20.ids"
jq -c --slurpfile ids <(jq -R . "$OUT/traj20.ids") 'select(.traj as $t | $ids | index($t))' "$TASKS" > "$OUT/tasks20.jsonl"
arm FULLB "$OUT/tasks20.jsonl" "$OUT/traj20.jsonl" "$OUT" --retained "$OUT/retained.json"
python3 -I "$H/score.py" "$TASKS" "$OUT" "$OUT/score.json" --arms FULL,POOL8K,POOL16K,FULLB > "$OUT/score.out"
echo ALL_DONE > "$OUT/DONE"
