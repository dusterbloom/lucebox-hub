#!/bin/bash
# Smoke (d): OFF on order A and order B (make_tasks.py), one lock hold, then the cmp_d.py gate. Exit code = the gate's.
# usage: bash run_d.sh <luce_server> <out dir>
set -u
S=/mnt/kimi-k3/split-work/rlru4k/smoke_d
BIN=$1; O=$2
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
mkdir -p $O
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
exec 9>/mnt/kimi-k3/build/3090.lock; flock 9
echo "=== $(date -Is) smoke d lock taken"; md5sum "$BIN"
for X in A B; do
  idle; echo "=== $(date -Is) order $X start"
  E1_BIN=$BIN python3 -I /mnt/kimi-k3/split-work/rlru4/agentic_eval/run_arm.py QK16KND $S/tasks.jsonl $TRAJ $O/$X --retained $S/ids_$X.json --probes-only --no-wait
  rc=$?; echo "=== $(date -Is) order $X end rc=$rc"
done
python3 -I $S/cmp_d.py $O/A/QK16KND.jsonl $O/B/QK16KND.jsonl
