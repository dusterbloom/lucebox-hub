#!/bin/bash
# Exact-mass rank dump on the engine's own LRU state: one probes-only arm over the 951, no reselect (answers = OFF), with
# LUCE_KVFLASH_EXACT_DUMP writing mass_r / mass_end per request.   usage: bash run_dump.sh <luce_server>
set -u
D=/mnt/kimi-k3/split-work/rlru
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
exec >> $D/run_dump.log 2>&1
echo "=== $(date -Is) waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; BIN='"$BIN"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'
  echo "=== $(date -Is) lock taken"
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
  { echo "binary $BIN"; md5sum "$BIN"; md5sum $D/agentic_eval/*.py $D/retained.json; } > $D/provenance_dump.txt
  # Smoke (rows not counted): 5 probes; one dump line per request with both mass arrays, or no full arm.
  mkdir -p $D/smoke/exact_dump
  LUCE_KVFLASH_EXACT_DUMP=$D/smoke/exact_dump E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/smoke --retained $D/smoke_ids.json --probes-only --no-wait
  NREQ=$(wc -l < $D/smoke/QK16KND.jsonl); NDUMP=$(grep -c mass_end $D/smoke/exact_dump/exact_dump.jsonl)
  echo "smoke: $NREQ requests, $NDUMP dump lines"
  if [ "$NREQ" != "$NDUMP" ] || grep -q "\"error\"" $D/smoke/QK16KND.jsonl; then echo "=== $(date -Is) SMOKE FAILED"; exit 1; fi
  mkdir -p $D/dump/exact_dump
  export LUCE_KVFLASH_EXACT_DUMP=$D/dump/exact_dump
  echo "=== $(date -Is) dump arm start"
  E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/dump --retained $D/retained.json --probes-only --no-wait
  echo "=== $(date -Is) dump arm end rc=$?"
'
echo "=== $(date -Is) lock released"
