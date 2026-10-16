#!/bin/bash
# Drift repro (README caveat 1): 08db p36 probes (rewound under ON), then f9b0 p12 probes (mode 4 skips; p12:7 drifted in the full run).
# Arms on one binary in one lock hold: off, on (k=2), noop (mode 4 rewind + re-prefill, no reselect).
# usage: bash run_drift.sh <luce_server>
set -u
D=/mnt/kimi-k3/split-work/rlru4
O=$D/drift
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
exec >> $O/run_drift.log 2>&1
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
arm() {  # <label>; env comes from the caller
  idle; echo "=== $(date -Is) $1 start"
  E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $O/$1 --retained $O/ids.json --probes-only --no-wait
  rc=$?; echo "=== $(date -Is) $1 end rc=$rc"; }
echo "=== $(date -Is) waiting for the 3090 lock"; md5sum "$BIN"
exec 9>/mnt/kimi-k3/build/3090.lock; flock 9
echo "=== $(date -Is) lock taken"
arm off
LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=2 arm on
LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_REWIND_NOOP=1 arm noop
echo "=== $(date -Is) done, lock released"
