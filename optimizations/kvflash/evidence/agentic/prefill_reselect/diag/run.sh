#!/bin/bash
# Diagnostic (not preregistered, no verdict): the engine's actual answer-time pool vs the replay, 67 probes (ids.json).
#   off = QK16KND; m1 = reselect at the last turn cut (the failed run); m2 = reselect at end-1; m2t128 = m2 with the last
#   128 chunks (8K tokens) protected. All with KVFLASH_DEBUG_RESIDENT=1.
# usage: bash run.sh <luce_server binary with mode 2>
set -u
D=/mnt/kimi-k3/split-work/prefill_reselect_diag
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
exec >> $D/run.log 2>&1
echo "=== $(date -Is) waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; BIN='"$BIN"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'
  echo "=== $(date -Is) lock taken"
  { echo "binary $BIN"; md5sum "$BIN"; md5sum $D/agentic_eval/*.py $D/ids.json; } > $D/provenance.txt
  export KVFLASH_DEBUG_RESIDENT=1
  for ARM in off m1 m2 m2t128; do
    until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
    echo "=== $(date -Is) arm $ARM start"
    unset LUCE_KVFLASH_PREFILL_RESELECT LUCE_KVFLASH_RESELECT_TAIL_CHUNKS
    case $ARM in m1) export LUCE_KVFLASH_PREFILL_RESELECT=1;; m2) export LUCE_KVFLASH_PREFILL_RESELECT=2;;
                 m2t128) export LUCE_KVFLASH_PREFILL_RESELECT=2 LUCE_KVFLASH_RESELECT_TAIL_CHUNKS=128;; esac
    E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/$ARM --retained $D/ids.json --probes-only --no-wait
    echo "=== $(date -Is) arm $ARM end rc=$?"
  done
'
echo "=== $(date -Is) lock released"
