#!/bin/bash
# Smoke (a) failed on p36:7 (mode-3 no-op answer differs from OFF, 0 page events). Control: mode 2 with the same no-op tail has
# the same batch split (prefill to end-1, then a 1-token batch) but no restore or rewind. If it also differs from OFF, the cause
# is the batch split; if it matches OFF, the rewind is not exact.   usage: bash run_split_ctl.sh <luce_server>
set -u
D=/mnt/kimi-k3/split-work/rlru
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
exec >> $D/run_split_ctl.log 2>&1
echo "=== $(date -Is) waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; BIN='"$BIN"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
  echo "=== $(date -Is) lock taken"; md5sum "$BIN"
  LUCE_KVFLASH_PREFILL_RESELECT=2 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RESELECT_TAIL_CHUNKS=254 E1_BIN=$BIN \
    python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/smoke_split --retained $D/smoke_ids.json --probes-only --no-wait
  grep "prefill reselect" $D/smoke_split/QK16KND.server.log
  python3 -I $D/cmp_smoke.py $D/smoke_off/QK16KND.jsonl $D/smoke_split/QK16KND.jsonl $D/smoke_split/QK16KND.server.log
  python3 -I $D/cmp_smoke.py $D/smoke_noop/QK16KND.jsonl $D/smoke_split/QK16KND.jsonl $D/smoke_split/QK16KND.server.log
  echo "=== $(date -Is) done"
'
