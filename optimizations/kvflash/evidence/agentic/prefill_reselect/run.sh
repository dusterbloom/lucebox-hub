#!/bin/bash
# Prefill-end reselect, end to end on the 3090: two probes-only arms, one binary, one lock hold.
#   OFF = QK16KND (B0 POOL16K flags + LUCE_KVFLASH_POLICY=qk, no dump)
#   ON  = QK16KND + LUCE_KVFLASH_PREFILL_RESELECT=1
# usage: bash run.sh <luce_server binary>
set -u
D=/mnt/kimi-k3/split-work/prefill_reselect_run
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
mkdir -p /mnt/kimi-k3/build/tmp
exec >> $D/run.log 2>&1
echo "=== $(date -Is) waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; BIN='"$BIN"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'
  echo "=== $(date -Is) lock taken"
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
  { echo "binary $BIN"; md5sum "$BIN"; ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
    md5sum $D/agentic_eval/*.py $D/retained.json; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; } > $D/provenance.txt
  # Preflight (rows not counted): 5 ON probes; abort unless reselect fires with page events and no row errors.
  echo "=== $(date -Is) smoke start"
  LUCE_KVFLASH_PREFILL_RESELECT=1 E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/smoke --retained $D/smoke_ids.json --probes-only --no-wait
  grep "prefill reselect" $D/smoke/QK16KND.server.log | tail -8
  if ! grep -qE "prefill reselect @[0-9]+: [1-9]" $D/smoke/QK16KND.server.log || grep -q "\"error\"" $D/smoke/QK16KND.jsonl; then
    echo "=== $(date -Is) SMOKE FAILED, no arms run"; exit 1; fi
  echo "=== $(date -Is) smoke ok"
  for ARM in off on; do
    until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
    echo "=== $(date -Is) arm $ARM start"
    if [ $ARM = on ]; then export LUCE_KVFLASH_PREFILL_RESELECT=1; else unset LUCE_KVFLASH_PREFILL_RESELECT; fi
    E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/$ARM --retained $D/retained.json --probes-only --no-wait
    echo "=== $(date -Is) arm $ARM end rc=$?"
  done
'
echo "=== $(date -Is) lock released"
