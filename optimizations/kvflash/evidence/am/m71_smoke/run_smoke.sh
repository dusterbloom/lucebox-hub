#!/bin/bash
# M7.1 smoke (not scored, no accuracy claim): AM off vs AM on (ratio 0.20, select-only, queries later), same binary,
# QK16KND 16K pool config, 3 longest trajectories (57 probes), --no-kv-cache, KVFLASH_* and LUCE_KVFLASH_* cleared by
# run_arm_smoke.py. One server at a time, each arm under the 3090 lock.
set -u
D=/mnt/kimi-k3/kvflash-gate-m7/m71_smoke
BIN=/mnt/kimi-k3/build/m7-cuda/luce_server
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
if [ -z "${ARM:-}" ]; then
  exec >> $D/run_smoke.log 2>&1
  { echo "binary $BIN"; md5sum $BIN; ldd $BIN | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
    git -C /home/peppi/Dev/lucebox-builds/wt-m7 log -1 --format="commit %H"; md5sum $D/run_arm_smoke.py $D/ids.json; } > $D/provenance.txt
  for A in AMOFF AMON; do
    echo "=== $(date -Is) $A: waiting for the 3090 lock"
    ARM=$A flock /mnt/kimi-k3/build/3090.lock bash "$0" || { echo "=== $(date -Is) $A failed"; exit 1; }
    echo "=== $(date -Is) $A: lock released"
  done
  exit 0
fi
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
echo "=== $(date -Is) $ARM start"
E1_BIN=$BIN python3 -I $D/run_arm_smoke.py $ARM $TASKS $TRAJ $D/$ARM --retained $D/ids.json --probes-only --no-wait
echo "=== $(date -Is) $ARM end rc=$?"
