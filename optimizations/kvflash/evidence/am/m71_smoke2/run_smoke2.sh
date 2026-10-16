#!/bin/bash
# M7.1 smoke 2 (idle fit). Same layout as smoke 1. Arms: AMOFF and AMOFF2 (AM off twice: the noise band), AMON (ratio
# 0.20 select-only, 4 s client think time between requests, the server idle in between), AMON0 (no think time: a
# request arrives while the idle pass starts). Think time is the same sleep on the AM-off arms. One server at a time,
# each arm under the 3090 lock; --no-kv-cache; KVFLASH_* and LUCE_KVFLASH_* cleared by the runner.
set -u
D=/mnt/kimi-k3/kvflash-gate-m7/m71_smoke2
BIN=/mnt/kimi-k3/build/m7-cuda/luce_server
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
if [ -z "${ARM:-}" ]; then
  exec >> $D/run_smoke2.log 2>&1
  { echo "binary $BIN"; md5sum $BIN; ldd $BIN | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
    git -C /home/peppi/Dev/lucebox-builds/wt-m7 log -1 --format="commit %H"; md5sum $D/run_arm_smoke2.py $D/ids.json; } > $D/provenance.txt
  for A in AMOFF AMON AMOFF2 AMON0; do
    echo "=== $(date -Is) $A: waiting for the 3090 lock"
    ARM=$A flock /mnt/kimi-k3/build/3090.lock bash "$0" || { echo "=== $(date -Is) $A failed"; exit 1; }
    echo "=== $(date -Is) $A: lock released"
  done
  exit 0
fi
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
T=4; if [ "$ARM" = AMON0 ]; then T=0; fi
echo "=== $(date -Is) $ARM start (think ${T}s)"
AE_THINK_S=$T E1_BIN=$BIN python3 -I $D/run_arm_smoke2.py $ARM $TASKS $TRAJ $D/$ARM --retained $D/ids.json --probes-only --no-wait
echo "=== $(date -Is) $ARM end rc=$?"
