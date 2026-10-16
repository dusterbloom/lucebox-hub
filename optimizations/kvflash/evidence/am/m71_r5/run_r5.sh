#!/bin/bash
# R5 on the GPU: default-off m71 vs main, within-pool probes (prompt <= 16320 tokens), --kvflash 16384 with no env.
# parity contract: same GGUF, same --max-ctx 52224, same KV type default, temp 0 seed 1, same template, same prefix-cache
# flags, same request order, same 16K pool; main has no disk KV cache by default, m71 gets --no-kv-cache.
set -u
D=/mnt/kimi-k3/kvflash-gate-m7/m71_r5
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
M71=/mnt/kimi-k3/build/m7-cuda/luce_server
MAIN=/mnt/kimi-k3/build/main-tip-cuda/luce_server
if [ -z "${ARM:-}" ]; then
  exec >> $D/run_r5.log 2>&1
  { for B in $MAIN $M71; do echo "binary $B"; md5sum $B; ldd $B | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done; done
    git -C /home/peppi/Dev/lucebox-builds/wt-main-tip log -1 --format="main %H"; git -C /home/peppi/Dev/lucebox-builds/wt-m7 log -1 --format="m71 %H"
    md5sum $D/run_arm_r5.py $D/ids_within_pool.json; } > $D/provenance.txt
  for A in R5MAIN R5M71; do
    echo "=== $(date -Is) $A: waiting for the 3090 lock"
    ARM=$A flock /mnt/kimi-k3/build/3090.lock bash "$0" || { echo "=== $(date -Is) $A failed"; exit 1; }
    echo "=== $(date -Is) $A: lock released"
  done
  exit 0
fi
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
if [ "$ARM" = R5MAIN ]; then BIN=$MAIN; else BIN=$M71; fi
echo "=== $(date -Is) $ARM start ($BIN)"
E1_BIN=$BIN python3 -I $D/run_arm_r5.py $ARM $TASKS $TRAJ $D/$ARM --retained $D/ids_within_pool.json --probes-only --no-wait
echo "=== $(date -Is) $ARM end rc=$?"
