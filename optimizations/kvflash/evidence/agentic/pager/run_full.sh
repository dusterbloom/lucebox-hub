#!/bin/bash
# usage: run_full.sh   (caller holds the 3090 lock). Arms in order: packed, then legacy; same binary, on6ref env.
set -u
BIN=/mnt/kimi-k3/build/pager-cuda/luce_server
OUT=/mnt/kimi-k3/split-work/pager/full
K=/mnt/kimi-k3/split-work/rlru4k/v2
R4=/mnt/kimi-k3/split-work/rlru4
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
for v in $(env | grep -E '^(LUCE_)?KVFLASH_' | cut -d= -f1); do unset $v; done
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
md5sum $BIN
arm() {
  idle; echo "=== $(date -Is) $1 start; env: $(env | grep -E '^(LUCE_)?KVFLASH_' | sort | tr '\n' ' ')"
  E1_BIN=$BIN python3 -I $R4/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $OUT/$1 --retained $R4/retained.json --probes-only --no-wait
  echo "=== $(date -Is) $1 end rc=$? errored_rows=$(grep -c '"error"' $OUT/$1/QK16KND.jsonl)"
  python3 -I $K/resident_cmp.py $K/off $OUT/$1 > $OUT/$1.resident_cmp.txt 2>&1; echo "=== resident_cmp $1 rc=$?"; }
on6() { LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=6 \
    LUCE_KVFLASH_SCORE_POOL_RECENT=1 LUCE_KVFLASH_LOG_RESIDENT=1 arm "$@"; }
on6 packed
LUCE_KVFLASH_PAGER_LEGACY=1 on6 legacy
