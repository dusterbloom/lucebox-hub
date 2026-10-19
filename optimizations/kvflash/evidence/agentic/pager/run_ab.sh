#!/bin/bash
# Pager decode A/B (packed vs legacy copy path), same binary f619149b, ABBA blocks, compaction before each block.
# Question: is packed's -1.9 tok/s paired decode gap in full/ (packed ran right after a build, no compaction) real?
set -u
BIN=/mnt/kimi-k3/build/pager-cuda/luce_server
P=/mnt/kimi-k3/split-work/pager
R4=/mnt/kimi-k3/split-work/rlru4
for v in $(env | grep -E '^(LUCE_)?KVFLASH_' | cut -d= -f1); do unset $v; done
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
compact() { (cd /mnt/c && /mnt/c/Windows/System32/wsl.exe -u root sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>&1 | tr -d '\000'); }
md5sum $BIN
blk() {  # blk <name> [legacy]
  idle; compact; echo "=== $(date -Is) $1 start"
  env ${2:+LUCE_KVFLASH_PAGER_LEGACY=1} LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=6 \
    LUCE_KVFLASH_SCORE_POOL_RECENT=1 E1_BIN=$BIN python3 -I $R4/agentic_eval/run_arm.py QK16KND /mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl \
    /mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl $P/ab/$1 --retained $P/ab_retained.json --probes-only --no-wait
  echo "=== $(date -Is) $1 end rc=$?"; }
blk A1_packed; blk B1_legacy legacy; blk B2_legacy legacy; blk A2_packed
