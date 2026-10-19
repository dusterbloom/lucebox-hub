#!/bin/bash
# Smoke (not eval rows): V2PU flags with a generous preamble reserve, to read the preamble end and check uq hit/pin logs.
set -u
F=/mnt/kimi-k3/build/pinuq-cuda-28806315
D=/mnt/kimi-k3/split-work/decision
for v in $(env | grep -E '^(LUCE_)?KVFLASH_' | cut -d= -f1); do unset $v; done
export LD_LIBRARY_PATH=$F E1_BIN=$F/luce_server
export LUCE_KVFLASH_LOG_RESIDENT=1 LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RECALL_BLOCKS=6 LUCE_KVFLASH_RESELECT_SCORER=exact \
  LUCE_KVFLASH_SCORER_DEVICE=gpu LUCE_KVFLASH_SCORE_POOL_RECENT=1 LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2 \
  LUCE_KVFLASH_PREAMBLE_ROWS=8192 LUCE_KVFLASH_UQ_QUERIES=64
echo "=== $(date -Is) smoke start; env: $(env | grep -E '^LUCE_KVFLASH_' | sort | tr '\n' ' ')"
python3 -I $D/agentic_eval/run_arm.py QK16KND $D/smoke/decision_tasks.jsonl /mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl $D/smoke/v2pu --no-wait
echo "=== $(date -Is) smoke end rc=$?"
