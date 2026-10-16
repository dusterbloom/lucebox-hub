#!/bin/bash
# Mode-4 R-LRU answer arms (PREREG.md): smokes, then OFF and ON over the 951. One binary, one lock hold.
#   OFF = QK16KND (B0 POOL16K flags + LUCE_KVFLASH_POLICY=qk)
#   ON  = OFF + LUCE_KVFLASH_PREFILL_RESELECT=4 + LUCE_KVFLASH_RESELECT_SCORER=exact + LUCE_KVFLASH_RECALL_BLOCKS=<k>
# usage: bash run_arms4.sh <luce_server> <k>
set -u
D=/mnt/kimi-k3/split-work/rlru4
BIN=$1; K=$2
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
FIT=chatcmpl-905125c7406f0c2f85fa42f3bef6b39d:p12:0
exec >> $D/run_arms4.log 2>&1
echo "=== $(date -Is) mode 4 k $K, waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; BIN='"$BIN"'; K='"$K"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'; FIT='"$FIT"'
  idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
  arm() {  # <label> <retained ids> <out dir>; ON env comes from the caller
    idle; echo "=== $(date -Is) $1 start"
    E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $3 --retained $2 --probes-only --no-wait
    echo "=== $(date -Is) $1 end rc=$?"; }
  echo "=== $(date -Is) lock taken"
  idle
  { echo "binary $BIN mode 4 k $K"; md5sum "$BIN"; ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
    md5sum $D/agentic_eval/*.py $D/retained.json $D/smoke_ids4.json $D/*.py; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; } > $D/provenance_arms.txt
  # Smoke (a): rewind + re-prefill with no reselect must reproduce the first pass bit for bit; the fitting probe must skip and equal OFF.
  arm smoke_off $D/smoke_ids4.json $D/smoke/off
  LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_REWIND_NOOP=1 arm smoke_noop $D/smoke_ids4.json $D/smoke/noop
  python3 -I $D/cmp_smoke4.py $D/smoke/off/QK16KND.jsonl $D/smoke/noop/QK16KND.jsonl $D/smoke/noop/QK16KND.server.log $FIT || { echo "=== $(date -Is) SMOKE A FAILED, no arms run"; exit 1; }
  # Smoke (b): ON fires the exact scorer with k and re-prefills every time it reselects; no failure, no row errors.
  LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K arm smoke_on $D/smoke_ids4.json $D/smoke/on
  L=$D/smoke/on/QK16KND.server.log
  grep "prefill reselect" $L | tail -8
  if ! grep -q "mode=4 scorer=exact k=$K " $L || grep -q "prefill reselect.*failed" $L || grep -q "\"error\"" $D/smoke/on/QK16KND.jsonl ||
     [ "$(grep -c "mode=4 scorer=exact k=$K " $L)" != "$(grep -c "mode=4 @[0-9]*: re-prefill [0-9]* tokens ok" $L)" ]; then
    echo "=== $(date -Is) SMOKE B FAILED, no arms run"; exit 1; fi
  echo "=== $(date -Is) smokes ok"
  arm off $D/retained.json $D/off
  LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K arm on $D/retained.json $D/on
'
echo "=== $(date -Is) lock released"
