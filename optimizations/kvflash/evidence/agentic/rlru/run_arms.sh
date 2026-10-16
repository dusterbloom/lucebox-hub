#!/bin/bash
# R-LRU answer arms (PREREG.md step 2): smokes, then OFF and ON over the 951. One binary, one lock hold.
#   OFF = QK16KND (B0 POOL16K flags + LUCE_KVFLASH_POLICY=qk)
#   ON  = OFF + LUCE_KVFLASH_PREFILL_RESELECT=<mode> + LUCE_KVFLASH_RESELECT_SCORER=exact + LUCE_KVFLASH_RECALL_BLOCKS=<k>
# usage: bash run_arms.sh <luce_server> <mode 1|3> <k>
set -u
D=/mnt/kimi-k3/split-work/rlru
BIN=$1; MODE=$2; K=$3
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
exec >> $D/run_arms.log 2>&1
echo "=== $(date -Is) mode $MODE k $K, waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; BIN='"$BIN"'; MODE='"$MODE"'; K='"$K"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'
  idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
  arm() {  # <label> <retained ids> <out dir>; ON env comes from the caller
    idle; echo "=== $(date -Is) $1 start"
    E1_BIN=$BIN python3 -I $D/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $3 --retained $2 --probes-only --no-wait
    echo "=== $(date -Is) $1 end rc=$?"; }
  on() { LUCE_KVFLASH_PREFILL_RESELECT=$MODE LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K arm "$@"; }
  echo "=== $(date -Is) lock taken"
  idle
  { echo "binary $BIN mode $MODE k $K"; md5sum "$BIN"; ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
    md5sum $D/agentic_eval/*.py $D/retained.json $D/smoke_ids.json; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; } > $D/provenance_arms.txt
  # Smokes (rows not counted). (a) mode 3 only: rewind + re-run with a no-op reselect must reproduce OFF exactly.
  if [ "$MODE" = 3 ]; then
    arm smoke_off $D/smoke_ids.json $D/smoke_off
    LUCE_KVFLASH_PREFILL_RESELECT=3 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RESELECT_TAIL_CHUNKS=254 arm smoke_noop $D/smoke_ids.json $D/smoke_noop
    python3 -I $D/cmp_smoke.py $D/smoke_off/QK16KND.jsonl $D/smoke_noop/QK16KND.jsonl $D/smoke_noop/QK16KND.server.log || { echo "=== $(date -Is) SMOKE A FAILED, no arms run"; exit 1; }
  fi
  # (b) ON fires the exact scorer with k, no reselect failure, no row errors.
  on smoke_on $D/smoke_ids.json $D/smoke_on
  L=$D/smoke_on/QK16KND.server.log
  grep "prefill reselect" $L | tail -8
  if ! grep -q "scorer=exact k=$K " $L || grep -q "prefill reselect.*failed" $L || grep -q "\"error\"" $D/smoke_on/QK16KND.jsonl ||
     { [ "$MODE" = 3 ] && [ "$(grep -c "rewind 1 token" $L)" != "$(grep -c "rerun ok" $L)" ]; }; then
    echo "=== $(date -Is) SMOKE B FAILED, no arms run"; exit 1; fi
  echo "=== $(date -Is) smokes ok"
  arm off $D/retained.json $D/off
  on on $D/retained.json $D/on
'
echo "=== $(date -Is) lock released"
