#!/bin/bash
# Selector v2 end to end (PREREG_e2e.md): the selv2 binary in mode 4 at k=6, exact scorer, recency pool; v2 adds
# LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2. Phases: shadow (flags off = prs8 ON6 on the smoke ids),
# smoke (v2 on every reselect), v2 (951 probes), on6ref (flags off, 951 probes, same binary and session).
# The 3090 lock per phase; a failed arm, smoke or check stops the run.   usage: bash run_e2e.sh
set -u
E=/mnt/kimi-k3/split-work/selv2/e2e
K=/mnt/kimi-k3/split-work/rlru4k/v2
R4=/mnt/kimi-k3/split-work/rlru4
BIN=/mnt/kimi-k3/build/selv2-cuda/luce_server
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
if [ -z "${PHASE:-}" ]; then
  exec >> $E/run_e2e.log 2>&1
  if env | grep -qE '^(LUCE_)?KVFLASH_'; then echo "=== $(date -Is) inherited KVFLASH env, refusing:"; env | grep -E '^(LUCE_)?KVFLASH_'; exit 1; fi
  for P in shadow smoke v2 on6ref; do
    echo "=== $(date -Is) phase $P: waiting for the 3090 lock"
    PHASE=$P flock /mnt/kimi-k3/build/3090.lock bash "$0" || { echo "=== $(date -Is) phase $P failed, no further phases"; exit 1; }
    echo "=== $(date -Is) phase $P: lock released"
  done
  exit 0
fi
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
arm() {  # <label> <retained ids> <out dir>; env comes from the caller's prefix
  idle; echo "=== $(date -Is) $1 start; env: $(env | grep -E '^(LUCE_)?KVFLASH_' | sort | tr '\n' ' ')"
  E1_BIN=$BIN python3 -I $R4/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $3 --retained $2 --probes-only --no-wait
  rc=$?; n=$(grep -c '"error"' $3/QK16KND.jsonl); echo "=== $(date -Is) $1 end rc=$rc errored_rows=$n"
  [ $rc = 0 ] && [ $n = 0 ]; }
on6() { LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=6 \
    LUCE_KVFLASH_SCORE_POOL_RECENT=1 LUCE_KVFLASH_LOG_RESIDENT=1 arm "$@"; }
sel2() { LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2 on6 "$@"; }
echo "=== $(date -Is) phase $PHASE: lock taken"
{ echo "phase $PHASE binary $BIN"; md5sum "$BIN"; } >> $E/provenance_e2e.txt
case $PHASE in
  shadow)
    { ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
      md5sum $R4/agentic_eval/*.py $R4/retained.json $R4/smoke_ids4.json $K/resident_cmp.py $E/*.py $E/*.sh $E/PREREG_e2e.md
      nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; } >> $E/provenance_e2e.txt
    on6 shadow_on6 $R4/smoke_ids4.json $E/shadow/on6 || exit 1
    python3 -I $E/gate_e2e.py shadow $K/smoke/on6 $E/shadow/on6 || { echo "=== SHADOW FAILED"; exit 1; } ;;
  smoke)
    sel2 smoke_v2 $R4/smoke_ids4.json $E/smoke/v2 || exit 1
    grep -E "selector v2|prefill reselect (mode=4 scorer|picked units)" $E/smoke/v2/QK16KND.server.log | tail -6
    python3 -I $E/gate_e2e.py v2 $E/smoke/v2 || { echo "=== SMOKE FAILED"; exit 1; }
    python3 -I $K/resident_cmp.py $K/smoke/off $E/smoke/v2 || { echo "=== SMOKE STRUCTURAL CHECK FAILED"; exit 1; } ;;
  v2)
    sel2 v2 $R4/retained.json $E/v2 || exit 1
    python3 -I $E/gate_e2e.py v2 $E/v2 || { echo "=== V2 LOG GATE FAILED"; exit 1; }
    python3 -I $K/resident_cmp.py $K/off $E/v2 || { echo "=== STRUCTURAL CHECK FAILED on v2"; exit 1; } ;;
  on6ref)
    on6 on6ref $R4/retained.json $E/on6ref || exit 1
    python3 -I $K/resident_cmp.py $K/off $E/on6ref || { echo "=== STRUCTURAL CHECK FAILED on on6ref"; exit 1; } ;;
esac
