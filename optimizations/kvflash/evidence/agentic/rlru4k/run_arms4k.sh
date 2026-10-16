#!/bin/bash
# Larger-k mode-4 arms (PREREG.md): smokes (a)-(c), OFF, gate (d), ON2, ON6, ON10; one binary; the 3090 lock per phase.
# A failed arm (nonzero run_arm.py exit or any errored row) stops the run.
# usage: bash run_arms4k.sh <luce_server>
set -u
D=/mnt/kimi-k3/split-work/rlru4k
R4=/mnt/kimi-k3/split-work/rlru4
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
FIT=chatcmpl-905125c7406f0c2f85fa42f3bef6b39d:p12:0
if [ -z "${PHASE:-}" ]; then
  exec >> $D/run_arms4k.log 2>&1
  if env | grep -qE '^(LUCE_)?KVFLASH_'; then echo "=== $(date -Is) inherited KVFLASH env, refusing:"; env | grep -E '^(LUCE_)?KVFLASH_'; exit 1; fi
  for P in smoke off gate_d on2 on6 on10; do
    echo "=== $(date -Is) phase $P: waiting for the 3090 lock"
    PHASE=$P flock /mnt/kimi-k3/build/3090.lock bash "$0" "$BIN" || { echo "=== $(date -Is) phase $P failed, no further phases"; exit 1; }
    echo "=== $(date -Is) phase $P: lock released"
  done
  exit 0
fi
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
arm() {  # <label> <retained ids> <out dir>; mode-4 env comes from the caller's prefix
  idle; echo "=== $(date -Is) $1 start; env: $(env | grep -E '^(LUCE_)?KVFLASH_' | sort | tr '\n' ' ')"
  E1_BIN=$BIN python3 -I $R4/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $3 --retained $2 --probes-only --no-wait
  rc=$?; n=$(grep -c '"error"' $3/QK16KND.jsonl); echo "=== $(date -Is) $1 end rc=$rc errored_rows=$n"
  [ $rc = 0 ] && [ $n = 0 ]; }
on() {  # <k> <label> <ids> <out>
  K=$1; shift; LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K arm "$@"; }
gate_b() {  # <k> <dir>: scorer line with k, one re-prefill ok per reselect, no failure
  L=$2/QK16KND.server.log
  grep -q "mode=4 scorer=exact k=$1 " $L && ! grep -q "prefill reselect.*failed" $L &&
    [ "$(grep -c "mode=4 scorer=exact k=$1 " $L)" = "$(grep -c "mode=4 @[0-9]*: re-prefill [0-9]* tokens ok" $L)" ]; }
echo "=== $(date -Is) phase $PHASE: lock taken"
{ echo "phase $PHASE binary $BIN"; md5sum "$BIN"; } >> $D/provenance_arms.txt
case $PHASE in
  smoke)
    { ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
      md5sum $R4/agentic_eval/*.py $R4/retained.json $R4/smoke_ids4.json $R4/drift/ids.json $D/*.py $D/*.sh; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; } >> $D/provenance_arms.txt
    # (a) noop rewind bit-exact; the fitting probe skips and equals OFF
    arm smoke_off $R4/smoke_ids4.json $D/smoke/off || exit 1
    LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_REWIND_NOOP=1 arm smoke_noop $R4/smoke_ids4.json $D/smoke/noop || exit 1
    python3 -I $R4/cmp_smoke4.py $D/smoke/off/QK16KND.jsonl $D/smoke/noop/QK16KND.jsonl $D/smoke/noop/QK16KND.server.log $FIT || { echo "=== SMOKE A FAILED"; exit 1; }
    # (b) every k fires and re-prefills
    for K in 2 6 10; do
      on $K smoke_on$K $R4/smoke_ids4.json $D/smoke/on$K || exit 1
      grep "prefill reselect" $D/smoke/on$K/QK16KND.server.log | tail -3
      gate_b $K $D/smoke/on$K || { echo "=== SMOKE B FAILED at k=$K"; exit 1; }
    done
    # (c) drift ids: OFF and ON2 agree on every request mode 4 skipped (OFF here is also smoke (d)'s reduced-history run)
    arm drift_off $R4/drift/ids.json $D/smoke/drift_off || exit 1
    on 2 drift_on2 $R4/drift/ids.json $D/smoke/drift_on2 || exit 1
    python3 -I $D/gate_cd.py c $D/smoke/drift_off $D/smoke/drift_on2 || { echo "=== SMOKE C FAILED"; exit 1; }
    echo "=== $(date -Is) smokes a-c ok" ;;
  off) arm off $R4/retained.json $D/off ;;
  gate_d) python3 -I $D/gate_cd.py d $D/smoke/drift_off $D/off || { echo "=== SMOKE D FAILED, no ON arm"; exit 1; } ;;
  on2) on 2 on2 $R4/retained.json $D/on2 ;;
  on6) on 6 on6 $R4/retained.json $D/on6 ;;
  on10) on 10 on10 $R4/retained.json $D/on10 ;;
esac
