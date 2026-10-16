#!/bin/bash
# rlru4k-v2 (PREREG_v2.md): mode 4 with the recency scoring pool, on prs8 for every arm; the 3090 lock per phase.
# Phases: shadow (prs8 flag off reproduces prs7's picks), smoke (a, b, c, e), off, gate_d, on2, on6, on10.
# Every ON phase ends with the structural check (resident set at each request's first prefill batch == OFF's).
# A failed arm, smoke or check stops the run.
# usage: bash run_v2.sh <prs8 luce_server>
set -u
D=/mnt/kimi-k3/split-work/rlru4k/v2
K4=/mnt/kimi-k3/split-work/rlru4k
R4=/mnt/kimi-k3/split-work/rlru4
BIN=$1
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
FIT=chatcmpl-905125c7406f0c2f85fa42f3bef6b39d:p12:0
if [ -z "${PHASE:-}" ]; then
  exec >> $D/run_v2.log 2>&1
  if env | grep -qE '^(LUCE_)?KVFLASH_'; then echo "=== $(date -Is) inherited KVFLASH env, refusing:"; env | grep -E '^(LUCE_)?KVFLASH_'; exit 1; fi
  for P in shadow smoke off gate_d on2 on6 on10; do
    echo "=== $(date -Is) phase $P: waiting for the 3090 lock"
    PHASE=$P flock /mnt/kimi-k3/build/3090.lock bash "$0" "$BIN" || { echo "=== $(date -Is) phase $P failed, no further phases"; exit 1; }
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
off() { LUCE_KVFLASH_LOG_RESIDENT=1 arm "$@"; }
on() {  # <k> <label> <ids> <out>: the v2 ON arm (flag on, resident log on)
  K=$1; shift; LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K \
    LUCE_KVFLASH_SCORE_POOL_RECENT=1 LUCE_KVFLASH_LOG_RESIDENT=1 arm "$@"; }
on_old() {  # <k> <label> <ids> <out>: prs7's ON arm exactly (flag off), for the shadow check
  K=$1; shift; LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K arm "$@"; }
gate_b() {  # <k> <dir>: scorer line with k, one re-prefill ok per reselect, no failure,
  # and a recency line on every pooled request (pooled = overflowing; both lines are logged under kvf_paged)
  L=$2/QK16KND.server.log
  grep -q "mode=4 scorer=exact k=$1 " $L && ! grep -q "prefill reselect.*failed" $L &&
    [ "$(grep -c "mode=4 scorer=exact k=$1 " $L)" = "$(grep -c "mode=4 @[0-9]*: re-prefill [0-9]* tokens ok" $L)" ] &&
    [ "$(grep -c "prefill reselect recent-pool @" $L)" = "$(grep -c "\[kvflash\] resident @" $L)" ]; }
echo "=== $(date -Is) phase $PHASE: lock taken"
{ echo "phase $PHASE binary $BIN"; md5sum "$BIN"; } >> $D/provenance_v2.txt
case $PHASE in
  shadow)
    { ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
      md5sum $R4/agentic_eval/*.py $R4/retained.json $R4/smoke_ids4.json $R4/drift/ids.json $K4/diag27.json $K4/gate_cd.py $D/*.py $D/*.sh
      nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; } >> $D/provenance_v2.txt
    # prs8 with the flag off must pick exactly what prs7 picked (fast scorer + default-off recency are pick-neutral)
    for K in 2 6 10; do
      on_old $K shadow_on$K $R4/smoke_ids4.json $D/shadow/on$K || exit 1
      python3 -I $D/shadow_cmp.py $K4/smoke/on$K $D/shadow/on$K || { echo "=== SHADOW FAILED at k=$K"; exit 1; }
    done
    on_old 2 shadow_drift_on2 $R4/drift/ids.json $D/shadow/drift_on2 || exit 1
    python3 -I $D/shadow_cmp.py $K4/smoke/drift_on2 $D/shadow/drift_on2 || { echo "=== SHADOW FAILED on drift ids"; exit 1; }
    echo "=== $(date -Is) shadow ok" ;;
  smoke)
    # (a) noop rewind bit-exact under the flag; the fitting probe skips and equals OFF
    off smoke_off $R4/smoke_ids4.json $D/smoke/off || exit 1
    LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_REWIND_NOOP=1 LUCE_KVFLASH_SCORE_POOL_RECENT=1 \
      LUCE_KVFLASH_LOG_RESIDENT=1 arm smoke_noop $R4/smoke_ids4.json $D/smoke/noop || exit 1
    python3 -I $R4/cmp_smoke4.py $D/smoke/off/QK16KND.jsonl $D/smoke/noop/QK16KND.jsonl $D/smoke/noop/QK16KND.server.log $FIT || { echo "=== SMOKE A FAILED"; exit 1; }
    python3 -I $D/resident_cmp.py $D/smoke/off $D/smoke/noop || { echo "=== SMOKE E FAILED (noop)"; exit 1; }
    # (b) every k fires, re-prefills, logs the recency step; (e) resident sets equal OFF's
    for K in 2 6 10; do
      on $K smoke_on$K $R4/smoke_ids4.json $D/smoke/on$K || exit 1
      grep -E "prefill reselect (mode=4 scorer|recent-pool)" $D/smoke/on$K/QK16KND.server.log | tail -4
      gate_b $K $D/smoke/on$K || { echo "=== SMOKE B FAILED at k=$K"; exit 1; }
      python3 -I $D/resident_cmp.py $D/smoke/off $D/smoke/on$K || { echo "=== SMOKE E FAILED at k=$K"; exit 1; }
    done
    # (c) drift ids: OFF and ON2 agree on every request mode 4 skipped
    off drift_off $R4/drift/ids.json $D/smoke/drift_off || exit 1
    on 2 drift_on2 $R4/drift/ids.json $D/smoke/drift_on2 || exit 1
    python3 -I $K4/gate_cd.py c $D/smoke/drift_off $D/smoke/drift_on2 $D/smoke/drift_on2 || { echo "=== SMOKE C FAILED"; exit 1; }
    python3 -I $D/resident_cmp.py $D/smoke/drift_off $D/smoke/drift_on2 || { echo "=== SMOKE E FAILED (drift)"; exit 1; }
    echo "=== $(date -Is) smokes a, b, c, e ok" ;;
  off) off off $R4/retained.json $D/off ;;
  gate_d) python3 -I $K4/gate_cd.py d $D/smoke/drift_off $D/off $D/smoke/drift_on2 || { echo "=== SMOKE D FAILED, no ON arm"; exit 1; } ;;
  on2|on6|on10)
    K=${PHASE#on}
    on $K $PHASE $R4/retained.json $D/$PHASE || exit 1
    python3 -I $D/resident_cmp.py $D/off $D/$PHASE || { echo "=== STRUCTURAL CHECK FAILED on $PHASE"; exit 1; } ;;
esac
