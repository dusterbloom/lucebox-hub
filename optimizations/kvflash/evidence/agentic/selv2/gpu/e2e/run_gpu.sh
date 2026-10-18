#!/bin/bash
# Selector v2 with the GPU scorer end to end (PREREG_gpu.md): one binary (selv2gpu-cuda) for every phase. Phases: shadow
# (gpu + LUCE_KVFLASH_SCORER_SHADOW=1 on the smoke ids), gpu (951 probes), on6ref (flags off, 951 probes, same session).
# The 3090 lock per phase; a failed arm or check stops the run.   usage: bash run_gpu.sh   (PHASES="gpu on6ref" to resume)
set -u
E=/mnt/kimi-k3/split-work/selv2/gpu
K=/mnt/kimi-k3/split-work/rlru4k/v2
R4=/mnt/kimi-k3/split-work/rlru4
BIN=/mnt/kimi-k3/build/selv2gpu-cuda/luce_server
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
if [ -z "${PHASE:-}" ]; then
  exec >> $E/run_gpu.log 2>&1
  if env | grep -qE '^(LUCE_)?KVFLASH_'; then echo "=== $(date -Is) inherited KVFLASH env, refusing:"; env | grep -E '^(LUCE_)?KVFLASH_'; exit 1; fi
  for P in ${PHASES:-shadow gpu on6ref}; do
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
gpu() { LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2 LUCE_KVFLASH_SCORER_DEVICE=gpu on6 "$@"; }
echo "=== $(date -Is) phase $PHASE: lock taken"
{ echo "phase $PHASE binary $BIN"; md5sum "$BIN"; } >> $E/provenance_gpu.txt
case $PHASE in
  shadow)
    { ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done
      md5sum $R4/agentic_eval/*.py $R4/retained.json $R4/smoke_ids4.json $K/resident_cmp.py $E/*.py $E/*.sh $E/PREREG_gpu.md \
        $E/../e2e/gate_e2e.py
      nvidia-smi --query-gpu=name,driver_version --format=csv; } >> $E/provenance_gpu.txt
    LUCE_KVFLASH_SCORER_SHADOW=1 gpu shadow_gpu $R4/smoke_ids4.json $E/shadow_pre || exit 1
    python3 -I $E/gate_gpu.py shadow $E/shadow_pre || { echo "=== SHADOW FAILED"; exit 1; }
    python3 -I $K/resident_cmp.py $K/smoke/off $E/shadow_pre || { echo "=== SHADOW STRUCTURAL CHECK FAILED"; exit 1; } ;;
  gpu)
    gpu gpu $R4/retained.json $E/v2 || exit 1
    python3 -I $E/gate_gpu.py gpu $E/v2 || { echo "=== GPU LOG GATE FAILED"; exit 1; }
    python3 -I $K/resident_cmp.py $K/off $E/v2 || { echo "=== STRUCTURAL CHECK FAILED on gpu"; exit 1; } ;;
  on6ref)
    on6 on6ref $R4/retained.json $E/on6ref || exit 1
    python3 -I $K/resident_cmp.py $K/off $E/on6ref || { echo "=== STRUCTURAL CHECK FAILED on on6ref"; exit 1; }
    python3 -I $E/gate_gpu.py ident $E/on6ref $E/../e2e/on6ref || { echo "=== FLAGS-OFF IDENTITY FAILED"; exit 1; } ;;
esac
