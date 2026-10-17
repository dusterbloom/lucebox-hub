#!/bin/bash
# Selector v2 derisk (PREREG_selv2.md): ../rlru/run_dump.sh with LUCE_KVFLASH_EXACT_DUMP_MQ=64 on the selv2 binary.
# One probes-only arm over the same retained list, no reselect (answers = OFF).   usage: bash run_mq_dump.sh
set -u
D=/mnt/kimi-k3/split-work/selv2
H=/mnt/kimi-k3/split-work/rlru
BIN=/mnt/kimi-k3/build/selv2-cuda/luce_server
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
exec >> $D/run_mq_dump.log 2>&1
echo "=== $(date -Is) waiting for the 3090 lock"
flock /mnt/kimi-k3/build/3090.lock bash -c '
  set -u
  D='"$D"'; H='"$H"'; BIN='"$BIN"'; TASKS='"$TASKS"'; TRAJ='"$TRAJ"'
  echo "=== $(date -Is) lock taken"
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
  { echo "binary $BIN"; md5sum "$BIN"; md5sum $H/agentic_eval/*.py $H/retained.json $H/smoke_ids.json $D/*.py $D/*.sh $D/PREREG_selv2.md; } > $D/provenance_mq.txt
  export LUCE_KVFLASH_EXACT_DUMP_MQ=64
  # Smoke (rows not counted): 5 probes; one line per request in both dump files, or no full arm.
  mkdir -p $D/smoke/exact_dump
  LUCE_KVFLASH_EXACT_DUMP=$D/smoke/exact_dump E1_BIN=$BIN python3 -I $H/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/smoke --retained $H/smoke_ids.json --probes-only --no-wait
  NREQ=$(wc -l < $D/smoke/QK16KND.jsonl); NDUMP=$(grep -c mass_end $D/smoke/exact_dump/exact_dump.jsonl); NMQ=$(grep -c "\"n_q\"" $D/smoke/exact_dump/exact_dump_mq.jsonl)
  echo "smoke: $NREQ requests, $NDUMP dump lines, $NMQ mq lines"
  if [ "$NREQ" != "$NDUMP" ] || [ "$NREQ" != "$NMQ" ] || grep -q "\"error\"" $D/smoke/QK16KND.jsonl; then echo "=== $(date -Is) SMOKE FAILED"; exit 1; fi
  mkdir -p $D/dump/exact_dump
  export LUCE_KVFLASH_EXACT_DUMP=$D/dump/exact_dump
  echo "=== $(date -Is) dump arm start"
  E1_BIN=$BIN python3 -I $H/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ $D/dump --retained $H/retained.json --probes-only --no-wait
  echo "=== $(date -Is) dump arm end rc=$?"
'
echo "=== $(date -Is) lock released"
