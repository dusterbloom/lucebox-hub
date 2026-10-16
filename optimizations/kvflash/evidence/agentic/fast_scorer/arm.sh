#!/bin/bash
# usage: arm.sh <luce_server> <ids.json> <outdir> <k|off> <shadow|ref|none>   (caller holds the 3090 lock)
set -u
BIN=$1; IDS=$2; OUT=$3; K=$4; SH=$5
for v in $(compgen -e | grep -E '^(LUCE_)?KVFLASH_'); do unset "$v"; done
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done
mkdir -p "$OUT"
md5sum "$BIN" > "$OUT/provenance.txt"
ldd "$BIN" | grep -E "ggml|llama" | while read -r name _ path _; do md5sum "$path"; done >> "$OUT/provenance.txt"
echo "=== $(date -Is) start $OUT k=$K mode=$SH"
if [ "$K" != "off" ]; then
  export LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=$K
  [ "$SH" = shadow ] && export LUCE_KVFLASH_SCORER_SHADOW=1
  [ "$SH" = ref ] && export LUCE_KVFLASH_SCORER_REF=1
fi
E1_BIN=$BIN python3 -I /mnt/kimi-k3/build/scorer-run/agentic_eval/run_arm.py QK16KND $TASKS $TRAJ "$OUT" --retained "$IDS" --probes-only --no-wait
echo "=== $(date -Is) end rc=$?"
