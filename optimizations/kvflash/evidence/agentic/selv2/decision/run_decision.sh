#!/bin/bash
# Decision-step eval (PREREG_decision.md). Caller holds /mnt/kimi-k3/build/3090.lock. One frozen binary, arms in prereg order.
set -u
F=/mnt/kimi-k3/build/pinuq-cuda-28806315
D=/mnt/kimi-k3/split-work/decision
TRAJ=/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl
TASKS=/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl
R=6144
export LD_LIBRARY_PATH=$F E1_BIN=$F/luce_server
md5sum $F/luce_server $F/*.so.0
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "GPU busy, re-poll in 60s"; sleep 60; done; }
compact() { (cd /mnt/c && /mnt/c/Windows/System32/wsl.exe -u root sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>&1 | tr -d '\000'); }
clean() { for v in $(env | grep -E '^(LUCE_)?KVFLASH_' | cut -d= -f1); do unset $v; done; }
ON6="LUCE_KVFLASH_LOG_RESIDENT=1 LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=6 LUCE_KVFLASH_SCORE_POOL_RECENT=1"
V2="$ON6 LUCE_KVFLASH_SCORER_DEVICE=gpu LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2"
declare -A ENV=( [FULL]="" [ON6]="$ON6" [V2]="$V2" [V2P]="$V2 LUCE_KVFLASH_PREAMBLE_ROWS=$R"
                 [V2PU]="$V2 LUCE_KVFLASH_PREAMBLE_ROWS=$R LUCE_KVFLASH_UQ_QUERIES=64" [OFFP]="LUCE_KVFLASH_LOG_RESIDENT=1 LUCE_KVFLASH_PREAMBLE_ROWS=$R" )
arm() {  # arm <name> <out_dir> [run_arm extra args...]
  local name=$1 out=$2; shift 2
  clean; idle; compact
  local mode=QK16KND; [ "$name" = FULL ] && mode=FULL
  echo "=== $(date -Is) $out start; env: ${ENV[$name]}"
  env ${ENV[$name]} python3 -I $D/agentic_eval/run_arm.py $mode "$@" --no-wait
  echo "=== $(date -Is) $out end rc=$? errored_rows=$(grep -c '"error"' $(ls $D/arms/$(basename $out)/*.jsonl))"
}
# Deviation 1 order: reference and floor first, then the primary pair, the probe guard, then the secondary arms
for a in FULL OFFP ON6 V2PU; do arm $a $D/arms/$a $D/decision_tasks.jsonl $TRAJ $D/arms/$a; done
for a in ON6 V2PU; do arm $a $D/arms/probe_$a $TASKS $TRAJ $D/arms/probe_$a --retained $D/retained_paged.json --probes-only; done
for a in V2 V2P; do arm $a $D/arms/$a $D/decision_tasks.jsonl $TRAJ $D/arms/$a; done
echo "=== $(date -Is) all arms done"
