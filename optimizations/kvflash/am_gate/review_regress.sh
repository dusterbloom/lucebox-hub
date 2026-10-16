#!/bin/bash
# usage: review_regress.sh <binary> <outdir> [scenario...]  -- takes the 3090 lock itself
[ -n "$M7_LOCKED" ] || exec flock /mnt/kimi-k3/build/3090.lock env M7_LOCKED=1 "$0" "$@"
BIN=$1; OUT=$2; shift 2
G=$(dirname "$(readlink -f "$0")")
export TMPDIR=$OUT/tmp KV_BIN=$BIN KV_MODEL=/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf KV_MAX_CTX=131072 KV_PORT=18092 KV_EXTRA=""
mkdir -p $TMPDIR
for t in "${@:-hook draft oversize bypass}"; do
  for s in $t; do
    export KV_ROOT=$OUT/$s; mkdir -p $KV_ROOT
    python3 -I $G/review_regress.py $s > $KV_ROOT/result.txt 2>&1; rc=$?; echo "$s exit=$rc"; [ $rc -eq 0 ] || fail=1
  done
done
exit ${fail:-0}
