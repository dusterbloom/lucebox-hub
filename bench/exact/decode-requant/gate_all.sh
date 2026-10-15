#!/bin/bash
# Runs the candidate arms of the quality gate one after another (after any running luce_server has exited),
# scores each against the q8_0 baseline dump as soon as it finishes and deletes the candidate dump (1.2 GB each at
# 2400 positions; the box disk is near full). usage: gate_all.sh <outdir> <tail> <drop> <type>...
set -u
cd "$(dirname "$0")"
OUT=$1; TAIL=$2; DROP=$3; shift 3
while pgrep -f "luce_server --model" > /dev/null; do sleep 5; done
for t in "$@"; do
  bash qgate.sh "$t" "$OUT" "$TAIL" "$DROP" LUCE_DECODE_REQ="$t" > "$OUT/gate_$t.log" 2>&1
  echo "== $t vs q8_0"
  python3 kl_gate.py "$OUT/q8_0.bin" "$OUT/$t.bin" && rm -f "$OUT/$t.bin"
done
echo GATE_DONE
