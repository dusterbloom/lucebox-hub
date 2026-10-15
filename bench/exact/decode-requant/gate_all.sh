#!/bin/bash
# Runs the candidate arms of the quality gate one after another (after any running luce_server has exited),
# scores each against the q8_0 baseline dump as soon as it finishes and deletes the candidate dump (1.2 GB each at
# 2400 positions; the box disk is near full). An arm is <gdn_type> or <gdn_type>:<attn_type>
# (LUCE_DECODE_REQ_ATTN; 0 = attention layers stay exact).
# usage: gate_all.sh <outdir> <tail> <drop> <arm>...
set -u
cd "$(dirname "$0")"
OUT=$1; TAIL=$2; DROP=$3; shift 3
while pgrep -f "luce_server --model" > /dev/null; do sleep 5; done
for arm in "$@"; do
  t=${arm%%:*}; set_env=""
  [ "$arm" != "$t" ] && set_env="LUCE_DECODE_REQ_ATTN=${arm#*:}"
  name=${arm/:/_}
  bash qgate.sh "$name" "$OUT" "$TAIL" "$DROP" LUCE_DECODE_REQ="$t" $set_env > "$OUT/gate_$name.log" 2>&1
  echo "== $name vs q8_0"
  python3 kl_gate.py "$OUT/q8_0.bin" "$OUT/$name.bin" && rm -f "$OUT/$name.bin"
done
echo GATE_DONE
