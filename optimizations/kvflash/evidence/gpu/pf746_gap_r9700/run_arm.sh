#!/bin/bash
# usage: [KV_POOL=.. PASTE_CHARS=.. PASTE_IDS=.. KV_MAX_CTX=..] run_arm.sh ARM TAG
#   ARM A = merge binary, --prefill-compression auto   (kvflash/pf746-merge, the gap)
#   ARM B = merge+fix binary, --prefill-compression auto (kvflash/pf746-merge-fix)
#   ARM C = merge+fix binary, --prefill-compression off  (exact-answer reference)
# Conventions of ~/kvflash-gate/run_gate.sh: one server at a time (flock on ~/kvflash-gate/server.lock), R9700 only
# (HIP_VISIBLE_DEVICES=0), default KV cache type, temp 0 seed 1, nothing in /tmp (TMPDIR under ~/pf746/tmp).
set -u
ARM=$1; TAG=$2
P=$HOME/pf746; G=$HOME/kvflash-gate
case $ARM in A) BIN=$P/bin/luce_server-A;; B|C) BIN=$P/bin/luce_server-B;; *) echo "bad ARM" >&2; exit 2;; esac
D=$P/out/$TAG; mkdir -p $D $P/tmp
export TMPDIR=$P/tmp KV_ROOT=$D KV_BIN=$BIN HIP_VISIBLE_DEVICES=0
export KV_MODEL=$HOME/models/Qwen3.8-27B-UD-Q4_K_M.gguf KV_DRAFTER=$HOME/models/Qwen3.5-0.8B-BF16.gguf
exec 9>$G/server.lock; flock -n 9 || { echo "another gate holds the lock" >&2; exit 3; }
pgrep -x luce_server >/dev/null && { echo "a luce_server is already running" >&2; exit 3; }
{
  echo "arm=$ARM tag=$TAG date=$(date -Is) host=$(hostname)"
  echo "bin=$BIN md5=$(md5sum $BIN | cut -d' ' -f1)"; cat $BIN.prov
  echo "target=$KV_MODEL md5=$(md5sum $KV_MODEL | cut -d' ' -f1)"
  echo "drafter=$KV_DRAFTER md5=$(md5sum $KV_DRAFTER | cut -d' ' -f1)"
  echo "KV_POOL=${KV_POOL:-24576} KV_MAX_CTX=${KV_MAX_CTX:-65536} PASTE_CHARS=${PASTE_CHARS:-28000} PASTE_IDS=${PASTE_IDS:-} PASTE_SKIP=${PASTE_SKIP:-0}"
} | tee $D/provenance_$ARM.txt
cd $P/scripts && python3 -I pf746_gap.py $ARM 2>&1 | tee $D/client_$ARM.out
exit ${PIPESTATUS[0]}
