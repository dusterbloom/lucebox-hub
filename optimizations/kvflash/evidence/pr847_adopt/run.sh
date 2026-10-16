#!/bin/bash
# usage: run.sh   -- both arms under the 3090 lock, results in /mnt/kimi-k3/build/pr847_adopt/<arm>/
[ -n "$LOCKED" ] || exec flock /mnt/kimi-k3/build/3090.lock env LOCKED=1 "$0" "$@"
O=/mnt/kimi-k3/build/pr847_adopt
export TMPDIR=$O/tmp KV_MODEL=/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf KV_MAX_CTX=131072 KV_PORT=18093 KV_EXTRA=""
mkdir -p $TMPDIR
for arm in 7281314f c0a311ad; do
  export KV_ROOT=$O/$arm KV_BIN=$O/luce_server_$arm
  mkdir -p $KV_ROOT
  { echo "start $(date -Is)"; echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"; md5sum $KV_BIN; } > $KV_ROOT/provenance.txt
  python3 -I $O/repro847.py $arm > $KV_ROOT/driver.out 2>&1
  echo "$arm exit=$?" >> $KV_ROOT/provenance.txt
  echo "end $(date -Is)" >> $KV_ROOT/provenance.txt
done
