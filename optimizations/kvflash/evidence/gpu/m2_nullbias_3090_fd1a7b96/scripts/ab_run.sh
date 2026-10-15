#!/bin/bash
# usage: ab_run.sh <run name> <luce_server binary>
O=/mnt/kimi-k3/kvflash-gate-m2prefill
mkdir -p $O/$1 $O/tmp
export KV_EXTRA=--no-kv-cache TMPDIR=$O/tmp KV_ROOT=$O/$1 KV_BIN=$2 KV_MODEL=/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf KV_MAX_CTX=131072 KV_PORT=18090
python3 -I /mnt/kimi-k3/build/kvoff-parity/gate4_live.py A > $O/$1/driver.out 2>&1
echo exit=$?
