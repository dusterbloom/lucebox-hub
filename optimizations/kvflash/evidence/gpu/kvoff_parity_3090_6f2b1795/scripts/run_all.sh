#!/bin/bash
# Runs main, E, main, E. usage: flock /mnt/kimi-k3/build/3090.lock run_all.sh
O=/mnt/kimi-k3/kvflash-gate-kvoff-6f2b1795
B=/mnt/kimi-k3/build/diskrestore-repro/bin
S=$(dirname "$(readlink -f "$0")")
mkdir -p $O/tmp
run() { # name binary
  mkdir -p $O/$1 $O/$1/kvcache
  echo "== $1 idle check: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" | tee -a $O/idle.txt
  KV_EXTRA=--no-kv-cache TMPDIR=$O/tmp KV_ROOT=$O/$1 KV_BIN=$2 KV_CACHE_DIR=$O/$1/kvcache \
  KV_MODEL=/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf KV_MAX_CTX=131072 KV_PORT=18190 \
  python3 -I $S/gate4_live.py A > $O/$1/driver.out 2>&1
  echo "$1 exit=$?" | tee -a $O/idle.txt
}
run main-run1 $B/luce_server-main-6f2b1795
run E-run1 $B/luce_server-E-09bde623-gate
run main-run2 $B/luce_server-main-6f2b1795
run E-run2 $B/luce_server-E-09bde623-gate
