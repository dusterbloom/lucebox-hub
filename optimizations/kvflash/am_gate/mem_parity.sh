#!/bin/bash
# Default-off memory parity: same flags, main vs m7 (no --kvflash-am-ratio). Takes the 3090 lock itself.
# Prints the server's own GPU memory (nvidia-smi compute-apps, by pid) after load and after a 64-token request.
[ -n "$M7_LOCKED" ] || exec flock /mnt/kimi-k3/build/3090.lock env M7_LOCKED=1 "$0" "$@"
D=/mnt/kimi-k3/build
G=$(dirname "$(readlink -f "$0")")
O=/mnt/kimi-k3/kvflash-gate-m7/mem
mkdir -p $O/tmp
for arm in "main $D/main-6f2b-ab/luce_server" "m7 ${M7BIN:-$D/m7-gate/luce_server_m7c}"; do
  set -- $arm
  mkdir -p $O/$1
  TMPDIR=$O/tmp KV_ROOT=$O/$1 KV_BIN=$2 KV_MODEL=/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf \
    KV_MAX_CTX=131072 KV_PORT=18091 KV_EXTRA="--no-kv-cache --kvflash 8192" python3 -I $G/mem_probe.py $1
done
