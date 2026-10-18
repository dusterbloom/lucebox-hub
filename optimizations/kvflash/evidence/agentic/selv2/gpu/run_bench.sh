#!/bin/bash
# GPU scorer derisk (../../../../GPU_SCORER.md): bench_kvflash_gpu_mass at the E2E v2 sealed-chunk median (514), max (576)
# and the 52224-token max-ctx (816), q4_0 K (the server's cache type) and q8_0, 64 and 16 queries. One run at a time
# under the 3090 lock.   usage: bash run_bench.sh <bench binary>   (build line in GPU_SCORER.md)
set -u
D=$(dirname "$0")
{ md5sum "$1"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > $D/bench.out
for a in "514 64 mmq 10 q4_0" "576 64 mmq 10 q4_0" "816 64 mmq 8 q4_0" "514 16 mmq 10 q4_0" \
         "514 64 mmq 10 q8_0" "576 64 mmq 10 q8_0" "1024 64 mmq 6 q8_0"; do
  echo "### $a" >> $D/bench.out
  flock /mnt/kimi-k3/build/3090.lock "$1" $a >> $D/bench.out 2>&1
done
