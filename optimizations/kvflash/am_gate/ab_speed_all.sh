#!/bin/bash
# Run under: flock /mnt/kimi-k3/build/3090.lock <this script>
# Interleaved: for each of 5 rounds run main, m7 (compaction off, the graph under test), m7on (informational).
D=/mnt/kimi-k3/build
O=/mnt/kimi-k3/kvflash-gate-m7/speed
G=/home/peppi/Dev/lucebox-builds/wt-m7/optimizations/kvflash/am_gate
MAINBIN=$D/main-6f2b-ab/luce_server
M7BIN=${M7BIN:-$D/m7-gate/luce_server_m7}
mkdir -p $O $O/tmp
P=$O/PROVENANCE.txt
{
echo "flock command: flock /mnt/kimi-k3/build/3090.lock $0"
echo "start: $(date -Is)"
echo "workload: ab_speed.py, 5 distinct ~19K-token prompts per server run (salts 11..15), max_tokens 256, temperature 0, Qwen3.8-27B-UD-Q4_K_M, --max-ctx 131072, --kvflash 8192, LUCE_KVFLASH_POLICY=lru, --no-kv-cache, default KV type"
echo "arms: main = 6f2b1795; m7 = $M7BIN built from branch kvflash/2.0b-m7-auto-compact $(git -C /home/peppi/Dev/lucebox-builds/wt-m7 rev-parse --short HEAD), --kvflash-am-ratio unset (nothing compacted); m7on = same binary with --kvflash-am-ratio 0.10 (informational)"
echo "main binary: $MAINBIN md5 $(md5sum < $MAINBIN | cut -d' ' -f1)"
md5sum $D/main-6f2b-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/main-6f2b-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
echo "m7 binary: $M7BIN md5 $(md5sum < $M7BIN | cut -d' ' -f1)"
md5sum $D/m7-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/m7-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
} > $P
for i in 1 2 3 4 5; do
  for arm in "main $MAINBIN" "m7 $M7BIN" "m7on $M7BIN"; do
    set -- $arm
    name=${1}_$i
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    echo "run $name: bin $2 | $(date -Is) | compute apps before start: [$apps]" >> $P
    waited=0
    while [ -n "$apps" ] && [ $waited -lt 3600 ]; do sleep 20; waited=$((waited+20)); apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; done
    if [ -n "$apps" ]; then echo "ABORT: GPU busy before $name" >> $P; exit 1; fi
    if [ $waited -gt 0 ]; then echo "run $name: waited ${waited}s for another GPU user (not ours), now idle" >> $P; fi
    mkdir -p $O/$name
    if [ "$1" = m7on ]; then export M7_EXTRA="--kvflash-am-ratio 0.10"; else export M7_EXTRA=""; fi
    TMPDIR=$O/tmp KV_ROOT=$O/$name KV_BIN=$2 KV_MODEL=/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf KV_MAX_CTX=131072 KV_PORT=18090 KV_EXTRA="--no-kv-cache --kvflash 8192" \
      python3 -I $G/ab_speed.py > $O/$name/driver.out 2>&1 || { echo "ABORT: $name failed (see $O/$name/driver.out)" >> $P; exit 1; }
  done
done
echo "end: $(date -Is)" >> $P
python3 -I $G/ab_speed_cmp.py > $O/cmp.out 2>&1; echo "ab_speed_cmp exit=$?" >> $P
