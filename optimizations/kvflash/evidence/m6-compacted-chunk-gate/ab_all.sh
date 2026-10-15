#!/bin/bash
# Run under: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m6c-ab/ab_all.sh
D=/mnt/kimi-k3/build
O=/mnt/kimi-k3/kvflash-gate-m6c
MAINBIN=$D/main-6f2b-ab/luce_server
M6BIN=$D/m6c-cuda/luce_server
mkdir -p $O
P=$O/PROVENANCE.txt
{
echo "flock command: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m6c-ab/ab_all.sh"
echo "start: $(date -Is)"
echo "workload: ab_m6.py, 5 distinct ~19K-token prompts per server run (salts 11..15), max_tokens 256, temperature 0, Qwen3.8-27B-UD-Q4_K_M, --max-ctx 131072, --kvflash 8192, LUCE_KVFLASH_POLICY=lru, --no-kv-cache, default KV type"
echo "arms: main = 6f2b1795 (wt-main-6f2b); m6c = branch kvflash/2.0b-m6-compacted-chunk $(git -C /home/peppi/Dev/lucebox-builds/wt-m6c rev-parse --short HEAD), nothing compacted; neutral = same binary, LUCE_KVFLASH_TEST_COMPACT=20:0 (chunk 20 behind the append head marked compacted, K/V untouched, beta 0)"
echo "main binary: $MAINBIN md5 $(md5sum < $MAINBIN | cut -d' ' -f1) runpath ggml dir $D/main-6f2b-cuda/deps/llama.cpp/ggml/src"
md5sum $D/main-6f2b-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/main-6f2b-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
echo "m6c binary: $M6BIN md5 $(md5sum < $M6BIN | cut -d' ' -f1) runpath ggml dir $D/m6c-cuda/deps/llama.cpp/ggml/src"
md5sum $D/m6c-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/m6c-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
echo "drop_caches/compact_memory: sudo needs a password in this session, so NOT done; page cache was warm for all arms (model file re-read each run)"
} > $P
for i in 1 2 3 4 5; do
  for arm in "main $MAINBIN" "m6c $M6BIN" "neutral $M6BIN"; do
    set -- $arm
    name=${1}_$i
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    echo "run $name: bin $2 | $(date -Is) | compute apps before start: [$apps]" >> $P
    waited=0
    while [ -n "$apps" ] && [ $waited -lt 3600 ]; do sleep 20; waited=$((waited+20)); apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; done
    if [ -n "$apps" ]; then echo "ABORT: GPU busy before $name" >> $P; exit 1; fi
    if [ $waited -gt 0 ]; then echo "run $name: waited ${waited}s for another GPU user (not ours), now idle" >> $P; fi
    if [ "$1" = neutral ]; then export M6_TEST_COMPACT=20:0; else unset M6_TEST_COMPACT; fi
    $D/m6c-ab/ab_run.sh $name $2 >> $P || { echo "ABORT: $name failed (see $O/$name/driver.out)" >> $P; exit 1; }
  done
done
unset M6_TEST_COMPACT
echo "end: $(date -Is)" >> $P
python3 -I $D/m6c-ab/ab_cmp.py > $O/cmp.out 2>&1; echo "ab_cmp exit=$?" >> $P
