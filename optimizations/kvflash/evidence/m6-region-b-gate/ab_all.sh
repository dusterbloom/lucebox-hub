#!/bin/bash
# Run under: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m6-ab/ab_all.sh
D=/mnt/kimi-k3/build
O=/mnt/kimi-k3/kvflash-gate-m6
MAINBIN=$D/main-6f2b-ab/luce_server
M6BIN=$D/m6-cuda/luce_server
mkdir -p $O
P=$O/PROVENANCE.txt
{
echo "flock command: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m6-ab/ab_all.sh"
echo "start: $(date -Is)"
echo "workload: ab_m6.py, 5 distinct ~20K-token prompts per server run (salts 11..15), max_tokens 256, temperature 0 seed 1, Qwen3.8-27B-UD-Q4_K_M, --max-ctx 131072, --kvflash 8192, LUCE_KVFLASH_POLICY=lru, --no-kv-cache, default KV type; LUCE_KVFLASH_B_TOKENS=2048 set on both arms (main ignores it)"
echo "arms: main = 6f2b1795 (wt-main-6f2b), m6 = branch kvflash/2.0b-m6-region-b $(git -C /home/peppi/Dev/lucebox-builds/wt-m6 rev-parse --short HEAD) (B_L = 2048 rows per layer allocated, empty)"
echo "main binary: $MAINBIN md5 $(md5sum < $MAINBIN | cut -d' ' -f1) runpath ggml dir $D/main-6f2b-cuda/deps/llama.cpp/ggml/src"
md5sum $D/main-6f2b-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/main-6f2b-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
echo "m6 binary: $M6BIN md5 $(md5sum < $M6BIN | cut -d' ' -f1) runpath ggml dir $D/m6-cuda/deps/llama.cpp/ggml/src"
md5sum $D/m6-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/m6-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
echo "drop_caches/compact_memory: sudo needs a password in this session, so NOT done; page cache was warm for both arms (model file re-read each run)"
} > $P
for i in 1 2 3 4 5; do
  for arm in "main $MAINBIN" "m6 $M6BIN"; do
    set -- $arm
    name=${1}_$i
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    echo "run $name: bin $2 | $(date -Is) | compute apps before start: [$apps]" >> $P
    if [ -n "$apps" ]; then echo "ABORT: GPU busy before $name" >> $P; exit 1; fi
    $D/m6-ab/ab_run.sh $name $2 >> $P
  done
done
echo "end: $(date -Is)" >> $P
