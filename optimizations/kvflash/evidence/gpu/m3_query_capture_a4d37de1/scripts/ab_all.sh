#!/bin/bash
# Run under: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m3-ab/ab_all.sh
D=/mnt/kimi-k3/build
O=/mnt/kimi-k3/kvflash-gate-m3
M2BIN=$D/m2-fd1a7b96-ab/luce_server
M3BIN=$D/m3-cuda/luce_server
mkdir -p $O
P=$O/PROVENANCE.txt
{
echo "flock command: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m3-ab/ab_all.sh"
echo "start: $(date -Is)"
echo "workload: gate4_live.py A (turn 1 ~11.6K tokens cold prefill), Qwen3.8-27B-UD-Q4_K_M, --max-ctx 131072, --no-kv-cache, no --kvflash, default KV type, temp default of the chat client (greedy, answers compared byte for byte)"
echo "arms: m2 = M2 tip fd1a7b96, m3 = M3 capture off (default), m3cap = M3 with LUCE_KVFLASH_AM_CAPTURE=1"
echo "m2 binary: $M2BIN md5 $(md5sum < $M2BIN | cut -d' ' -f1) runpath ggml dir $D/m2-cuda/deps/llama.cpp/ggml/src"
md5sum $D/m2-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/m2-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
echo "m3 binary: $M3BIN md5 $(md5sum < $M3BIN | cut -d' ' -f1) runpath ggml dir $D/m3-cuda/deps/llama.cpp/ggml/src"
md5sum $D/m3-cuda/deps/llama.cpp/ggml/src/lib*.so.0.9.11 $D/m3-cuda/deps/llama.cpp/ggml/src/ggml-cuda/lib*.so.0.9.11
} > $P
for i in 1 2 3 4 5; do
  for arm in "m2 $M2BIN 0" "m3 $M3BIN 0" "m3cap $M3BIN 1"; do
    set -- $arm
    name=${1}_$i
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    echo "run $name: bin $2 AM_CAP=$3 | $(date -Is) | compute apps before start: [$apps]" >> $P
    if [ -n "$apps" ]; then echo "ABORT: GPU busy before $name" >> $P; exit 1; fi
    $D/m3-ab/ab_run.sh $name $2 $3 >> $P
  done
done
echo "end: $(date -Is)" >> $P
