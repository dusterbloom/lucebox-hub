#!/bin/bash
# Run under: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/kvoff-parity/ab_all.sh
D=/mnt/kimi-k3/build
O=/mnt/kimi-k3/kvflash-gate-m2prefill
mkdir -p $O
P=$O/PROVENANCE.txt
{
echo "flock command: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/kvoff-parity/ab_all.sh"
echo "start: $(date -Is)"
echo "workload: gate4_live.py A, Qwen3.8-27B-UD-Q4_K_M, --max-ctx 131072, --no-kv-cache, no --kvflash, default KV type"
echo "main: $D/main-6f2b-ab/luce_server md5 $(md5sum < $D/main-6f2b-ab/luce_server | cut -d' ' -f1) commit 6f2b17958ebb49f2b7060ac0a40ce6509aa90039"
echo "m2:   $D/m2-fd1a7b96-ab/luce_server md5 $(md5sum < $D/m2-fd1a7b96-ab/luce_server | cut -d' ' -f1) commit fd1a7b96 (kvflash/2.0b-m2-kv-bias)"
} > $P
for i in 1 2 3 4 5; do
  for arm in "main main-6f2b-ab" "m2 m2-fd1a7b96-ab"; do
    set -- $arm
    name=${1}_$i
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    echo "run $name: bin $D/$2/luce_server md5 $(md5sum < $D/$2/luce_server | cut -d' ' -f1) | $(date -Is) | compute apps before start: [$apps]" >> $P
    if [ -n "$apps" ]; then echo "ABORT: GPU busy before $name" >> $P; exit 1; fi
    $D/kvoff-parity/ab_run.sh $name $D/$2/luce_server >> $P
  done
done
echo "end: $(date -Is)" >> $P
