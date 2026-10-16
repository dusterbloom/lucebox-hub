#!/bin/bash
B=/mnt/kimi-k3/build/scorer-cuda/luce_server
R=/mnt/kimi-k3/build/scorer-run
exec >> $R/seq1.log 2>&1
echo "=== $(date -Is) waiting for lock"
flock /mnt/kimi-k3/build/3090.lock $R/arm.sh $B $R/smoke_ids4.json $R/smoke_k2_shadow_pin 2 1
