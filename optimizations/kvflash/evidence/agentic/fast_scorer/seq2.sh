#!/bin/bash
B=/mnt/kimi-k3/build/scorer-bin/luce_server
R=/mnt/kimi-k3/build/scorer-run
exec >> $R/seq2.log 2>&1
echo "=== $(date -Is) waiting for lock"
exec 9>/mnt/kimi-k3/build/3090.lock; flock 9
echo "=== $(date -Is) lock taken"
$R/arm.sh $B $R/smoke_ids4.json $R/s_k2_shadow 2 shadow
$R/arm.sh $B $R/smoke_ids4.json $R/s_k10_shadow 10 shadow
$R/arm.sh $B $R/drift_ids.json $R/d_k2_shadow 2 shadow
$R/arm.sh $B $R/drift_ids.json $R/d_k10_shadow 10 shadow
$R/arm.sh $B $R/smoke_ids4.json $R/s_k2_new 2 none
$R/arm.sh $B $R/smoke_ids4.json $R/s_k2_ref 2 ref
echo "=== $(date -Is) done, lock released"
