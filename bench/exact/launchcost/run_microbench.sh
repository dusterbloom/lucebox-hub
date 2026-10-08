#!/bin/bash
# Sweep the graph_launch_cost microbench: N nodes x shape x knob.
# DO NOT RUN until coordinator says "GPU free". GPU jobs only via gpu_exec.sh.
# Usage: XENV="HIP_FORCE_DEV_KERNARG=1" TAG=kernarg bash run_microbench.sh
set -u
B=~/qwen4exp-launchcost
O=$B/logs
mkdir -p $O
TAG=${TAG:-base}
for n in 256 1024 2048; do
  for shape in empty chain chain1m; do
    env ${XENV:-} HIP_VISIBLE_DEVICES=1 TMPDIR=/home/duster/qwen4exp-f16/scratch \
      ~/qwen4exp-f16/gpu_exec.sh $B/graph_launch_cost $n $shape 200 \
      > $O/${TAG}_n${n}_${shape}.log 2>&1
    echo "tag=$TAG n=$n shape=$shape rc=$? $(grep -E "^shape=" $O/${TAG}_n${n}_${shape}.log)"
  done
done
