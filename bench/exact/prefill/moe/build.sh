#!/bin/bash
# Build the routed gate/up microbench on lucebox6 against the rsynced ggml tree and gufo's static lib.
# usage: build.sh [extra hipcc flags...]   (e.g. -DMOE_EXP='"exp.cuh"')
set -e
cd /home/duster/qwen4exp-f16/scratch/moe2
G=/home/duster/qwen4exp-f16/scratch/moe2/src/ggml
/opt/rocm/bin/hipcc -O3 -DNDEBUG --offload-arch=gfx1151 -DGGML_USE_HIP -DGGML_HIP_NO_VMM -Wno-unused-result -Wno-deprecated-declarations \
    -I $G/include -I $G/src -I $G/src/ggml-cuda -I /opt/rocm/include \
    "$@" moe2_bench.hip -L/home/duster/qwen4exp-f16/scratch/moe2/gufo_objs -l:libgufo_kernels.a -o moe_bench 2>&1 | grep -v "warning:" | grep -vE "^\s+[0-9]+ \||^\s+\||^\s+\^|generated when compiling" | head -60
ls -la moe_bench
