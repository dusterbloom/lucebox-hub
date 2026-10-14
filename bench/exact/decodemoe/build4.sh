#!/bin/bash
# Build the T=1 routed gate/up microbench on lucebox4 (gfx1151 = HIP device 1) from the box ggml tree,
# the local mmvq.cu/mmvq-routed-gu.cuh copies, and gufo's vendored mmvq sources (MIT).
set -e
D=/home/duster/qwen4exp-f16/scratch/decodemoe
G=/home/duster/qwen4exp-exact-stack/src/server/deps/llama.cpp/ggml
HIPCC=/opt/rocm/bin/hipcc
COMMON="-O3 --offload-arch=gfx1151 -std=gnu++17 -DNDEBUG -D__HIP_PLATFORM_AMD__=1 -Wno-unused-result -Wno-deprecated-declarations"
OURS="-x hip -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DGGML_CUDA_FA_ALL_QUANTS -DGGML_BACKEND_BUILD -DGGML_SHARED -DGGML_BACKEND_SHARED -fPIC -I$D/ours -I$G/src -I$G/src/ggml-cuda -I$G/include -I$G/../../../src/hip_compat"
GM=/home/duster/qwen4exp-f16/scratch/gufo/src/models/qwen38_flash_next/kernels/rocm/mmq
cd $D
mkdir -p gufo
if [ "${1:-}" != "nommvq" ]; then
  $HIPCC -c $COMMON $OURS ours/mmvq.cu -o ours/mmvq.o 2>&1 | grep -E "error|Error" | head -20 || true
fi
$HIPCC -c $COMMON $OURS stubs.hip -o stubs.o 2>&1 | grep -E "error" | head || true
for f in mmvq moe_wave64 qfn_ggml_stubs; do
  [ -f gufo/$f.o ] || $HIPCC -c $COMMON -x hip -I$GM $GM/$f.hip.cpp -o gufo/$f.o 2>&1 | grep -E "error" | head || true
done
$HIPCC -c $COMMON -x hip -I$GM gufo_shim.hip -o gufo_shim.o 2>&1 | grep -E "error" | head || true
$HIPCC -c $COMMON $OURS -Rpass-analysis=kernel-resource-usage gu_bench.hip -o gu_bench.o 2> gu_bench.res.log || { grep -E "error" gu_bench.res.log | head; exit 1; }
$HIPCC $COMMON gu_bench.o ours/mmvq.o stubs.o gufo_shim.o gufo/mmvq.o gufo/moe_wave64.o gufo/qfn_ggml_stubs.o -o gu_bench 2>&1 | grep -E "error|undefined" | head -20 || true
ls -la gu_bench
