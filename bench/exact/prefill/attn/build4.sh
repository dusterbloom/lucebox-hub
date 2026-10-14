#!/bin/bash
# Build the prefill attention microbench on lucebox4 (gfx1151 = HIP device 1) from the box ggml tree, the
# local qsa.cu copy (ours/qsa.cu, with the exported launchers) and gufo's compiled kernels object (MIT).
set -e
D=/home/duster/qwen4exp-f16/scratch/attn
G=/home/duster/qwen4exp-exact-stack/src/server/deps/llama.cpp/ggml
HIPCC=/opt/rocm/bin/hipcc
COMMON="-O3 --offload-arch=gfx1151 -std=gnu++17 -DNDEBUG -D__HIP_PLATFORM_AMD__=1 -Wno-unused-result -Wno-deprecated-declarations"
OURS="-x hip -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DGGML_CUDA_FA_ALL_QUANTS -DGGML_BACKEND_BUILD -DGGML_SHARED -DGGML_BACKEND_SHARED -fPIC -I$D/ours -I$G/src -I$G/src/ggml-cuda -I$G/include -I$G/../../../src/hip_compat"
GB=/home/duster/qwen4exp-f16/scratch/gufo/build/release
GK=$GB/src/models/qwen38_flash_next/CMakeFiles/gufo_qwen38_flash_next.dir/kernels/rocm
GOBJ="$GK/kernels.hip.cpp.o $GK/w8a8_wave64.hip.o $GB/CMakeFiles/gufo_core.dir/src/core/quant/ggml_dequant.cpp.o"
cd $D
$HIPCC -c $COMMON $OURS -Rpass-analysis=kernel-resource-usage ours/qsa.cu -o ours/qsa.o 2> ours/qsa.res.log || { grep -E "error" ours/qsa.res.log | head -20; exit 1; }
[ -f stubs.o ] || $HIPCC -c $COMMON $OURS stubs.hip -o stubs.o
[ -f gufo_shim.o ] || $HIPCC -c $COMMON -x hip gufo_shim.hip -o gufo_shim.o
$HIPCC -c $COMMON -x hip attn_bench.hip -o attn_bench.o
$HIPCC $COMMON attn_bench.o ours/qsa.o stubs.o gufo_shim.o $GOBJ -o attn_bench -lamdhip64 2>&1 | grep -E "error|undefined" | head -20 || true
ls -la attn_bench
# rope microbench: includes ours/rope.cu directly (rope_bench.hip), no gufo objects needed
$HIPCC -c $COMMON $OURS -I$D rope_bench.hip -o rope_bench.o 2>&1 | grep -E "error" | head -20 || true
$HIPCC $COMMON rope_bench.o stubs.o -o rope_bench -lamdhip64 2>&1 | grep -E "error|undefined" | head -20 || true
ls -la rope_bench
