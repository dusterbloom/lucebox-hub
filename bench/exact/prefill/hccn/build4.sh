#!/bin/bash
# Build the HC combine microbench on lucebox4 (gfx1151 = HIP device 1) from the box ggml tree, the local
# hc-cn.cu copy (ours/hc-cn.cu) and gufo's compiled kernels object (MIT). Stubs shared with ../attn.
set -e
D=/home/duster/qwen4exp-f16/scratch/hccn
A=/home/duster/qwen4exp-f16/scratch/attn
G=/home/duster/qwen4exp-exact-stack/src/server/deps/llama.cpp/ggml
HIPCC=/opt/rocm/bin/hipcc
COMMON="-O3 --offload-arch=gfx1151 -std=gnu++17 -DNDEBUG -D__HIP_PLATFORM_AMD__=1 -Wno-unused-result -Wno-deprecated-declarations"
OURS="-x hip -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DGGML_CUDA_FA_ALL_QUANTS -DGGML_BACKEND_BUILD -DGGML_SHARED -DGGML_BACKEND_SHARED -fPIC -I$D -I$G/src -I$G/src/ggml-cuda -I$G/include -I$G/../../../src/hip_compat"
GB=/home/duster/qwen4exp-f16/scratch/gufo/build/release
GK=$GB/src/models/qwen38_flash_next/CMakeFiles/gufo_qwen38_flash_next.dir/kernels/rocm
GOBJ="$GK/kernels.hip.cpp.o $GK/w8a8_wave64.hip.o $GB/CMakeFiles/gufo_core.dir/src/core/quant/ggml_dequant.cpp.o"
cd $D
[ -f gufo_shim.o ] || $HIPCC -c $COMMON -x hip gufo_shim.hip -o gufo_shim.o
$HIPCC -c $COMMON $OURS -Rpass-analysis=kernel-resource-usage hccn_bench.hip -o hccn_bench.o 2> hccn.res.log || { grep -E "error" hccn.res.log | head -20; exit 1; }
$HIPCC $COMMON hccn_bench.o $A/stubs.o gufo_shim.o $GOBJ -o hccn_bench -lamdhip64 2>&1 | grep -E "error|undefined" | head -20 || true
ls -la hccn_bench
grep -A6 "hc_combine_norm_f32_all\|hc_combine_norm_moe_f32_all" hccn.res.log | grep -E "Function Name|VGPRs:|Occupancy|ScratchSize" | head -20
