#!/bin/bash
# Build + run the standalone bit-identity test for the LUCE_QWEN_LAUNCH_CUT
# candidate kernel (rms_norm_mul_q8_1_f32, launch-cut.cu). Pure HIP test, no
# ggml graph involved -- only needs libggml-hip.so for the exported symbol
# ggml_cuda_test_launch_cut_rms_norm_mul_q8_1.
#
# DO NOT RUN until the coordinator says "GPU free" -- another agent is timing
# on this box. Build dir is a FRESH configure, never `cp -a`'d from another
# build dir (see KNOCKOUT-HC.md's CMAKE_HOME_DIRECTORY pitfall note).
#
# Usage (on lucebox4, duster@100.115.193.112):
#   bash build_launch_cut_test.sh
set -euo pipefail

R=~/qwen4exp-launch-cut
S=$R/src
B=$R/build1151
G=$S/server/deps/llama.cpp/ggml

mkdir -p "$R"

# Step 1: fresh cmake configure in $B, same options as
# ~/qwen4exp-exact-stack/build1151's CMakeCache (confirm by diffing
# CMakeCache.txt's non-path entries before building anything else).
#   cmake -S "$S/server" -B "$B" <same -D flags as build1151> \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build "$B" -j32 --target ggml-hip ggml-base ggml-cpu ggml

HIPCC=/opt/rocm-7.2.2/lib/llvm/bin/clang++

"$HIPCC" -O3 -DNDEBUG --offload-arch=gfx1151 -std=gnu++17 \
  -I"$G/include" -I"$G/src" -I"$G/src/ggml-cuda" \
  -D__HIP_PLATFORM_AMD__=1 -isystem /opt/rocm/include \
  -o "$R/test_launch_cut_bitexact" \
  "$S/bench/exact/test_launch_cut_bitexact.cpp" \
  -L/opt/rocm/lib \
  -Wl,-rpath,/opt/rocm/lib:/opt/rocm/lib64:"$B/deps/llama.cpp/ggml/src":"$B/deps/llama.cpp/ggml/src/ggml-hip" \
  "$B/deps/llama.cpp/ggml/src/ggml-hip/libggml-hip.so" \
  /opt/rocm/lib/libamdhip64.so

"$R/test_launch_cut_bitexact"
