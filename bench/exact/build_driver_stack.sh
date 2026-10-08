#!/bin/bash
# Build the stack driver (driver_shared_epilogue.cpp) against the ~/qwen4exp-exact-stack tree.
set -euo pipefail
R=/home/duster/qwen4exp-exact-stack; S=$R/src; B=$R/build1151; G=$S/server/deps/llama.cpp/ggml
/usr/bin/c++ $(cat $R/inc.txt) \
  -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DLUCE_HAVE_DRAFT_TOPK=1 \
  -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -isystem /opt/rocm/include -O3 -DNDEBUG -std=gnu++17 \
  -o $R/driver.o -c $S/driver_shared_epilogue.cpp
cd $B
/opt/rocm-7.2.2/lib/llvm/bin/clang++ -O3 -DNDEBUG --offload-arch=gfx1151 -L/opt/rocm/lib -o $R/driver-stack \
  -Wl,-rpath,/opt/rocm/lib:/opt/rocm/lib64:$B/deps/llama.cpp/ggml/src:$B/deps/llama.cpp/ggml/src/ggml-hip \
  $R/driver.o libluce_common.a deps/llama.cpp/ggml/src/libggml.so deps/llama.cpp/ggml/src/ggml-hip/libggml-hip.so \
  deps/llama.cpp/ggml/src/libggml-cpu.so /opt/rocm/lib/libamdhip64.so deps/llama.cpp/ggml/src/libggml-base.so \
  libjpeg-turbo-prefix/lib/libjpeg.a libimage_codec_png.a -ldl /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so -lpthread
sha256sum $R/driver-stack $B/deps/llama.cpp/ggml/src/ggml-hip/libggml-hip.so.0.9.11
