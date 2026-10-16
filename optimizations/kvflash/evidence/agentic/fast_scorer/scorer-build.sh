#!/bin/bash
# usage: scorer-build.sh <target...>
B=/mnt/kimi-k3/build/scorer-cuda
if [ ! -f $B/CMakeCache.txt ]; then
  cmake -S /home/peppi/Dev/lucebox-builds/wt-scorer/server -B $B -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DLUCE_TESTS=ON -DLUCE_SERVER=ON -DLUCE_GPU_BACKEND=cuda > /mnt/kimi-k3/build/scorer-cuda.cfg.log 2>&1
fi
nice -n 19 cmake --build $B -j4 --target "$@"
