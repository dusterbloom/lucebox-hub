#!/bin/bash
# usage: build.sh SRCDIR TAG   builds SRCDIR/server (HIP gfx1201, same flags as the earlier run) and archives ~/pf746r/bin/luce_server-TAG
set -e
S=$1; TAG=$2
export TMPDIR=$HOME/pf746r/tmp ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
mkdir -p ~/pf746r/bin
cd $S; git log --oneline -1
cd server
cmake -B build-hip -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1201 -DLUCE_HIP_SM80_EQUIV=ON -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm
nice -n 10 cmake --build build-hip --target luce_server -j12
md5sum build-hip/luce_server
cp build-hip/luce_server ~/pf746r/bin/luce_server-$TAG
echo "$TAG $(git -C $S rev-parse HEAD) $(md5sum build-hip/luce_server | cut -d' ' -f1)" > ~/pf746r/bin/luce_server-$TAG.prov
echo archived
