#!/bin/bash
# usage: build.sh BRANCH TAG   builds BRANCH from ~/lucebox.git in ~/pf746/src (HIP gfx1201, same flags as ~/kvflash-gate/build.sh); archives ~/pf746/bin/luce_server-TAG
set -e
B=$1; TAG=$2
export TMPDIR=$HOME/pf746/tmp
cd ~/pf746/src; git fetch -q origin "$B"; git checkout -q --detach FETCH_HEAD; git log --oneline -1
cd server; export ROCM_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
cmake -B build-hip -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1201 -DLUCE_HIP_SM80_EQUIV=ON -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm
nice -n 10 cmake --build build-hip --target luce_server -j16
md5sum build-hip/luce_server
cp build-hip/luce_server ~/pf746/bin/luce_server-$TAG; echo "$B $(git -C ~/pf746/src rev-parse HEAD) $(md5sum build-hip/luce_server | cut -d" " -f1)" > ~/pf746/bin/luce_server-$TAG.prov
echo archived ~/pf746/bin/luce_server-$TAG
