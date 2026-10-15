#!/bin/bash
# Build gemvbench against the box build tree and run it under rocprofv3 for the decode-copy shapes.
# usage: gemv.sh <outdir> [ENV...]   e.g. gemv.sh base   /   gemv.sh v1 LUCE_KQ_DENSE=1
# SHAPES="q5_K:2560:10240 ..." overrides the default list (type:K:rows).
set -u
R=/home/duster/qwen4exp-exact-stack; B=$R/build1151; G=$B/deps/llama.cpp/ggml/src
S=/home/duster/qwen4exp-f16/scratch/decreq/gemv
RP=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/bin/rocprofv3
LIB=/home/duster/gufo-parity/rocprof-root/opt/rocm-7.2.2/lib
OUT=$S/$1; shift; EXTRA="$*"
mkdir -p $OUT
if [ ! -x $S/gemvbench ] || [ $S/gemvbench.cpp -nt $S/gemvbench ]; then
  /usr/bin/c++ -O2 -std=gnu++17 -I$R/src/server/deps/llama.cpp/ggml/include -o $S/gemvbench $S/gemvbench.cpp \
    -L$G -L$G/ggml-hip -Wl,-rpath,$G:$G/ggml-hip:/opt/rocm/lib -lggml -lggml-base -lggml-cpu -lggml-hip || exit 1
fi
md5sum $G/ggml-hip/libggml-hip.so.0.9.11 > $OUT/lib.md5
SHAPES=${SHAPES:-"q8_0:2560:10240 q5_K:2560:10240 q6_K:2560:10240 q5_1:2560:10240 q8_0:2560:6144 q5_K:2560:6144 q8_0:6144:2560 q5_K:6144:2560 q6_K:6144:2560 q8_0:2560:12288 q6_K:2560:12288 q8_0:2560:512 q6_K:2560:512"}
for s in $SHAPES; do
  IFS=: read ty k rows <<< "$s"
  d=$OUT/${ty}_${k}x${rows}; mkdir -p $d
  env HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$LIB:/opt/rocm/lib $EXTRA $RP --kernel-trace --output-format csv -d $d -o kt -- $S/gemvbench $ty $k $rows 8 20 > $d/out.txt 2>&1
  line=$(grep ^gemvbench $d/out.txt); mb=$(echo "$line" | sed -E 's/.*bytes=([0-9.]+) MB.*/\1/')
  echo "$line"
  python3 $S/ktsum.py $d/kt_kernel_trace.csv $mb 8
  rm -f $d/kt_kernel_trace.csv
done
