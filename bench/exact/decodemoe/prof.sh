#!/bin/bash
# rocprofv3 kernel trace of gu_bench on lucebox4 (gfx1151 = HIP device 1); prints per-kernel stats.
cd /home/duster/qwen4exp-f16/scratch/decodemoe
ROCPROF=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/bin/rocprofv3
rm -rf prof
env HIP_VISIBLE_DEVICES=1 TMPDIR=$PWD LD_LIBRARY_PATH=/home/duster/gufo-parity/rocprof-root/opt/rocm-7.2.2/lib:/opt/rocm/lib:/opt/rocm/lib64 \
  $ROCPROF --kernel-trace --stats --output-format csv -d prof -o gu -- ./gu_bench "$@" > prof_run.log 2>&1
f=$(find prof -name "*kernel_stats.csv" | head -1)
echo "$f"
python3 stats.py "$f"
