#!/bin/bash
# DEBUG_HIP_GRAPH_BATCH_SIZE sweep (candidate #2 in KNOBS.md): 1, 64, 512, 4096.
# DO NOT RUN until coordinator says "GPU free". GPU jobs only via gpu_exec.sh.
set -u
for bs in 1 64 512 4096; do
  TAG=batchsz_$bs XENV="DEBUG_HIP_GRAPH_BATCH_SIZE=$bs" bash ~/qwen4exp-launchcost/run_microbench.sh
done
