#!/bin/bash
# Run the full microbench knob sweep from KNOBS.md in priority order.
# DO NOT RUN until coordinator says "GPU free". GPU jobs only via gpu_exec.sh.
set -u
M=~/qwen4exp-launchcost/run_microbench.sh

echo "=== baseline (no knob) ==="
TAG=base bash $M

echo "=== 1: DEBUG_CLR_GRAPH_PACKET_CAPTURE=1 ==="
TAG=pktcap_1 XENV="DEBUG_CLR_GRAPH_PACKET_CAPTURE=1" bash $M
echo "=== 1b: DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 (in case default sense is inverted) ==="
TAG=pktcap_0 XENV="DEBUG_CLR_GRAPH_PACKET_CAPTURE=0" bash $M

echo "=== 2: DEBUG_HIP_GRAPH_BATCH_SIZE sweep ==="
bash ~/qwen4exp-launchcost/run_batch_size_sweep.sh

echo "=== 3: DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0 (flagged risk, microbench only here) ==="
TAG=hdpflush_0 XENV="DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0" bash $M

echo "=== 4: HIP_FORCE_DEV_KERNARG=1 ==="
TAG=kernarg_1 XENV="HIP_FORCE_DEV_KERNARG=1" bash $M

echo "=== 3+4 combined: HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0 ==="
TAG=kernarg_hdp_combo XENV="HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0" bash $M

echo "=== 5: DEBUG_HIP_KERNARG_COPY_OPT=1 ==="
TAG=kcopy_1 XENV="DEBUG_HIP_KERNARG_COPY_OPT=1" bash $M
echo "=== 5: DEBUG_CLR_BLIT_KERNARG_OPT=1 ==="
TAG=blitkarg_1 XENV="DEBUG_CLR_BLIT_KERNARG_OPT=1" bash $M

echo "=== 6: DEBUG_HIP_FORCE_GRAPH_QUEUES=1 ==="
TAG=forcegq_1 XENV="DEBUG_HIP_FORCE_GRAPH_QUEUES=1" bash $M
echo "=== 6: DEBUG_HIP_DYNAMIC_QUEUES=1 ==="
TAG=dynq_1 XENV="DEBUG_HIP_DYNAMIC_QUEUES=1" bash $M
echo "=== 6: DEBUG_HIP_FORCE_ASYNC_QUEUE=1 ==="
TAG=asyncq_1 XENV="DEBUG_HIP_FORCE_ASYNC_QUEUE=1" bash $M

echo "=== 7: DEBUG_CLR_BATCH_CPU_SYNC_SIZE / DEBUG_CLR_MAX_BATCH_SIZE (microbench exploration) ==="
TAG=clrbatch_sync_1 XENV="DEBUG_CLR_BATCH_CPU_SYNC_SIZE=1" bash $M
TAG=clrbatch_max_4096 XENV="DEBUG_CLR_MAX_BATCH_SIZE=4096" bash $M

echo "=== 8: GPU_MAX_HW_QUEUES=1 (negative control) ==="
TAG=hwq_1 XENV="GPU_MAX_HW_QUEUES=1" bash $M

echo "=== 9: HSA_ENABLE_INTERRUPT=0 ==="
TAG=noint_0 XENV="HSA_ENABLE_INTERRUPT=0" bash $M

echo "=== 10: confirm AMD_SERIALIZE_KERNEL/AMD_LOG_LEVEL unset in baseline env ==="
env | grep -E "AMD_SERIALIZE_KERNEL|AMD_LOG_LEVEL" || echo "confirmed unset"
