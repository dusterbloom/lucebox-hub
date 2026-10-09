#!/bin/bash
# Native-mode (n=256 reps=8) timing/contract-check run, full 38.41 env.
# Usage: run_native.sh <model.gguf> <tag> <run_index>
set -u
R=/home/duster/qwen4exp-exact-stack; A=/home/duster/pr823-release-20261007/artifacts; O=/home/duster/qwen4exp-q6/logs
MODEL="$1"; TAG="$2"; IDX="$3"
mkdir -p $O
env HIP_VISIBLE_DEVICES=1 TMPDIR=/home/duster/qwen4exp-f16/scratch LUCE_QWEN_GRAPH_SH=1 LUCE_QWEN_HC_DOWN_INJECT=1 LUCE_QWEN_SHARED_OVERLAP=1 \
  LUCE_QWEN_QSA_CONT_ELISION=1 LUCE_QWEN_EXACT_ROUTER_SUFFIX=1 LUCE_QWEN_EXPERT_ROW_WARPS=8 LUCE_HOST_PREFILL_GUARDS=1 \
  LUCE_QWEN_HC_SCALE_SILU=1 LUCE_QWEN_HC_LO_Q8=1 LUCE_QWEN_PRODUCER_Q8=1 LUCE_QWEN_HC_UPMIX_ROW8=1 LUCE_QWEN_SHARED_EPILOGUE=1 \
  LUCE_QWEN_GDN_AB_EXACT=1 LUCE_GPU_ARGMAX=1 MEASURE_GPU_ARGMAX=1 MEASURE_FOLLOW=$A/follow.ids \
  ~/qwen4exp-f16/gpu_exec.sh $R/driver-stack "$MODEL" $A/prompt.ids 256 8 > $O/${TAG}_${IDX}.log 2>&1
echo "tag=$TAG idx=$IDX rc=$?"
