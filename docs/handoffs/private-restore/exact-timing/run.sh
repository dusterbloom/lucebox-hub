#!/bin/bash
# Time the exact 38.41 reconstruction: original driver, env and args; fresh process per rep.
set -u
R=/home/duster/qwen4exp-private-exact; A=/home/duster/pr823-release-20261007/artifacts; O=/home/duster/qwen4exp-f16/private-exact
MODEL=/home/duster/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
for rep in ${REPS:-1 2 3}; do
  env HIP_VISIBLE_DEVICES=1 TMPDIR=/home/duster/qwen4exp-f16/scratch LUCE_QWEN_GRAPH_SH=1 LUCE_QWEN_HC_DOWN_INJECT=1 LUCE_QWEN_SHARED_OVERLAP=1 \
    LUCE_QWEN_QSA_CONT_ELISION=1 LUCE_QWEN_EXACT_ROUTER_SUFFIX=1 LUCE_QWEN_EXPERT_ROW_WARPS=8 LUCE_HOST_PREFILL_GUARDS=1 \
    LUCE_QWEN_HC_SCALE_SILU=1 LUCE_QWEN_HC_LO_Q8=1 LUCE_QWEN_PRODUCER_Q8=1 LUCE_QWEN_HC_UPMIX_ROW8=1 LUCE_QWEN_SHARED_EPILOGUE=1 \
    LUCE_QWEN_GDN_AB_EXACT=1 LUCE_GPU_ARGMAX=1 MEASURE_GPU_ARGMAX=1 MEASURE_FOLLOW=$A/follow.ids \
    ~/qwen4exp-f16/gpu_exec.sh $R/driver-exact-v4 "$MODEL" $A/prompt.ids 256 8 > $O/rep$rep.log 2>&1
  echo "rep=$rep rc=$? $(grep -E "^\[measure\]" $O/rep$rep.log | tr "\n" " " | cut -c1-400)"
done
