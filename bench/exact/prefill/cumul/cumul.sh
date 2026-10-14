#!/bin/bash
# Cumulative same-session run on lucebox4: (A) luce with every 2026-10-14 kill switch set to legacy,
# (B) luce default (HEAD), (C) gufo default, (D) gufo GUFO_DECODE_Q5=0 (decode only, like-for-like bytes).
# Arms alternate A/B/C/D across restarts; per restart: p4096 warm-up (dropped), REPS x p4096, REPS x p16384,
# 16-token decode warm-up (dropped), REPS x decode128; greedy-64 on p4096 and decode512 in round 0.
# usage: cumul.sh <outdir> <rounds> [reps]
set -u
S=/home/duster/qwen4exp-f16/scratch
TP=$S/gufo/textparity
LUCE=/home/duster/qwen4exp-exact-stack/build1151/luce_server
GUFO=/home/duster/qwen4exp-f16/scratch/gufo/build/release/gufo
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
LEGACY_ENV="LUCE_GDN_LEGACY=1 LUCE_MMB_Q8F16_RASTER=16 LUCE_GATED_NORM_WAVE=0 LUCE_GDN_CONV_QKNORM=0 LUCE_QWEN_EXACT_STACK=0 LUCE_MMVQ_MOE_GU2W=0 LUCE_MMB_GLU_LEGACY=1 LUCE_QSA_UNION_LEGACY=1 LUCE_QSA_ROWS_LEGACY=1 LUCE_ROPE_TABLE_LEGACY=1"
OUT=$1; ROUNDS=$2; REPS=${3:-5}
PORT=18097
mkdir -p $OUT
md5sum $LUCE $(dirname $LUCE)/deps/llama.cpp/ggml/src/ggml-hip/libggml-hip.so.0.9.11 $GUFO $MODEL > $OUT/binary.md5
{ echo "A: $LEGACY_ENV"; echo "B: (default)"; echo "C: gufo --cache-ram-bytes 1"; echo "D: gufo GUFO_DECODE_Q5=0 (decode only)"; } > $OUT/arms.txt

start_server() {  # $1 = arm $2 = round
  local arm=$1 extra=""
  case $arm in
    A) extra="$LEGACY_ENV" ;;
    D) extra="GUFO_DECODE_Q5=0" ;;
  esac
  if [ "$arm" = A ] || [ "$arm" = B ]; then
    env HIP_VISIBLE_DEVICES=1 GGML_CUDA_GRAPH_STATS=1 $extra nohup $LUCE --model "$MODEL" --port $PORT --max-ctx 20480 --prefix-cache-slots 0 \
        > $OUT/serve_${arm}_$2.log 2>&1 &
  else
    env HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/rocm-7.2.2/lib $extra nohup $GUFO serve --port $PORT llm -m "$MODEL" --context 20480 \
        --served-model-name bench --think off --max-pending-per-client 8 --cache-ram-bytes 1 -v > $OUT/serve_${arm}_$2.log 2>&1 &
  fi
  SPID=$!
  for i in $(seq 1 900); do
    curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && return 0
    kill -0 $SPID 2>/dev/null || { echo "server died $arm $2"; tail -20 $OUT/serve_${arm}_$2.log; return 1; }
    sleep 1
  done
  echo "server start timeout $arm $2"; return 1
}

stop_server() {
  kill $SPID 2>/dev/null
  for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || { sleep 3; return 0; }; sleep 1; done
  kill -9 $SPID 2>/dev/null; sleep 3
}

run_arm() {  # $1 = arm, $2 = round
  local arm=$1 r=$2
  start_server $arm $r || return 1
  if [ "$arm" != D ]; then
    python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt > /dev/null   # warm-up, dropped
    for i in $(seq 1 $REPS); do python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt >> $OUT/${arm}${r}_p4096.jsonl; done
    for i in $(seq 1 $REPS); do python3 $TP/text_bench_client.py $PORT $TP/prompt_p16384_r0.txt >> $OUT/${arm}${r}_p16384.jsonl; done
  fi
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r1.txt 16 > /dev/null   # warm-up, dropped
  for i in $(seq 1 $REPS); do python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 >> $OUT/${arm}${r}_decode.jsonl; done
  if [ "$r" = 0 ] && [ "$arm" != D ]; then
    python3 $S/decode/greedy64.py $PORT $TP/prompt_p4096_r0.txt 64 > $OUT/${arm}_greedy64_p4096.json
    python3 $S/decode/greedy64.py $PORT $TP/prompt_decode512_r0.txt 64 > $OUT/${arm}_greedy64_d512.json
  fi
  stop_server
}

for r in $(seq 0 $((ROUNDS-1))); do
  for arm in A B C D; do run_arm $arm $r || exit 1; done
done
echo CUMUL_DONE
