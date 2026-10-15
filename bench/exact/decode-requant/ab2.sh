#!/bin/bash
# One-binary A/B of luce arms on lucebox4, arms alternating across restarts. Each arm is NAME=ENV/FLAGS
# (ENV: space-separated VAR=val, FLAGS: extra luce_server flags; either may be empty). Per restart: p4096
# warm-up (dropped), REPS x p4096, REPS x p16384, 16-token decode warm-up (dropped), REPS x decode128;
# greedy-64 on p4096 and decode512 in round 0; VmRSS and the chunk lines of the server log per restart (mem.txt).
# usage: ab2.sh <outdir> <rounds> <reps> NAME=ENV/FLAGS ...   e.g. A= 'B=LUCE_DECODE_REQ=q8_0/--chunk 15616'
set -u
S=/home/duster/qwen4exp-f16/scratch
TP=$S/gufo/textparity
LUCE=/home/duster/qwen4exp-exact-stack/build1151/luce_server
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
OUT=$1; ROUNDS=$2; REPS=$3; shift 3
PORT=18097
mkdir -p $OUT
md5sum $LUCE $(dirname $LUCE)/deps/llama.cpp/ggml/src/ggml-hip/libggml-hip.so.0.9.11 $MODEL > $OUT/binary.md5
printf '%s\n' "$@" > $OUT/arms.txt

start_server() {  # $1 = arm spec, $2 = round
  local name=${1%%=*} spec=${1#*=}
  local envs=${spec%%/*} flags=""
  [ "$spec" != "$envs" ] && flags=${spec#*/}
  env HIP_VISIBLE_DEVICES=1 GGML_CUDA_GRAPH_STATS=1 $envs nohup $LUCE --model "$MODEL" --port $PORT --max-ctx 20480 --prefix-cache-slots 0 $flags \
      > $OUT/serve_${name}_$2.log 2>&1 &
  SPID=$!
  for i in $(seq 1 900); do
    curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && return 0
    kill -0 $SPID 2>/dev/null || { echo "server died $name $2"; tail -20 $OUT/serve_${name}_$2.log; return 1; }
    sleep 1
  done
  echo "server start timeout $name $2"; return 1
}

stop_server() {
  kill $SPID 2>/dev/null
  for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || { sleep 3; return 0; }; sleep 1; done
  kill -9 $SPID 2>/dev/null; sleep 3
}

run_arm() {  # $1 = arm spec, $2 = round
  local name=${1%%=*} r=$2
  start_server "$1" $r || return 1
  python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt > /dev/null   # warm-up, dropped
  { echo "arm $name round $r"; grep VmRSS /proc/$SPID/status; grep -E "chunk-auto|chunk  " $OUT/serve_${name}_$r.log; } >> $OUT/mem.txt
  for i in $(seq 1 $REPS); do python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt >> $OUT/${name}${r}_p4096.jsonl; done
  for i in $(seq 1 $REPS); do python3 $TP/text_bench_client.py $PORT $TP/prompt_p16384_r0.txt >> $OUT/${name}${r}_p16384.jsonl; done
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r1.txt 16 > /dev/null   # warm-up, dropped
  for i in $(seq 1 $REPS); do python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 >> $OUT/${name}${r}_decode.jsonl; done
  if [ "$r" = 0 ]; then
    python3 $S/decode/greedy64.py $PORT $TP/prompt_p4096_r0.txt 64 > $OUT/${name}_greedy64_p4096.json
    python3 $S/decode/greedy64.py $PORT $TP/prompt_decode512_r0.txt 64 > $OUT/${name}_greedy64_d512.json
  fi
  stop_server
}

for r in $(seq 0 $((ROUNDS-1))); do
  for arm in "$@"; do run_arm "$arm" $r || exit 1; done
done
echo AB_DONE
