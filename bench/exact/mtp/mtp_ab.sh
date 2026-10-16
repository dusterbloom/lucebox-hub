#!/bin/bash
# One-binary A/B of MTP end-to-end decode on lucebox4, arms alternating across restarts. Each arm is
# NAME=ENV/FLAGS (ENV: space-separated VAR=val, FLAGS: extra luce_server flags; either may be empty).
# Per restart: one dropped warm-up per workload, then REPS x 256-token greedy decode on code16k (the textparity
# p16384 slice), prose16k and counting; the server's [qwen4exp-mtp] line (drafts / accepted / tokens_per_step /
# draft_ms) of every request is kept next to the client's decode tok/s. With PREFILL=1 also p4096 / p16384
# prefill tok/s (text_bench_client). Round 0 also stores the greedy-256 texts for the identity check.
# usage: mtp_ab.sh <outdir> <rounds> <reps> NAME=ENV/FLAGS ...
#   e.g. mtp_ab.sh out 2 5 'new=' 'legacy=LUCE_VERIFY_GRAPH_LEGACY=1' 'ar=/--verify-width 1 --no-draft'
set -u
S=/home/duster/qwen4exp-f16/scratch
TP=$S/gufo/textparity
P=$S/mtp-e2e/prompts
LUCE=${LUCE:-/home/duster/qwen4exp-exact-stack/build1151/luce_server}
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
# The sidecar is not auto-discovered here (<repo>/MTP is searched two levels up from the model, lucebox-models has
# it one level up), so every arm gets --draft unless its flags contain --no-draft (plain AR without the sidecar).
DRAFT=${DRAFT:-/home/duster/lucebox-models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
OUT=$1; ROUNDS=$2; REPS=$3; shift 3
PORT=${PORT:-18098}
NTOK=${NTOK:-256}
PREFILL=${PREFILL:-0}
WORKLOADS=${WORKLOADS-code16k prose16k counting}   # WORKLOADS="" with PREFILL=1: prefill only
MAXCTX=${MAXCTX:-20480}                             # code64k (textparity p16384 r0+r1+r2+r0, ~66K tokens) needs 81920
mkdir -p $OUT
md5sum $LUCE $(dirname $LUCE)/deps/llama.cpp/ggml/src/ggml-hip/libggml-hip.so.0.9.11 $MODEL > $OUT/binary.md5
printf '%s\n' "$@" > $OUT/arms.txt
declare -A WL=( [code16k]=$TP/prompt_p16384_r0.txt [prose16k]=$P/prose16k.txt [counting]=$P/counting.txt
                [code64k]=$S/mtppolicy/code64k.txt )

start_server() {  # $1 = arm spec, $2 = round
  local name=${1%%=*} spec=${1#*=}
  local envs=${spec%%/*} flags=""
  [ "$spec" != "$envs" ] && flags=${spec#*/}
  local draft="--draft $DRAFT"
  [[ "$flags" == *--no-draft* ]] && { draft=""; flags=${flags//--no-draft/}; }
  env HIP_VISIBLE_DEVICES=1 GGML_CUDA_GRAPH_STATS=1 $envs nohup $LUCE --model "$MODEL" $draft --port $PORT --max-ctx $MAXCTX --prefix-cache-slots 0 $flags \
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

decode() {  # $1 = arm name, $2 = round, $3 = workload, $4 = out file (or /dev/null for a warm-up)
  local log=$OUT/serve_${1}_$2.log
  local before=$(grep -c "qwen4exp-mtp\]" $log)
  local line
  line=$(python3 $TP/decode_bench_client.py $PORT ${WL[$3]} $NTOK) || return 1
  local mtp=$(grep "qwen4exp-mtp\]" $log | tail -n +$((before + 1)) | head -1)
  [ "$4" = /dev/null ] || printf '%s\t%s\n' "$line" "$mtp" >> $4
}

run_arm() {  # $1 = arm spec, $2 = round
  local name=${1%%=*} r=$2
  start_server "$1" $r || return 1
  { echo "arm $name round $r"; grep VmRSS /proc/$SPID/status; grep -E "chunk-auto|chunk  |verify-width|mtp" $OUT/serve_${name}_$r.log | head -5; } >> $OUT/mem.txt
  for wl in $WORKLOADS; do
    decode $name $r $wl /dev/null || { echo "arm $name round $r $wl failed"; stop_server; return 1; }   # warm-up, dropped
    for i in $(seq 1 $REPS); do decode $name $r $wl $OUT/${name}${r}_${wl}.tsv || { echo "arm $name round $r $wl failed"; stop_server; return 1; }; done
  done
  if [ "$PREFILL" = 1 ]; then
    python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt > /dev/null
    for i in $(seq 1 $REPS); do python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt >> $OUT/${name}${r}_p4096.jsonl; done
    for i in $(seq 1 $REPS); do python3 $TP/text_bench_client.py $PORT $TP/prompt_p16384_r0.txt >> $OUT/${name}${r}_p16384.jsonl; done
  fi
  if [ "$r" = 0 ]; then
    for wl in $WORKLOADS; do python3 $S/decode/greedy64.py $PORT ${WL[$wl]} $NTOK > $OUT/${name}_greedy_${wl}.json; done
  fi
  stop_server
}

for r in $(seq 0 $((ROUNDS-1))); do
  for arm in "$@"; do run_arm "$arm" $r || echo "arm ${arm%%=*} round $r incomplete"; done
done
echo AB_DONE
