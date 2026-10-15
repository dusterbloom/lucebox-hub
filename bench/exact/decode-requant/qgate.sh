#!/bin/bash
# Teacher-forced logits dump of one arm on lucebox4 (quality gate for LUCE_DECODE_REQ): starts luce_server with the
# arm's env, runs every slice prompt through the decode path with the last TAIL tokens forced (LUCE_TEACHER_*),
# appends the per-step logits to <outdir>/<arm>.bin, stops the server. The dump's step 0 is the prefill's logits.
# usage: qgate.sh <arm-name> <outdir> <tail> <drop> [env assignments...]   e.g. qgate.sh q5_1 out 600 5 LUCE_DECODE_REQ=q5_1
set -u
S=/home/duster/qwen4exp-f16/scratch
LUCE=/home/duster/qwen4exp-exact-stack/build1151/luce_server
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
ARM=$1; OUT=$2; TAIL=$3; DROP=$4; shift 4
PORT=18098
mkdir -p $OUT
rm -f $OUT/$ARM.bin
env HIP_VISIBLE_DEVICES=1 LUCE_TEACHER_TAIL=$TAIL LUCE_TEACHER_DROP=$DROP LUCE_TEACHER_LOGITS=$OUT/$ARM.bin "$@" \
    nohup $LUCE --model "$MODEL" --port $PORT --max-ctx 20480 --prefix-cache-slots 0 > $OUT/serve_$ARM.log 2>&1 &
SPID=$!
for i in $(seq 1 900); do
  curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && break
  kill -0 $SPID 2>/dev/null || { echo "server died $ARM"; tail -20 $OUT/serve_$ARM.log; exit 1; }
  sleep 1
done
for p in $S/decreq/slice_*.txt; do
  python3 $S/decreq/teacher_client.py $PORT $p $TAIL > $OUT/${ARM}_$(basename $p .txt).json || echo "request failed $p"
done
kill $SPID 2>/dev/null
for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
ls -la $OUT/$ARM.bin
grep -E "decode .* copies|exact-stack defaults|LUCE_DECODE_REQ" $OUT/serve_$ARM.log | head -3
