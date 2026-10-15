#!/bin/bash
# Fusion audit traces on lucebox4: rocprofv3 --kernel-trace of luce_server with the match/debug logs on
# (LUCE_OP_TRACE, LUCE_QSA_GATE_DEBUG, LUCE_NORM_ROPE_DEBUG, LUCE_PLE_DEBUG: stderr only, kernels unchanged).
#   pf   : one warm-up p4096 request on distinct text, then the measured p4096 request
#   dec  : 16-token warm-up then 2 x decode128 (prompt_decode512), HEAD defaults
#   req  : same as dec with LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k
# usage: trace.sh <pf|dec|req> [outdir-root]   (lock ~/BOX_LOCK must be held by the caller)
set -u
S=/home/duster/qwen4exp-f16/scratch; TP=$S/gufo/textparity
LUCE=${LUCE:-/home/duster/qwen4exp-exact-stack/build1151/luce_server}
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
RP=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/bin/rocprofv3
LIB=/home/duster/gufo-parity/rocprof-root/opt/rocm-7.2.2/lib
[ -d $LIB ] || LIB=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/lib
ARM=$1; ROOT=${2:-$S/audit}; O=$ROOT/$ARM; PORT=18111
mkdir -p $O
DBG="LUCE_OP_TRACE=1 LUCE_QSA_GATE_DEBUG=1 LUCE_NORM_ROPE_DEBUG=1 LUCE_PLE_DEBUG=1"
EXTRA=""; [ "$ARM" = req ] && EXTRA="LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k"
md5sum $LUCE > $O/binary.md5
# rocprofv3 7.2.2 spams a "W<date> ... tool.cpp" warning per HSA event (13 GB per request on a full disk): drop them
( env HIP_VISIBLE_DEVICES=1 TMPDIR=$O LD_LIBRARY_PATH=$LIB:/opt/rocm/lib:/opt/rocm/lib64 $DBG $EXTRA \
  $RP --kernel-trace --output-format csv -d $O -o kt -- $LUCE --model $MODEL --port $PORT --max-ctx 20480 --prefix-cache-slots 0 2>&1 \
  | grep --line-buffered -v "^W2026" > $O/serve.log ) &
P=$!
for i in $(seq 1 900); do curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && break; kill -0 $P 2>/dev/null || { echo died; tail -20 $O/serve.log; exit 1; }; sleep 1; done
if [ "$ARM" = pf ]; then
  python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r1.txt > $O/req1.json
  echo "=== MEASURED REQUEST STARTS ===" >> $O/serve.log; sleep 3
  python3 $TP/text_bench_client.py $PORT $TP/prompt_p4096_r0.txt > $O/req2.json
  cat $O/req2.json
else
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r1.txt 16 > $O/warm.jsonl
  echo "=== MEASURED DECODE STARTS ===" >> $O/serve.log
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 > $O/decode.jsonl
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 >> $O/decode.jsonl
  cat $O/decode.jsonl
fi
SPID=$(pgrep -f "luce_server --model .* --port $PORT ")
[ -n "$SPID" ] && kill $SPID
for i in $(seq 1 240); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -0 $SPID 2>/dev/null && kill -9 $SPID
sleep 2; ls -la $O; du -sh $O
echo TRACE_DONE $ARM
