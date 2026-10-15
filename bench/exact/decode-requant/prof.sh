#!/bin/bash
# rocprofv3 kernel trace of luce_server decode, one arm per server: 16-token warm-up then 2 x decode128 (prompt_decode512_r0).
set -u
S=/home/duster/qwen4exp-f16/scratch; TP=$S/gufo/textparity
LUCE=/home/duster/qwen4exp-exact-stack/build1151/luce_server
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
RP=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/bin/rocprofv3
LIB=/home/duster/gufo-parity/rocprof-root/opt/rocm-7.2.2/lib
[ -d $LIB ] || LIB=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/lib
PORT=18098
run() {
  O=$S/decreq/prof/$1; mkdir -p $O
  env HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$LIB:/opt/rocm/lib:/opt/rocm/lib64 $2 \
    $RP --kernel-trace --output-format csv -d $O -o kt -- $LUCE --model $MODEL --port $PORT --max-ctx 20480 --prefix-cache-slots 0 > $O/serve.log 2>&1 &
  P=$!
  for i in $(seq 1 900); do curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && break; kill -0 $P 2>/dev/null || { echo died $1; tail -20 $O/serve.log; return 1; }; sleep 1; done
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r1.txt 16 > $O/warm.jsonl
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 > $O/decode.jsonl
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 >> $O/decode.jsonl
  cat $O/decode.jsonl
  kill $P; for i in $(seq 1 240); do kill -0 $P 2>/dev/null || break; sleep 1; done; sleep 3
  ls -la $O; du -sh $O
}
run req "LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k"
run exact ""
echo PROF_DONE
