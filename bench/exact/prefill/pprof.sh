#!/bin/bash
# rocprofv3 kernel trace of one prefill request on lucebox4: luce_server started under the profiler with the
# given env (e.g. LUCE_MMB_GLU_LEGACY=1), one warm-up request on distinct text, then the measured request.
# usage: pprof.sh <outdir> <prompt p4096|p16384> [extra env]
set -u
S=/home/duster/qwen4exp-f16/scratch
TP=$S/gufo/textparity
LUCE=/home/duster/qwen4exp-exact-stack/build1151/luce_server
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
ROCPROF=/home/duster/pr823-release-20261007/artifacts/profiler-root/opt/rocm-7.2.2/bin/rocprofv3
OUT=$1; P=$2; EXTRA=${3:-}; PORT=18103
mkdir -p $OUT
env HIP_VISIBLE_DEVICES=1 TMPDIR=$OUT LD_LIBRARY_PATH=/home/duster/gufo-parity/rocprof-root/opt/rocm-7.2.2/lib:/opt/rocm/lib:/opt/rocm/lib64 $EXTRA \
  nohup $ROCPROF --kernel-trace --output-format csv -d $OUT -o run -- \
  $LUCE --model "$MODEL" --port $PORT --max-ctx 20480 --prefix-cache-slots 0 > /dev/null 2>&1 &
RPID=$!
for i in $(seq 1 900); do
  curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && break
  kill -0 $RPID 2>/dev/null || { echo "profiled server died"; exit 1; }
  sleep 1
done
python3 $TP/text_bench_client.py $PORT $TP/prompt_${P}_r1.txt > $OUT/req1.json
sleep 3
python3 $TP/text_bench_client.py $PORT $TP/prompt_${P}_r0.txt > $OUT/req2.json
SPID=$(pgrep -f "luce_server --model .* --port $PORT ")
[ -n "$SPID" ] && kill $SPID
for i in $(seq 1 180); do kill -0 $RPID 2>/dev/null || break; sleep 1; done
for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -0 $SPID 2>/dev/null && kill -9 $SPID
echo PPROF_DONE
