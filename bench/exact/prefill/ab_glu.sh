#!/bin/bash
# End-to-end A/B of the single-launch routed gate/up GEMM: one binary, LUCE_MMB_GLU_LEGACY=1 (old)
# vs unset (new, the default), arms alternating across restarts. Same protocol as decode/ab.sh:
# per restart: warm-up decode, REPS x decode128, 1 warm-up + REPS x p4096, 1 warm-up + REPS x p16384,
# greedy-64 on two prompts.
# usage: [ARMS="new old"] [OLD_ENV=... NEW_ENV=...] ab_glu.sh <outdir> <reps> <rounds>
#   (F16 kernel A/B: OLD_ENV=LUCE_MMB_GLU_F16=0 NEW_ENV=LUCE_MMB_GLU_F16=1)
set -u
S=/home/duster/qwen4exp-f16/scratch
TP=$S/gufo/textparity
D=$S/decode
LUCE=/home/duster/qwen4exp-exact-stack/build1151/luce_server
MODEL=/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
OUT=$1; REPS=$2; ROUNDS=$3; PORT=18102
mkdir -p $OUT
md5sum $LUCE > $OUT/binary.md5
sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && echo "drop_caches ok" > $OUT/dropcaches.txt || echo "drop_caches unavailable" > $OUT/dropcaches.txt
for r in $(seq 1 $ROUNDS); do
for arm in ${ARMS:-old new}; do
  TAG=${arm}${r}
  if [ $arm = old ]; then EXTRA="${OLD_ENV:-LUCE_MMB_GLU_LEGACY=1}"; else EXTRA="${NEW_ENV:-LUCE_MMB_GLU_LEGACY=0}"; fi
  env HIP_VISIBLE_DEVICES=1 GGML_CUDA_GRAPH_STATS=1 $EXTRA nohup $LUCE --model "$MODEL" --port $PORT --max-ctx 20480 --prefix-cache-slots 0 > $OUT/serve_$TAG.log 2>&1 &
  SPID=$!
  for i in $(seq 1 900); do
    curl -s -o /dev/null -m 2 http://127.0.0.1:$PORT/v1/models && break
    kill -0 $SPID 2>/dev/null || { echo "server died $TAG"; tail -5 $OUT/serve_$TAG.log; exit 1; }
    sleep 1
  done
  python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r1.txt 16 > /dev/null
  for i in $(seq 1 $REPS); do
    python3 $TP/decode_bench_client.py $PORT $TP/prompt_decode512_r0.txt 128 >> $OUT/${TAG}_decode.jsonl
  done
  for p in p4096 p16384; do
    python3 $TP/text_bench_client.py $PORT $TP/prompt_${p}_r1.txt > /dev/null
    for i in $(seq 1 $REPS); do
      python3 $TP/text_bench_client.py $PORT $TP/prompt_${p}_r0.txt >> $OUT/${TAG}_${p}.jsonl
    done
  done
  python3 $D/greedy64.py $PORT $TP/prompt_decode512_r0.txt 64 > $OUT/${TAG}_greedy64_d512.json
  python3 $D/greedy64.py $PORT $TP/prompt_p4096_r0.txt 64 > $OUT/${TAG}_greedy64_p4096.json
  kill $SPID; for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null
  sleep 3
done
done
echo AB_DONE
