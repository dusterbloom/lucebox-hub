#!/usr/bin/env bash
# TDD GREEN gate for EB canvas blocking (diffusion_decoder.cpp run_eb_generate).
# RED (pre-blocking, established by the 2026-06-15 sweep): C=2048 → content_len=0,
#   step ms=6457, server glibc-abort under back-to-back load, ~64 tok/s.
# GREEN (blocked at <=512): non-empty content, every forward C<=512 & step<=600ms,
#   no crash across 3 back-to-back long gens, >=120 tok/s.
set +u
BIN=/home/peppi/Dev/dg-l2/server/build/dflash_server
MODEL=/home/peppi/models/diffusiongemma-26b/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
ARGS="--host 0.0.0.0 --port 18100 --max-ctx 131072 --cache-type-k q4_0 --cache-type-v q4_0 --model-name google/diffusiongemma-26B-A4B-it"
LOG=/tmp/dg_eb_test.log
URL=http://localhost:18100/v1/chat/completions
PROMPT="Write a detailed multi-paragraph technical explanation of how a modern operating system manages virtual memory, paging, the TLB, and page replacement."

for p in $(pgrep -x dflash_server); do kill -TERM "$p"; done
for i in $(seq 1 40); do pgrep -x dflash_server >/dev/null || break; sleep 0.5; done
for i in $(seq 1 30); do M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "${M:-9999}" -lt 2000 ] && break; sleep 1; done
: > "$LOG"
setsid "$BIN" "$MODEL" $ARGS >"$LOG" 2>&1 </dev/null &
for i in $(seq 1 120); do curl -s -m2 http://localhost:18100/health 2>/dev/null | grep -q ok && break; sleep 0.5; done

clen=0; tps=0; fin=""
for r in 1 2 3; do                         # 3 back-to-back long gens (crash repro)
  R=$(curl -s -m120 "$URL" -H 'Content-Type: application/json' -d "{\"model\":\"google/diffusiongemma-26B-A4B-it\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":2048,\"temperature\":0.0,\"stream\":false}")
  if [ "$r" = 1 ]; then
    clen=$(echo "$R" | jq -r '(.choices[0].message.content//"")|length')
    tps=$(echo  "$R" | jq -r '.usage.timings.decode_tokens_per_sec//0')
    fin=$(echo  "$R" | jq -r '.choices[0].finish_reason//"?"')
    echo "$R" | jq -r '.choices[0].message.content//""' | head -c 280 > /tmp/dg_eb_sample.txt
  fi
done
alive=$(pgrep -x dflash_server >/dev/null && echo yes || echo DEAD)

# Max canvas C and max step ms across all forwards this run.
maxC=$(grep 'dg-timing' "$LOG" | sed -n 's/.*C=\([0-9]*\).*/\1/p' | sort -n | tail -1)
maxstep=$(grep 'dg-timing' "$LOG" | sed -n 's/.*step ms=\([0-9.]*\).*/\1/p' | sort -n | tail -1)

echo "----- EB blocking GREEN gate -----"
echo "content_len=$clen  finish=$fin  decode_tps=$tps  alive=$alive  maxC=$maxC  max_step_ms=$maxstep"
echo "sample: $(cat /tmp/dg_eb_sample.txt)"
pass=1
awk "BEGIN{exit !($clen>0)}"            || { echo "FAIL: content empty"; pass=0; }
[ "$alive" = yes ]                      || { echo "FAIL: server crashed"; pass=0; }
awk "BEGIN{exit !(${maxC:-99999}<=512)}"      || { echo "FAIL: canvas not blocked (maxC=$maxC)"; pass=0; }
awk "BEGIN{exit !(${maxstep:-99999}<=600)}"   || { echo "FAIL: step over budget (maxstep=$maxstep)"; pass=0; }
awk "BEGIN{exit !(${tps:-0}>=120)}"           || { echo "FAIL: tok/s under 120 (tps=$tps)"; pass=0; }
[ "$pass" = 1 ] && echo "RESULT: GREEN" || echo "RESULT: RED"
