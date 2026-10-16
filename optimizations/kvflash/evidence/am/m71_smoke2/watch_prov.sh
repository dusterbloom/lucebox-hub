#!/bin/bash
# Records the md5 of the binary actually loaded when each of AMOFF2 and AMON0 starts.
D=/mnt/kimi-k3/kvflash-gate-m7/m71_smoke2
for A in AMOFF2 AMON0; do
  until grep -q "$A start" $D/run_smoke2.log; do sleep 5; done
  for i in $(seq 1 60); do
    P=$(pgrep -f "m7-cuda/luce_server" | head -1); [ -n "$P" ] && break; sleep 5
  done
  { echo "== $A $(date -Is) pid=${P:-none}"
    [ -n "$P" ] && md5sum /proc/$P/exe
    md5sum /mnt/kimi-k3/build/m7-cuda/luce_server; } >> $D/provenance2.txt
  until grep -q "$A: lock released" $D/run_smoke2.log; do sleep 10; done
done
