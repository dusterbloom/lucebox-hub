#!/bin/bash
# usage: vram_sample.sh OUTFILE  samples GPU0 VRAM used (bytes) every second until killed
while true; do
  u=$(/opt/rocm/bin/rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[0\].*Used" | grep -o '[0-9]*$')
  echo "$(date +%s.%N) $u" >> $1
  sleep 1
done
