#!/bin/bash
# PREREG R1b Amendment 3: F-UNI, GPU type matched to the orig cell of each article range. --start is relative to the split start.
cd "$(dirname "$0")" || exit 1
L() { name=$1; gpu=$2; shift 2
  KVFLASH_GPU=$gpu nohup modal run r1b.py::cells --cells FUNI "$@" --out r1b_funi_$name > r1b_funi_$name.log 2>&1 &
}
L s10 "H100!" --split sel --target 0.1 --start 0 --n 20
L s05 "H200" --split sel --target 0.05 --start 0 --n 20
L h10a "H200" --split held --target 0.1 --start 0 --n 10
L h10b "H100!" --split held --target 0.1 --start 10 --n 10
L h10c "H200" --split held --target 0.1 --start 20 --n 10
L h05 "H200" --split held --target 0.05 --start 0 --n 30
wait
