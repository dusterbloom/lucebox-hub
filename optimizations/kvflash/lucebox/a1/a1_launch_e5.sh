#!/bin/bash
# runs ON lucebox6, AFTER E3/E2 has exited: diagnostic E5, shipping arm at ratio 0.25, probes only, pilot trajectories 1 to 6.
# The pilot FULL-CHK rows are copied unedited as the reference, so only the SHIP arm runs. Output ~/r1e/out/e5/.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/e5
PY=~/r1e/venv/bin/python
if [ -e $O/pid ]; then echo "e5 dir already started"; exit 1; fi
if [ -e $HOME/r1e/out/e3e2/pid ] && kill -0 $(cat $HOME/r1e/out/e3e2/pid) 2>/dev/null; then echo "e3e2 still running"; exit 1; fi
$PY - <<'EOF' || { echo "launch gate FAILED"; exit 1; }
import subprocess, sys
sys.path.insert(0, "/home/duster/r1e/code/a1")
import a1_core as c
m = c.parse_free(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
print("free:", m)
sys.exit(0 if c.launch_ok(m) else 1)
EOF
mkdir -p $O
cp -n ~/r1e/out/a1_pilot/FULL.jsonl $O/FULL.jsonl
{
  echo "E5 diagnostic (ratio 0.25), launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- reference rows copied unedited from a1_pilot"
  md5sum ~/r1e/out/a1_pilot/FULL.jsonl $O/FULL.jsonl
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_run.py --out $O --n 6 --ratio 0.25 --gate3-after 99 --early-at 99"
  free -m
} > $O/provenance.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_run.py --out $O --n 6 --ratio 0.25 --gate3-after 99 --early-at 99 > $O/stdout.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.out 2>&1 < /dev/null &
echo "e5 pid $PID, watchdog pid $!"
