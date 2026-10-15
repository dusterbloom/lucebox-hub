#!/bin/bash
# runs ON lucebox6: diagnostics E3 + E2 (PREREG "Diagnostics E2, E3, E5 before A2"), probes only, pilot trajectories 1 to 6.
# One build pass per trajectory; reads ~/r1e/out/a1_pilot/{FULL,SHIP}.jsonl, never writes there. Output ~/r1e/out/e3e2/.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/e3e2
PY=~/r1e/venv/bin/python
if [ -e $O/pid ]; then echo "e3e2 dir already started"; exit 1; fi
$PY - <<'EOF' || { echo "launch gate FAILED"; exit 1; }
import subprocess, sys
sys.path.insert(0, "/home/duster/r1e/code/a1")
import a1_core as c
m = c.parse_free(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
print("free:", m)
sys.exit(0 if c.launch_ok(m) else 1)
EOF
mkdir -p $O
{
  echo "E3/E2 diagnostics, launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- reference rows (read only)"
  md5sum ~/r1e/out/a1_pilot/FULL.jsonl ~/r1e/out/a1_pilot/SHIP.jsonl
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_e.py --out $O --n 6 --ship-check-n 1"
  free -m
} > $O/provenance.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_e.py --out $O --n 6 --ship-check-n 1 > $O/stdout.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.out 2>&1 < /dev/null &
echo "e3e2 pid $PID, watchdog pid $!"
