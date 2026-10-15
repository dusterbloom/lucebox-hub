#!/bin/bash
# runs ON lucebox6, AFTER E3/E2, E5, E6, E7, the A3 tests and A3 stage 1 have exited (it runs before A3 stage 2) (order in PREREG): diagnostic E9, the exact-mass router (E2 ranking plus exact-key mass, no answers), probes only, pilot trajectories 1 to 6.
# The pilot FULL-CHK rows are copied unedited as the reference, so no answer is generated. Output ~/r1e/out/e9/.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/e9
PY=~/r1e/venv/bin/python
if [ -e $O/pid ]; then echo "e9 dir already started"; exit 1; fi
for d in e3e2 e5 e6 e7 e8 e9 a3sel a3; do if [ -e $HOME/r1e/out/$d/pid ] && kill -0 $(cat $HOME/r1e/out/$d/pid) 2>/dev/null; then echo "$d still running"; exit 1; fi; done
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
  echo "E9 diagnostic (exact-mass router), launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- reference rows copied unedited from a1_pilot"
  md5sum ~/r1e/out/a1_pilot/FULL.jsonl $O/FULL.jsonl
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_e9.py --out $O --n 6"
  free -m
} > $O/provenance.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_e9.py --out $O --n 6 > $O/stdout.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.out 2>&1 < /dev/null &
echo "e9 pid $PID, watchdog pid $!"
