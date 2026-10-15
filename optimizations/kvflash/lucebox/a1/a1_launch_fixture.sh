#!/bin/bash
# runs ON lucebox6, AFTER E7 has exited and before A2: real-data reference fixtures for the engine's AM gates (M3, M4, M5): the shipping fit's
# inputs and outputs on 3 blocks, raw float32/int32 files plus manifest.json per case. Output ~/r1e/out/am_fixtures/ (about 1.1 GB, under 15 min).
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/am_fixtures
PY=~/r1e/venv/bin/python
if [ -e $O/pid ]; then echo "am_fixtures dir already started"; exit 1; fi
for d in e3e2 e5 e6 e7 a1; do if [ -e $HOME/r1e/out/$d/pid ] && kill -0 $(cat $HOME/r1e/out/$d/pid) 2>/dev/null; then echo "$d still running"; exit 1; fi; done
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
  echo "am_fixtures dump, launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: am_fixture_dump.py --out $O"
  free -m
} > $O/provenance.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/am_fixture_dump.py --out $O > $O/stdout.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.out 2>&1 < /dev/null &
echo "am_fixtures pid $PID, watchdog pid $!"
