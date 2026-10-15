#!/bin/bash
# runs ON lucebox6, AFTER A3 stage 1 has exited and a1_a3_report.py sel has written ~/r1e/out/a3sel/a3_report.json (one job at a time):
# A3 stage 2, the held-out proper run (PREREG "A3 routed exact fetch"), probes only, trajectories 7 to 60, ONE configuration.
# No argument: the window W is read by the python side from the stage-1 report's `selection.selected` (the report must be complete,
# exactly the registered arms), and that report's sha256 is written into $O/manifest.json. W cannot be given.
# FULL rows: copied unedited from the A1 proper run (~/r1e/out/a1/FULL.jsonl, read only) for trajectories 7-20 and never recomputed;
# the rest are computed with the A1 FULL arm. Output ~/r1e/out/a3/. Early look at 20 held-out trajectories (99.9%), as A1 Amendment 1.
# Resume: rerun this script after a stop. It refuses while the recorded pid is alive; a stale pid is a stop and is resumed. The
# manifest (code md5s, arguments, selection digest) must match the first attempt's or the python side refuses; a trajectory killed
# mid-way is redone. Every attempt has its own id and its own provenance, stdout, run log and watchdog output.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/a3
A1=$HOME/r1e/out/a1
SEL=$HOME/r1e/out/a3sel
PY=~/r1e/venv/bin/python
ATT=$(date +%Y%m%dT%H%M%S)
if [ -e $O/pid ] && kill -0 "$(cat $O/pid)" 2>/dev/null; then echo "a3 is running (pid $(cat $O/pid))"; exit 1; fi
if pgrep -f 'a1_(run|e|e67|a3)\.py' > /dev/null; then echo "another a1 job is running"; exit 1; fi
if [ -n "$1" ]; then echo "no arguments: the window is the stage-1 selection"; exit 1; fi
$PY - <<'EOF' || { echo "launch gate FAILED"; exit 1; }
import subprocess, sys
sys.path.insert(0, "/home/duster/r1e/code/a1")
import a1_core as c
m = c.parse_free(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
print("free:", m)
sys.exit(0 if c.launch_ok(m) else 1)
EOF
mkdir -p $O
[ -e $O/pid ] && echo "attempt $ATT resumes after a stopped attempt (stale pid $(cat $O/pid))"
{
  echo "A3 stage 2 (held-out proper run), attempt $ATT, launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- reference rows (read only)"
  md5sum $A1/FULL.jsonl
  echo "--- stage-1 report"
  md5sum $SEL/a3_report.json
  sha256sum $SEL/a3_report.json
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_a3.py --stage proper --out $O --ref-dir $A1 --sel-report $SEL/a3_report.json --early-at 20 --cap 16384 --attempt $ATT"
  free -m
} > $O/provenance.$ATT.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_a3.py --stage proper --out $O --ref-dir $A1 --sel-report $SEL/a3_report.json --early-at 20 --cap 16384 --attempt $ATT > $O/stdout.$ATT.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.$ATT.out 2>&1 < /dev/null &
echo "a3 attempt $ATT pid $PID, watchdog pid $!"
