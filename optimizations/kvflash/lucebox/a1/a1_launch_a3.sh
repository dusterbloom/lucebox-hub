#!/bin/bash
# runs ON lucebox6, AFTER A3 stage 1 has exited and a1_a3_report.py sel has written ~/r1e/out/a3sel/a3_report.json (one job at a time):
# A3 stage 2, the held-out proper run (PREREG "A3 routed exact fetch"), probes only, trajectories 7 to 60, ONE configuration.
#   a1_launch_a3.sh              W and k from the stage-1 report's `selection.selected`
#   a1_launch_a3.sh 4096 4       W and k given (must be the stage-1 choice; it is recorded in provenance.txt)
# FULL rows: copied unedited from the A1 proper run (~/r1e/out/a1/FULL.jsonl, read only) for trajectories 7-20; the rest are computed
# with the A1 FULL arm. Output ~/r1e/out/a3/. Early look at 20 held-out trajectories (99.9%), as A1 Amendment 1.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/a3
A1=$HOME/r1e/out/a1
SEL=$HOME/r1e/out/a3sel
PY=~/r1e/venv/bin/python
if [ -e $O/pid ]; then echo "a3 dir already started"; exit 1; fi
if pgrep -f 'a1_(run|e|e67|a3)\.py' > /dev/null; then echo "another a1 job is running"; exit 1; fi
W=${1:-}
K=${2:-}
if [ -z "$W" ] || [ -z "$K" ]; then
  read W K < <($PY - $SEL/a3_report.json <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["selection"]["selected"]
if not s:
    sys.exit("stage 1 selected no configuration: amend the PREREG first")
w, k = s.split("_")
print(w[1:], k[1:])
EOF
  ) || { echo "could not read the stage-1 selection"; exit 1; }
fi
echo "configuration: W=$W k=$K"
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
  echo "A3 stage 2 (held-out proper run), launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES; W=$W k=$K"
  echo "--- reference rows (read only)"
  md5sum $A1/FULL.jsonl
  echo "--- stage-1 report"
  md5sum $SEL/a3_report.json
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_a3.py --stage proper --out $O --ref-dir $A1 --w $W --k $K --early-at 20 --cap 16384"
  free -m
} > $O/provenance.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_a3.py --stage proper --out $O --ref-dir $A1 --w $W --k $K --early-at 20 --cap 16384 > $O/stdout.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.out 2>&1 < /dev/null &
echo "a3 pid $PID, watchdog pid $!"
