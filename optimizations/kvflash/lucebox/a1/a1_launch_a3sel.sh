#!/bin/bash
# runs ON lucebox6, AFTER any other GPU job has exited (one job at a time): A3 stage 1, configuration selection (PREREG "A3 routed exact fetch"),
# probes only, pilot trajectories 1 to 6, W in {2, 6} blocks (1,280 and 3,840 tokens); k fills the 16,384-row budget at every cut.
# Reads ~/r1e/out/a1_pilot/FULL.jsonl as the reference, never writes there. Output ~/r1e/out/a3sel/.
# Resume: rerun this script after a stop. It refuses while the recorded pid is alive; a stale pid is a stop and is resumed. The run
# manifest (code md5s, arguments) must match the first attempt's or the python side refuses. Every attempt has its own id and its
# own provenance, stdout, run log and watchdog output; nothing from an earlier attempt is overwritten.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/a3sel
PILOT=$HOME/r1e/out/a1_pilot
PY=~/r1e/venv/bin/python
ATT=$(date +%Y%m%dT%H%M%S)
if [ -e $O/pid ] && kill -0 "$(cat $O/pid)" 2>/dev/null; then echo "a3sel is running (pid $(cat $O/pid))"; exit 1; fi
if pgrep -f 'a1_(run|e|e67|a3)\.py' > /dev/null; then echo "another a1 job is running"; exit 1; fi
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
  echo "A3 stage 1 (selection), attempt $ATT, launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- reference rows (read only)"
  md5sum $PILOT/FULL.jsonl
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_a3.py --stage sel --out $O --ref-dir $PILOT --n 6 --ws 2,6 --cap 16384 --attempt $ATT"
  free -m
} > $O/provenance.$ATT.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_a3.py --stage sel --out $O --ref-dir $PILOT --n 6 --ws 2,6 --cap 16384 --attempt $ATT > $O/stdout.$ATT.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.$ATT.out 2>&1 < /dev/null &
echo "a3sel attempt $ATT pid $PID, watchdog pid $!"
