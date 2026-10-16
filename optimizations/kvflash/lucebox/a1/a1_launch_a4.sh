#!/bin/bash
# runs ON lucebox6, AFTER A3 stage 2 (and any other GPU job) has exited, one job at a time (PREREG "A4 R-LRU offline pilot"):
# the R-LRU offline pilot, probes only, pilot trajectories 1 to 6. Arms A4_K4 and A4_K0 (64-row sink, no prefix; one build pass) and A4_K4P (exact
# prefix P; its own build pass), k = 4 or 0, 16,384-row cap at every cut. Reads ~/r1e/out/a1_pilot/FULL.jsonl as the reference (copied unedited
# into the output directory, md5 checked). Output ~/r1e/out/a4/.
# Resume: rerun this script after a stop. It refuses while the recorded pid is alive; a stale pid is a stop and is resumed. The run manifest
# (code md5s, arguments) must match the first attempt's or the python side refuses. Every attempt has its own id and its own provenance, stdout,
# run log and watchdog output; nothing from an earlier attempt is overwritten.
source ~/r1e/code/r1e/env.sh           # HIP_VISIBLE_DEVICES=1
export A1_TP_BUILD=$HOME/r1e/code/a1/tool_parser/build
O=$HOME/r1e/out/a4
PILOT=$HOME/r1e/out/a1_pilot
PY=~/r1e/venv/bin/python
ATT=$(date +%Y%m%dT%H%M%S%N)
if [ -e $O/pid ] && kill -0 "$(cat $O/pid)" 2>/dev/null; then echo "a4 is running (pid $(cat $O/pid))"; exit 1; fi
for d in e3e2 e5 e6 e7 e8 e9 a3sel a3; do if [ -e $HOME/r1e/out/$d/pid ] && kill -0 $(cat $HOME/r1e/out/$d/pid) 2>/dev/null; then echo "$d still running"; exit 1; fi; done
if pgrep -f 'a1_(run|e|e9|e67|a3|a4)\.py' > /dev/null; then echo "another a1 job is running"; exit 1; fi
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
  echo "A4 R-LRU offline pilot, attempt $ATT, launched $(date -Is); HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
  echo "--- reference rows copied unedited from a1_pilot (registered md5 93a29de59549954ebd9d46ffd9214047)"
  md5sum $PILOT/FULL.jsonl   # the copy in $O is made atomically by a1_a4.py (ensure_reference) and checked there
  echo "--- code md5"
  bash ~/r1e/md5set.sh ~/r1e/code/a1 ~/r1e/code/am_qwen35
  md5sum $A1_TP_BUILD/tool_parser_cli $A1_TP_BUILD/tokenize_cli ~/r1e/data/a1/*.jsonl ~/r1e/data/a1/*.jinja ~/r1e/data/a1/*.gguf ~/r1e/data/a1/request_ids.sha256
  echo "--- cmd: a1_a4.py --out $O --ref-dir $PILOT --n 6 --cap 16384 --attempt $ATT"
  free -m
} > $O/provenance.$ATT.txt
cd ~/r1e/code/repo
setsid nohup $PY ~/r1e/code/a1/a1_a4.py --out $O --ref-dir $PILOT --n 6 --cap 16384 --attempt $ATT > $O/stdout.$ATT.log 2>&1 < /dev/null &
PID=$!
echo $PID > $O/pid
setsid nohup $PY ~/r1e/code/a1/a1_watchdog.py $PID $O > $O/watchdog.$ATT.out 2>&1 < /dev/null &
echo "a4 attempt $ATT pid $PID, watchdog pid $!"
