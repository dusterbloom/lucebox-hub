"""The dump arm's own answers (no reselect, about OFF) by distance.   python3 -I off_score.py [arm dir]"""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "dump")
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
ids = set(json.load(open(os.path.join(D, "retained.json"))))
rows = [r for r in map(json.loads, open(os.path.join(ARM, "QK16KND.jsonl"))) if r["id"] in ids]
ok = {r["id"]: probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]) for r in rows}
old = [i for i in ok if tasks[i]["distance_tok"] > 16384]
print(f"{len(ok)} probes: correct {sum(ok.values()):.0f}, within 16K {sum(ok[i] for i in ok if i not in old):.0f}/{len(ok) - len(old)}, "
      f"beyond {sum(ok[i] for i in old):.0f}/{len(old)}")
