"""Smoke (d) inputs. OFF runs f9b0's p12 probes (they fit the pool) after two different overflow histories:
  A: 08db's p36 probes, then f9b0 (the recorded order of ../../rlru4/drift)
  B: 08db's p12, p24 and p36 probes, then f9b0 (closer to the full run's history)
Both histories overflow, so f9b0's primer takes the pooled restore at the same rung in both (a fitting history would take the
dense path at an exact length instead). The only difference is the physical layout the history leaves behind.
run_arm.py runs tasks in task-file order (--retained is a set).
    python3 -I make_tasks.py   -> tasks.jsonl, ids_A.json, ids_B.json"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
A_T, F_T = "chatcmpl-08db2cebdcf5a106502475fba3d0c14f", "chatcmpl-f9b0f6072abe7d7d31ee12992f5fd055"
drift = json.load(open(os.path.join(D, "..", "..", "rlru4", "drift", "ids.json")))
retained = json.load(open(os.path.join(D, "..", "..", "rlru4", "retained.json")))
rows = [l for l in open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl")]
open(os.path.join(D, "tasks.jsonl"), "w").writelines([l for l in rows if json.loads(l)["traj"] == A_T] + [l for l in rows if json.loads(l)["traj"] == F_T])
json.dump(drift, open(os.path.join(D, "ids_A.json"), "w"))
json.dump(drift + [i for i in retained if i.startswith(A_T + ":p12:") or i.startswith(A_T + ":p24:")], open(os.path.join(D, "ids_B.json"), "w"))
print(len(drift), "ids in A,", len(json.load(open(os.path.join(D, "ids_B.json")))), "in B")
