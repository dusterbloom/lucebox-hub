"""M6 gate: 5 distinct ~20K-token prompts, greedy, one server per run, no compaction.
Both arms run with --kvflash 8192 --no-kv-cache, LUCE_KVFLASH_POLICY=lru and the same
LUCE_KVFLASH_B_TOKENS (main ignores it). Writes m6_run.json (texts + timings)."""
import sys, os, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlib import *

N_PROMPTS = int(os.environ.get("M6_PROMPTS", "5"))
T = int(os.environ.get("M6_TOKENS", "20000"))
env = {"LUCE_KVFLASH_POLICY": "lru", "LUCE_KVFLASH_B_TOKENS": os.environ.get("M6_B", "2048")}
log = f"{ROOT}/server.log"
rec = []
with Server(log, env=env):
    chat([{"role": "user", "content": filler(1500, salt=99)}], max_tokens=4)  # warm-up, not recorded
    for i in range(N_PROMPTS):
        pos = os.path.getsize(log)
        q = "\nQuestion: continue this list with the next 30 records in exactly the same format and write nothing else."
        o = chat([{"role": "user", "content": filler(T, salt=11 + i) + q}], max_tokens=256)
        time.sleep(0.7)
        with open(log, errors="replace") as f:
            f.seek(pos)
            seg = f.read().splitlines()
        rec.append(dict(prompt=i, text=o["text"], prompt_tokens=o["prompt_tokens"], ct=o["completion_tokens"],
                        cache=[l for l in seg if "chat CACHE" in l], done=[l for l in seg if "chat DONE" in l]))
        print(json.dumps({k: v for k, v in rec[-1].items() if k != "text"}), flush=True)
json.dump(rec, open(f"{ROOT}/m6_run.json", "w"), indent=1)
