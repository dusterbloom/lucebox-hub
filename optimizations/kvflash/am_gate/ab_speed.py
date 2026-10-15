"""M7 speed check: 5 distinct ~19K-token prompts per server run, greedy, one server per run.
All arms: --kvflash 8192 --no-kv-cache, LUCE_KVFLASH_POLICY=lru, default KV type. Arms differ only in
the binary and in M7_EXTRA (e.g. "--kvflash-am-ratio 0.10" for the compaction-on arm).
Writes speed_run.json (texts + the server's own CACHE and DONE lines per prompt) and, when compaction
is on, the per-prompt "[kvflash-am] prefill:" summaries."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlib import *

N_PROMPTS = int(os.environ.get("M7_PROMPTS", "5"))
T = int(os.environ.get("M7_TOKENS", "20000"))
extra = os.environ.get("M7_EXTRA", "").split()
log = f"{ROOT}/server.log"
rec = []
with Server(log, env={"LUCE_KVFLASH_POLICY": "lru"}, extra=extra):
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
                        cache=[l for l in seg if "chat CACHE" in l], done=[l for l in seg if "chat DONE" in l],
                        am=[l for l in seg if "[kvflash-am] prefill:" in l]))
        print(json.dumps({k: v for k, v in rec[-1].items() if k != "text"}), flush=True)
json.dump(rec, open(f"{ROOT}/speed_run.json", "w"), indent=1)
