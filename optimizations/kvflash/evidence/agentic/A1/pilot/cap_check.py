"""Which probe failures can come from the 64-token answer cap?  python3 -I a1_trunc.py
Exact match after normalize() means a capped answer can only lose a correct gold if the gold itself does not fit.
Rows: A1 pilot FULL/SHIP (trajectories 1-6) and B0 FULL/POOL16K probe rows on the same trajectories."""
import json
import sys

sys.path.insert(0, "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/agentic_eval")
import probes  # noqa: E402


EV = "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/evidence/agentic/"

tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
N = probes.normalize


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if tasks.get(r["id"], {}).get("kind") == "probe"}


def ok(r):
    return "error" not in r and probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"])


def gold_len(i):
    return len(tasks[i]["answer"]) // 3  # ponytail: ~3 chars per token for code, no tokenizer on this host


full, ship = load(EV + "A1/pilot/FULL.jsonl"), load(EV + "A1/pilot/SHIP.jsonl")
trajs = {r["traj"] for r in full.values()}
b0f = {i: r for i, r in load(EV + "B0_3090/FULL.jsonl").items() if r["traj"] in trajs}
b0p = {i: r for i, r in load(EV + "B0_3090/POOL16K.jsonl").items() if r["traj"] in trajs}

ids = sorted(full)
long_gold = [i for i in ids if gold_len(i) >= 60]
print(f"probes {len(ids)}; gold tokens max {max(map(gold_len, ids))}; gold >= 60 tokens: {len(long_gold)}")
for name, R in (("A1 FULL", full), ("A1 SHIP", ship), ("B0 FULL", b0f), ("B0 POOL16K", b0p)):
    capped = [r for r in R.values() if r["completion_tokens"] >= 64 or r["finish"] == "length"]
    cut_gold = [r for r in capped if N(tasks[r["id"]]["answer"]).startswith(N(r.get("content") or "")[:-8])]
    overran = [r for r in capped if N(r.get("content") or "").startswith(N(tasks[r["id"]]["answer"]))]
    think = sum(1 for r in R.values() if "<think>" in (r.get("content") or "") or (r.get("reasoning") or ""))
    print(f"{name:11s} rows {len(R):4d} correct {sum(map(ok, R.values())):4d} at cap {len(capped):3d} "
          f"(cap cut a gold prefix {len(cut_gold)}, gold then kept going {len(overran)}) "
          f"max tokens {max(r['completion_tokens'] for r in R.values())} with reasoning/think {think}")

print("\nA1 SHIP rows at the cap, retained probes (FULL correct):")
for i in sorted(ship):
    r = ship[i]
    if (r["completion_tokens"] >= 64 or r["finish"] == "length") and ok(full[i]):
        t = tasks[i]
        print(f"- {t['type']:12s} gold {gold_len(i):2d} tok {t['answer'][:60]!r}\n    SHIP {(r.get('content') or '')[:150]!r}")
