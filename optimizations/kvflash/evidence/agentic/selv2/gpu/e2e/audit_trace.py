"""Before/after on the real coding traces: on6ref (ON6) vs gpu (selector v2 + GPU scorer), same binary 20e33b31.
Leak checks, held-out split, and concrete examples with picks vs fact chunks.
    python3 -I audit_trace.py"""
import hashlib
import json
import math
import re
import sys

R4 = "/mnt/kimi-k3/split-work/rlru4"
sys.path.insert(0, f"{R4}/agentic_eval")
import probes  # noqa: E402

G = "/mnt/kimi-k3/split-work/selv2/gpu"
W = "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/evidence/agentic"
CT, OLD = 64, 16384
SEL = re.compile(r"mode=4 scorer=exact k=(\d+) tail=(\d+) @(\d+) \(q tok \d+\) picked=\[([0-9,]*)\]")
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
ids = json.load(open(f"{R4}/retained.json"))
spans = {r["id"]: r["span"]["fact"] for r in json.load(open(f"{W}/E1_router_recall/summary.per_probe.json"))}
trajs = {r["trajectory_id"]: r for r in map(json.loads, open("/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl"))}


def arm(d, unit):
    rows = [json.loads(l) for l in open(f"{d}/QK16KND.jsonl")]
    blocks = open(f"{d}/QK16KND.server.log").read().split("] chat START ")[1:]
    assert len(rows) == len(blocks)
    out = {}
    for r, b in zip(rows, blocks):
        if r.get("kind") != "probe":
            continue
        m = SEL.search(b)
        sel = None
        if m:
            tail, at = int(m[2]), int(m[3])
            sel = {"fwc": (at - 1) // CT - tail, "picked": sorted(int(x) for x in m[4].split(",") if x), "at": at}
        out[r["id"]] = {"ans": r.get("content"), "ok": probes.score_exact(r.get("content") or "", tasks[r["id"]]["answer"]),
                        "sel": sel, "unit": unit, "fits": "mode=4: skip, fits" in b}
    return out


A = {"on6": arm(f"{G}/on6ref", 10), "v2": arm(f"{G}/v2", 2)}
full = {r["id"]: r for r in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/full/FULL.jsonl")) if r.get("kind") == "probe"}
split = lambda i: "AB"[int(hashlib.sha1(i.split(":")[0].encode()).hexdigest(), 16) % 2]
band = lambda i: "beyond" if tasks[i]["distance_tok"] > OLD else "within"


def held(a, i):
    s = A[a][i]["sel"]
    if not s or i not in spans:
        return "fits" if A[a][i]["fits"] else "no-sel"
    fa, fb = spans[i]
    cs = range(fa // CT, fb // CT + 1)
    if fb // CT >= s["fwc"]:
        return "in window"
    h = [c // A[a][i]["unit"] in s["picked"] for c in cs]
    return "held" if all(h) else "part" if any(h) else "missed"


def sign_p(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n)


print("== 1. answer leaks into the question (normalized answer is a substring of the question)")
leak = [i for i in ids if probes.normalize(tasks[i]["answer"]) in tasks[i]["question"]]
print(f"   {len(leak)} of {len(ids)}", leak[:5])
print("== 2. full KV on the current harness (FULL arm, older binary 13c65b5c; retained = FULL-correct by construction)")
print(f"   FULL correct on retained: {sum(probes.score_exact(full[i].get('content') or '', tasks[i]['answer']) for i in ids)}/{len(ids)}")
print("== 3. guess floor: correct answers when the fact is NOT held after reselect")
for a in A:
    cl = {}
    for i in ids:
        if band(i) == "beyond":
            k = held(a, i)
            n, c = cl.get(k, (0, 0))
            cl[k] = (n + 1, c + A[a][i]["ok"])
    print(f"   {a}: beyond class -> (probes, correct) {dict(sorted(cl.items()))}")
print("== 4. held-out split: v2 settings were chosen on split A (PREREG_selv2); E2E per split")
for sp in "AB":
    for b in ("beyond", "within"):
        S = [i for i in ids if split(i) == sp and band(i) == b]
        on6, v2 = (sum(A[a][i]["ok"] for i in S) for a in ("on6", "v2"))
        w = sum(A["v2"][i]["ok"] and not A["on6"][i]["ok"] for i in S)
        l_ = sum(A["on6"][i]["ok"] and not A["v2"][i]["ok"] for i in S)
        print(f"   split {sp} {b:6}: n {len(S):3}  on6 {on6:3}  v2 {v2:3}  discordant v2 {w} on6 {l_}  sign p {sign_p(w, l_):.2g}"
              f"  trajs {len({tasks[i]['traj'] for i in S})}")
print("== 5. beyond by probe type (on6 -> v2 correct / n)")
for t in probes.TYPES:
    S = [i for i in ids if band(i) == "beyond" and tasks[i]["type"] == t]
    print(f"   {t:13} n {len(S):3}  {sum(A['on6'][i]['ok'] for i in S):3} -> {sum(A['v2'][i]['ok'] for i in S):3}")
print("== 6. beyond by distance (tokens from source to cut)")
for lo, hi in ((16384, 24576), (24576, 32768), (32768, 1 << 30)):
    S = [i for i in ids if lo < tasks[i]["distance_tok"] <= hi]
    print(f"   ({lo}, {hi if hi < 1 << 29 else 'max'}] n {len(S):3}  on6 {sum(A['on6'][i]['ok'] for i in S):3} -> v2 {sum(A['v2'][i]['ok'] for i in S):3}")
print("== 6b. picks spent on the preamble (system prompt + tool schemas + task, tokens < boundaries['2']), per reselect")
for a in A:
    share, n = [], 0
    for i in ids:
        s = A[a][i]["sel"]
        if not s:
            continue
        pre = trajs[tasks[i]["traj"]]["boundaries"]["2"]
        u = A[a][i]["unit"]
        share.append(sum((x + 1) * u * CT <= pre for x in s["picked"]) / len(s["picked"]))
    share.sort()
    print(f"   {a}: {len(share)} reselects; median share of picked units wholly inside the preamble {share[len(share) // 2]:.2f}"
          f" (p10 {share[len(share) // 10]:.2f}, p90 {share[9 * len(share) // 10]:.2f})")


def show(i):
    t, tr = tasks[i], trajs[tasks[i]["traj"]]
    src = tr["messages"][t["msg_index"]]["content"]
    k = src.find(t["answer"])
    call = next((tc["function"]["arguments"] for m in tr["messages"][: t["msg_index"]] for tc in m.get("tool_calls") or []
                 if tc["id"] == tr["messages"][t["msg_index"]].get("tool_call_id")), "?")
    fa, fb = spans.get(i, (None, None))
    print(f"\n--- {i}  [{t['type']}, split {split(i)}, {band(i)}]")
    print(f"  trajectory: {len(tr['messages'])} messages; cut after message {t['cut_index'] - 1} at {t['cut_tok']} tokens; "
          f"source = tool result msg {t['msg_index']} at ~{t['src_tok']} tokens (distance {t['distance_tok']})")
    print(f"  source tool call: {call[:160]}")
    print(f"  source excerpt  : ...{src[max(0, k - 120):k]!r}>>>{t['answer']!r}<<<{src[k + len(t['answer']):k + len(t['answer']) + 80]!r}...")
    print(f"  question (appended user turn; with the suffix + template tail it is ~the last 64 prompt tokens v2 scores with):\n    {t['question']!r}")
    print(f"  gold: {t['answer']!r}   fact tokens [{fa}, {fb}] = chunks {fa // CT if fa is not None else '?'}-{fb // CT if fb is not None else '?'}")
    for a in A:
        r = A[a][i]
        s = r["sel"] or {}
        u = r["unit"]
        fu = sorted({c // u for c in range(fa // CT, fb // CT + 1)}) if fa is not None else []
        print(f"  {a:3}: unit {u} chunks, window from chunk {s.get('fwc')}; fact unit(s) {fu}; picked {s.get('picked')}"
              f"\n       -> {held(a, i)}; answer {r['ans']!r} {'OK' if r['ok'] else 'WRONG'}")


def pick(cond, n, key=lambda i: i):
    return sorted([i for i in ids if cond(i)], key=key)[:n]


print("\n== 7. examples")
gain = [i for i in ids if band(i) == "beyond" and A["v2"][i]["ok"] and not A["on6"][i]["ok"] and split(i) == "B"]
seen, ex = set(), []
for i in sorted(gain, key=lambda i: hashlib.md5(i.encode()).hexdigest()):
    if tasks[i]["type"] not in seen:
        seen.add(tasks[i]["type"])
        ex.append(i)
for i in ex[:3]:
    show(i)
print("\n  -- every probe v2 loses (on6 right, v2 wrong):")
for i in ids:
    if A["on6"][i]["ok"] and not A["v2"][i]["ok"]:
        show(i)
