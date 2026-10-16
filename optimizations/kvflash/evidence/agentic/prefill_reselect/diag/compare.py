"""Engine answer-time pool (KVFLASH_DEBUG_RESIDENT) vs the replay from E1's dump, per arm, on the diagnostic probes.
    python3 -I compare.py <worktree>
Per probe: Jaccard(actual, replay); fact chunk / fact+-2 / whole source message resident in the ACTUAL pool; share of the last 256
chunks resident; answer correct. Replay: off = E1's LRU resident set; m1, m2 = reselect W=4 on scores_last (the simulated policy)."""
import json
import lzma
import os
import re
import statistics as st
import sys

W_TREE = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(W_TREE, "optimizations/kvflash/evidence/agentic/E1_router_recall")
sys.path.insert(0, os.path.join(D, "agentic_eval"))
import probes  # noqa: E402

CT, BLOCKS = 64, 256
tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
recs = {r["id"]: r for r in json.load(open(os.path.join(E1, "summary.per_probe.json")))}
rows = [json.loads(l) for l in lzma.open(os.path.join(E1, "rows_QK16K.jsonl.xz"), "rt")]
dumps = [json.loads(l) for l in lzma.open(os.path.join(E1, "qk_dump.jsonl.xz"), "rt")]
matched, di = {}, 0
for r in rows:
    if "error" in r or r.get("prompt_tokens") is None:
        continue
    while di < len(dumps) and dumps[di]["prompt_len"] != r["prompt_tokens"]:
        di += 1
    if di < len(dumps):
        matched[r["id"]] = dumps[di]
        di += 1


def ranges(s):
    out = set()
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def replay(d, arm):
    if arm == "off":
        return set(d["resident"]) | {d["committed"] // CT}
    n, cur, s, w = d["n_chunks"], d["committed"] // CT, d["scores_last"], 128 if arm.endswith("t128") else 4
    prot = {0} | {c for c in range(n + 1) if c > cur - 1 - w}
    rest = sorted((c for c in range(n) if c not in prot), key=lambda c: (-s[c], c))
    return prot | set(rest[:max(0, BLOCKS - len(prot))])


DBG = re.compile(r"\[kvflash-dbg\] after prefill kv_offset=(-?\d+) committed=(\d+) resident=(\S*)")
for arm in ("off", "m1", "m2", "m2t128"):
    p = os.path.join(D, arm, "QK16KND.jsonl")
    if not os.path.exists(p):
        continue
    reqs = [json.loads(l) for l in open(p)]
    dbg = [m.groups() for m in map(DBG.search, open(os.path.join(D, arm, "QK16KND.server.log"))) if m]
    if len(dbg) != len(reqs):
        print(f"{arm}: {len(dbg)} debug lines for {len(reqs)} requests; alignment by order is unsafe, skipping")
        continue
    stats = []
    for r, (_, committed, res_s) in zip(reqs, dbg):
        if r.get("kind") != "probe" or r["id"] not in matched:
            continue
        i, res, t = r["id"], ranges(res_s), tasks[r["id"]]
        assert abs(int(committed) - r["prompt_tokens"]) <= 1, (i, committed, r["prompt_tokens"])
        rp = replay(matched[i], arm)
        fa, fb = recs[i]["span"]["fact"]
        ma, mb = recs[i]["span"]["msg"]
        cur = int(committed) // CT
        stats.append({"id": i, "within": t["distance_tok"] <= 16384, "ok": probes.score_exact(r.get("content") or "", t["answer"]),
                      "jac": len(res & rp) / len(res | rp), "fact": all(c in res for c in range(fa // CT, fb // CT + 1)),
                      "near": all(c in res for c in range(fa // CT - 2, fb // CT + 3) if c <= cur),
                      "msg": all(c in res for c in range(ma // CT, min(mb // CT, cur) + 1)),
                      "rp_fact": all(c in rp for c in range(fa // CT, fb // CT + 1)),
                      "rp_msg": all(c in rp for c in range(ma // CT, min(mb // CT, cur) + 1)),
                      "recent": sum(c in res for c in range(cur - BLOCKS + 1, cur + 1)) / BLOCKS})
    for name, S in (("within 16K", [s for s in stats if s["within"]]), ("beyond 16K", [s for s in stats if not s["within"]])):
        if not S:
            continue
        print(f"{arm:3} {name:10} n {len(S):>3}  ok {sum(s['ok'] for s in S):>3}  jaccard(actual, replay) median {st.median(s['jac'] for s in S):.2f}  "
              f"fact {sum(s['fact'] for s in S):>3}  fact+-2 {sum(s['near'] for s in S):>3}  msg {sum(s['msg'] for s in S):>3}  "
              f"recent share median {st.median(s['recent'] for s in S):.2f}  | replay fact {sum(s['rp_fact'] for s in S):>3} msg {sum(s['rp_msg'] for s in S):>3}")
    json.dump(stats, open(os.path.join(D, arm + "_pool.json"), "w"))
