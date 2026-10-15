"""E1 analysis: recall@k of the target-QK index at prefill end, on the retained B0 probes.

Usage: python3 -I analyze_e1.py <rows.jsonl> <qk_dump.jsonl> <summary.json> [--tasks T] [--traj TR] [--retained R] [--b0 POOL16K.jsonl]

Matching: the harness sends one primer request and then its probes per (trajectory, cut); every request is one
do_prefill, so the dump has one line per request in request order. A row and a dump line match when
dump.prompt_len == row.prompt_tokens (the full prompt token count); rows without a line are reported.

Fact position: tasks carry src_tok = the start of the assistant call that produced the tool message, not the
fact. The fact is located by rendering the trajectory with the harness template, finding the answer text (for
`defname` the stated line) in the tool message, and mapping its characters to tokens with the harness tokenizer.
A fact hits at k when any 64-token chunk overlapping its token span is in the top k of the sealed chunks.
`lit` is the literal PREREG rule: the chunks covering [src_tok, src_tok + gold length], which is the start of the
source message, not the fact. `msg` is the generous variant: any chunk overlapping the whole source message.

Rank: 1 + number of sealed chunks with a strictly larger score than the best-scoring covering chunk.
A covering chunk without pooled keys (score -2) counts as not found. Facts in the unsealed tail are
`tail` (always resident, no rank)."""
import bisect
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "agentic_eval"))
sys.path.insert(0, AE)

KS = (8, 16, 32, 64, 128)
RANK_BINS = ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9))
OLD = 16384


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rank_of(scores, cover):
    """Best (smallest) rank of any covering chunk; None when every covering chunk is unscorable (-2) or none sealed."""
    cs = [scores[c] for c in cover if c < len(scores) and scores[c] > -1.999]
    if not cs:
        return None
    best = max(cs)
    return 1 + sum(1 for s in scores if s > best)


def hit_at(rank, k):
    return rank is not None and rank <= k


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "min": xs[0], "max": xs[-1],
            "mean": sum(xs) / len(xs)}


def summarize(recs, variant, which):
    """recs: probe records. variant: 'last' | 'max'. which: 'fact' | 'msg'."""
    rk = [r["rank"][which][variant] for r in recs]
    tail = [r for r in recs if r["tail"][which]]
    out = {"n": len(recs), "n_tail_unsealed": len(tail)}
    scored = [r for r in recs if not r["tail"][which]]
    out["n_ranked"] = len(scored)
    out["n_no_score"] = sum(1 for r in scored if r["rank"][which][variant] is None)
    rks = [r["rank"][which][variant] for r in scored]
    out["recall"] = {}
    for k in KS:
        # tail facts are resident and always reachable: counted as hits (they never need the router)
        h = sum(1 for x in rks if hit_at(x, k)) + len(tail)
        ch = sum(min(1.0, k / r["n_chunks"]) for r in recs)
        out["recall"][str(k)] = {"hits": h, "n": len(recs), "rate": h / len(recs) if recs else None,
                                 "chance_rate": ch / len(recs) if recs else None}
    out["rank_hist"] = {("%d-%d" % b if b[1] < 10**9 else "%d+" % b[0]): sum(1 for x in rks if x is not None and b[0] <= x <= b[1])
                        for b in RANK_BINS}
    out["rank_hist"]["none"] = sum(1 for x in rks if x is None)
    out["rank_stats"] = quantiles([x for x in rks if x is not None])
    out["rank_frac_stats"] = quantiles([x / r["n_chunks"] for x, r in zip(rks, scored) if x is not None])
    # non-resident at prefill end: the probes a router would have to fetch
    nr = [r for r in scored if not r["resident"][which]]
    out["n_resident_at_end"] = len(scored) - len(nr) + len(tail)
    out["non_resident"] = {"n": len(nr), "recall": {str(k): {"hits": sum(1 for r in nr if hit_at(r["rank"][which][variant], k)),
                                                             "n": len(nr)} for k in KS}}
    return out


def locate_facts(task, traj, cache):
    """Token spans (fact, message) of the probe's source in the full render. Returns dict or None."""
    import convert
    tid = task["traj"]
    if tid not in cache:
        msgs, tools = traj["messages"], traj["tools"]
        full = convert.render(msgs, tools)
        enc = convert.tokenizer().encode(full)
        cache[tid] = (full, [e for (_, e) in enc.offsets], msgs, tools, {})
    full, ends, msgs, tools, rc = cache[tid]
    i = task["msg_index"]
    j = max(k for k in range(i) if msgs[k]["role"] == "assistant")
    if j not in rc:
        rc[j] = len(convert.render(msgs[:j], tools))
    start = rc[j]
    nxt = [k for k in range(i + 1, len(msgs)) if msgs[k]["role"] == "assistant"]
    end = rc[nxt[0]] if nxt and nxt[0] in rc else None
    if end is None and nxt:
        rc[nxt[0]] = len(convert.render(msgs[:nxt[0]], tools))
        end = rc[nxt[0]]
    if end is None:
        end = len(full)
    ans = task["answer"]
    pos = -1
    if task["type"] == "defname":
        m = re.search(r"line (\d+)\?", task["question"])
        if m:
            lm = re.compile(r"^\s*%s\t" % m.group(1), re.M).search(full, start, end)
            if lm:
                pos = full.find(ans, lm.end(), end)
    if pos < 0:
        pos = full.find(ans, start, end)
    if pos < 0:
        return None
    tok = lambda c: bisect.bisect_right(ends, c)
    f0, f1 = tok(pos), tok(pos + len(ans) - 1)
    return {"fact": (f0, f1), "msg": (tok(start), tok(end - 1)), "lit": (tok(start), tok(start) + f1 - f0),
            "src_tok_check": tok(start)}


def main():
    rows_p, dump_p, out_p = sys.argv[1:4]
    tasks_p = arg("--tasks", "/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl")
    traj_p = arg("--traj", "/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl")
    ret_p = arg("--retained", os.path.join(HERE, "..", "B0_3090", "retained.json"))
    b0_p = arg("--b0", os.path.join(HERE, "..", "B0_3090", "POOL16K.jsonl"))
    retained = set(json.load(open(ret_p)))
    tasks = {t["id"]: t for t in map(json.loads, open(tasks_p)) if t["kind"] == "probe" and t["id"] in retained}
    trajs = {}
    for l in open(traj_p):
        r = json.loads(l)
        trajs[r["trajectory_id"]] = r
    rows = [json.loads(l) for l in open(rows_p)]
    dumps = [json.loads(l) for l in open(dump_p)]
    b0 = {}
    if os.path.exists(b0_p):
        b0 = {r["id"]: r for r in map(json.loads, open(b0_p))}

    # match rows to dump lines in order
    di = 0
    matched, unmatched_rows, skipped_dumps = {}, [], 0
    for r in rows:
        if "error" in r or r.get("prompt_tokens") is None:
            unmatched_rows.append(r["id"])
            continue
        while di < len(dumps) and dumps[di]["prompt_len"] != r["prompt_tokens"]:
            di += 1
            skipped_dumps += 1
        if di < len(dumps):
            matched[r["id"]] = dumps[di]
            di += 1
        else:
            unmatched_rows.append(r["id"])

    cache, recs, bad = {}, [], []
    for tid, t in tasks.items():
        d = matched.get(tid)
        if d is None:
            bad.append((tid, "no dump line"))
            continue
        loc = locate_facts(t, trajs[t["traj"]], cache)
        if loc is None:
            bad.append((tid, "answer not found in source message"))
            continue
        n = d["n_chunks"]
        ct = d["chunk_tokens"]
        res = set(d["resident"])
        sl, sm = d["scores_last"], d["scores_max"]
        rec = {"id": tid, "type": t["type"], "distance_tok": t["distance_tok"], "n_chunks": n, "prompt_len": d["prompt_len"],
               "n_prefilled": d["committed"] - d["kv_offset"], "n_q_tokens": d["n_q_tokens"], "n_pooled": d["n_pooled"],
               "rank": {}, "tail": {}, "resident": {}}
        for which in ("fact", "lit", "msg"):
            a, b = loc[which]
            cover = list(range(a // ct, b // ct + 1))
            sealed_cover = [c for c in cover if c < n]
            rec["tail"][which] = len(sealed_cover) == 0
            rec["resident"][which] = any(c in res for c in sealed_cover)
            rec["rank"][which] = {"last": rank_of(sl, sealed_cover), "max": rank_of(sm, sealed_cover) if sm else None}
        rec["span"] = loc
        if b0.get(tid):
            rec["b0_answer_equal"] = (b0[tid].get("content") == next((r.get("content") for r in rows if r["id"] == tid), None))
        recs.append(rec)

    subsets = {"older_than_16K": [r for r in recs if r["distance_tok"] > OLD],
               "within_16K": [r for r in recs if r["distance_tok"] <= OLD], "all": recs}
    out = {"n_tasks_retained": len(tasks), "n_rows": len(rows), "n_dump_lines": len(dumps), "skipped_dump_lines": skipped_dumps,
           "unmatched_rows": len(unmatched_rows), "unusable_probes": bad[:50], "n_unusable": len(bad), "n_probes_analysed": len(recs),
           "k_chunks": list(KS), "chunk_tokens": 64, "variants": {}}
    for variant in ("last", "max"):
        feasible = all(r["n_q_tokens"] > 0 for r in recs) if recs else False
        if variant == "max" and not feasible:
            out["variants"][variant] = "not feasible (no per-token queries captured)"
            continue
        out["variants"][variant] = {which: {name: summarize(sub, variant, which) for name, sub in subsets.items()}
                                    for which in ("fact", "lit", "msg")}
    out["n_q_tokens"] = quantiles([r["n_q_tokens"] for r in recs])
    out["n_prefilled"] = quantiles([r["n_prefilled"] for r in recs])
    out["n_pooled_lt_n_chunks"] = sum(1 for r in recs if r["n_pooled"] < r["n_chunks"])
    if b0:
        eq = [r["b0_answer_equal"] for r in recs if "b0_answer_equal" in r]
        out["answers_equal_to_B0_POOL16K"] = {"equal": sum(eq), "n": len(eq)}
    json.dump(out, open(out_p, "w"), indent=1)
    json.dump(recs, open(out_p.replace(".json", ".per_probe.json"), "w"))
    print(json.dumps({k: out[k] for k in out if k != "variants"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
