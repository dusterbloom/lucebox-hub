"""Diagnostics E3 (oracle exact source block) and E2 (router recall) of PREREG "Diagnostics E2, E3, E5 before A2".
One build pass per trajectory (the A1 shipping state via r2_core.gen_fblk_window with window 0 and an on_block hook that keeps each
block's exact K/V), then per retained probe: E3 answers with the source blocks swapped back to exact, E2 block ranks from two routers.
  python3 a1_e.py --out ~/r1e/out/e3e2 --pilot-dir ~/r1e/out/a1_pilot [--n 6] [--ship-check-n 1]
Helpers above the driver are pure (torch only) and tested in test_a1_e.py."""
import argparse
import json
import math
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as core  # noqa: E402

BLOCK, CHUNK = 640, 64
KS = (1, 2, 4, 8, 16)
FAR = 16384                                  # "more than 16K tokens back"


# ---------------------------------------------------------------- pure helpers

def overlap_blocks(lo, hi, n_blocks, block=BLOCK):
    """Sealed blocks (0 .. n_blocks - 1, `block` tokens each) that overlap the token span [lo, hi]; the open tail is not a block."""
    return [j for j in range(n_blocks) if j * block <= hi and (j + 1) * block > lo]


def locate_span(decode, ids, lo, hi, answer):
    """(first, last) token indices (absolute, inclusive) of the first occurrence of `answer` in the text of ids[lo:hi], or None.
    `decode(list_of_ids) -> str`; token boundaries come from the decoded length of growing prefixes (a gold string may start or end
    mid-token)."""
    text = decode(ids[lo:hi])
    at = text.find(answer)
    if at < 0:
        return None

    def first_token_reaching(chars):                 # smallest t such that the first t tokens decode to >= `chars` characters
        a, b = 1, hi - lo
        while a < b:
            m = (a + b) // 2
            if len(decode(ids[lo:lo + m])) >= chars:
                b = m
            else:
                a = m + 1
        return a

    return lo + first_token_reaching(at + 1) - 1, lo + first_token_reaching(at + len(answer)) - 1


def oracle_triples(ship, comp, exact, swap, tail, linear):
    """The shipping triples with the blocks in `swap` replaced (not duplicated) by their exact K/V. `comp[j]` / `exact[j]` hold, per
    layer, the block's (K, beta, V) (None on linear layers); the open tail (`tail` rows) and the DeltaNet entries come from `ship`."""
    swap = set(swap)
    out = []
    for i, t in enumerate(ship):
        if i in linear:
            out.append(t)
            continue
        pieces = [(exact if j in swap else comp)[j][i] for j in range(len(comp))]
        if tail:
            pieces.append(tuple(x[:, :, -tail:] for x in t))
        out.append(tuple(torch.cat([p[c] for p in pieces], 2) for c in range(3)))
    return tuple(out)


def router_mass(q, keys, beta, spans):
    """(n, B): per question token, each block's share of the attention mass its compacted slots receive, sum over its slots of
    exp(q.k/sqrt(d) + beta) normalised over the blocks (max-subtracted), mean over layers and query heads. q: (L, H, n, d); keys:
    per layer (Hkv, S, d); beta: per layer (Hkv, S); spans: [(start, end)] slot ranges of the blocks."""
    L, H, n, d = q.shape
    acc = 0
    for l in range(L):
        g = H // keys[l].shape[0]
        k = keys[l].float().repeat_interleave(g, 0)
        b = beta[l].float().repeat_interleave(g, 0)
        logit = q[l].float() @ k.transpose(1, 2) / math.sqrt(d) + b[:, None, :]         # (H, n, S)
        e = torch.exp(logit - logit.max(-1, keepdim=True).values)
        mass = torch.stack([e[..., s:t].sum(-1) for s, t in spans], -1)                  # (H, n, B)
        acc = acc + (mass / mass.sum(-1, keepdim=True)).mean(0)
    return acc / L


def pooled_chunk_keys(layer_keys, chunk=CHUNK):
    """kvflash_qk pooling: per layer K (Hkv, n, d) with n a multiple of `chunk` -> (C, L, Hkv, d): the post-RoPE keys summed over
    each chunk and L2-normalised with the engine's +1e-6."""
    out = []
    for K in layer_keys:
        Hkv, n, d = K.shape
        assert n % chunk == 0
        s = K.float().view(Hkv, n // chunk, chunk, d).sum(2).permute(1, 0, 2)            # (C, Hkv, d)
        out.append(s / (s.norm(dim=-1, keepdim=True) + 1e-6))
    return torch.stack(out, 1)


def qk_chunk_scores(pooled, q):
    """kvflash_qk_chunk_scores for every question token: (C, n) = mean over layers of the max over the query heads of the cosine
    between the (+1e-6 normalised) query and the chunk's pooled key of the head's KV group. pooled: (C, L, Hkv, d); q: (L, H, n, d)."""
    L, H, n, d = q.shape
    qn = q.float() / (q.float().norm(dim=-1, keepdim=True) + 1e-6)
    pk = pooled.repeat_interleave(H // pooled.shape[2], 2)                               # (C, L, H, d)
    cos = torch.einsum("clhd,lhnd->clhn", pk, qn)
    return cos.max(2).values.mean(1)


def block_scores(chunk_scores, per_block):
    """(C, n) chunk scores -> (B, n) block scores: the max over each block's chunks."""
    C, n = chunk_scores.shape
    return chunk_scores.view(C // per_block, per_block, n).max(1).values


def aggregate_tokens(scores):
    """(B, n) -> {"max": max over question tokens, "last": the last token only}, each (B,)."""
    return {"max": scores.max(1).values, "last": scores[:, -1]}


def best_rank(scores, sources):
    """1-based rank of the best-scoring source block (ties do not push it down)."""
    best = max(float(scores[j]) for j in sources)
    return 1 + int((scores > best).sum())


def recall_table(ranks, ks=KS):
    if not ranks:
        return {"n": 0}
    qs = statistics.quantiles(ranks, n=4) if len(ranks) > 1 else [ranks[0]] * 3
    return {"n": len(ranks), "recall": {str(k): sum(r <= k for r in ranks) / len(ranks) for k in ks},
            "median_rank": float(statistics.median(ranks)), "mean_rank": sum(ranks) / len(ranks), "quartiles": qs,
            "best": min(ranks), "worst": max(ranks)}


# ---------------------------------------------------------------- driver

ROUTERS = ("a_max", "a_last", "b_max", "b_last")


def _done_ids(path):
    return {r["id"] for r in core.read_rows(path)}


def run_trajectory(ctx, out, pilot_dir, tid, traj, tasks, full_rows, ship_rows, check_ship, log):
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    paths = {n: os.path.join(out, n + ".jsonl") for n in ("E3", "E3SPEC", "e2", "ship_check")}
    ret = core.retained_probe_ids(H, tasks, full_rows)
    plan = core.restrict(core.plan_trajectory(H, traj, tasks), ret)
    if not plan.cuts:
        return {"traj": tid, "n_rows": 0}
    done = _done_ids(paths["e2"])
    exact, comp, pooled = [], [], []
    full_layers = [i for i in range(len(S.layer_types)) if i not in S.linear]

    def on_block(j, a, b, ex, cp):
        exact.append(ex)
        comp.append(cp)
        pooled.append(pooled_chunk_keys([ex[i][0][0] for i in full_layers]))

    segs = core.segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    stats = {"traj": tid, "n_rows": 0, "build_s": 0.0, "answer_s": 0.0, "e2_s": 0.0}
    t_mark = time.time()
    for k, t_tr, t_len, info in r2.gen_fblk_window(R, segs, list(range(len(segs))), 0, on_block=on_block):
        stats["build_s"] += time.time() - t_mark
        cut, nb, tail = plan.cuts[k], info["blocks"], info["tail_tokens"]
        assert len(comp) == nb and t_len == cut.end
        spans, pos = [], 0
        for j in range(nb):
            n_slots = comp[j][full_layers[0]][0].shape[2]
            spans.append((pos, pos + n_slots))
            pos += n_slots
        for item in cut.items:
            t = item.task
            if t["id"] in done:
                continue
            t0 = time.time()
            gold_len = len(H.ids(t["answer"]))
            lo = t["src_tok"]
            span = locate_span(lambda sel: S.tok.decode(sel, skip_special_tokens=False), plan.stream, lo, cut.end, t["answer"])
            located = span is not None
            s_tok, e_tok = span if located else (lo, lo + gold_len)
            sw = {"E3": overlap_blocks(s_tok, e_tok, nb), "E3SPEC": overlap_blocks(lo, lo + gold_len, nb)}
            cache = {}
            for arm, blocks in sw.items():
                key = tuple(blocks)
                if key not in cache:
                    trip = oracle_triples(t_tr, comp, exact, blocks, tail, S.linear)
                    tm = {}
                    gen = r2.answer_ids(S, trip, t_len, item.suffix, item.max_new, timing=tm)
                    cache[key] = (gen, tm)
                    del trip
                gen, tm = cache[key]
                text = S.tok.decode(core.strip_stop(gen, S.eos), skip_special_tokens=False)
                extra = {"cut_index": cut.cut_index, "ctx_len": t_len, "decode_tokens": len(gen), "swapped_blocks": blocks,
                         "located": located, "src_span": [s_tok, e_tok], "n_blocks": nb,
                         "timings": {"prefill_ms": round(tm["prefill_s"] * 1000, 1), "decode_ms": round(tm["decode_s"] * 1000, 1)}}
                core.append_row(paths[arm], core.make_row(item, arm, gen, text, ctx.parse(text, plan.tools), S.eos, extra))
                stats["n_rows"] += 1
            stats["answer_s"] += time.time() - t0
            if tid in check_ship:
                pr = check_ship[tid].get(t["id"])
                gen = r2.answer_ids(S, t_tr, t_len, item.suffix, item.max_new)
                core.append_row(paths["ship_check"], {"id": t["id"], "traj": tid, "equal": pr is not None and pr == gen,
                                                      "pilot_row": pr is not None, "n": len(gen)})
            t1 = time.time()
            row = {"id": t["id"], "traj": tid, "type": t.get("type"), "distance_tok": t["distance_tok"], "ctx_len": t_len,
                   "n_blocks": nb, "src_blocks": sw["E3"], "located": located, "eligible": bool(located and sw["E3"])}
            if row["eligible"]:
                ids = torch.tensor([item.suffix], dtype=torch.long).to(S.dev)
                _, _, q = r2.extend(S.model, S.layer_types, t_tr, t_len, ids, want_queries=True)
                keys = [t_tr[i][0][0, :, :pos] for i in full_layers]
                betas = [t_tr[i][1][0, :, :pos] for i in full_layers]
                a = router_mass(q, keys, betas, spans)                                    # (n, B)
                b = block_scores(qk_chunk_scores(torch.cat(pooled), q), BLOCK // CHUNK)   # (B, n)
                scores = {"a": a.T, "b": b}
                row["ranks"] = {f"{r}_{agg}": best_rank(v, sw["E3"]) for r, m in scores.items()
                                for agg, v in aggregate_tokens(m).items()}
                del q
            core.append_row(paths["e2"], row)
            stats["e2_s"] += time.time() - t1
        a_, r_ = core.mem_gib()
        log(f"[e3e2] {tid[-8:]} cut {k + 1}/{len(segs)} ctx {t_len} blocks {nb} peak alloc {a_:.1f} reserved {r_:.1f} GiB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot-dir", default=os.path.expanduser("~/r1e/out/a1_pilot"))
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ship-check-n", type=int, default=1, help="re-answer the shipping arm on the first N trajectories and compare to the pilot")
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    import tool_parse
    H = core.Harness(core.default_paths())
    m = core.load_model(a.model, a.mem_cap_gib, log)
    ctx = SimpleNamespace(S=m.S, R=m.R, H=H, parse=tool_parse.parse)
    trajs = H.load_trajs()
    order = list(trajs)[: a.n]
    tasks = H.load_tasks()
    full_rows = core.read_rows(os.path.join(a.pilot_dir, "FULL.jsonl"))
    ship_rows = core.read_rows(os.path.join(a.pilot_dir, "SHIP.jsonl"))
    check = {tid: {r["id"]: r["gen_ids"] for r in ship_rows if r["traj"] == tid} for tid in order[: a.ship_check_n]}
    for tid in order:
        tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
        st = run_trajectory(ctx, a.out, a.pilot_dir, tid, trajs[tid], tp, full_rows, ship_rows, check, log)
        core.append_row(os.path.join(a.out, "progress.jsonl"), {**st, "arm": "E3E2", "peak_alloc_gib": core.mem_gib()[0],
                                                              "peak_reserved_gib": core.mem_gib()[1]})
    sc = core.read_rows(os.path.join(a.out, "ship_check.jsonl"))
    bad = [r["id"] for r in sc if not r["equal"]]
    with open(os.path.join(a.out, "ship_check.json"), "w") as f:
        json.dump({"checked": len(sc), "equal": len(sc) - len(bad), "mismatch_ids": bad}, f, indent=1)
    log(f"[e3e2] ship check: {len(sc) - len(bad)} of {len(sc)} equal to the pilot's SHIP rows")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
