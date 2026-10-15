"""A3 routed exact fetch (PREREG "A3 routed exact fetch"). The KVFlash 2.0b design on the B0 agentic probes, offline.

State at a cut (r2_core.gen_fblk_window, one build pass per trajectory and window W):
  P  the sealed blocks overlapping the rendered system message and tool schema stay exact (E7, `exact_blocks`);
  W  the last window_blocks(W) sealed blocks stay exact;
  the other sealed 640-token blocks are compacted to 64 slots (ratio 0.10), their exact K/V kept aside (on_block, as E3);
  the open tail and the DeltaNet state are as the generator leaves them.
Per probe: (1) routing pass: prefill the question suffix under that state, capture its queries (r2_core.extend), score every
compacted block (not P, not W) with the compacted-slot attention mass at the last question token (a1_e.router_mass); the pass's KV
is dropped. (2) fetch: the top k blocks swap to exact (a1_e.oracle_triples: the compacted entry is replaced), k lowered per cut so
that rows <= CAP. (3) answer: r2_core.answer_ids on the swapped state, exactly as A1.
  stage sel:    python3 a1_a3.py --stage sel --out OUT --ref-dir PILOT_DIR [--ws 4096,8192 --ks 4,8]   (6 pilot trajectories)
  stage proper: python3 a1_a3.py --stage proper --out OUT --ref-dir A1_DIR --w W --k K                  (trajectories 7-60)
Resumable per trajectory, window and arm: rerun the same command. Helpers above the driver are pure and tested in test_a1_a3.py."""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as core  # noqa: E402
import a1_e  # noqa: E402
import a1_e67 as e67  # noqa: E402

BLOCK, SLOTS = 640, 64                       # sealed block tokens; slots a compacted block keeps (ratio 0.10)
CAP = 16384                                  # rows of POOL16K (B0): the equal-memory budget
KV_BYTES_PER_TOKEN = 64 * 1024               # 16 full-attention layers x 4 KV heads x 256 x K and V x 2 bytes
SEL_WS, SEL_KS = (4096, 8192), (4, 8)
PILOT_N = 6                                  # the pilot trajectories; stage proper runs the rest
RECALL_KS = a1_e.KS


# ---------------------------------------------------------------- pure: names, row budget, candidates

def window_blocks(w_tokens, block=BLOCK):
    """W is a token budget; the window is the largest whole number of blocks inside it (4096 -> 6 blocks, 8192 -> 12)."""
    return w_tokens // block


def arm_name(w_tokens, k):
    return f"A3_W{w_tokens}_K{k}"


def route_name(w_tokens):
    return f"A3_W{w_tokens}_route"


def exact_set(nb, p, wb):
    """Sealed blocks that are exact at the cut without any fetch: the first p (system prefix) and the last wb (window)."""
    return set(range(min(p, nb))) | set(range(max(0, nb - wb), nb))


def candidates(nb, p, wb):
    """The compacted sealed blocks, the only ones the router may rank (never P, never W; the open tail is not a block)."""
    ex = exact_set(nb, p, wb)
    return [j for j in range(nb) if j not in ex]


def rows_at(nb, tail, p, wb, k, block=BLOCK, slots=SLOTS):
    """Rows per full-attention layer when k candidate blocks are fetched: exact blocks and fetched blocks at `block`, the other
    compacted blocks at `slots`, plus the open tail."""
    n_ex = len(exact_set(nb, p, wb))
    n_c = nb - n_ex
    assert 0 <= k <= n_c, (k, n_c)
    return block * (n_ex + k) + slots * (n_c - k) + tail


def choose_k(nb, tail, p, wb, k_req, cap=CAP, block=BLOCK, slots=SLOTS):
    """Largest k <= k_req (and <= the number of candidates) with rows <= cap. When even k = 0 is over the cap the cut is
    flagged over_budget and runs at k = 0."""
    n_c = len(candidates(nb, p, wb))
    r0 = rows_at(nb, tail, p, wb, 0, block, slots)
    k_fit = (cap - r0) // (block - slots) if r0 <= cap else 0
    k = max(0, min(k_req, n_c, k_fit))
    r = rows_at(nb, tail, p, wb, k, block, slots)
    return {"k_req": k_req, "k": k, "rows": r, "rows_k0": r0, "n_cand": n_c, "over_budget": r > cap, "lowered": k < min(k_req, n_c)}


def rank_candidates(scores, cand):
    """Candidate block ids by descending score; ties go to the lower block index."""
    s = [float(x) for x in scores]
    return [cand[i] for i in sorted(range(len(cand)), key=lambda i: (-s[i], cand[i]))]


def rank_of(ranked, sources):
    """1-based position of the best-ranked source block in `ranked`, or None when no source is a candidate."""
    pos = [ranked.index(j) + 1 for j in sources if j in ranked]
    return min(pos) if pos else None


def source_status(located, src_blocks, nb, p, wb):
    """E3's located-block definition, seen from the A3 state: where the fact's block(s) are before any fetch."""
    if not located:
        return "not_located"
    if not src_blocks:
        return "tail"
    return "candidate" if any(j in candidates(nb, p, wb) for j in src_blocks) else "exact_resident"


def a3_state(ship, comp, exact, nb, tail, p, wb, fetch, linear):
    """The state the answer sees: P, W and the fetched blocks exact (their compacted entry replaced, E3's swap), the other blocks
    compacted, the open tail and DeltaNet entries from `ship` (the generator's cut state). fetch=[] is `ship` itself."""
    return a1_e.oracle_triples(ship, comp, exact, exact_set(nb, p, wb) | set(fetch), tail, linear)


def select_config(table):
    """Stage 1 rule: highest retention, ties by fewer mean rows, among eligible configurations (no cut over the row cap at k = 0).
    `selected_unrestricted` ignores eligibility, for the report. table: {"W4096_K4": {correct, n, mean_rows, eligible}}."""
    def key(name):
        v = table[name]
        return (-v["correct"] / v["n"], v["mean_rows"], name)
    ok = sorted((n for n in table if table[n]["eligible"]), key=key)
    return {"selected": ok[0] if ok else None, "selected_unrestricted": min(table, key=key) if table else None,
            "order_eligible": ok}


def final_verdict(ci, bar=core.BAR):
    """PREREG A1 section 6 (A3 inherits it): PASS lo > bar; FAIL hi < bar, or INCONCLUSIVE with the point under the bar."""
    if ci["lo"] > bar:
        return "PASS"
    if ci["hi"] < bar or ci["mean"] < bar:
        return "FAIL"
    return "INCONCLUSIVE"


def seed_reference(ref_path, out_path, traj_ids):
    """Copy the reference FULL rows of `traj_ids` (raw lines, unedited) into out_path unless it already exists."""
    if os.path.exists(out_path):
        return 0
    keep, n = set(traj_ids), 0
    with open(ref_path) as f, open(out_path, "w") as g:
        for line in f:
            if line.strip() and json.loads(line)["traj"] in keep:
                g.write(line if line.endswith("\n") else line + "\n")
                n += 1
    return n


# ---------------------------------------------------------------- routing, fetch, answer (one probe)

def _full_rows(triples, full_layers):
    return triples[full_layers[0]][0].shape[2]


def route_a_last(S, st, suffix):
    """Routing pass. Prefill the question suffix under the cut state and capture its queries; the pass's KV is dropped. Score each
    candidate block by the attention mass its compacted slots get at the LAST question token (a1_e.router_mass: softmax over the
    candidate slots only, mean over layers and query heads). Returns (cand, scores over cand or None, timings)."""
    import r2_core as r2
    cand = candidates(st.nb, st.p, st.wb)
    if not cand:
        return cand, None, {"route_s": 0.0, "router_s": 0.0}
    ids = torch.tensor([suffix], dtype=torch.long).to(S.dev)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t0 = time.time()
    _, _, q = r2.extend(S.model, S.layer_types, st.triples, st.t_len, ids, want_queries=True)   # the new triples are not kept
    sync()
    t1 = time.time()
    fl = st.full_layers
    keys = [torch.cat([st.comp[j][i][0][0] for j in cand], 1) for i in fl]
    betas = [torch.cat([st.comp[j][i][1][0] for j in cand], 1) for i in fl]
    spans, pos = [], 0
    for j in cand:
        n = st.comp[j][fl[0]][0].shape[2]
        spans.append((pos, pos + n))
        pos += n
    a = a1_e.router_mass(q, keys, betas, spans)                                                  # (n, C)
    del q
    sync()
    return cand, a[-1], {"route_s": round(t1 - t0, 3), "router_s": round(time.time() - t1, 3)}


def answer_item(S, st, item, ks, cap=CAP, block=BLOCK, slots=SLOTS):
    """One probe: route once, then for each requested k fetch the top k_eff blocks and answer. Identical fetch sets are answered
    once. Returns {cand, ranked, route, arms: {k_req: budget fields + fetched + gen + timing}}."""
    import r2_core as r2
    cand, a_last, rt = route_a_last(S, st, item.suffix)
    ranked = rank_candidates(a_last, cand) if cand else []
    arms, done = {}, {}
    for k_req in ks:
        bud = choose_k(st.nb, st.tail, st.p, st.wb, k_req, cap, block, slots)
        fetched = ranked[:bud["k"]]
        key = tuple(sorted(fetched))
        if key not in done:
            trip = a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, fetched, st.linear)
            got = _full_rows(trip, st.full_layers)
            assert got == bud["rows"], f"state has {got} rows, budget says {bud['rows']}"
            tm = {}
            gen = r2.answer_ids(S, trip, st.t_len, item.suffix, item.max_new, timing=tm)
            done[key] = (gen, tm)
            del trip
        gen, tm = done[key]
        arms[k_req] = {**bud, "fetched": list(fetched), "fetch_bytes": bud["k"] * block * KV_BYTES_PER_TOKEN, "gen": gen, "timing": tm}
    return {"cand": cand, "ranked": ranked, "route": rt, "arms": arms}


# ---------------------------------------------------------------- driver

def _ids(path):
    return {r["id"] for r in core.read_rows(path)}


def run_trajectory(ctx, out, W, ks, tid, traj, tasks, full_rows, log, cap=CAP):
    """One trajectory, one window: build the P+W state once, answer every retained probe for every k in `ks`. Rows go to
    A3_W{W}_K{k}.jsonl (harness row schema) and A3_W{W}_route.jsonl (router ranks, fetch plan per k). Finished probes are skipped."""
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    wb = window_blocks(W)
    paths = {k: os.path.join(out, arm_name(W, k) + ".jsonl") for k in ks}
    rpath = os.path.join(out, route_name(W) + ".jsonl")
    done = {k: _ids(paths[k]) for k in ks}
    done_route = _ids(rpath)
    ret = core.retained_probe_ids(H, tasks, full_rows)
    todo = {t["id"] for t in tasks if t["id"] in ret and (t["id"] not in done_route or any(t["id"] not in done[k] for k in ks))}
    if not todo:
        return None
    plan = core.restrict(core.plan_trajectory(H, traj, [t for t in tasks if t["id"] in todo]), todo)
    end = e67.system_end(plan.stream, S.tok.convert_tokens_to_ids("<|im_end|>"))
    assert S.tok.decode(plan.stream[:3], skip_special_tokens=False).startswith("<|im_start|>system")
    p = e67.prefix_blocks(end)
    full_layers = [i for i in range(len(S.layer_types)) if i not in S.linear]
    exact, comp = [], []
    segs = core.segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    stats = {"traj": tid, "W": W, "window_blocks": wb, "p_blocks": p, "system_end_tok": end, "n_rows": 0, "build_s": 0.0,
             "answer_s": 0.0}
    log(f"[a3] {tid[-8:]} W {W} ({wb} blocks): system+tools end at token {end}, {p} exact prefix blocks")
    t_mark = time.time()
    for k_, trip, t_len, info in r2.gen_fblk_window(R, segs, list(range(len(segs))), wb, exact_blocks=p,
                                                    on_block=lambda j, a, b, ex, cp: (exact.append(ex), comp.append(cp))):
        stats["build_s"] += time.time() - t_mark
        cut, nb, tail = plan.cuts[k_], info["blocks"], info["tail_tokens"]
        assert len(comp) == nb and t_len == cut.end
        st = SimpleNamespace(triples=trip, t_len=t_len, nb=nb, tail=tail, comp=comp, exact=exact, p=p, wb=wb, linear=S.linear,
                             full_layers=full_layers)
        for item in cut.items:
            t = item.task
            t0 = time.time()
            res = answer_item(S, st, item, ks, cap)
            lo = t["src_tok"]
            gold_len = len(H.ids(t["answer"]))
            span = a1_e.locate_span(lambda sel: S.tok.decode(sel, skip_special_tokens=False), plan.stream, lo, cut.end, t["answer"])
            src = a1_e.overlap_blocks(*span, nb) if span else []
            status = source_status(span is not None, src, nb, p, wb)
            wall = round(time.time() - t0, 2)
            for k in ks:
                if t["id"] in done[k]:
                    continue
                a = res["arms"][k]
                text = S.tok.decode(core.strip_stop(a["gen"], S.eos), skip_special_tokens=False)
                extra = {"cut_index": cut.cut_index, "ctx_len": t_len, "decode_tokens": len(a["gen"]), "W": W, "window_blocks": wb,
                         "p_blocks": p, "n_blocks": nb, "tail_tokens": tail, "k_req": k, "k": a["k"], "fetched_blocks": a["fetched"],
                         "rows": a["rows"], "rows_k0": a["rows_k0"], "over_budget": a["over_budget"], "n_cand": a["n_cand"],
                         "fetch_bytes": a["fetch_bytes"], "route_s": res["route"]["route_s"], "probe_wall_s": wall,
                         "src_blocks": src, "src_status": status,
                         "timings": {"prefill_ms": round(a["timing"]["prefill_s"] * 1000, 1),
                                     "decode_ms": round(a["timing"]["decode_s"] * 1000, 1)}}
                core.append_row(paths[k], core.make_row(item, arm_name(W, k), a["gen"], text, ctx.parse(text, plan.tools), S.eos, extra))
                stats["n_rows"] += 1
            if t["id"] not in done_route:
                row = {"id": t["id"], "traj": tid, "type": t.get("type"), "distance_tok": t["distance_tok"], "ctx_len": t_len,
                       "cut_index": cut.cut_index, "W": W, "n_blocks": nb, "tail_tokens": tail, "p_blocks": p, "window_blocks": wb,
                       "n_cand": len(res["cand"]), "located": span is not None, "src_span": list(span) if span else [lo, lo + gold_len],
                       "src_blocks": src, "src_status": status, "rank_a_last": rank_of(res["ranked"], src),
                       "ranked_top16": res["ranked"][:16], **res["route"], "probe_wall_s": wall,
                       "per_k": {str(k): {kk: res["arms"][k][kk] for kk in ("k", "rows", "rows_k0", "over_budget", "fetched", "fetch_bytes")}
                                 | {"source_exact_at_answer": bool(src) and all(
                                     j in exact_set(nb, p, wb) or j in res["arms"][k]["fetched"] for j in src)} for k in ks}}
                core.append_row(rpath, row)
            stats["answer_s"] += wall
        a_, r_ = core.mem_gib()
        log(f"[a3] {tid[-8:]} W {W} cut {k_ + 1}/{len(segs)} ctx {t_len} blocks {nb} tail {tail} peak alloc {a_:.1f} reserved {r_:.1f} GiB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    stats["build_s"], stats["answer_s"] = round(stats["build_s"], 1), round(stats["answer_s"], 1)
    stats["peak_alloc_gib"], stats["peak_reserved_gib"] = (round(x, 1) for x in core.mem_gib())
    return stats


def run_sel(ctx, out, ref_dir, ws, ks, n, log):
    """Stage 1: the first n (6) pilot trajectories, one build pass per window, the pilot FULL rows as reference (read only)."""
    H = ctx.H
    trajs, tasks = H.load_trajs(), H.load_tasks()
    full = core.read_rows(os.path.join(ref_dir, "FULL.jsonl"))
    for tid in list(trajs)[:n]:
        tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
        fr = [r for r in full if r["traj"] == tid]
        assert fr, f"no reference FULL rows for {tid} in {ref_dir}"
        for W in ws:
            t0 = time.time()
            st = run_trajectory(ctx, out, W, ks, tid, trajs[tid], tp, fr, log)
            if st:
                core.append_row(os.path.join(out, "progress.jsonl"), {**st, "arm": f"A3_W{W}", "wall_s": round(time.time() - t0, 1)})
    return {"stopped": None, "trajectories": list(trajs)[:n]}


def run_proper(ctx, out, ref_dir, W, k, early_at, log, cap=CAP):
    """Stage 2: trajectories PILOT_N+1 .. end in file order. FULL rows come from the A1 proper run where they exist, the rest are
    computed with the A1 FULL arm (r2_core.gen_fullchk, same flags). One Haybittle-Peto look at `early_at` held-out trajectories."""
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    trajs, all_tasks = H.load_trajs(), H.load_tasks()
    order = list(trajs)[PILOT_N:]
    fpath = os.path.join(out, "FULL.jsonl")
    log(f"[a3] seeded {seed_reference(os.path.join(ref_dir, 'FULL.jsonl'), fpath, order)} reference FULL rows from {ref_dir}")
    arm = arm_name(W, k)
    apath = os.path.join(out, arm + ".jsonl")
    by = {}
    for t in all_tasks:
        by.setdefault(t["traj"], []).append(t)
    finished, summary = [], {"stopped": None, "W": W, "k": k, "arm": arm}
    try:
        for tid in order:
            tp = [t for t in by.get(tid, []) if t["kind"] == "probe"]
            todo = core.pending(tp, _ids(fpath))
            if todo:                                              # the A1 FULL arm, for trajectories the A1 run did not reach
                plan = core.restrict(core.plan_trajectory(H, trajs[tid], todo), {t["id"] for t in todo})
                st = core.run_unit(S, R, plan, r2.gen_fullchk, ctx.parse, lambda row: core.append_row(fpath, row), "FULL", log=log,
                                   empty_cache=True)
                core.append_row(os.path.join(out, "progress.jsonl"), st)
            fr = [r for r in core.read_rows(fpath) if r["traj"] == tid]
            t0 = time.time()
            st = run_trajectory(ctx, out, W, (k,), tid, trajs[tid], tp, fr, log, cap)
            if st:
                core.append_row(os.path.join(out, "progress.jsonl"), {**st, "arm": arm, "wall_s": round(time.time() - t0, 1)})
            finished.append(tid)
            done = set(finished)
            chk = core.stage_checks(H, [t for t in all_tasks if t["traj"] in done], core.read_rows(fpath), core.read_rows(apath),
                                    finished, gate3_after=10 ** 9, early_at=early_at)
            with open(os.path.join(out, "checks.json"), "w") as f:
                json.dump(chk, f, indent=1)
            if "early" in chk and len(finished) == early_at:
                log(f"[a3] early look at {early_at}: {chk['early']}")
                if chk["early"]["verdict"] != "CONTINUE":
                    raise core.StageStop("early-" + chk["early"]["verdict"], chk["early"])
    except core.StageStop as e:
        summary["stopped"] = {"reason": e.reason, "detail": e.detail}
        log(f"[a3] stage stopped: {e.reason}")
    summary["trajectories"] = finished
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("sel", "proper"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dir", required=True, help="sel: the pilot dir (FULL.jsonl, read only); proper: the A1 proper dir")
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--n", type=int, default=PILOT_N, help="sel: how many pilot trajectories")
    ap.add_argument("--ws", default=",".join(map(str, SEL_WS)))
    ap.add_argument("--ks", default=",".join(map(str, SEL_KS)))
    ap.add_argument("--w", type=int, help="proper: the selected window in tokens")
    ap.add_argument("--k", type=int, help="proper: the selected k")
    ap.add_argument("--early-at", type=int, default=core.EARLY_AT)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    a = ap.parse_args(argv)
    if a.stage == "proper" and (a.w is None or a.k is None):
        ap.error("--stage proper needs --w and --k (the selected configuration)")
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    import tool_parse
    H = core.Harness(core.default_paths())
    m = core.load_model(a.model, a.mem_cap_gib, log)
    ctx = SimpleNamespace(S=m.S, R=m.R, H=H, parse=tool_parse.parse)
    if a.stage == "sel":
        summary = run_sel(ctx, a.out, a.ref_dir, [int(x) for x in a.ws.split(",")], [int(x) for x in a.ks.split(",")], a.n, log)
    else:
        summary = run_proper(ctx, a.out, a.ref_dir, a.w, a.k, a.early_at, log, a.cap)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[a3] done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
