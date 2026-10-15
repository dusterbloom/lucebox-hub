"""A3 routed exact fetch (PREREG "A3 routed exact fetch"). The KVFlash 2.0b design on the B0 agentic probes, offline.

State at a cut (r2_core.gen_fblk_window, one build pass per trajectory and window W):
  P  the sealed blocks overlapping the rendered system message and tool schema stay exact (E7, `exact_blocks`);
  W  the last W sealed blocks stay exact;
  the other sealed 640-token blocks are compacted to 64 slots (ratio 0.10), their exact K/V kept aside (on_block, as E3);
  the open tail and the DeltaNet state are as the generator leaves them.
Per probe: (1) routing pass: prefill the question suffix under that state, capture its queries (r2_core.extend), score every
compacted block (not P, not W) with the compacted-slot attention mass at the last question token (a1_e.router_mass); the pass's KV
is dropped. (2) fetch: the top k blocks swap to exact (a1_e.oracle_triples: the compacted entry is replaced), k = the most blocks
that keep rows <= CAP (capped at the candidates). (3) answer: r2_core.answer_ids on the swapped state, exactly as A1.
  stage sel:    python3 a1_a3.py --stage sel --out OUT --ref-dir PILOT_DIR [--ws 2,6]   (6 pilot trajectories; W in blocks)
  stage proper: python3 a1_a3.py --stage proper --out OUT --ref-dir A1_DIR --sel-report SEL/a3_report.json   (trajectories 7-60)
k is not a parameter: at each cut it is the largest number of router-ordered candidates that keeps the rows, the question and the
answer's decode tokens within CAP. Stage 2 takes W from the validated stage-1 report (its sha256 goes into manifest.json).
Resumable per trajectory and window: rerun the same command; the manifest must match, a trajectory without a done marker is redone. Helpers above the driver are pure and tested in test_a1_a3.py."""
import argparse
import contextlib
import functools
import hashlib
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
SEL_WBS = (2, 6)                             # window grid, in blocks (1,280 and 3,840 tokens)
PILOT_N = 6                                  # the pilot trajectories; stage proper runs the rest
RECALL_KS = a1_e.KS


# ---------------------------------------------------------------- pure: names, row budget, candidates

def arm_name(wb):
    return f"A3_W{wb}"


def route_name(wb):
    return f"A3_W{wb}_route"


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


def choose_k(nb, tail, p, wb, reserve=0, cap=CAP, block=BLOCK, slots=SLOTS):
    """The budget fills: k is the largest number of router-ordered candidates with
        rows + reserve <= cap,   rows = P + W + slots*(compacted not fetched) + block*k + tail,
    capped at the candidates. `reserve` is what the answer appends to the cut: the question (longest suffix at the cut) plus the
    maximum new tokens, because B0's pool holds those inside its rows. When even k = 0 is over the cap the cut is flagged
    over_budget and runs at k = 0 (the configuration is then ineligible)."""
    n_c = len(candidates(nb, p, wb))
    r0 = rows_at(nb, tail, p, wb, 0, block, slots)
    k = min(n_c, (cap - reserve - r0) // (block - slots)) if r0 + reserve <= cap else 0
    r = rows_at(nb, tail, p, wb, k, block, slots)
    return {"k": k, "rows": r, "reserve": reserve, "peak_rows": r + reserve, "rows_k0": r0, "n_cand": n_c,
            "over_budget": r + reserve > cap, "lowered": k < n_c}


def fetch_coverage(src, nb, p, wb, fetched):
    """How much of the fact's candidate source blocks the fetch restored: None (no candidate source block), "none", "partial", "all"."""
    cs = [j for j in src if j in candidates(nb, p, wb)]
    if not cs:
        return None
    n = sum(1 for j in cs if j in fetched)
    return "none" if n == 0 else "all" if n == len(cs) else "partial"


class _RowGuard:
    """Wraps the model: counts the tokens fed to the cache and asserts rows <= cap at every call (question prefill, each decode step)."""

    def __init__(self, model, rows, cap):
        self.model, self.rows, self.cap = model, rows, cap

    def __call__(self, *a, **k):
        ids = k["input_ids"] if "input_ids" in k else a[0]
        self.rows += ids.shape[-1]
        assert self.rows <= self.cap, f"answer state reached {self.rows} rows, cap {self.cap}"
        return self.model(*a, **k)

    def __getattr__(self, name):
        return getattr(self.model, name)


@contextlib.contextmanager
def row_guard(S, rows, cap):
    """Within the block every S.model call is checked against the cap, starting from `rows` already held."""
    real = S.model
    S.model = _RowGuard(real, rows, cap)
    try:
        yield
    finally:
        S.model = real


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
    table: {"W6": {correct, n, mean_rows, eligible}}."""
    def key(name):
        v = table[name]
        return (-v["correct"] / v["n"], v["mean_rows"], name)
    ok = sorted((n for n in table if table[n]["eligible"]), key=key)
    return {"selected": ok[0] if ok else None, "order_eligible": ok}


def final_verdict(ci, bar=core.BAR):
    """PREREG A1 section 6 (A3 inherits it): PASS lo > bar; FAIL hi < bar, or INCONCLUSIVE with the point under the bar."""
    if ci["lo"] > bar:
        return "PASS"
    if ci["hi"] < bar or ci["mean"] < bar:
        return "FAIL"
    return "INCONCLUSIVE"


def seed_reference(ref_path, out_path, traj_ids):
    """Copy the reference FULL rows of `traj_ids` (raw lines, unedited) into out_path unless it already exists. The copy goes to a
    file in the same directory and is renamed into place, so out_path is either absent or complete."""
    if os.path.exists(out_path):
        return 0
    keep, n, tmp = set(traj_ids), 0, out_path + ".part"
    with open(ref_path) as f, open(tmp, "w") as g:
        for line in f:
            if line.strip() and json.loads(line)["traj"] in keep:
                g.write(line if line.endswith("\n") else line + "\n")
                n += 1
        g.flush()
        os.fsync(g.fileno())
    os.replace(tmp, out_path)
    return n


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_selection(path):
    """The stage-1 report stage 2 must follow: returns (window_blocks, sha256). Refuses a report that is not a complete stage-1
    report of exactly the registered windows, or that selected nothing."""
    rep = json.load(open(path))
    names = sorted(f"W{w}" for w in SEL_WBS)
    if rep.get("stage") != "sel" or not rep.get("complete") or sorted(rep.get("arms", {})) != names:
        raise SystemExit(f"{path} is not a complete stage-1 report of {names}")
    sel = (rep.get("selection") or {}).get("selected")
    if sel not in rep["arms"]:
        raise SystemExit(f"{path} selected no configuration ({sel!r}): amend the PREREG first")
    return int(sel[1:]), sha256_file(path)


def code_md5(paths):
    return {os.path.basename(p): hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths}


def claim_manifest(path, fp):
    """Run fingerprint (stage, arguments, selection digest, code md5s). The first start writes it; a rerun must match it exactly or
    the run refuses, so resumed rows always come from the same code, arguments and selection."""
    if not os.path.exists(path):
        with open(path + ".part", "w") as f:
            json.dump(fp, f, indent=1, sort_keys=True)
        os.replace(path + ".part", path)
        return "new"
    old = json.load(open(path))
    diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
    if diff:
        raise SystemExit(f"{path} was written by a different run ({', '.join(diff)} differ): refusing to resume")
    return "match"


def purge_partial(paths, done_trajs):
    """Drop the rows of trajectories without a done marker (a run killed mid-trajectory): that trajectory is redone, never kept."""
    for p in paths:
        rows = core.read_rows(p)
        kept = [r for r in rows if r["traj"] in done_trajs]
        if len(kept) != len(rows):
            with open(p + ".part", "w") as f:
                for r in kept:
                    f.write(json.dumps(r) + "\n")
            os.replace(p + ".part", p)


# ---------------------------------------------------------------- routing, fetch, answer (one probe)

def _full_rows(triples, full_layers):
    return triples[full_layers[0]][0].shape[2]


def route_a_last(S, st, suffix, cap=None):
    """Routing pass. Prefill the question suffix under the cut state and capture its queries; the pass's KV is dropped. Score each
    candidate block by the attention mass its compacted slots get at the LAST question token (a1_e.router_mass: softmax over the
    candidate slots only, mean over layers and query heads). Returns (cand, scores over cand or None, timings)."""
    import r2_core as r2
    cand = candidates(st.nb, st.p, st.wb)
    if not cand:
        return cand, None, {"route_s": 0.0, "router_s": 0.0}
    if cap is not None:                                  # the routing pass holds the cut's rows plus the question
        assert _full_rows(st.triples, st.full_layers) + len(suffix) <= cap, "routing pass over the row cap"
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


def answer_item(S, st, item, cap=CAP, block=BLOCK, slots=SLOTS):
    """One probe: route once, fetch the top k blocks (k fills the budget, choose_k, which reserves the cut's longest question plus
    the maximum new tokens: st.reserve, else this item's), answer under a row guard. Returns
    {cand, ranked, route, budget, fetched, fetch_bytes, gen, timing}."""
    import r2_core as r2
    reserve = getattr(st, "reserve", None) or len(item.suffix) + item.max_new
    bud = choose_k(st.nb, st.tail, st.p, st.wb, reserve, cap, block, slots)
    checked = None if bud["over_budget"] else cap                    # an over-budget cut is flagged and run, not aborted
    cand, a_last, rt = route_a_last(S, st, item.suffix, checked)
    ranked = rank_candidates(a_last, cand) if cand else []
    fetched = ranked[:bud["k"]]
    trip = a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, fetched, st.linear)
    got = _full_rows(trip, st.full_layers)
    assert got == bud["rows"], f"state has {got} rows, budget says {bud['rows']}"
    tm = {}
    with row_guard(S, got, checked if checked is not None else 10 ** 12):
        gen = r2.answer_ids(S, trip, st.t_len, item.suffix, item.max_new, timing=tm)
    del trip
    return {"cand": cand, "ranked": ranked, "route": rt, "budget": bud, "fetched": list(fetched),
            "fetch_bytes": bud["k"] * block * KV_BYTES_PER_TOKEN, "gen": gen, "timing": tm}


# ---------------------------------------------------------------- driver

def _ids(path):
    return {r["id"] for r in core.read_rows(path)}


def _paths(out, wb):
    return (os.path.join(out, arm_name(wb) + ".jsonl"), os.path.join(out, route_name(wb) + ".jsonl"),
            os.path.join(out, f"done_{arm_name(wb)}.jsonl"))


def prepare_window(out, wb):
    """Resume: keep only the rows of trajectories with a done marker; a trajectory killed mid-way is redone from scratch."""
    path, rpath, dpath = _paths(out, wb)
    purge_partial([path, rpath], {r["traj"] for r in core.read_rows(dpath)})


def run_trajectory(ctx, out, wb, tid, traj, tasks, full_rows, log, attempt, cap=CAP):
    """One trajectory, one window of `wb` blocks: build the P+W state once, answer every retained probe (k fills the budget). Rows go
    to A3_W{wb}.jsonl (harness row schema) and A3_W{wb}_route.jsonl (router ranks, fetch plan), both stamped with the attempt id; a
    done marker closes the trajectory. A trajectory with a marker is skipped; one without (prepare_window) has no rows left."""
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    path, rpath, dpath = _paths(out, wb)
    if tid in {r["traj"] for r in core.read_rows(dpath)}:
        return None
    assert not [r for r in core.read_rows(path) + core.read_rows(rpath) if r["traj"] == tid], f"{tid}: rows without a done marker"
    ret = core.retained_probe_ids(H, tasks, full_rows)
    todo = {t["id"] for t in tasks if t["id"] in ret}
    if not todo:
        return None
    plan = core.restrict(core.plan_trajectory(H, traj, [t for t in tasks if t["id"] in todo]), todo)
    end = e67.system_end(plan.stream, S.tok.convert_tokens_to_ids("<|im_end|>"))
    assert S.tok.decode(plan.stream[:3], skip_special_tokens=False).startswith("<|im_start|>system")
    p = e67.prefix_blocks(end)
    full_layers = [i for i in range(len(S.layer_types)) if i not in S.linear]
    exact, comp = [], []
    segs = core.segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    stats = {"traj": tid, "window_blocks": wb, "p_blocks": p, "system_end_tok": end, "n_rows": 0, "build_s": 0.0,
             "answer_s": 0.0, "attempt": attempt}
    log(f"[a3] {tid[-8:]} W {wb} blocks: system+tools end at token {end}, {p} exact prefix blocks")
    t_mark = time.time()
    for k_, trip, t_len, info in r2.gen_fblk_window(R, segs, list(range(len(segs))), wb, exact_blocks=p,
                                                    on_block=lambda j, a, b, ex, cp: (exact.append(ex), comp.append(cp))):
        stats["build_s"] += time.time() - t_mark
        cut, nb, tail = plan.cuts[k_], info["blocks"], info["tail_tokens"]
        assert len(comp) == nb and t_len == cut.end
        reserve = max(len(i.suffix) + i.max_new for i in cut.items)              # one k per cut: room for its longest question + answer
        st = SimpleNamespace(triples=trip, t_len=t_len, nb=nb, tail=tail, comp=comp, exact=exact, p=p, wb=wb, linear=S.linear,
                             full_layers=full_layers, reserve=reserve)
        for item in cut.items:
            t = item.task
            t0 = time.time()
            res = answer_item(S, st, item, cap)
            lo = t["src_tok"]
            gold_len = len(H.ids(t["answer"]))
            span = a1_e.locate_span(lambda sel: S.tok.decode(sel, skip_special_tokens=False), plan.stream, lo, cut.end, t["answer"])
            src = a1_e.overlap_blocks(*span, nb) if span else []
            status = source_status(span is not None, src, nb, p, wb)
            wall = round(time.time() - t0, 2)
            bud = res["budget"]
            fields = {"attempt": attempt, "k": bud["k"], "rows": bud["rows"], "reserve": bud["reserve"], "peak_rows": bud["peak_rows"],
                      "rows_k0": bud["rows_k0"], "over_budget": bud["over_budget"], "n_cand": bud["n_cand"],
                      "fetch_bytes": res["fetch_bytes"], "probe_wall_s": wall, "window_blocks": wb, "p_blocks": p, "n_blocks": nb,
                      "tail_tokens": tail, "src_blocks": src, "src_status": status}
            text = S.tok.decode(core.strip_stop(res["gen"], S.eos), skip_special_tokens=False)
            extra = {**fields, "cut_index": cut.cut_index, "ctx_len": t_len, "decode_tokens": len(res["gen"]),
                     "fetched_blocks": res["fetched"], "route_s": res["route"]["route_s"],
                     "timings": {"prefill_ms": round(res["timing"]["prefill_s"] * 1000, 1),
                                 "decode_ms": round(res["timing"]["decode_s"] * 1000, 1)}}
            core.append_row(path, core.make_row(item, arm_name(wb), res["gen"], text, ctx.parse(text, plan.tools), S.eos, extra))
            stats["n_rows"] += 1
            row = {**fields, "id": t["id"], "traj": tid, "type": t.get("type"), "distance_tok": t["distance_tok"], "ctx_len": t_len,
                   "cut_index": cut.cut_index, "n_cand": len(res["cand"]), "located": span is not None,
                   "src_span": list(span) if span else [lo, lo + gold_len],
                   "rank_a_last": rank_of(res["ranked"], src), "ranked_top16": res["ranked"][:16], **res["route"],
                   "fetched": res["fetched"],
                   "source_exact_at_answer": bool(src) and all(j in exact_set(nb, p, wb) or j in res["fetched"] for j in src),
                   "source_fetch_coverage": fetch_coverage(src, nb, p, wb, res["fetched"])}
            core.append_row(rpath, row)
            stats["answer_s"] += wall
        a_, r_ = core.mem_gib()
        log(f"[a3] {tid[-8:]} W {wb} cut {k_ + 1}/{len(segs)} ctx {t_len} blocks {nb} tail {tail} peak alloc {a_:.1f} reserved {r_:.1f} GiB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    stats["build_s"], stats["answer_s"] = round(stats["build_s"], 1), round(stats["answer_s"], 1)
    stats["peak_alloc_gib"], stats["peak_reserved_gib"] = (round(x, 1) for x in core.mem_gib())
    core.append_row(dpath, {"traj": tid, "attempt": attempt, "n_rows": stats["n_rows"]})          # last: the trajectory is complete
    return stats


def run_sel(ctx, out, ref_dir, wbs, n, log, attempt, cap=CAP):
    """Stage 1: the first n (6) pilot trajectories, one build pass per window (in blocks), the pilot FULL rows as reference (read only)."""
    H = ctx.H
    trajs, tasks = H.load_trajs(), H.load_tasks()
    full = core.read_rows(os.path.join(ref_dir, "FULL.jsonl"))
    for wb in wbs:
        prepare_window(out, wb)
    for tid in list(trajs)[:n]:
        tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
        fr = [r for r in full if r["traj"] == tid]
        assert fr, f"no reference FULL rows for {tid} in {ref_dir}"
        for wb in wbs:
            t0 = time.time()
            st = run_trajectory(ctx, out, wb, tid, trajs[tid], tp, fr, log, attempt, cap)
            if st:
                core.append_row(os.path.join(out, "progress.jsonl"), {**st, "arm": arm_name(wb), "wall_s": round(time.time() - t0, 1)})
    return {"stopped": None, "trajectories": list(trajs)[:n], "attempt": attempt}


def full_gen():
    """The A1 FULL arm with the per-block allocator cleanup bound INTO the generator (run_unit's own empty_cache is the per-cut one)."""
    import r2_core as r2
    return functools.partial(r2.gen_fullchk, empty_cache=True)


def compute_full(ctx, plan, fpath, log):
    return core.run_unit(ctx.S, ctx.R, plan, full_gen(), ctx.parse, lambda row: core.append_row(fpath, row), "FULL", log=log,
                         empty_cache=True)


def run_proper(ctx, out, ref_dir, wb, early_at, log, attempt, cap=CAP):
    """Stage 2: trajectories PILOT_N+1 .. end in file order. FULL rows come from the A1 proper run where they exist (never recomputed),
    the rest are computed with the A1 FULL arm. One Haybittle-Peto look at `early_at` held-out trajectories."""
    H = ctx.H
    trajs, all_tasks = H.load_trajs(), H.load_tasks()
    order = list(trajs)[PILOT_N:]
    ref = os.path.join(ref_dir, "FULL.jsonl")
    fpath = os.path.join(out, "FULL.jsonl")
    reused = {r["traj"] for r in core.read_rows(ref)} & set(order)
    log(f"[a3] seeded {seed_reference(ref, fpath, order)} reference FULL rows from {ref_dir}")
    arm = arm_name(wb)
    apath = os.path.join(out, arm + ".jsonl")
    prepare_window(out, wb)
    by = {}
    for t in all_tasks:
        by.setdefault(t["traj"], []).append(t)
    finished, summary = [], {"stopped": None, "window_blocks": wb, "arm": arm, "attempt": attempt}
    try:
        for tid in order:
            tp = [t for t in by.get(tid, []) if t["kind"] == "probe"]
            todo = core.pending(tp, _ids(fpath))
            assert tid not in reused or not todo, f"{tid}: reference FULL rows from {ref_dir} are incomplete; they are never recomputed"
            if todo:                                              # the A1 FULL arm, for trajectories the A1 run did not reach
                plan = core.restrict(core.plan_trajectory(H, trajs[tid], todo), {t["id"] for t in todo})
                core.append_row(os.path.join(out, "progress.jsonl"), {**compute_full(ctx, plan, fpath, log), "attempt": attempt})
            fr = [r for r in core.read_rows(fpath) if r["traj"] == tid]
            t0 = time.time()
            st = run_trajectory(ctx, out, wb, tid, trajs[tid], tp, fr, log, attempt, cap)
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
    ap.add_argument("--ws", default=",".join(map(str, SEL_WBS)), help="sel: window sizes in blocks")
    ap.add_argument("--sel-report", help="proper: the stage-1 a3_report.json; W is read from it, never given")
    ap.add_argument("--early-at", type=int, default=core.EARLY_AT)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--attempt", default=time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}")
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    a = ap.parse_args(argv)
    if a.stage == "proper" and not a.sel_report:
        ap.error("--stage proper needs --sel-report (the window is the stage-1 selection, not an argument)")
    if a.stage == "sel" and a.cap != CAP:
        ap.error(f"--cap {a.cap}: selection runs at the registered {CAP} rows only")
    os.makedirs(a.out, exist_ok=True)
    wb, sel_sha = load_selection(a.sel_report) if a.stage == "proper" else (None, None)
    import r2_core as r2
    here = os.path.dirname(os.path.abspath(__file__))
    fp = {"stage": a.stage, "ws": a.ws if a.stage == "sel" else None, "n": a.n if a.stage == "sel" else None, "window_blocks": wb,
          "early_at": a.early_at, "cap": a.cap, "model": a.model, "sel_report_sha256": sel_sha,
          "code_md5": code_md5([os.path.join(here, f) for f in ("a1_a3.py", "a1_e.py", "a1_core.py", "a1_e67.py")] + [r2.__file__])}
    logf = open(os.path.join(a.out, f"run.{a.attempt}.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    log(f"[a3] attempt {a.attempt}: manifest {claim_manifest(os.path.join(a.out, 'manifest.json'), fp)}")
    import tool_parse
    H = core.Harness(core.default_paths())
    m = core.load_model(a.model, a.mem_cap_gib, log)
    ctx = SimpleNamespace(S=m.S, R=m.R, H=H, parse=tool_parse.parse)
    if a.stage == "sel":
        summary = run_sel(ctx, a.out, a.ref_dir, [int(x) for x in a.ws.split(",")], a.n, log, a.attempt, a.cap)
    else:
        summary = run_proper(ctx, a.out, a.ref_dir, wb, a.early_at, log, a.attempt, a.cap)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[a3] done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
