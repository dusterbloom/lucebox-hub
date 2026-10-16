"""A4, the R-LRU offline pilot (PREREG "A4 R-LRU offline pilot"). The candidate engine design, measured on the real HF state, no compaction:
  state   an exact, contiguous recent window of whole 640-token blocks, plus the open tail, plus (primary arms) a 64-row sink = rows 0-63
          of block 0, or (arm K4P) the exact system and tool prefix P of A3 and E7; DeltaNet state carried through every token.
  evicted every older sealed block. It is neither attended nor compacted; its exact K/V, as computed while it was resident, is kept on host.
  routing the question's last-token query on the evicted blocks' exact keys (a1_e9.exact_mass, "x_last"), normalised over the evicted blocks.
  fetch   the top k blocks come back exact (a1_a3.a3_state -> a1_e.oracle_triples, A3's insertion path), then the probe is answered.
Arms: A4_K4 (k = 4, sink, primary), A4_K0 (k = 0, sink: pure LRU), A4_K4P (k = 4, P exact: A3's prefix; reported, never primary).
A4_K4 and A4_K0 share one build pass per trajectory; A4_K4P has its own.
  python3 a1_a4.py --out ~/r1e/out/a4 --ref-dir ~/r1e/out/a1_pilot
Pure helpers (inequalities, cut plan, `<|im_end|>` finders, ranks, source status) are above the driver and tested in test_a1_a4.py. Reads a1_a3 / a1_e /
a1_e9 / a1_e67 / a1_core / r2_core and changes none of them."""
import argparse
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_a3  # noqa: E402
import a1_core as core  # noqa: E402
import a1_e  # noqa: E402
import a1_e9 as e9  # noqa: E402
import a1_e67 as e67  # noqa: E402

BLOCK, CAP, SINK, K_REQ = a1_a3.BLOCK, a1_a3.CAP, 64, 4
KV_BYTES_PER_TOKEN = a1_a3.KV_BYTES_PER_TOKEN
ARMS = {"A4_K4": {"build": "S", "k": K_REQ}, "A4_K0": {"build": "S", "k": 0}, "A4_K4P": {"build": "P", "k": K_REQ}}
ROUTED = ("A4_K4", "A4_K4P")
RANKERS = ("x_last", "x_imend", "x_ucontent", "x_lit5", "x_max", "b_last")


class OverBudget(RuntimeError):
    """A state that cannot fit the row cap even with the smallest window. On this grid only a bug gets here: the run stops."""


def build_cfg(build, p):
    """S: no prefix, a SINK-row sink. P: the first p blocks stay exact (block 0 is among them, so the sink is moot)."""
    return {"pinned": 0, "sink": SINK} if build == "S" else {"pinned": p, "sink": 0}


def arms_of(build):
    return [a for a, s in ARMS.items() if s["build"] == build]


# ---------------------------------------------------------------- pure: rows, build-time eviction, cut plan

def state_rows(full_ids, sink, tail, block=BLOCK):
    """Rows per full-attention layer of a state that holds the whole blocks `full_ids`, the open tail, and the first `sink` rows of
    block 0 unless block 0 is itself among the whole blocks (then its sink rows are already counted: no row is counted twice)."""
    ids = set(full_ids)
    return block * len(ids) + (sink if sink and 0 not in ids else 0) + tail


def build_evictions(resident, new_id, pinned, sink, cap=CAP, block=BLOCK):
    """Block-granular LRU at build time. Before block `new_id` is appended, evict the oldest unpinned whole blocks (ids >= pinned) until
        (BUILD)   block * (P + W + 1) + sink * [block 0 not whole] + (block - 1)  <=  cap
    i.e. state_rows(resident - evicted + {new_id}, sink, tail = block - 1) <= cap: the held blocks, the one being appended, and the
    largest open tail that can follow it must fit. `resident` = whole-block ids held now. Returns the ids to evict, oldest first."""
    keep, out = sorted(resident), []
    while state_rows(keep + [new_id], sink, block - 1, block) > cap:
        old = [j for j in keep if j >= pinned]
        if not old:
            raise OverBudget(f"block {new_id}: {len(keep)} pinned blocks + 1 + tail {block - 1} exceed {cap} rows")
        out.append(old[0])
        keep.remove(old[0])
    return out


def cut_plan(nb, tail, pinned, w_build, reserve, k_req, sink, cap=CAP, block=BLOCK):
    """The state at a cut, from the build's resident window. P = min(pinned, nb) pinned blocks; the window is the newest w unpinned blocks,
    w <= w_build, the largest w for which, with k = min(k_req, candidates),
        (CUT)   block * (P + w + k) + sink * [block 0 not whole] + tail + reserve  <=  cap
    The bound charges every fetched block `block` rows and the sink whenever block 0 is not in the pinned set or the window. If block 0 is
    fetched its `sink` rows were already resident, so the actual rows are `sink` fewer (state_rows of the final set). Candidates = every
    sealed block that is neither pinned nor in the window, in id order. Raises OverBudget when even w = 0 does not fit."""
    P = min(pinned, nb)
    for w in range(min(w_build, nb - P), -1, -1):
        window = list(range(nb - w, nb))
        cands = list(range(P, nb - w))
        k = min(k_req, len(cands))
        base = sorted(set(range(P)) | set(window))
        bound = block * (len(base) + k) + (sink if sink and 0 not in base else 0) + tail
        if bound + reserve <= cap:
            return {"w": w, "w_build": w_build, "window": window, "cands": cands, "k": k, "base": base, "rows_bound": bound,
                    "rows_base": state_rows(base, sink, tail, block), "reserve": reserve, "peak_bound": bound + reserve}
    raise OverBudget(f"nb {nb} tail {tail} pinned {P} reserve {reserve}: no window fits {cap} rows")


def final_rows(plan, fetched, sink, tail, block=BLOCK):
    """Actual rows of the answer state: the base blocks plus the fetched ones (block 0 fetched adds `block - sink` rows, not `block`)."""
    return state_rows(set(plan["base"]) | set(fetched), sink, tail, block)


def candidate_start(j, sink, block=BLOCK):
    """First row of block j that is a candidate row: block 0 keeps its sink rows resident, so only rows sink.. are evicted/fetched."""
    return sink if (j == 0 and sink) else 0


def fetch_rows(fetched, sink, block=BLOCK):
    return sum(block - candidate_start(j, sink, block) for j in fetched)


def piece_list(host, full_ids, sink, block=BLOCK):
    """Per-layer pieces of the state in logical order: the sink rows of block 0 (when block 0 is not whole), then the whole blocks."""
    ids = sorted(set(full_ids))
    pieces = [slice_piece(host[0], 0, sink)] if sink and 0 not in ids else []
    return pieces + [host[j] for j in ids]


def slice_piece(entry, a, b):
    """rows a..b of a per-layer (K, beta, V) entry (None on DeltaNet layers); a copy."""
    return tuple(None if t is None else tuple(x[:, :, a:b].clone() for x in t) for t in entry)


def to_device(entry, dev):
    return tuple(None if t is None else tuple(x.to(dev) for x in t) for t in entry)


def imend_index(ids, im_start, im_end):
    """Index in the question suffix `ids` of the <|im_end|> that closes the user message: the last <|im_end|> before the LAST <|im_start|>
    (the assistant header). The engine's prompt ends `...content<|im_end|>\\n<|im_start|>assistant\\n`, so its "end - 5" query token is this
    one. None when the suffix has no such pair."""
    starts = [i for i, t in enumerate(ids) if t == im_start]
    if not starts:
        return None
    ends = [i for i in range(starts[-1]) if ids[i] == im_end]
    return ends[-1] if ends else None


def ucontent_index(ids, im_start, im_end):
    """The last content token of the user message: the token just before that <|im_end|> (one token earlier than imend_index)."""
    i = imend_index(ids, im_start, im_end)
    return i - 1 if i is not None and i >= 1 else None


def lit5_index(ids):
    """The literal fifth token from the end of the suffix (the engine's 'end - 5'), clamped to the first token."""
    return max(0, len(ids) - 5)


# ---------------------------------------------------------------- pure: sources, ranks

def source_blocks(span, nb, sink, block=BLOCK):
    """E3's located-span blocks, minus block 0 when the whole span lies inside the sink rows (that fact is resident, not routable)."""
    if span is None:
        return []
    blocks = a1_e.overlap_blocks(span[0], span[1], nb, block)
    if sink and blocks and blocks[0] == 0 and span[1] < sink:
        blocks = blocks[1:]
    return blocks


def source_status(located, span, nb, cands, sink, block=BLOCK):
    """not_located | candidate (an evicted block holds the fact) | resident (pinned prefix, sink, window or open tail)."""
    if not located:
        return "not_located"
    return "candidate" if any(j in cands for j in source_blocks(span, nb, sink, block)) else "resident"


def block_max(chunk_scores, counts):
    """(C, n) chunk scores -> (B, n) block scores: the max over each block's chunks; counts[i] = chunks of block i (block 0 may be short)."""
    out, pos = [], 0
    for c in counts:
        out.append(chunk_scores[pos:pos + c].max(0).values)
        pos += c
    assert pos == chunk_scores.shape[0]
    return torch.stack(out)


def rank_row(x, b, cand, src, imend, ucontent, lit5):
    """Ranks of the best source block among the candidates only. x: (n, C) exact mass normalised over `cand`; b: (C, n) cosine block
    scores of `cand`; src: block ids (those not in cand are ignored); imend / ucontent / lit5: query token positions (the first two may be None).
    None when no source block is a candidate."""
    pos = [cand.index(j) for j in src if j in cand]
    if not pos:
        return None
    xt = x.T
    return {"x_last": a1_e.best_rank(xt[:, -1], pos), "x_imend": a1_e.best_rank(xt[:, imend], pos) if imend is not None else None,
            "x_ucontent": a1_e.best_rank(xt[:, ucontent], pos) if ucontent is not None else None,
            "x_lit5": a1_e.best_rank(xt[:, lit5], pos), "x_max": a1_e.best_rank(xt.max(1).values, pos),
            "b_last": a1_e.best_rank(b[:, -1], pos)}


# ---------------------------------------------------------------- build: LRU generator and host store

def gen_lru(R, segs, ks, pinned, sink, cap=CAP, block=BLOCK, on_block=None):
    """Like r2_core.gen_fblk_window, but older blocks are evicted, not compacted. Every block is prefilled exact behind what the cache holds
    at that moment (pinned blocks, sink rows, the window) with the true positions; the DeltaNet state is carried through every block. Before
    a block is appended, build_evictions drops the oldest unpinned whole blocks. `on_block(j, a, b, exact)` gets each block's exact
    per-layer (K, beta, V) at seal time. Yields (k, triples, ctx_len, info): the build state (pinned/sink + window + the open tail, whose
    K/V were computed behind that window) with info = {tail_tokens, blocks, resident (whole-block ids), pinned, sink}."""
    import r2_core as r2
    S = R.S
    stream = torch.cat([s.ids for s in segs], 1)
    ends = [sum(s.n for s in segs[:i + 1]) for i in range(len(segs))]
    n_layers = len(S.layer_types)
    resident, sink_piece, lin, orig_len, consumed, built = {}, None, None, 0, 0, 0

    def compose():
        if lin is None:
            return None
        pieces = ([sink_piece] if sink_piece is not None and 0 not in resident else []) + [resident[j] for j in sorted(resident)]
        return tuple(lin[i] if i in S.linear else tuple(torch.cat([pc[i][c] for pc in pieces], 2) for c in range(3))
                     for i in range(n_layers))

    for k in sorted(ks):
        blocks, new_end = r2.sealed_blocks(consumed, ends[k], block)
        for a, b in blocks:
            for e in build_evictions(sorted(resident), built, pinned, sink, cap, block):
                del resident[e]
            n = b - a
            state, orig_len, _ = r2.extend(S.model, S.layer_types, compose(), orig_len, stream[:, a:b].to(S.dev))
            exact = tuple(None if i in S.linear else tuple(x[:, :, -n:].clone() for x in state[i]) for i in range(n_layers))
            resident[built] = exact
            if built == 0 and sink:
                sink_piece = slice_piece(exact, 0, sink)
            if on_block is not None:
                on_block(built, a, b, exact)
            lin = {i: state[i] for i in S.linear}
            del state
            built += 1
        consumed = new_end
        tail = stream[:, consumed:ends[k]]
        t_tr, t_len = compose(), orig_len
        if tail.shape[1]:
            t_tr, t_len, _ = r2.extend(S.model, S.layer_types, t_tr, orig_len, tail.to(S.dev))
        yield k, t_tr, t_len, {"tail_tokens": int(tail.shape[1]), "blocks": built, "resident": sorted(resident), "pinned": pinned, "sink": sink}


class HostStore:
    """Every sealed block's exact per-layer (K, beta, V) on host, as computed while the block was resident (the evicted blocks' archive), and
    the pooled chunk keys of its exact keys (kvflash_qk, for b_last)."""

    def __init__(self, full_layers, chunk=a1_e.CHUNK):
        self.full_layers, self.chunk = full_layers, chunk
        self.host, self.pooled = [], []

    def on_block(self, j, a, b, exact):
        assert j == len(self.host)
        self.host.append(tuple(None if t is None else tuple(x.detach().to("cpu", copy=True) for x in t) for t in exact))
        self.pooled.append(a1_e.pooled_chunk_keys([exact[i][0][0] for i in self.full_layers], self.chunk).to("cpu"))

    def cand_keys(self, cand, sink, dev):
        """{block: per full layer (Hkv, rows, d) exact keys on `dev`} for the candidate rows only (block 0 without its sink rows)."""
        return {j: [self.host[j][i][0][0][:, candidate_start(j, sink):].to(dev) for i in self.full_layers] for j in cand}

    def cand_pooled(self, cand, sink, dev):
        """(chunks, counts) of the candidates' pooled keys, block 0 without the chunks of its sink rows."""
        assert not sink or sink % self.chunk == 0
        parts = [self.pooled[j][candidate_start(j, sink) // self.chunk:] for j in cand]
        return torch.cat(parts).to(dev), [p.shape[0] for p in parts]


def cut_state(ship, host, full_ids, sink, tail, linear, dev):
    """The answer-side state: `ship`'s DeltaNet entries and open tail with the pieces of `full_ids` (and the sink rows) in front, inserted by
    A3's path (a1_a3.a3_state -> a1_e.oracle_triples with every piece swapped in, E3's swap)."""
    pieces = [to_device(p, dev) for p in piece_list(host, full_ids, sink)]
    return a1_a3.a3_state(ship, pieces, pieces, len(pieces), tail, 0, len(pieces), [], linear)


# ---------------------------------------------------------------- routing and answer (one probe)

def route_probe(S, base, t_len, suffix, cand, keys, pooled, counts, im_start, im_end):
    """Routing pass: prefill the question suffix under the cut's base state (no fetch yet), capture its queries, drop the pass's KV.
    x = exact-key mass per candidate (a1_e9.exact_mass, normalised over the candidates), b = kvflash_qk cosine per candidate block."""
    ids = torch.tensor([suffix], dtype=torch.long).to(S.dev)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t0 = time.time()
    import r2_core as r2
    _, _, q = r2.extend(S.model, S.layer_types, base, t_len, ids, want_queries=True)
    sync()
    t1 = time.time()
    x = e9.exact_mass(q, keys, cand)
    b = block_max(a1_e.qk_chunk_scores(pooled, q), counts)
    del q
    sync()
    return x.cpu(), b.cpu(), {"route_s": round(t1 - t0, 3), "router_s": round(time.time() - t1, 3),
                              "imend": imend_index(suffix, im_start, im_end),
                              "ucontent": ucontent_index(suffix, im_start, im_end), "lit5": lit5_index(suffix)}


def answer_state(S, st, plan, fetched, suffix, max_new, cap, block=BLOCK):
    """Answer one probe on the base state plus the fetched blocks (A3's insertion path). The rows are asserted against the plan and the
    cap, and the answer runs under A3's row guard."""
    import r2_core as r2
    full = sorted(set(plan["base"]) | set(fetched))
    trip = cut_state(st.ship, st.host, full, st.sink, st.tail, st.linear, S.dev)
    got = a1_a3._full_rows(trip, st.full_layers)
    assert got == final_rows(plan, fetched, st.sink, st.tail, block) <= plan["rows_bound"], (got, plan["rows_bound"])
    assert got + plan["reserve"] <= cap, f"answer state {got} + reserve {plan['reserve']} over {cap}"
    tm = {}
    with a1_a3.row_guard(S, got, cap):
        gen = r2.answer_ids(S, trip, st.t_len, suffix, max_new, timing=tm)
    del trip
    return gen, tm, got


# ---------------------------------------------------------------- driver

def _paths(out, build):
    arms = arms_of(build)
    f = {a: os.path.join(out, a + ".jsonl") for a in arms}
    f.update({a + "_route": os.path.join(out, a + "_route.jsonl") for a in arms if a in ROUTED})
    f["cuts"] = os.path.join(out, f"A4_cuts_{build}.jsonl")
    return f, os.path.join(out, f"done_{build}.jsonl")


def prepare_build(out, build, log=print):
    files, dpath = _paths(out, build)
    for p in [*files.values(), dpath]:
        a1_a3.repair_tail(p, log)
    a1_a3.purge_partial(list(files.values()), {r["traj"] for r in core.read_rows(dpath)})


def ensure_reference(ref_dir, out):
    """The pilot FULL-CHK rows, copied unedited into the output directory (renamed into place) and checked against the registered md5."""
    src, dst = os.path.join(ref_dir, "FULL.jsonl"), os.path.join(out, "FULL.jsonl")
    a1_a3.check_reference(src, a1_a3.PILOT_REF_MD5)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
    a1_a3.check_reference(dst, a1_a3.PILOT_REF_MD5)
    return dst


def run_build(ctx, out, build, tid, traj, tasks, full_rows, log, attempt, cap=CAP):
    """One trajectory, one build pass: the S build serves A4_K4 and A4_K0, the P build serves A4_K4P. A done marker closes the pass."""
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    files, dpath = _paths(out, build)
    if tid in {r["traj"] for r in core.read_rows(dpath)}:
        return None
    ret = core.retained_probe_ids(H, tasks, full_rows)
    todo = {t["id"] for t in tasks if t["id"] in ret}
    if not todo:
        return None
    plan = core.restrict(core.plan_trajectory(H, traj, [t for t in tasks if t["id"] in todo]), todo)
    im_end, im_start = S.tok.convert_tokens_to_ids("<|im_end|>"), S.tok.convert_tokens_to_ids("<|im_start|>")
    end = e67.system_end(plan.stream, im_end)
    p = e67.prefix_blocks(end)
    cfg = build_cfg(build, p)
    pinned, sink = cfg["pinned"], cfg["sink"]
    full_layers = [i for i in range(len(S.layer_types)) if i not in S.linear]
    store = HostStore(full_layers)
    segs = core.segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    stats = {"traj": tid, "build": build, "p_blocks": p, "system_end_tok": end, "n_rows": 0, "build_s": 0.0, "answer_s": 0.0,
             "attempt": attempt}
    log(f"[a4] {tid[-8:]} build {build}: pinned {pinned} blocks, sink {sink} rows, system+tools end at token {end} ({p} blocks)")
    t_mark = time.time()
    for k_, ship, t_len, info in gen_lru(R, segs, list(range(len(segs))), pinned, sink, cap, BLOCK, store.on_block):
        stats["build_s"] += time.time() - t_mark
        cut, nb, tail = plan.cuts[k_], info["blocks"], info["tail_tokens"]
        assert len(store.host) == nb and t_len == cut.end
        w_build = len([j for j in info["resident"] if j >= pinned])
        assert [j for j in info["resident"] if j >= pinned] == list(range(nb - w_build, nb)), info      # the window is contiguous and newest
        reserve = max(len(i.suffix) + i.max_new for i in cut.items)
        plans = {a: cut_plan(nb, tail, pinned, w_build, reserve, ARMS[a]["k"], sink, cap) for a in arms_of(build)}
        st = SimpleNamespace(ship=ship, host=store.host, sink=sink, tail=tail, t_len=t_len, linear=S.linear, full_layers=full_layers)
        for a, pl in plans.items():
            core.append_row(files["cuts"], {"traj": tid, "build": build, "arm": a, "cut_index": cut.cut_index, "ctx_len": t_len,
                                            "n_blocks": nb, "tail": tail, "pinned": pinned, "sink": sink, "window_build": w_build,
                                            "window_cut": pl["w"], "n_cand": len(pl["cands"]), "k": pl["k"], "reserve": reserve,
                                            "rows_base": pl["rows_base"], "rows_bound": pl["rows_bound"], "peak_bound": pl["peak_bound"],
                                            "attempt": attempt})
        bases, keys = {}, {}
        for a, pl in plans.items():
            bases[a] = cut_state(ship, store.host, pl["base"], sink, tail, S.linear, S.dev)
            if a in ROUTED and pl["k"] and pl["cands"]:
                keys[a] = (store.cand_keys(pl["cands"], sink, S.dev), *store.cand_pooled(pl["cands"], sink, S.dev))
        for item in cut.items:
            t = item.task
            t0 = time.time()
            gold_len = len(H.ids(t["answer"]))
            lo = t["src_tok"]
            span = a1_e.locate_span(lambda sel: S.tok.decode(sel, skip_special_tokens=False), plan.stream, lo, cut.end, t["answer"])
            shared = {}
            for a, pl in plans.items():
                spec = ARMS[a]
                cands, fetched, ranks, rt, ranked = pl["cands"], [], None, {"route_s": 0.0, "router_s": 0.0}, []
                src = source_blocks(span, nb, sink)
                status = source_status(span is not None, span, nb, cands, sink)
                if a in keys:
                    assert a1_a3._full_rows(bases[a], full_layers) + len(item.suffix) <= cap, "routing pass over the row cap"
                    kd, pooled, counts = keys[a]
                    x, b, rt = route_probe(S, bases[a], t_len, item.suffix, cands, kd, pooled, counts, im_start, im_end)
                    ranked = a1_a3.rank_candidates(x[-1], cands)
                    fetched = ranked[:pl["k"]]
                    ranks = rank_row(x, b, cands, src, rt["imend"], rt["ucontent"], rt["lit5"])
                key = tuple(sorted(set(pl["base"]) | set(fetched)))
                if key in shared and not fetched:                       # identical state (no fetch, same blocks): same greedy answer, reused
                    gen, tm, got = shared[key]
                    reused = True
                else:
                    gen, tm, got = answer_state(S, st, pl, fetched, item.suffix, item.max_new, cap, BLOCK)
                    reused = False
                    if not fetched:
                        shared[key] = (gen, tm, got)
                wall = round(time.time() - t0, 2)
                text = S.tok.decode(core.strip_stop(gen, S.eos), skip_special_tokens=False)
                fields = {"attempt": attempt, "build": build, "k": len(fetched), "k_plan": pl["k"], "rows": got, "reserve": reserve,
                          "peak_rows": got + reserve, "n_cand": len(cands), "window_build": w_build, "window_cut": pl["w"],
                          "fetch_bytes": fetch_rows(fetched, sink) * KV_BYTES_PER_TOKEN, "n_blocks": nb, "tail_tokens": tail,
                          "src_blocks": src, "src_cand_blocks": [j for j in src if j in cands], "src_status": status, "answer_reused": reused, "probe_wall_s": wall}
                extra = {**fields, "cut_index": cut.cut_index, "ctx_len": t_len, "decode_tokens": len(gen), "fetched_blocks": fetched,
                         "route_s": rt["route_s"], "timings": {"prefill_ms": round(tm["prefill_s"] * 1000, 1),
                                                               "decode_ms": round(tm["decode_s"] * 1000, 1)}}
                core.append_row(files[a], core.make_row(item, a, gen, text, ctx.parse(text, plan.tools), S.eos, extra))
                stats["n_rows"] += 1
                if a in ROUTED:
                    e5, uc = rt.get("imend"), rt.get("ucontent")
                    core.append_row(files[a + "_route"], {
                        **fields, "id": t["id"], "arm": a, "traj": tid, "type": t.get("type"), "distance_tok": t["distance_tok"],
                        "ctx_len": t_len, "cut_index": cut.cut_index, "located": span is not None,
                        "src_span": list(span) if span else [lo, lo + gold_len], "ranks": ranks, "ranked_top16": ranked[:16],
                        "fetched": fetched, "route_s": rt["route_s"], "router_s": rt["router_s"],
                        "imend": {"index": e5, "from_end": len(item.suffix) - e5 if e5 is not None else None,
                                  "token": S.tok.decode([item.suffix[e5]], skip_special_tokens=False) if e5 is not None else None},
                        "ucontent": {"index": uc, "token": S.tok.decode([item.suffix[uc]], skip_special_tokens=False) if uc is not None else None},
                        "lit5": {"index": rt.get("lit5"), "token": S.tok.decode([item.suffix[rt["lit5"]]], skip_special_tokens=False)
                                 if "lit5" in rt else None}})
                stats["answer_s"] += time.time() - t0
                t0 = time.time()
        del bases, keys
        a_, r_ = core.mem_gib()
        log(f"[a4] {tid[-8:]} {build} cut {k_ + 1}/{len(segs)} ctx {t_len} blocks {nb} window {w_build} peak alloc {a_:.1f} reserved {r_:.1f} GiB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    stats["build_s"], stats["answer_s"] = round(stats["build_s"], 1), round(stats["answer_s"], 1)
    stats["peak_alloc_gib"], stats["peak_reserved_gib"] = (round(x, 1) for x in core.mem_gib())
    core.append_row(dpath, {"traj": tid, "attempt": attempt, "n_rows": stats["n_rows"]})
    return stats


def run_pilot(ctx, out, ref_dir, n, log, attempt, cap=CAP):
    H = ctx.H
    trajs, tasks = H.load_trajs(), H.load_tasks()
    ref = ensure_reference(ref_dir, out)
    full = core.read_rows(ref)
    a1_a3.check_universe(a1_a3.universe_ids(H, tasks, full, set(list(trajs)[:n])), a1_a3.PILOT_PROBES, "pilot reference")
    for build in ("S", "P"):
        prepare_build(out, build, log)
    for tid in list(trajs)[:n]:
        tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
        fr = [r for r in full if r["traj"] == tid]
        assert fr, f"no reference FULL rows for {tid}"
        for build in ("S", "P"):
            t0 = time.time()
            st = run_build(ctx, out, build, tid, trajs[tid], tp, fr, log, attempt, cap)
            if st:
                core.append_row(os.path.join(out, "progress.jsonl"), {**st, "arm": "A4_" + build, "wall_s": round(time.time() - t0, 1)})
    return {"stopped": None, "trajectories": list(trajs)[:n], "attempt": attempt}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dir", required=True, help="the pilot dir (FULL.jsonl, read only)")
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--n", type=int, default=a1_a3.PILOT_N)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--attempt", default=a1_a3.new_attempt_id())
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    a = ap.parse_args(argv)
    if a.cap != CAP:
        ap.error(f"--cap {a.cap}: A4 runs at the registered {CAP} rows only")
    if a.n != a1_a3.PILOT_N:
        ap.error(f"--n {a.n}: A4 runs over the registered {a1_a3.PILOT_N} pilot trajectories only")
    os.makedirs(a.out, exist_ok=True)
    import r1b_core
    import r2_core as r2
    here = os.path.dirname(os.path.abspath(__file__))
    paths = core.default_paths()
    fp = {"stage": "a4", "n": a.n, "cap": a.cap, "sink": SINK, "arms": {k: v for k, v in ARMS.items()}, "model": a.model,
          "ref_dir": a.ref_dir, "ref_full_md5": a1_a3.md5_file(os.path.join(a.ref_dir, "FULL.jsonl")),
          "data_md5": {k: a1_a3.md5_file(paths[k]) for k in ("traj", "tasks", "template", "tokenizer")},
          "code_md5": a1_a3.code_md5([os.path.join(here, f) for f in ("a1_a4.py", "a1_a4_report.py", "a1_launch_a4.sh", "a1_a3.py", "a1_e.py", "a1_e9.py", "a1_core.py", "a1_e67.py")]
                                     + [r2.__file__, r1b_core.__file__])}
    logf = open(os.path.join(a.out, f"run.{a.attempt}.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    log(f"[a4] attempt {a.attempt}: manifest {a1_a3.claim_manifest(os.path.join(a.out, 'manifest.json'), fp)}")
    import tool_parse
    H = core.Harness(paths)
    m = core.load_model(a.model, a.mem_cap_gib, log)
    ctx = SimpleNamespace(S=m.S, R=m.R, H=H, parse=tool_parse.parse)
    summary = run_pilot(ctx, a.out, a.ref_dir, a.n, log, a.attempt, a.cap)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[a4] done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
