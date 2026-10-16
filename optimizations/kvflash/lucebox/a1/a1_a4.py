"""A4, the R-LRU offline pilot (PREREG "A4 R-LRU offline pilot"). The candidate engine design, measured on the real HF state, no compaction:
  state   an exact, contiguous recent window of whole 640-token blocks, plus the open tail, plus (primary arms) a 64-row sink = rows 0-63
          of block 0, or (arm K4P) the exact system and tool prefix P of A3 and E7; DeltaNet state carried through every token.
  evicted every older sealed block. It is neither attended nor compacted; its exact K/V, as computed while it was resident, is kept on host.
  routing the question's last-token query on the evicted blocks' exact keys (a1_e9.exact_mass, "x_last"), normalised over the evicted blocks.
  fetch   the top k blocks come back exact (a1_a3.a3_state -> a1_e.oracle_triples, A3's insertion path), then the probe is answered.
Arms: A4_K4 (k = 4, sink, primary), A4_K0 (k = 0, sink: pure LRU), A4_K4P (k = 4, P exact: A3's prefix; reported, never primary).
A4_K4 and A4_K0 share one build pass per trajectory; A4_K4P has its own.
Amendment 1 (PREREG "A4 Amendment 1"): compact-on-evict arms A4_CT, A4_CTK4 (build CT), A4_CA (build CA), A4_CT0 (build CT0). An evicted block is
kept, compacted to ratio 0.2 (CT_SLOTS of 640 rows per KV head) into a frozen compacted region charged to the same 16,384-row budget (CtCompactor).
  python3 a1_a4.py --out ~/r1e/out/a4 --ref-dir ~/r1e/out/a1_pilot
Pure helpers (inequalities, cut plan, `<|im_end|>` finders, ranks, source status) are above the driver and tested in test_a1_a4.py. Reads a1_a3 / a1_e /
a1_e9 / a1_e67 / a1_core / r2_core and changes none of them."""
import argparse
import contextlib
import hashlib
import inspect
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
ARMS = {"A4_K4": {"build": "S", "k": K_REQ}, "A4_K0": {"build": "S0", "k": 0}, "A4_K4P": {"build": "P", "k": K_REQ}}
ROUTED = ("A4_K4", "A4_K4P", "A4_CTK4")
CT_SLOTS, MIN_WINDOW, Q_CAP = BLOCK // 5, 2, 1024      # ratio 0.2 of a block per KV head; the smallest window (blocks); the most delayed queries per fit
# Amendment 1. A build fixes how an evicted block is compacted; later blocks are prefilled behind it, so each build is its own pass per trajectory:
#   CT selection only on the delayed queries, CA the FUNI fit on the delayed queries, CT0 selection only on the block's own seal-time queries.
CT_BUILDS = {"CT": {"fit": False, "queries": "delayed"}, "CA": {"fit": True, "queries": "delayed"}, "CT0": {"fit": False, "queries": "own"}}
CT_ARMS = {"A4_CT": {"build": "CT", "k": 0}, "A4_CTK4": {"build": "CT", "k": K_REQ}, "A4_CA": {"build": "CA", "k": 0}, "A4_CT0": {"build": "CT0", "k": 0}}
ALL_ARMS = {**ARMS, **CT_ARMS}
# Run order (arm-major, all 6 trajectories per build, so the CT-vs-K0 answer lands first): A4_K0, A4_CT + A4_CTK4, A4_CT0, A4_CA, then A4_K4, A4_K4P.
# A4_K0 has its own build pass (S0, the same state as S); the original A4_K4 keeps S.
BUILD_ORDER = ("S0", "CT", "CT0", "CA", "S", "P")
RANKERS = ("x_last", "x_imend", "x_ucontent", "x_lit5", "x_max", "b_last")


class OverBudget(RuntimeError):
    """A state that cannot fit the row cap even with the smallest window. On this grid only a bug gets here: the run stops."""


def build_cfg(build, p):
    """S: no prefix, a SINK-row sink. P: the first p blocks stay exact (block 0 is among them, so the sink is moot)."""
    return {"pinned": 0, "sink": SINK} if build in ("S", "S0") else {"pinned": p, "sink": 0}


def arms_of(build):
    return [a for a, s in ALL_ARMS.items() if s["build"] == build]


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


# ---------------------------------------------------------------- Amendment 1 guards: rows over every layer, state digest, dependency digest

def rows_all(triples, full_layers):
    """Rows per full-attention layer, asserted equal across ALL full layers and across K, beta and V (a_a3._full_rows looks at layer 0 only)."""
    ns = {triples[i][c].shape[2] for i in full_layers for c in range(3)}
    assert len(ns) == 1, f"full-attention layers hold different row counts {sorted(ns)}"
    return ns.pop()


def build_guard(triples, S, add, cap):
    """Before a build forward: the cache about to be extended by `add` tokens holds at most `cap` rows on every full-attention layer."""
    if triples is not None:
        rows = rows_all(triples, [i for i in range(len(S.layer_types)) if i not in S.linear])
        assert rows + add <= cap, f"build forward would reach {rows} + {add} rows, cap {cap}"


def state_digest(triples):
    """sha256 over every layer's tensors (K, beta, V of the 16 full-attention layers, the DeltaNet conv and recurrent state), shapes and dtypes
    included. The no-eviction invariant is compared by this digest between A4_K0 and the compact-on-evict builds."""
    h = hashlib.sha256()
    for i, t in enumerate(triples):
        for c, x in enumerate(t):
            x = x.detach().cpu().contiguous()
            h.update(f"{i}.{c}:{tuple(x.shape)}:{x.dtype}".encode())
            if x.numel():
                h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def digest_files(paths):
    """sha256 over (basename, content) of the files, sorted by path."""
    h = hashlib.sha256()
    for q in sorted(set(paths)):
        h.update(os.path.basename(q).encode())
        h.update(open(q, "rb").read())
    return h.hexdigest()


def compaction_deps():
    """Fingerprint of what a CT/CA/CT0 fit depends on beyond our own files: the resolved sources of the compactor (registry, the method and
    algorithm classes, the context-prefill query generator), the cache and query-capture overlay (models.qwen3_5 and models.generate), the config
    loaders, and the resolved algorithm and query configs of cell FUNI. Goes into the manifest: a change between attempts refuses to resume."""
    import importlib
    import evaluation  # noqa: F401
    import r1b_core
    from evaluation.configs.utils import load_algorithm_config, load_query_config
    cfg_file, name, qcfg, _ = r1b_core.CELLS["FUNI"]
    method = r1b_core.method_for("FUNI", None)
    names = ["compaction.compaction_methods.registry", "compaction.query_generation.context_prefill", "models.qwen3_5", "models.qwen3_5.query_capture",
             "models.generate", "evaluation.configs.utils", type(method).__module__, method.algorithm_class.__module__]
    files = set()
    for n in names:
        f = inspect.getsourcefile(importlib.import_module(n))
        files.add(f)
        if os.path.basename(f) == "__init__.py":                                   # a package: the overlay's modules sit next to it
            files.update(os.path.join(os.path.dirname(f), x) for x in os.listdir(os.path.dirname(f)) if x.endswith(".py"))
    cfgs = json.dumps({"algorithm": load_algorithm_config(cfg_file)[name], "query": load_query_config(qcfg)}, sort_keys=True, default=repr)
    return {"files": digest_files(files), "n_files": len(files), "configs": hashlib.sha256(cfgs.encode()).hexdigest(), "cell": "FUNI", "method": name}


# ---------------------------------------------------------------- pure: compact-on-evict budget (Amendment 1)

def ct_slots(j, sink, block=BLOCK):
    """Slots per KV head of compacted block j: a fifth of its candidate rows (block 0 keeps its sink rows resident, so 576 rows give 115)."""
    return (block - candidate_start(j, sink, block)) // 5


def ct_bound(n_whole, zero_whole, n_comp, k, sink, tail, block=BLOCK):
    """Rows bound of a state with `n_whole` whole blocks, `n_comp` compacted blocks charged CT_SLOTS each and `k` of them swapped for their
    exact block (640 instead of CT_SLOTS): block*(n_whole+k) + sink*[block 0 not whole] + CT_SLOTS*(n_comp-k) + tail. No reserve."""
    return block * (n_whole + k) + (sink if sink and not zero_whole else 0) + CT_SLOTS * (n_comp - k) + tail


def ct_rows(window, comp, fetched, sink, tail, block=BLOCK):
    """Actual rows: whole blocks (window and fetched), the sink rows unless block 0 is whole, the real slots of the compacted blocks not fetched."""
    whole = set(window) | set(fetched)
    return (block * len(whole) + (sink if sink and 0 not in whole else 0) + tail
            + sum(ct_slots(j, sink, block) for j in comp if j not in whole))


def ct_build_step(resident, compacted, new_id, sink, cap=CAP, block=BLOCK, min_window=MIN_WINDOW):
    """Before block `new_id` is appended: evict the oldest window blocks (each is compacted, so it costs CT_SLOTS rows) until
        (CT-BUILD)  block*(W+1) + sink*[block 0 not whole] + CT_SLOTS*C + (block-1)  <=  cap
    with W resident whole blocks and C compacted ones. The window never goes under `min_window` blocks (counting the new one); when that
    floor is reached and it still does not fit, the oldest compacted block is dropped.
    # ponytail: the drop is the ceiling of the compacted region; a dropped block leaves the GPU state and is never routable (its exact
    # copy stays in the host archive). Counted by the caller.
    Returns (evict, drop), both oldest first; an evicted block may also be dropped at once (then it is not compacted)."""
    keep, comp, evict, drop = sorted(resident), sorted(compacted), [], []
    while ct_bound(len(keep) + 1, 0 in keep or new_id == 0, len(comp), 0, sink, block - 1, block) > cap:
        if len(keep) + 1 > min_window:
            evict.append(keep.pop(0))
            comp.append(evict[-1])
        elif comp:
            drop.append(comp.pop(0))
        else:
            raise OverBudget(f"block {new_id}: window floor {min_window} blocks and no compacted block left exceed {cap} rows")
    return evict, drop


def ct_cut_plan(nb, tail, resident, compacted, reserve, k_req, sink, cap=CAP, block=BLOCK, min_window=MIN_WINDOW):
    """The state at a cut from the build's window `resident` and compacted set. The largest window w <= len(resident), w >= min(min_window,
    len(resident)), for which, with the oldest window blocks beyond w compacted at the cut and k = min(k_req, compacted),
        (CT-CUT)  ct_bound(w, ., C, k) + tail + reserve  <=  cap
    (C counts the compacted blocks after the cut's own evictions). At the window floor the oldest compacted blocks are dropped (for this cut only)
    until it fits. With nothing compacted anywhere the plan is asserted equal to A4_K0's cut_plan: compaction is active only when the pool is full."""
    res, comp0 = sorted(resident), sorted(compacted)
    wb = len(res)
    floor = min(min_window, wb)
    for w in range(wb, floor - 1, -1):
        window, cut_ev = res[wb - w:], res[:wb - w]
        comp = comp0 + cut_ev
        n_drop = 0
        while True:
            k = min(k_req, len(comp) - n_drop)
            bound = ct_bound(w, 0 in window, len(comp) - n_drop, k, sink, tail, block)
            if bound + reserve <= cap:
                dropped = comp[:n_drop]
                left = comp[n_drop:]
                plan = {"w": w, "w_build": wb, "window": window, "comp": left, "cands": left, "cut_compact": [j for j in cut_ev if j in left],
                        "dropped_cut": dropped, "k": k, "base": window, "rows_bound": bound, "rows_base": ct_rows(window, left, [], sink, tail, block),
                        "reserve": reserve, "peak_bound": bound + reserve}
                if not comp0 and not cut_ev:
                    k0 = cut_plan(nb, tail, 0, wb, reserve, 0, sink, cap, block)
                    assert plan["window"] == k0["window"] and plan["rows_bound"] == k0["rows_bound"], (plan, k0)    # the A4_K0 state exactly
                return plan
            if w > floor or n_drop == len(comp):
                break
            n_drop += 1
    raise OverBudget(f"nb {nb} tail {tail} reserve {reserve}: no window of at least {floor} blocks fits {cap} rows")


def ct_piece_list(host, comp, window, fetched, sink, block=BLOCK):
    """Per-layer pieces of a compact-on-evict state in logical order: the sink rows of block 0 (unless block 0 is whole), then every block
    present, whole (window or fetched, the host copy) or compacted (`comp[j]`, frozen)."""
    whole = set(window) | set(fetched)
    pieces = [slice_piece(host[0], 0, sink)] if sink and 0 not in whole else []
    return pieces + [host[j] if j in whole else comp[j] for j in sorted(whole | set(comp))]


def logical_pieces(sink_piece, resident, comp):
    """gen_lru's cache pieces in logical order: the sink rows unless block 0 is resident, then each block resident (exact) or compacted."""
    return ([sink_piece] if sink_piece is not None and 0 not in resident else []) + [
        resident[j] if j in resident else comp[j] for j in sorted(set(resident) | set(comp))]


def delayed_blocks(j, upto, mode):
    """Blocks whose prefill queries fit block j, evicted when `upto` blocks have been built: the ones sealed after it and before the one being
    appended (j+1 .. upto-1), or, for the immediate control, its own."""
    return list(range(j + 1, upto)) if mode == "delayed" else [j]


def subsample_idx(n, cap=Q_CAP):
    """At most `cap` evenly spaced token indices out of n (midpoints of cap equal strides; all of them when n <= cap)."""
    return list(range(n)) if n <= cap else [(2 * i + 1) * n // (2 * cap) for i in range(cap)]


def gather_queries(parts, idx):
    """Tokens `idx` (indices into the concatenation along the token axis) of the (L, H, n_i, d) tensors `parts`, concatenated in order."""
    out, off, pos = [], 0, 0
    for p in parts:
        n = p.shape[2]
        sel = []
        while pos < len(idx) and idx[pos] < off + n:
            sel.append(idx[pos] - off)
            pos += 1
        if sel:
            out.append(p[:, :, torch.tensor(sel, dtype=torch.long)])
        off += n
    return torch.cat(out, 2)


def q_summary(log):
    """Compactions so far, how many used fewer query tokens than were available (subsampled to Q_CAP), mean tokens available and used."""
    n = len(log)
    return {"compactions": n, "subsampled": sum(1 for r in log if r["n_avail"] > r["n_used"]), "q_avail_mean": sum(r["n_avail"] for r in log) / n if n else None,
            "q_used_mean": sum(r["n_used"] for r in log) / n if n else None}


@contextlib.contextmanager
def selection_only(R):
    """E8's selection-only compaction (a1_core.select_only) on the runner for the duration of the block; restored on exit."""
    orig = R.method.algorithm_class
    R.method.algorithm_class = core.select_only(orig)
    try:
        yield
    finally:
        R.method.algorithm_class = orig


class CtCompactor:
    """Compact-on-evict for one build pass: the post-RoPE queries of every sealed block are kept on host while it is in the window; a block
    leaving the window is compacted to CT_SLOTS per KV head on the delayed (or own) queries, subsampled to at most Q_CAP tokens, by the
    shipping compactor (`R._compact`, cell FUNI), either fitted (beta and values) or selection only (zero beta, original K/V rows)."""

    def __init__(self, R, stream, build, sink, block=BLOCK, q_cap=Q_CAP):
        self.R, self.stream, self.sink, self.block, self.q_cap = R, stream, sink, block, q_cap
        self.fit, self.mode = CT_BUILDS[build]["fit"], CT_BUILDS[build]["queries"]
        self.comp, self.queries, self.log, self.drops = {}, {}, [], 0

    def on_queries(self, j, q):
        self.queries[j] = q.detach().to("cpu")

    def make(self, j, entry, lin_src, upto, log=None):
        """The compacted per-layer (K, beta, V) of block j (None on DeltaNet layers) from its exact `entry`; nothing is stored. `lin_src[i]` supplies
        the DeltaNet entries the repo's cache needs. The tokens available and used are appended to `log` (default: the build log `self.log`;
        cut-time fits pass their own list so the two are never mixed)."""
        S, b = self.R.S, self.block
        cs = candidate_start(j, self.sink, b)
        n = b - cs
        parts = [self.queries[m] for m in delayed_blocks(j, upto, self.mode)]
        assert parts, f"block {j}: no queries at upto={upto}"
        avail = sum(p.shape[2] for p in parts)
        idx = subsample_idx(avail, self.q_cap)
        q = gather_queries(parts, idx).to(S.dev)
        sl = slice_piece(entry, cs, b) if cs else entry
        kv = self.R._pseudo_kv(tuple(lin_src[i] if i in S.linear else sl[i] for i in range(len(S.layer_types))), S.linear)
        ids = self.stream[:, j * b + cs:(j + 1) * b]
        with (contextlib.nullcontext() if self.fit else selection_only(self.R)):
            seg, _ = self.R._compact(kv, n // 5, range(0, n), S.tok.decode(ids[0]), queries=q)
        out = tuple(None if i in S.linear else tuple(x.detach() for x in seg[i]) for i in range(len(S.layer_types)))
        want = ct_slots(j, self.sink, b)
        assert all(x.shape[2] == want for o in out if o is not None for x in o), (j, want, [x.shape for o in out if o is not None for x in o][:3])
        (self.log if log is None else log).append({"block": j, "upto": upto, "n_avail": avail, "n_used": len(idx)})
        return out

    def add(self, j, entry, lin_src, upto):
        """Compact block j at its eviction and freeze it; its own queries are no longer needed."""
        self.comp[j] = self.make(j, entry, lin_src, upto)
        self.queries.pop(j, None)

    def drop(self, ids):
        for j in ids:
            self.comp.pop(j, None)
            self.queries.pop(j, None)
        self.drops += len(ids)


# ---------------------------------------------------------------- build: LRU generator and host store

def gen_lru(R, segs, ks, pinned, sink, cap=CAP, block=BLOCK, on_block=None, ct=None):
    """Like r2_core.gen_fblk_window, but older blocks are evicted, not compacted. Every block is prefilled exact behind what the cache holds
    at that moment (pinned blocks, sink rows, the window) with the true positions; the DeltaNet state is carried through every block. Before
    a block is appended, build_evictions drops the oldest unpinned whole blocks. `on_block(j, a, b, exact)` gets each block's exact
    per-layer (K, beta, V) at seal time. Yields (k, triples, ctx_len, info): the build state (pinned/sink + window + the open tail, whose
    K/V were computed behind that window) with info = {tail_tokens, blocks, resident (whole-block ids), pinned, sink}.
    `ct` (a CtCompactor, Amendment 1): an evicted block is compacted into ct.comp instead of dropped (ct_build_step), kept in the cache in
    logical order; the sealed blocks' queries are handed to ct; info adds `compacted` (ids) and `drops`. ct=None is the original generator."""
    import r2_core as r2
    S = R.S
    stream = torch.cat([s.ids for s in segs], 1)
    ends = [sum(s.n for s in segs[:i + 1]) for i in range(len(segs))]
    n_layers = len(S.layer_types)
    resident, sink_piece, lin, orig_len, consumed, built = {}, None, None, 0, 0, 0

    def compose():
        if lin is None:
            return None
        pieces = logical_pieces(sink_piece, resident, ct.comp if ct else {})
        return tuple(lin[i] if i in S.linear else tuple(torch.cat([pc[i][c] for pc in pieces], 2) for c in range(3))
                     for i in range(n_layers))

    for k in sorted(ks):
        blocks, new_end = r2.sealed_blocks(consumed, ends[k], block)
        for a, b in blocks:
            if ct is None:
                evict, drop = build_evictions(sorted(resident), built, pinned, sink, cap, block), []
            else:
                evict, drop = ct_build_step(sorted(resident), sorted(ct.comp), built, sink, cap, block)
            for e in evict:
                if ct is not None and e not in drop:
                    ct.add(e, resident[e], lin, built)
                del resident[e]
            if ct is not None:
                ct.drop(drop)
            n = b - a
            base = compose()
            build_guard(base, S, b - a, cap)
            state, orig_len, q = r2.extend(S.model, S.layer_types, base, orig_len, stream[:, a:b].to(S.dev), want_queries=ct is not None)
            if ct is not None:
                ct.on_queries(built, q)
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
            build_guard(t_tr, S, tail.shape[1], cap)
            t_tr, t_len, _ = r2.extend(S.model, S.layer_types, t_tr, orig_len, tail.to(S.dev))
        info = {"tail_tokens": int(tail.shape[1]), "blocks": built, "resident": sorted(resident), "pinned": pinned, "sink": sink}
        if ct is not None:
            info.update(compacted=sorted(ct.comp), drops=ct.drops)
        yield k, t_tr, t_len, info


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
    return pieces_state(ship, piece_list(host, full_ids, sink), tail, linear, dev)


def pieces_state(ship, pieces, tail, linear, dev):
    """`cut_state` for an explicit piece list (compact-on-evict states mix exact and compacted pieces)."""
    pieces = [to_device(p, dev) for p in pieces]
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
    got = rows_all(trip, st.full_layers)
    assert got == final_rows(plan, fetched, st.sink, st.tail, block) <= plan["rows_bound"], (got, plan["rows_bound"])
    assert got + plan["reserve"] <= cap, f"answer state {got} + reserve {plan['reserve']} over {cap}"
    tm = {}
    with a1_a3.row_guard(S, got, cap):
        gen = r2.answer_ids(S, trip, st.t_len, suffix, max_new, timing=tm)
    del trip
    return gen, tm, got


def route_compacted(S, base, t_len, suffix, cand, comp, full_layers):
    """CTK4's routing pass: the question suffix prefilled under the cut's base state, its queries captured, the pass's KV dropped. Score of each
    candidate = the mass its compacted slots (with their beta) get at the LAST question token, normalised over the candidates
    (a1_e.router_mass, mean over layers and query heads): `x_last` over the compacted slots. Returns (scores over cand, timings)."""
    import r2_core as r2
    ids = torch.tensor([suffix], dtype=torch.long).to(S.dev)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t0 = time.time()
    _, _, q = r2.extend(S.model, S.layer_types, base, t_len, ids, want_queries=True)
    sync()
    t1 = time.time()
    keys = [torch.cat([comp[j][i][0][0] for j in cand], 1) for i in full_layers]
    betas = [torch.cat([comp[j][i][1][0] for j in cand], 1) for i in full_layers]
    spans, pos = [], 0
    for j in cand:
        spans.append((pos, pos + comp[j][full_layers[0]][0].shape[2]))
        pos = spans[-1][1]
    x = a1_e.router_mass(q, keys, betas, spans)
    del q
    sync()
    return x[-1].cpu(), {"route_s": round(t1 - t0, 3), "router_s": round(time.time() - t1, 3)}


def answer_ct(S, st, plan, comp, fetched, suffix, max_new, cap, block=BLOCK):
    """answer_state for a compact-on-evict plan: window and fetched blocks exact, the other present blocks compacted (frozen), the sink rows
    when block 0 is not whole. Rows asserted against the plan's bound and the cap; the answer runs under A3's row guard."""
    import r2_core as r2
    trip = pieces_state(st.ship, ct_piece_list(st.host, comp, plan["window"], fetched, st.sink, block), st.tail, st.linear, S.dev)
    got = rows_all(trip, st.full_layers)
    assert got == ct_rows(plan["window"], plan["comp"], fetched, st.sink, st.tail, block) <= plan["rows_bound"], (got, plan["rows_bound"])
    assert got + plan["reserve"] <= cap, f"answer state {got} + reserve {plan['reserve']} over {cap}"
    tm = {}
    with a1_a3.row_guard(S, got, cap):
        gen = r2.answer_ids(S, trip, st.t_len, suffix, max_new, timing=tm)
    del trip
    return gen, tm, got


def ct_source_status(located, span, nb, window, comp, sink, block=BLOCK):
    """not_located | resident (a source block is in the window, the open tail or the sink) | compacted | dropped (neither)."""
    if not located:
        return "not_located"
    src = source_blocks(span, nb, sink, block)
    if not src or any(j in window or j >= nb for j in src):
        return "resident"
    return "compacted" if any(j in comp for j in src) else "dropped"


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
        no_ev = {a: not pl["cands"] and pl["w"] == nb for a, pl in plans.items()}                        # nothing evicted at this cut for this arm
        dig = state_digest(ship) if any(no_ev.values()) else None
        st = SimpleNamespace(ship=ship, host=store.host, sink=sink, tail=tail, t_len=t_len, linear=S.linear, full_layers=full_layers)
        for a, pl in plans.items():
            core.append_row(files["cuts"], {"traj": tid, "build": build, "arm": a, "cut_index": cut.cut_index, "ctx_len": t_len,
                                            "n_blocks": nb, "tail": tail, "pinned": pinned, "sink": sink, "window_build": w_build,
                                            "window_cut": pl["w"], "n_cand": len(pl["cands"]), "k": pl["k"], "reserve": reserve,
                                            "rows_base": pl["rows_base"], "rows_bound": pl["rows_bound"], "peak_bound": pl["peak_bound"],
                                            "digest": dig if no_ev[a] else None, "attempt": attempt})
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
                    assert rows_all(bases[a], full_layers) + len(item.suffix) <= cap, "routing pass over the row cap"
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


def run_build_ct(ctx, out, build, tid, traj, tasks, full_rows, log, attempt, cap=CAP):
    """Amendment 1: one trajectory, one compact-on-evict build (CT serves A4_CT and A4_CTK4; CA and CT0 serve their own arm). A done marker closes the pass."""
    H, S, R = ctx.H, ctx.S, ctx.R
    files, dpath = _paths(out, build)
    if tid in {r["traj"] for r in core.read_rows(dpath)}:
        return None
    ret = core.retained_probe_ids(H, tasks, full_rows)
    todo = {t["id"] for t in tasks if t["id"] in ret}
    if not todo:
        return None
    plan = core.restrict(core.plan_trajectory(H, traj, [t for t in tasks if t["id"] in todo]), todo)
    full_layers = [i for i in range(len(S.layer_types)) if i not in S.linear]
    store = HostStore(full_layers)
    segs = core.segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    ct = CtCompactor(R, torch.cat([s.ids for s in segs], 1), build, SINK)
    stats = {"traj": tid, "build": build, "n_rows": 0, "build_s": 0.0, "answer_s": 0.0, "attempt": attempt}
    log(f"[a4] {tid[-8:]} build {build}: compact on evict ({CT_BUILDS[build]}), sink {SINK} rows, {CT_SLOTS} slots per evicted block")
    t_mark = time.time()
    for k_, ship, t_len, info in gen_lru(R, segs, list(range(len(segs))), 0, SINK, cap, BLOCK, store.on_block, ct=ct):
        stats["build_s"] += time.time() - t_mark
        cut, nb, tail = plan.cuts[k_], info["blocks"], info["tail_tokens"]
        assert len(store.host) == nb and t_len == cut.end
        res = info["resident"]
        assert res == list(range(nb - len(res), nb)), info                                       # the window is contiguous and newest
        reserve = max(len(i.suffix) + i.max_new for i in cut.items)
        plans = {a: ct_cut_plan(nb, tail, res, info["compacted"], reserve, CT_ARMS[a]["k"], SINK, cap) for a in arms_of(build)}
        made, comps = {}, {}
        no_ev = {a: not pl["comp"] and not pl["dropped_cut"] and pl["w"] == nb and not info["drops"] for a, pl in plans.items()}
        dig = state_digest(ship) if any(no_ev.values()) else None
        for a, pl in plans.items():                     # the cut's own evictions are compacted on the queries available now, never stored
            cut_log = []                                # this arm's cut-time fits, kept apart from the build's log
            for j in pl["cut_compact"]:
                if j not in made:
                    rec = []
                    made[j] = (ct.make(j, to_device(store.host[j], S.dev), ship, nb, log=rec), rec[0])
                cut_log.append(made[j][1])
            comps[a] = {j: ct.comp[j] if j in ct.comp else made[j][0] for j in pl["comp"]}
            core.append_row(files["cuts"], {"traj": tid, "build": build, "arm": a, "cut_index": cut.cut_index, "ctx_len": t_len, "n_blocks": nb, "tail": tail,
                                            "sink": SINK, "window_build": len(res), "window_cut": pl["w"], "n_compacted": len(pl["comp"]),
                                            "n_cand": len(pl["cands"]), "n_cut_compact": len(pl["cut_compact"]), "drops_build": info["drops"],
                                            "drops_cut": len(pl["dropped_cut"]), "k": pl["k"], "reserve": reserve, "rows_base": pl["rows_base"],
                                            "rows_bound": pl["rows_bound"], "peak_bound": pl["peak_bound"], "q": q_summary(ct.log), "q_cut": q_summary(cut_log),
                                            "digest": dig if no_ev[a] else None, "attempt": attempt})
        st = SimpleNamespace(ship=ship, host=store.host, sink=SINK, tail=tail, t_len=t_len, linear=S.linear, full_layers=full_layers)
        bases = {a: pieces_state(ship, ct_piece_list(store.host, comps[a], pl["window"], [], SINK), tail, S.linear, S.dev) for a, pl in plans.items()}
        for item in cut.items:
            t = item.task
            t0 = time.time()
            gold_len = len(H.ids(t["answer"]))
            lo = t["src_tok"]
            span = a1_e.locate_span(lambda sel: S.tok.decode(sel, skip_special_tokens=False), plan.stream, lo, cut.end, t["answer"])
            src = source_blocks(span, nb, SINK)
            for a, pl in plans.items():
                cands, fetched, ranks, rt, ranked = pl["cands"], [], None, {"route_s": 0.0, "router_s": 0.0}, []
                status = ct_source_status(span is not None, span, nb, pl["window"], pl["comp"], SINK)
                if a in ROUTED and pl["k"]:
                    assert rows_all(bases[a], full_layers) + len(item.suffix) <= cap, "routing pass over the row cap"
                    x, rt = route_compacted(S, bases[a], t_len, item.suffix, cands, comps[a], full_layers)
                    ranked = a1_a3.rank_candidates(x, cands)
                    fetched = ranked[:pl["k"]]
                    ranks = {"x_last": a1_a3.rank_of(ranked, src)}
                gen, tm, got = answer_ct(S, st, pl, comps[a], fetched, item.suffix, item.max_new, cap)
                text = S.tok.decode(core.strip_stop(gen, S.eos), skip_special_tokens=False)
                fields = {"attempt": attempt, "build": build, "k": len(fetched), "k_plan": pl["k"], "rows": got, "reserve": reserve, "peak_rows": got + reserve,
                          "n_cand": len(cands), "n_compacted": len(pl["comp"]), "window_build": len(res), "window_cut": pl["w"], "drops_build": info["drops"],
                          "drops_cut": len(pl["dropped_cut"]), "fetch_bytes": fetch_rows(fetched, SINK) * KV_BYTES_PER_TOKEN, "n_blocks": nb, "tail_tokens": tail,
                          "src_blocks": src, "src_status": status, "answer_reused": False, "probe_wall_s": round(time.time() - t0, 2)}
                extra = {**fields, "cut_index": cut.cut_index, "ctx_len": t_len, "decode_tokens": len(gen), "fetched_blocks": fetched, "route_s": rt["route_s"],
                         "timings": {"prefill_ms": round(tm["prefill_s"] * 1000, 1), "decode_ms": round(tm["decode_s"] * 1000, 1)}}
                core.append_row(files[a], core.make_row(item, a, gen, text, ctx.parse(text, plan.tools), S.eos, extra))
                stats["n_rows"] += 1
                if a in ROUTED:
                    core.append_row(files[a + "_route"], {**fields, "id": t["id"], "arm": a, "traj": tid, "type": t.get("type"), "distance_tok": t["distance_tok"],
                                                          "ctx_len": t_len, "cut_index": cut.cut_index, "located": span is not None,
                                                          "src_span": list(span) if span else [lo, lo + gold_len], "ranks": ranks, "ranked_top16": ranked[:16],
                                                          "src_cand_blocks": [j for j in src if j in cands], "fetched": fetched, "route_s": rt["route_s"],
                                                          "router_s": rt["router_s"]})
                stats["answer_s"] += time.time() - t0
                t0 = time.time()
        del bases, comps, made
        a_, r_ = core.mem_gib()
        log(f"[a4] {tid[-8:]} {build} cut {k_ + 1}/{len(segs)} ctx {t_len} blocks {nb} window {len(res)} compacted {len(info['compacted'])} "
            f"drops {info['drops']} peak alloc {a_:.1f} reserved {r_:.1f} GiB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    stats["build_s"], stats["answer_s"] = round(stats["build_s"], 1), round(stats["answer_s"], 1)
    stats["q"] = q_summary(ct.log)
    stats["peak_alloc_gib"], stats["peak_reserved_gib"] = (round(x, 1) for x in core.mem_gib())
    core.append_row(dpath, {"traj": tid, "attempt": attempt, "n_rows": stats["n_rows"]})
    return stats


def run_pilot(ctx, out, ref_dir, n, log, attempt, cap=CAP):
    H = ctx.H
    trajs, tasks = H.load_trajs(), H.load_tasks()
    ref = ensure_reference(ref_dir, out)
    full = core.read_rows(ref)
    a1_a3.check_universe(a1_a3.universe_ids(H, tasks, full, set(list(trajs)[:n])), a1_a3.PILOT_PROBES, "pilot reference")
    for build in BUILD_ORDER:
        prepare_build(out, build, log)
    for build in BUILD_ORDER:                              # arm-major: one build over all trajectories, then the next (done markers are per build and trajectory)
        for tid in list(trajs)[:n]:
            tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
            fr = [r for r in full if r["traj"] == tid]
            assert fr, f"no reference FULL rows for {tid}"
            t0 = time.time()
            st = (run_build_ct if build in CT_BUILDS else run_build)(ctx, out, build, tid, trajs[tid], tp, fr, log, attempt, cap)
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
    fp = {"stage": "a4", "n": a.n, "cap": a.cap, "sink": SINK, "arms": ALL_ARMS, "ct": {"builds": CT_BUILDS, "slots": CT_SLOTS, "min_window": MIN_WINDOW, "q_cap": Q_CAP, "deps": compaction_deps()},
          "build_order": BUILD_ORDER, "model": a.model,
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
