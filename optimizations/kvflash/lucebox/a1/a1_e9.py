"""Diagnostic E9, the exact-mass router (PREREG "Diagnostic E9"). E2's ranking, re-run on the same 6 pilot trajectories, 118 probes and cuts,
with one more router: the attention mass the question puts on each block's EXACT keys (the 640 rows `on_block` keeps), zero beta.
In the same pass a_last / a_max (compacted slots with beta, E2) and b (kvflash_qk cosine, E2) are ranked on the identical state.
  python3 a1_e9.py --out ~/r1e/out/e9 [--n 6]      (FULL.jsonl, the pilot's FULL-CHK rows, must already be in --out)
Two sets of ranks per probe:
  ranks   E2's: over every sealed block of the cut (shipping state, window 0), routers a_last a_max x_last x_max b_last b_max;
  budget  A3's: per W in a1_a3.SEL_WBS, among the candidate blocks only (never P, never W), normalised over the candidates as A3's router
          is, with the k that a1_a3.choose_k gives at the cut (reserve = longest suffix + max_new); in_k = the source block is in the top k.
No answers are generated. Reads a1_a3 / a1_e / a1_e67 / a1_core; changes none of them. Pure helpers are tested in test_a1_e9.py."""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_a3  # noqa: E402
import a1_core as core  # noqa: E402
import a1_e  # noqa: E402
import a1_e67 as e67  # noqa: E402

ROUTERS = ("a_max", "a_last", "x_max", "x_last", "b_max", "b_last")     # E2's names plus x = exact mass
BUDGET_ROUTERS = ("a_last", "x_last", "x_max", "b_last")


# ---------------------------------------------------------------- pure helpers

def exact_mass(q, exact_k, blocks):
    """(n, len(blocks)): E2's router_mass on the blocks' exact keys with zero beta. q: (L, H, n, d); exact_k[j]: per full layer the block's
    (Hkv, rows, d) keys; the softmax is over the listed blocks only, each block's share is the sum over its rows."""
    L = q.shape[0]
    keys = [torch.cat([exact_k[j][l] for j in blocks], 1) for l in range(L)]
    betas = [torch.zeros(k.shape[:2], dtype=torch.float32, device=k.device) for k in keys]
    spans, pos = [], 0
    for j in blocks:
        n = exact_k[j][0].shape[1]
        spans.append((pos, pos + n))
        pos += n
    return a1_e.router_mass(q, keys, betas, spans)


def compacted_mass(q, comp_k, comp_b, blocks):
    """The same for the compacted slots with their beta (E2's a, A3's a_last before the last-token pick)."""
    L = q.shape[0]
    keys = [torch.cat([comp_k[j][l] for j in blocks], 1) for l in range(L)]
    betas = [torch.cat([comp_b[j][l] for j in blocks], 1) for l in range(L)]
    spans, pos = [], 0
    for j in blocks:
        n = comp_k[j][0].shape[1]
        spans.append((pos, pos + n))
        pos += n
    return a1_e.router_mass(q, keys, betas, spans)


def ranks_for(a, x, b, blocks, src):
    """1-based best source rank per router among `blocks`. a, x: (n, B) masses over `blocks`; b: (B, n) block scores of `blocks`;
    src: block ids (those not in `blocks` are ignored). Returns {router: rank} or None when no source block is in `blocks`."""
    pos = [blocks.index(j) for j in src if j in blocks]
    if not pos:
        return None
    out = {}
    for name, m in (("a", a.T), ("x", x.T), ("b", b)):
        for agg, v in a1_e.aggregate_tokens(m).items():
            out[f"{name}_{agg}"] = a1_e.best_rank(v, pos)
    return out


def budget_row(nb, tail, p, wb, reserve, ranks_cand):
    """A3's budget for the cut and whether the source block is inside the fetched k under each router. ranks_cand: ranks_for over the
    candidates (None: no source block among the candidates)."""
    bud = a1_a3.choose_k(nb, tail, p, wb, reserve)
    row = {"p": p, "k": bud["k"], "n_cand": bud["n_cand"], "reserve": reserve, "src_cand": ranks_cand is not None}
    if ranks_cand is not None:
        row["rank"] = {r: ranks_cand[r] for r in BUDGET_ROUTERS}
        row["in_k"] = {r: ranks_cand[r] <= bud["k"] for r in BUDGET_ROUTERS}
    return row


# ---------------------------------------------------------------- driver

class BlockStore:
    """What `on_block` keeps per sealed block (full-attention layers only): the exact keys, the compacted keys and beta, the pooled chunk keys."""

    def __init__(self, full_layers, chunk=a1_e.CHUNK):
        self.full_layers, self.chunk = full_layers, chunk
        self.exact_k, self.comp_k, self.comp_b, self.pooled = [], [], [], []

    def on_block(self, j, a, b, ex, cp):
        fl = self.full_layers
        self.exact_k.append([ex[i][0][0] for i in fl])
        self.comp_k.append([cp[i][0][0] for i in fl])
        self.comp_b.append([cp[i][1][0] for i in fl])
        self.pooled.append(a1_e.pooled_chunk_keys([ex[i][0][0] for i in fl], self.chunk))


def probe_rows(q, store, nb, tail, p, reserve, src, block=a1_e.BLOCK, chunk=a1_e.CHUNK, wbs=a1_a3.SEL_WBS):
    """One eligible probe: (ranks over every block, {str(W): budget row}) from the question queries q (L, H, n, d) and the blocks kept."""
    bsc = a1_e.block_scores(a1_e.qk_chunk_scores(torch.cat(store.pooled), q), block // chunk)            # (B, n)
    allb = list(range(nb))
    ranks = ranks_for(compacted_mass(q, store.comp_k, store.comp_b, allb), exact_mass(q, store.exact_k, allb), bsc, allb, src)
    budget = {}
    for wb in wbs:
        cand = a1_a3.candidates(nb, p, wb)
        rc = None
        if cand and any(j in cand for j in src):
            rc = ranks_for(compacted_mass(q, store.comp_k, store.comp_b, cand), exact_mass(q, store.exact_k, cand), bsc[cand], cand, src)
        budget[str(wb)] = budget_row(nb, tail, p, wb, reserve, rc)
    return ranks, budget



def run_trajectory(ctx, out, tid, traj, tasks, full_rows, log):
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    path = os.path.join(out, "e9.jsonl")
    done = {r["id"] for r in core.read_rows(path)}
    ret = core.retained_probe_ids(H, tasks, full_rows)
    plan = core.restrict(core.plan_trajectory(H, traj, tasks), ret)
    if not plan.cuts:
        return {"traj": tid, "n_rows": 0}
    full_layers = [i for i in range(len(S.layer_types)) if i not in S.linear]
    end = e67.system_end(plan.stream, S.tok.convert_tokens_to_ids("<|im_end|>"))
    p = e67.prefix_blocks(end)
    store = BlockStore(full_layers)
    on_block = store.on_block

    segs = core.segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    stats = {"traj": tid, "n_rows": 0, "build_s": 0.0, "route_s": 0.0, "system_end_tok": end, "prefix_blocks": p}
    t_mark = time.time()
    for k, t_tr, t_len, info in r2.gen_fblk_window(R, segs, list(range(len(segs))), 0, on_block=on_block):
        stats["build_s"] += time.time() - t_mark
        cut, nb, tail = plan.cuts[k], info["blocks"], info["tail_tokens"]
        assert len(store.exact_k) == nb and t_len == cut.end
        reserve = max(len(i.suffix) + i.max_new for i in cut.items)
        for item in cut.items:
            t = item.task
            if t["id"] in done:
                continue
            t1 = time.time()
            lo = t["src_tok"]
            span = a1_e.locate_span(lambda sel: S.tok.decode(sel, skip_special_tokens=False), plan.stream, lo, cut.end, t["answer"])
            located = span is not None
            src = a1_e.overlap_blocks(*span, nb) if located else []
            row = {"id": t["id"], "traj": tid, "type": t.get("type"), "distance_tok": t["distance_tok"], "ctx_len": t_len, "n_blocks": nb,
                   "tail": tail, "src_blocks": src, "located": located, "eligible": bool(located and src)}
            if row["eligible"]:
                ids = torch.tensor([item.suffix], dtype=torch.long).to(S.dev)
                _, _, q = r2.extend(S.model, S.layer_types, t_tr, t_len, ids, want_queries=True)
                row["ranks"], row["budget"] = probe_rows(q, store, nb, tail, p, reserve, src)
                del q
            core.append_row(path, row)
            stats["n_rows"] += 1
            stats["route_s"] += time.time() - t1
        a_, r_ = core.mem_gib()
        log(f"[e9] {tid[-8:]} cut {k + 1}/{len(segs)} ctx {t_len} blocks {nb} peak alloc {a_:.1f} reserved {r_:.1f} GiB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--n", type=int, default=6)
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
    tasks = H.load_tasks()
    full_rows = core.read_rows(os.path.join(a.out, "FULL.jsonl"))
    for tid in list(trajs)[: a.n]:
        tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
        st = run_trajectory(ctx, a.out, tid, trajs[tid], tp, [r for r in full_rows if r["traj"] == tid], log)
        core.append_row(os.path.join(a.out, "progress.jsonl"), {**st, "arm": "E9", "peak_alloc_gib": core.mem_gib()[0]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
