"""Stub tests for A4 (a1_a4.py, a1_a4_report.py): the build-time eviction inequality, the cut-time window / candidates / k under the reserve,
the `<|im_end|>` position finders, ranks restricted to the candidates, the sink bookkeeping (no row counted twice), the LRU generator against a
fake `extend`, and the report's counts. Pure functions and tensors only; no model."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
import a1_a3  # noqa: E402
import a1_a4 as a4  # noqa: E402
import a1_a4_report as rep  # noqa: E402

B, S, CAP = 640, 64, 16384


def simulate_build(nb, pinned, sink, cap=CAP, block=B):
    """Resident whole-block ids after nb appends, plus the largest rows (with the largest tail) seen after any append."""
    res, worst = [], 0
    for j in range(nb):
        for e in a4.build_evictions(res, j, pinned, sink, cap, block):
            res.remove(e)
        res.append(j)
        worst = max(worst, a4.state_rows(res, sink, block - 1, block))
    return res, worst


# ---------------------------------------------------------------- build-time inequality

def test_state_rows_counts_the_sink_once():
    assert a4.state_rows([3, 4], 64, 10) == 2 * B + 64 + 10                    # block 0 evicted: sink rows stand in for it
    assert a4.state_rows([0, 3], 64, 10) == 2 * B + 10                         # block 0 whole: its sink rows are among its 640
    assert a4.state_rows([0, 1], 0, 0) == 2 * B and a4.state_rows([2], 0, 5) == B + 5


def test_build_eviction_keeps_the_inequality_and_evicts_oldest_first():
    for pinned, sink in ((0, S), (10, 0)):
        res = []
        for j in range(70):
            ev = a4.build_evictions(res, j, pinned, sink)
            unpinned = [x for x in sorted(res) if x >= pinned]
            assert ev == unpinned[:len(ev)]                                    # oldest first, never a pinned block
            keep = [x for x in res if x not in ev] + [j]
            assert a4.state_rows(keep, sink, B - 1) <= CAP                     # (BUILD) holds after the append
            if ev:                                                             # minimal: keeping the last evicted block would break it
                assert a4.state_rows(keep + [ev[-1]], sink, B - 1) > CAP
            res = keep
        assert all(x in res for x in range(pinned))                           # pinned blocks survive
        win = [x for x in res if x >= pinned]
        assert win == list(range(70 - len(win), 70))                          # the window is contiguous and newest


def test_build_windows_at_the_registered_cap():
    res, worst = simulate_build(57, 0, S)                                       # sink build: 24 whole blocks + sink 64 + tail 639 = 16,063
    assert len(res) == 24 and res == list(range(33, 57)) and worst == 24 * B + S + (B - 1) == 16063 <= CAP
    res, worst = simulate_build(57, 10, 0)                                      # P build: 10 pinned + 14 window blocks
    assert len(res) == 24 and res[:10] == list(range(10)) and res[10:] == list(range(43, 57)) and worst == 24 * B + B - 1 <= CAP
    res, _ = simulate_build(19, 0, S)                                           # short context: nothing is evicted
    assert res == list(range(19))


def test_build_eviction_aborts_when_the_pinned_blocks_alone_do_not_fit():
    with pytest.raises(a4.OverBudget):
        a4.build_evictions(list(range(25)), 25, 25, 0)
    with pytest.raises(a4.OverBudget):
        a4.build_evictions([], 0, 0, 0, cap=B + 100)                          # one block + the largest tail do not fit


# ---------------------------------------------------------------- cut-time window, candidates, k under the reserve

def test_cut_plan_worked_numbers_nb57():
    nb, tail, reserve = 57, 64, 364
    w_s = len(simulate_build(nb, 0, S)[0])
    k4 = a4.cut_plan(nb, tail, 0, w_s, reserve, 4, S)
    assert (k4["w"], k4["k"], len(k4["cands"])) == (20, 4, 37) and k4["window"] == list(range(37, 57)) and k4["cands"] == list(range(0, 37))
    assert k4["rows_bound"] == B * 24 + S + tail == 15488 and k4["peak_bound"] == 15852 <= CAP
    k0 = a4.cut_plan(nb, tail, 0, w_s, reserve, 0, S)
    assert (k0["w"], k0["k"], len(k0["cands"])) == (24, 0, 33) and k0["rows_bound"] == B * 24 + S + tail
    w_p = len([j for j in simulate_build(nb, 10, 0)[0] if j >= 10])
    k4p = a4.cut_plan(nb, tail, 10, w_p, reserve, 4, 0)
    assert (k4p["w"], k4p["k"]) == (10, 4) and k4p["cands"] == list(range(10, 47)) and k4p["base"] == list(range(10)) + list(range(47, 57))
    assert k4p["rows_bound"] == B * 24 + tail and k4p["peak_bound"] <= CAP


def test_cut_plan_short_context_has_no_candidates_and_is_the_full_state():
    w = len(simulate_build(19, 0, S)[0])
    p = a4.cut_plan(19, 300, 0, w, 364, 4, S)
    assert p["cands"] == [] and p["k"] == 0 and p["base"] == list(range(19)) and p["rows_bound"] == 19 * B + 300   # block 0 whole: no sink row


@pytest.mark.parametrize("pinned,sink,k_req", [(0, S, 4), (0, S, 0), (10, 0, 4)])
def test_cut_plan_invariants_over_the_grid(pinned, sink, k_req):
    for nb in range(1, 70):
        res, _ = simulate_build(nb, pinned, sink)
        w_build = len([j for j in res if j >= pinned])
        for tail in (0, 1, 639):
            for reserve in (164, 364):
                pl = a4.cut_plan(nb, tail, pinned, w_build, reserve, k_req, sink)
                P = min(pinned, nb)
                assert pl["w"] <= w_build and pl["peak_bound"] <= CAP                         # (CUT)
                assert pl["window"] == list(range(nb - pl["w"], nb)) and pl["cands"] == list(range(P, nb - pl["w"]))
                assert pl["k"] == min(k_req, len(pl["cands"]))
                if pl["w"] < w_build:                                                         # maximal: one more window block would not fit
                    w1 = pl["w"] + 1
                    base = set(range(P)) | set(range(nb - w1, nb))
                    k1 = min(k_req, nb - P - w1)
                    assert B * (len(base) + k1) + (sink if sink and 0 not in base else 0) + tail + reserve > CAP
                # the actual rows of any fetch of k candidates never exceed the bound; fetching block 0 saves exactly the sink rows
                for fetched in (pl["cands"][:pl["k"]], pl["cands"][-pl["k"]:] if pl["k"] else []):
                    assert a4.final_rows(pl, fetched, sink, tail) <= pl["rows_bound"]
                if sink and pl["k"] and 0 in pl["cands"][:pl["k"]]:
                    assert a4.final_rows(pl, pl["cands"][:pl["k"]], sink, tail) == pl["rows_bound"] - sink


def test_cut_plan_aborts_when_even_an_empty_window_does_not_fit():
    with pytest.raises(a4.OverBudget):
        a4.cut_plan(57, 639, 0, 24, CAP, 4, S)
    with pytest.raises(a4.OverBudget):
        a4.cut_plan(40, 0, 10, 14, CAP - 10 * B + 1, 0, 0)


def test_the_k0_window_is_never_smaller_than_the_k4_window():
    for nb in range(25, 70):
        w = len(simulate_build(nb, 0, S)[0])
        assert a4.cut_plan(nb, 100, 0, w, 300, 0, S)["w"] >= a4.cut_plan(nb, 100, 0, w, 300, 4, S)["w"]


def test_fetch_rows_charge_block_zero_without_its_sink():
    assert a4.fetch_rows([0, 5], S) == (B - S) + B and a4.fetch_rows([0, 5], 0) == 2 * B and a4.fetch_rows([], S) == 0
    assert a4.candidate_start(0, S) == S and a4.candidate_start(3, S) == 0 and a4.candidate_start(0, 0) == 0


# ---------------------------------------------------------------- <|im_end|> position on a synthetic template

IM_START, IM_END, NL, ASSISTANT, THINK, THINK_END, NL2 = 1, 2, 3, 4, 5, 6, 7


def suffix(question):
    """<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n  (the B0 render, thinking off)."""
    return [IM_START, 99, NL] + question + [IM_END, NL, IM_START, ASSISTANT, NL, THINK, NL2, THINK_END, NL2]


def test_imend_is_the_user_turn_closer_and_ucontent_the_token_before_it():
    ids = suffix([10, 11, 12])
    i = a4.imend_index(ids, IM_START, IM_END)
    assert ids[i] == IM_END and i == len(ids) - 9                               # nine from the end with this generation prompt
    assert ids[a4.ucontent_index(ids, IM_START, IM_END)] == 12 and a4.ucontent_index(ids, IM_START, IM_END) == i - 1
    assert a4.lit5_index(ids) == len(ids) - 5 and ids[a4.lit5_index(ids)] == NL
    # the engine's prompt has no think block: there the closer is the literal fifth token from the end
    eng = [IM_START, 99, NL, 10, 11, IM_END, NL, IM_START, ASSISTANT, NL]
    assert a4.imend_index(eng, IM_START, IM_END) == len(eng) - 5 == a4.lit5_index(eng)
    assert a4.imend_index(suffix([10]), IM_START, IM_END) == 4 and a4.ucontent_index(suffix([10]), IM_START, IM_END) == 3
    assert a4.imend_index([IM_END, NL, IM_START], IM_START, IM_END) == 0 and a4.ucontent_index([IM_END, NL, IM_START], IM_START, IM_END) is None
    two = [IM_START, 99, NL, 50, IM_END, NL] + suffix([7, 8])                   # an earlier turn in the suffix: the LAST header wins
    assert two[a4.imend_index(two, IM_START, IM_END)] == IM_END and two[a4.ucontent_index(two, IM_START, IM_END)] == 8
    assert a4.imend_index(two, IM_START, IM_END) > 5
    assert a4.imend_index([10, 11, 12], IM_START, IM_END) is None and a4.lit5_index([1, 2]) == 0


# ---------------------------------------------------------------- ranks restricted to candidates; sources

def test_rank_row_ranks_among_the_candidates_only_and_ignores_resident_sources():
    cand = [4, 6, 9]
    x = torch.tensor([[0.2, 0.1, 0.7], [0.1, 0.6, 0.3], [0.5, 0.3, 0.2], [0.1, 0.2, 0.7], [0.6, 0.1, 0.3], [0.3, 0.4, 0.3]])   # (n = 6 tokens, 3)
    b = torch.tensor([[0.1] * 6, [0.9] * 6, [0.5] * 6])
    r = a4.rank_row(x, b, cand, [9, 2], imend=1, ucontent=0, lit5=2)                         # block 2 is not a candidate: ignored
    assert r == {"x_last": 2, "x_imend": 2, "x_ucontent": 1, "x_lit5": 3, "x_max": 1, "b_last": 2}
    assert a4.rank_row(x, b, cand, [2, 3], 1, 0, 2) is None                         # nothing routable
    assert a4.rank_row(x, b, cand, [6], None, None, 2)["x_imend"] is None


def test_block_max_handles_a_short_block_zero():
    cs = torch.tensor([[1.0, 5.0], [2.0, 0.0], [9.0, 1.0], [3.0, 3.0], [4.0, 8.0]])
    out = a4.block_max(cs, [2, 3])
    assert torch.equal(out, torch.tensor([[2.0, 5.0], [9.0, 8.0]]))
    with pytest.raises(AssertionError):
        a4.block_max(cs, [2, 2])


def test_source_blocks_and_status():
    assert a4.source_blocks(None, 5, S) == []
    assert a4.source_blocks((10, 30), 5, S) == []                              # a fact wholly inside the sink rows is resident
    assert a4.source_blocks((10, 100), 5, S) == [0] and a4.source_blocks((600, 700), 5, S) == [0, 1]
    assert a4.source_blocks((10, 30), 5, 0) == [0]                              # no sink (P arm): block 0 is just a block
    assert a4.source_status(False, None, 5, [0, 1], S) == "not_located"
    assert a4.source_status(True, (700, 710), 5, [0, 1], S) == "candidate"
    assert a4.source_status(True, (700, 710), 5, [], S) == "resident"            # in the window
    assert a4.source_status(True, (10, 30), 5, [0, 1], S) == "resident"          # in the sink
    assert a4.source_status(True, (3300, 3310), 5, [0, 1], S) == "resident"      # in the open tail (block 5 is not sealed)


# ---------------------------------------------------------------- pieces, insertion, host store (tensors)

HKV, D, NLAYER = 2, 4, 3                                                         # layer 1 is a DeltaNet layer


def entry(val, rows):
    return tuple(None if i == 1 else (torch.full((1, HKV, rows, D), float(val)), torch.zeros(1, HKV, rows), torch.full((1, HKV, rows, D), float(val) + 0.5))
                 for i in range(NLAYER))


def host_blocks(nb=6, rows=8):
    return [entry(j + 1, rows) for j in range(nb)]


def ship_of(host, tail=3):
    """The generator's cut state: every block whole, then `tail` rows of 99; the DeltaNet entry is a marker."""
    tr = []
    for i in range(NLAYER):
        if i == 1:
            tr.append((torch.ones(1), torch.zeros(1, 1, 0), torch.ones(1) * 7))
            continue
        t = (torch.full((1, HKV, tail, D), 99.0), torch.zeros(1, HKV, tail), torch.full((1, HKV, tail, D), 99.0))
        tr.append(tuple(torch.cat([host[j][i][c] for j in range(len(host))] + [t[c]], 2) for c in range(3)))
    return tuple(tr)


def test_piece_list_puts_the_sink_first_and_never_counts_a_row_twice():
    host = host_blocks()
    p = a4.piece_list(host, [3, 4], 4)
    assert [x[0][0].shape[2] for x in p] == [4, 8, 8] and torch.equal(p[0][0][0], host[0][0][0][:, :, :4]) and p[1] is host[3]
    p0 = a4.piece_list(host, [0, 3], 4)                                         # block 0 whole: no separate sink piece
    assert [x[0][0].shape[2] for x in p0] == [8, 8] and p0[0] is host[0]
    assert len(a4.piece_list(host, [2], 0)) == 1


def test_cut_state_inserts_fetched_blocks_exact_in_order_and_keeps_tail_and_deltanet():
    host = host_blocks()
    ship = ship_of(host)
    tail, sink, linear = 3, 4, {1}
    base = [4, 5]
    st = a4.cut_state(ship, host, base, sink, tail, linear, "cpu")
    assert st[0][0].shape[2] == a4.state_rows(base, sink, tail, 8)
    fetched = [1, 3]
    st2 = a4.cut_state(ship, host, sorted(set(base) | set(fetched)), sink, tail, linear, "cpu")
    assert st2[0][0].shape[2] == a4.state_rows([1, 3, 4, 5], sink, tail, 8)
    for i in (0, 2):
        for c in range(3):
            want = torch.cat([host[0][i][c][:, :, :sink]] + [host[j][i][c] for j in (1, 3, 4, 5)] + [ship[i][c][:, :, -tail:]], 2)
            assert torch.equal(st2[i][c], want)                                  # byte-equal to the host copies, in logical order, tail last
    assert st2[1] is ship[1]                                                      # DeltaNet entry comes from the ship state
    st3 = a4.cut_state(ship, host, [0, 4], sink, tail, linear, "cpu")             # fetching block 0 brings its remaining rows: 8, not 8 + 4
    assert st3[0][0].shape[2] == 2 * 8 + tail and torch.equal(st3[0][0][:, :, :8], host[0][0][0])


def test_host_store_trims_block_zero_for_keys_and_pooled_chunks():
    # a block of 8 rows = 2 chunks of 4; sink 4 = one chunk
    store = a4.HostStore([0, 2], chunk=4)
    for j in range(3):
        ex = entry(j + 1, 8)
        ex = tuple(None if t is None else (torch.randn(1, HKV, 8, D), t[1], t[2]) for t in ex)
        store.on_block(j, 0, 8, ex)
    keys = store.cand_keys([0, 2], 4, "cpu")
    assert keys[0][0].shape == (HKV, 4, D) and keys[2][0].shape == (HKV, 8, D)
    assert torch.equal(keys[0][1], store.host[0][2][0][0][:, 4:])
    pooled, counts = store.cand_pooled([0, 2], 4, "cpu")
    assert counts == [1, 2] and pooled.shape[0] == 3 and torch.equal(pooled[0], store.pooled[0][1])
    _, counts_nosink = store.cand_pooled([0, 2], 0, "cpu")
    assert counts_nosink == [2, 2]
    with pytest.raises(AssertionError):
        store.cand_pooled([0], 3, "cpu")                                           # the sink must be whole chunks


# ---------------------------------------------------------------- the LRU generator against a fake `extend`

def fake_extend(model, layer_types, triples, orig_len, ids, want_queries=False):
    """Stands in for r2_core.extend: appends one row per token to every full layer, K = token id + 0.001 * rows already held (so a block's
    values depend on what it was computed behind), V = K + 0.5, beta 0; the DeltaNet entry accumulates the token ids."""
    n = ids.shape[1]
    rows = 0 if triples is None else triples[0][0].shape[2]
    val = ids[0].float()[None, None, :, None].expand(1, HKV, n, D) + 0.001 * rows
    out = []
    for i, t in enumerate(layer_types):
        prev = None if triples is None else triples[i]
        if t == "linear_attention":
            acc = (prev[0] if prev is not None else torch.zeros(1)) + ids.sum().float()
            out.append((acc, torch.zeros(1, 1, 0), acc * 2))
        else:
            k = val if prev is None else torch.cat([prev[0], val], 2)
            b = torch.zeros(1, HKV, k.shape[2])
            v = val + 0.5 if prev is None else torch.cat([prev[2], val + 0.5], 2)
            out.append((k, b, v))
    return tuple(out), (orig_len or 0) + n, None


def lru_run(monkeypatch, pinned, sink, cap, block, ends, n_tokens=100, seen=None):
    import r2_core as r2

    def recording(model, layer_types, triples, orig_len, ids, *a, **k):
        if seen is not None:
            seen.append({"orig_len": orig_len, "ids": [int(v) for v in ids[0].tolist()], "triples": None if triples is None else tuple(
                tuple(x.clone() for x in t) for t in triples)})
        return fake_extend(model, layer_types, triples, orig_len, ids, *a, **k)
    monkeypatch.setattr(r2, "extend", recording)
    types = ["full_attention", "linear_attention", "full_attention"]
    Sx = SimpleNamespace(layer_types=types, linear={1}, dev="cpu", model=None)
    stream = torch.arange(1, n_tokens + 1)[None]
    segs, prev = [], 0
    for e in ends:
        segs.append(SimpleNamespace(ids=stream[:, prev:e], n=e - prev))
        prev = e
    store = a4.HostStore([0, 2], chunk=block)
    got = list(a4.gen_lru(SimpleNamespace(S=Sx), segs, list(range(len(ends))), pinned, sink, cap, block, store.on_block))
    return got, store, stream


def test_gen_lru_evicts_oldest_blocks_keeps_the_sink_and_computes_the_tail_behind_the_window(monkeypatch):
    block, sink, cap = 8, 4, 56                                                     # window of at most 4 whole blocks before an append
    got, store, stream = lru_run(monkeypatch, 0, sink, cap, block, [30, 100])
    (k0, t0, n0, i0), (k1, t1, n1, i1) = got
    assert (n0, i0["blocks"], i0["tail_tokens"], i0["resident"]) == (30, 3, 6, [0, 1, 2])          # nothing evicted yet
    assert t0[0][0].shape[2] == 30 and torch.equal(t0[0][0][0, 0, :, 0], (torch.arange(1, 31).float() + 0.001 * torch.cat(
        [torch.zeros(8), torch.full((8,), 8.0), torch.full((8,), 16.0), torch.full((6,), 24.0)])))
    assert i1["blocks"] == 12 and i1["tail_tokens"] == 4 and n1 == 100
    res = i1["resident"]
    assert res == list(range(12 - len(res), 12)) and len(res) == 5 and 0 not in res        # 4 held before the last append + the new block
    assert t1[0][0].shape[2] == a4.state_rows(res, sink, 4, block) == 5 * 8 + sink + 4
    assert t1[0][0].shape[2] + 7 <= cap
    # the sink rows are block 0's own rows, computed with nothing in front of them
    assert torch.equal(t1[0][0][0, 0, :sink, 0], torch.arange(1, sink + 1).float())
    # a resident block's rows equal its host copy; an evicted block's host copy still holds the values it had while resident
    first = res[0] * block
    assert torch.equal(t1[0][0][0, 0, sink:sink + block, 0], store.host[res[0]][0][0][0, 0, :, 0])
    assert torch.allclose(store.host[res[0]][0][0][0, 0, :, 0], torch.arange(first + 1, first + block + 1).float() + 0.001 * (sink + 4 * block), atol=1e-4)
    assert len(store.host) == 12 and all(store.host[j][0][0].shape == (1, HKV, block, D) for j in range(12))
    # the DeltaNet state is carried through every token, evicted or not
    assert float(t1[1][0]) == float(stream.sum())


def test_gen_lru_with_a_pinned_prefix_never_evicts_it(monkeypatch):
    block, cap = 8, 56
    got, store, _ = lru_run(monkeypatch, 2, 0, cap, block, [100])
    (_, tr, n, info), = got
    res = info["resident"]
    assert res[:2] == [0, 1] and res[2:] == list(range(12 - len(res[2:]), 12)) and len(res) == 6
    assert tr[0][0].shape[2] == a4.state_rows(res, 0, info["tail_tokens"], block)


# ---------------------------------------------------------------- the report's counts

def test_report_cells_pairs_and_router_tables():
    tasks = [{"id": f"p{i}", "traj": f"t{i % 3}", "distance_tok": d, "probe_type": ty}
             for i, (d, ty) in enumerate([(100, "line"), (5000, "line"), (9000, "name"), (20000, "name"), (3000, "line"), (30000, "name")])]
    ids = [t["id"] for t in tasks]
    vals, cl = [1, 1, 0, 1, 1, 0], [t["traj"] for t in tasks]
    c = rep.retention_cells(tasks, ids, vals, cl)
    assert c["overall"]["correct"] == 4 and c["overall"]["n"] == 6
    assert {k: (v["correct"], v["n"]) for k, v in c["by_distance"].items()} == {"0-4096": (2, 2), "4096-8192": (1, 1), "8192-16384": (0, 1), "16384-inf": (1, 2)}
    assert {k: (v["correct"], v["n"]) for k, v in c["by_type"].items()} == {"line": (3, 3), "name": (1, 3)}
    bc = rep.by_candidates([{"id": i, "n_cand": n} for i, n in zip(ids, [0, 0, 0, 5, 0, 7])], ids, vals, cl)
    assert (bc["none"]["correct"], bc["none"]["n"], bc["some"]["correct"], bc["some"]["n"]) == (3, 4, 1, 2)
    a, b = dict(zip(ids, [1, 1, 0, 1, 0, 0])), dict(zip(ids, [1, 0, 0, 1, 1, 1]))
    p = rep.paired(a, b)
    assert (p["n"], p["both"], p["only_a"], p["only_b"], p["neither"]) == (6, 2, 1, 2, 1) and 0 < p["mcnemar_exact_p"] <= 1

    def row(i, st, ranks, k, d, cov=(), fe=()):
        return {"id": i, "src_status": st, "ranks": ranks, "k": k, "k_plan": k, "distance_tok": d, "n_cand": 9,
                "src_cand_blocks": cov, "fetched": fe}
    full = {n: 1 for n in rep.a1_a4.RANKERS}
    route = [row("a", "candidate", {**full, "x_last": 3, "x_imend": None}, 4, 20000, [2, 5], [2, 7]),
             row("b", "candidate", {**full, "x_last": 9}, 4, 100, [3], [4]),
             row("c", "resident", None, 0, 100), row("d", "not_located", None, 0, 100)]
    t = rep.router_table(route)
    assert t["source_status"] == {"candidate": 2, "resident": 1, "not_located": 1} and t["source_candidate_with_ranks"] == 2
    assert t["x_last"]["all"]["recall"]["4"] == 0.5 and t["x_last"]["over_16K"]["n"] == 1 and t["x_last"]["inside_k_optimistic"] == [1, 2]
    assert t["x_last_fetch_coverage"] == {"none": 1, "partial": 1, "all": 0}
    assert rep.fetch_coverage([1, 2], [2, 1]) == "all" and rep.fetch_coverage([], [1]) is None
    assert t["x_imend"]["all"]["n"] == 1 and t["x_imend"]["inside_k_optimistic"] == [1, 1]            # a missing x_imend is left out, not counted as a miss
    cuts = [{"n_cand": 0, "window_build": 14, "window_cut": 14, "k": 0, "reserve": 200, "rows_bound": 9000, "peak_bound": 9200},
            {"n_cand": 30, "window_build": 24, "window_cut": 20, "k": 4, "reserve": 300, "rows_bound": 15488, "peak_bound": 15788}]
    ct = rep.cuts_table(cuts)
    assert ct["cuts"] == 2 and ct["cuts_with_candidates"] == 1 and ct["k_histogram"] == {"0": 1, "4": 1} and ct["window_cut_min"] == 14
    rows = [{"id": "a", "n_cand": 0, "content": "x", "rows": 10, "peak_rows": 12}, {"id": "b", "n_cand": 3, "content": "y", "rows": 10, "peak_rows": 14, "answer_reused": True}]
    assert rep.no_candidate_identity(rows, {"a": {"content": "x"}, "b": {"content": "z"}}) == {"n": 1, "content_equal_to_reference": 1}
    assert rep.row_stats(rows)["max_peak_rows"] == 14 and rep.row_stats(rows)["answers_reused"] == 1


def test_arms_and_builds():
    assert a1_a3.CAP == a4.CAP == CAP and a4.SINK == 64 and a4.K_REQ == 4
    assert a4.arms_of("S") == ["A4_K4", "A4_K0"] and a4.arms_of("P") == ["A4_K4P"]
    assert a4.build_cfg("S", 10) == {"pinned": 0, "sink": 64} and a4.build_cfg("P", 10) == {"pinned": 10, "sink": 0}


def reference_calls(ends, pinned, sink, cap, block):
    """Plain reference for gen_lru's extend calls: (kind, token ids the input cache must hold, orig_len, incoming token ids). Token ids are
    1-based stream positions. Eviction is the oldest unpinned block until the held blocks, the new one and a (block - 1) tail fit."""
    calls, res, built, consumed = [], [], 0, 0

    def prefix(held):
        if built == 0:                                                         # nothing computed yet: the very first call has no cache
            return []
        return (list(range(1, sink + 1)) if sink and 0 not in held else []) + [t for j in sorted(held) for t in range(j * block + 1, (j + 1) * block + 1)]
    for e in ends:
        while consumed + block <= e:
            while block * (len(res) + 1) + (sink if sink and 0 not in res + [built] else 0) + block - 1 > cap:
                res.remove(min(j for j in res if j >= pinned))
            calls.append(("block", prefix(res), consumed, list(range(consumed + 1, consumed + block + 1))))
            res.append(built)
            built += 1
            consumed += block
        if e > consumed:
            calls.append(("tail", prefix(res), consumed, list(range(consumed + 1, e + 1))))
    return calls


@pytest.mark.parametrize("pinned,sink", [(0, 4), (2, 0)])
def test_gen_lru_every_extend_call_sees_exactly_the_expected_prefix(monkeypatch, pinned, sink):
    """At every extend call the full input cache (every head, channel and full layer; the DeltaNet entry too), orig_len and the incoming
    tokens equal an independent reference schedule. A prefix that lags or leads the window by one block, or holds an evicted row, fails."""
    block, cap, ends = 8, 56, [20, 40, 60, 80, 100]
    seen = []
    lru_run(monkeypatch, pinned, sink, cap, block, ends, seen=seen)
    want = reference_calls(ends, pinned, sink, cap, block)
    assert len(want) >= 12 and len(seen) == len(want)
    assert max(len(w[1]) for w in want) < 100 - 3 * block                      # the window really is smaller than the stream: eviction happens
    for n, (call, (kind, prefix, orig, new)) in enumerate(zip(seen, want)):
        assert (call["orig_len"], call["ids"]) == (orig, new), (n, kind)
        tr = call["triples"]
        if not prefix:
            assert tr is None or tr[0][0].shape[2] == 0, n
            continue
        for i in (0, 2):                                                       # both full-attention layers, K and V, every head and channel
            k, bt, v = tr[i]
            assert k.shape[2] == len(prefix) == bt.shape[2] == v.shape[2], (n, kind, k.shape[2], len(prefix))
            assert torch.equal(k, k[:, :1, :, :1].expand_as(k)) and torch.equal(v, k + 0.5)
            assert [int(round(x)) for x in k[0, 0, :, 0].tolist()] == prefix, (n, kind)
        assert float(tr[1][0]) == orig * (orig + 1) / 2, n                      # DeltaNet: every token before this call, evicted or not


def test_answer_state_uses_the_given_block_size_for_the_row_check(monkeypatch):
    """With a non-640 block the answer-path row assertion must use that block size (it once defaulted to 640)."""
    import contextlib
    import r2_core as r2
    block, sink, tail = 24, 4, 6
    plan = a4.cut_plan(8, tail, 0, 2, 10, 2, sink, 10 ** 6, block)
    fetched = [1, 3]
    rows = a4.state_rows(set(plan["base"]) | set(fetched), sink, tail, block)
    assert rows != a4.state_rows(set(plan["base"]) | set(fetched), sink, tail)          # the 640 default would give a different count
    monkeypatch.setattr(a4, "cut_state", lambda *a, **k: "trip")
    monkeypatch.setattr(a1_a3, "_full_rows", lambda trip, layers: rows)
    monkeypatch.setattr(a1_a3, "row_guard", lambda S, got, cap: contextlib.nullcontext())
    monkeypatch.setattr(r2, "answer_ids", lambda S, trip, t_len, suffix, max_new, timing=None: (timing.update(prefill_s=0, decode_s=0), [1])[1])
    st = SimpleNamespace(ship=None, host=None, sink=sink, tail=tail, linear=set(), full_layers=[0], t_len=0)
    gen, tm, got = a4.answer_state(SimpleNamespace(dev="cpu"), st, plan, fetched, [1], 1, 10 ** 6, block)
    assert got == rows and gen == [1]
    with pytest.raises(AssertionError):
        a4.answer_state(SimpleNamespace(dev="cpu"), st, plan, fetched, [1], 1, 10 ** 6)       # default 640 disagrees with the 24-token rows
