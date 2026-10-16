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
    assert a4.arms_of("S") == ["A4_K4"] and a4.arms_of("S0") == ["A4_K0"] and a4.arms_of("P") == ["A4_K4P"]      # A4_K0 has its own build pass, run first
    assert a4.build_cfg("S0", 10) == a4.build_cfg("S", 10) and a4.BUILD_ORDER == ("S0", "CT", "CT0", "CA", "S", "P")
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
    monkeypatch.setattr(a4, "rows_all", lambda trip, layers: rows)
    monkeypatch.setattr(a1_a3, "row_guard", lambda S, got, cap: contextlib.nullcontext())
    monkeypatch.setattr(r2, "answer_ids", lambda S, trip, t_len, suffix, max_new, timing=None: (timing.update(prefill_s=0, decode_s=0), [1])[1])
    st = SimpleNamespace(ship=None, host=None, sink=sink, tail=tail, linear=set(), full_layers=[0], t_len=0)
    gen, tm, got = a4.answer_state(SimpleNamespace(dev="cpu"), st, plan, fetched, [1], 1, 10 ** 6, block)
    assert got == rows and gen == [1]
    with pytest.raises(AssertionError):
        a4.answer_state(SimpleNamespace(dev="cpu"), st, plan, fetched, [1], 1, 10 ** 6)       # default 640 disagrees with the 24-token rows


# ---------------------------------------------------------------- Amendment 1: compact on evict

def ct_build(nb, cap=CAP, sink=S, block=B):
    """ct_build_step over nb appends: (resident, compacted, drops, worst bound after an append)."""
    res, comp, drops, worst = [], [], 0, 0
    for j in range(nb):
        ev, dr = a4.ct_build_step(res, comp, j, sink, cap, block)
        for e in ev:
            res.remove(e)
            comp.append(e)
        for d in dr:
            comp.remove(d)
        drops += len(dr)
        res.append(j)
        worst = max(worst, a4.ct_bound(len(res), 0 in res, len(comp), 0, sink, block - 1, block))
    return res, comp, drops, worst


def test_ct_arms_and_builds():
    assert a4.CT_SLOTS == 128 == B // 5 and a4.MIN_WINDOW == 2 and a4.Q_CAP == 1024
    assert a4.arms_of("CT") == ["A4_CT", "A4_CTK4"] and a4.arms_of("CA") == ["A4_CA"] and a4.arms_of("CT0") == ["A4_CT0"]
    assert a4.arms_of("S") == ["A4_K4"] and "A4_CTK4" in a4.ROUTED and "A4_CT" not in a4.ROUTED       # the original arms are untouched
    assert a4.CT_BUILDS == {"CT": {"fit": False, "queries": "delayed"}, "CA": {"fit": True, "queries": "delayed"}, "CT0": {"fit": False, "queries": "own"}}
    assert a4.ct_slots(5, S) == 128 and a4.ct_slots(0, S) == 115 and a4.ct_slots(0, 0) == 128             # block 0 compacts its 576 candidate rows


@pytest.mark.parametrize("k_req", [0, 4])
def test_ct_where_nothing_is_evicted_the_state_is_a4_k0s_exactly(k_req):
    """The invariant: a cut that fits the budget is the A4_K0 state (full KV up to 24 blocks), for every tail and reserve."""
    checked = 0
    for nb in range(1, 25):
        res, comp, drops, _ = ct_build(nb)
        assert res == list(range(nb)) and comp == [] and drops == 0                                   # compaction is inactive until the pool is full
        for tail in (0, 1, 639):
            for reserve in (164, 364):
                k0 = a4.cut_plan(nb, tail, 0, nb, reserve, 0, S)
                ct = a4.ct_cut_plan(nb, tail, res, comp, reserve, k_req, S)
                if ct["cut_compact"]:                                                                 # the cut itself had to evict (24 blocks plus a full tail): not the invariant's case
                    assert k0["w"] < nb and ct["w"] == k0["w"]
                    continue
                checked += 1
                assert ct["window"] == k0["window"] and ct["rows_bound"] == k0["rows_bound"] and ct["k"] == 0 and ct["comp"] == []
                assert ct["rows_base"] == k0["rows_base"] == a4.state_rows(k0["base"], S, tail)
    assert checked >= 100


def test_ct_invariant_is_asserted_in_the_plan(monkeypatch):
    real = a4.cut_plan
    monkeypatch.setattr(a4, "cut_plan", lambda *a, **k: {**real(*a, **k), "rows_bound": real(*a, **k)["rows_bound"] + 1})
    with pytest.raises(AssertionError):
        a4.ct_cut_plan(10, 5, list(range(10)), [], 164, 0, S)


def test_ct_first_eviction_at_25_blocks_keeps_the_window_and_charges_the_compacted_block():
    res, comp, drops, worst = ct_build(25)
    assert res == list(range(1, 25)) and comp == [0] and drops == 0                                   # block 0 compacted, the window still 24 blocks
    assert worst == 24 * B + S + 128 + (B - 1) <= CAP
    res, comp, drops, worst = ct_build(40)                                                           # more compacted blocks squeeze the window
    assert len(res) < 24 and len(comp) == 40 - len(res) and drops == 0
    assert B * len(res) + S + 128 * len(comp) + (B - 1) <= CAP                                       # the inequality after the last append


def test_ct_budget_cap_holds_at_every_build_and_cut_and_for_any_fetch():
    for nb in range(1, 130):
        res, comp, drops, worst = ct_build(nb)
        assert worst <= CAP and len(res) >= min(a4.MIN_WINDOW, nb) and res == list(range(nb - len(res), nb))
        assert len(comp) + len(res) + drops == nb
        for tail in (0, 1, 639):
            for reserve in (164, 364):
                for k_req in (0, 4):
                    pl = a4.ct_cut_plan(nb, tail, res, comp, reserve, k_req, S)
                    assert pl["peak_bound"] == pl["rows_bound"] + reserve <= CAP and pl["w"] >= min(a4.MIN_WINDOW, len(res))
                    assert pl["k"] == min(k_req, len(pl["comp"])) and pl["comp"] == pl["cands"]
                    cs = pl["comp"]
                    assert a4.ct_rows(pl["window"], cs, [], S, tail) <= pl["rows_bound"]
                    # any k of the compacted blocks swapped for their exact block stays inside the bound, block 0 included
                    for pick in (cs[:pl["k"]], cs[len(cs) - pl["k"]:]):
                        assert len(pick) == pl["k"] and a4.ct_rows(pl["window"], cs, pick, S, tail) <= pl["rows_bound"]


def test_ct_the_drop_ceiling_drops_the_oldest_compacted_block_and_counts_it():
    cap = S + 2 * B + 128 * 3 + (B - 1)                                    # room for the minimum window of 2 and three compacted blocks
    res, comp, drops, worst = ct_build(40, cap)
    assert len(res) == 2 and comp == [35, 36, 37] and drops == 40 - 2 - 3 and worst <= cap
    # an evicted block can be dropped at once (the region is already at the ceiling): it is not compacted
    ev, dr = a4.ct_build_step([36, 37], [33, 34, 35], 38, S, cap)
    assert ev == [36] and dr == [33]
    with pytest.raises(a4.OverBudget):
        ct_build(40, S + 2 * B + (B - 1) - 1)                              # the floor window alone does not fit
    pl = a4.ct_cut_plan(40, 0, res, comp, 300, 0, S, cap + 300)
    assert pl["w"] == 2 and pl["dropped_cut"] == [] and pl["comp"] == [35, 36, 37]
    tight = a4.ct_cut_plan(40, 639, res, comp, 300, 0, S, cap + 300 - 128 + 1)       # one compacted block too many at the cut: the oldest goes, for this cut only
    assert tight["dropped_cut"] == [35] and tight["comp"] == [36, 37]


def test_ct_cut_evictions_compact_the_oldest_window_blocks_and_k4_swaps_inside_the_budget():
    res, comp, _, _ = ct_build(57)
    a = a4.ct_cut_plan(57, 639, res, comp, 364, 0, S)
    b = a4.ct_cut_plan(57, 639, res, comp, 364, 4, S)
    assert b["w"] < a["w"] and b["k"] == 4 and a["k"] == 0                                            # the fetch is paid for by a smaller window
    assert b["cut_compact"] == res[:len(res) - b["w"]] and b["comp"] == comp + b["cut_compact"]
    assert b["rows_bound"] + 364 <= CAP and a4.ct_bound(b["w"], False, len(b["comp"]), 4, S, 639) == b["rows_bound"]


def test_delayed_query_window_boundaries():
    assert a4.delayed_blocks(3, 7, "delayed") == [4, 5, 6]                  # sealed after it, before the block being appended
    assert a4.delayed_blocks(3, 5, "delayed") == [4] and a4.delayed_blocks(3, 4, "delayed") == []
    assert a4.delayed_blocks(3, 7, "own") == [3]                            # the immediate control: its own seal-time queries


def test_subsample_is_uniform_capped_and_gathers_the_right_tokens():
    assert a4.subsample_idx(1024) == list(range(1024)) and a4.subsample_idx(5, 10) == list(range(5))
    for n in (1025, 2 * B, 14720, 15360):
        idx = a4.subsample_idx(n)
        assert len(idx) == 1024 and idx == sorted(set(idx)) and 0 <= idx[0] and idx[-1] < n
        gaps = {b - a for a, b in zip(idx, idx[1:])}
        assert max(gaps) - min(gaps) <= 1 and idx[0] <= n / 1024 and n - 1 - idx[-1] <= n / 1024 + 1     # evenly spaced end to end
    parts = [torch.arange(n, dtype=torch.float32)[None, None, :, None].expand(2, 3, n, 4).clone() + off for n, off in ((5, 0), (7, 100), (3, 200))]
    got = a4.gather_queries(parts, [0, 4, 5, 11, 12, 14])
    assert got.shape == (2, 3, 6, 4) and got[0, 0, :, 0].tolist() == [0, 4, 100, 106, 200, 202]
    assert a4.q_summary([{"n_avail": 10, "n_used": 10}, {"n_avail": 30, "n_used": 12}]) == {"compactions": 2, "subsampled": 1, "q_avail_mean": 20, "q_used_mean": 11}
    assert a4.q_summary([])["compactions"] == 0


HK, DD, QL = 2, 3, 2


class FakeR:
    """Stands in for the repo's Runner: records every `_compact` call. Selection keeps the first `target` rows of each head with beta 0; the fit
    (algorithm class not wrapped) marks beta with 1 and doubles V."""

    def __init__(self, Sx):
        self.S, self.calls = Sx, []
        self.method = SimpleNamespace(algorithm_class=object)

    def _pseudo_kv(self, triples, linear, last=None):
        return tuple((t[0], t[2]) for t in triples)

    def _compact(self, kv, target, indices, text, queries=None):
        sel = self.method.algorithm_class is not object
        self.calls.append({"target": target, "n": len(indices), "queries": queries.clone(), "selection": sel})
        seg = []
        for i, (k, v) in enumerate(kv):
            if i in self.S.linear:
                seg.append((k, torch.zeros(1, 1, 0), v))
            else:
                seg.append((k[:, :, :target].clone(), torch.zeros(1, HK, target) if sel else torch.ones(1, HK, target),
                            (v if sel else 2 * v)[:, :, :target].clone()))
        return tuple(seg), {}


def ct_run(monkeypatch, build, cap, block, ends, n_tokens=120, sink=4, q_cap=a4.Q_CAP):
    import r2_core as r2
    monkeypatch.setattr(a4, "CT_SLOTS", block // 5)
    monkeypatch.setattr(a4.core, "select_only", lambda cls: type("SelectOnly", (cls,), {}))

    def ext(model, layer_types, triples, orig_len, ids, want_queries=False):
        out, olen, _ = fake_extend(model, layer_types, triples, orig_len, ids)
        q = ids[0].float()[None, None, :, None].expand(QL, 2, ids.shape[1], DD).clone() if want_queries else None
        return out, olen, q
    monkeypatch.setattr(r2, "extend", ext)
    types = ["full_attention", "linear_attention", "full_attention"]
    Sx = SimpleNamespace(layer_types=types, linear={1}, dev="cpu", model=None, tok=SimpleNamespace(decode=lambda ids: ""))
    R = FakeR(Sx)
    stream = torch.arange(1, n_tokens + 1)[None]
    segs, prev = [], 0
    for e in ends:
        segs.append(SimpleNamespace(ids=stream[:, prev:e], n=e - prev))
        prev = e
    ct = a4.CtCompactor(R, stream, build, sink, block, q_cap)
    store = a4.HostStore([0, 2], chunk=block)
    got, ct.snaps = [], []
    for out in a4.gen_lru(R, segs, list(range(len(ends))), 0, sink, cap, block, store.on_block, ct=ct):
        got.append(out)
        ct.snaps.append({j: (id(ct.comp[j]), ct.comp[j][0][0].clone()) for j in ct.comp})        # what the cut saw, to show later cuts leave it alone
    return got, store, ct, R


def test_ct_generator_compacts_on_evict_with_delayed_queries_and_freezes_the_block(monkeypatch):
    block, sink = 10, 5
    cap = sink + 6 * block + 2 * 2 + (block - 1)                           # six blocks (the new one included) plus two compacted ones fit; the five blocks of cut one fit whole
    got, store, ct, R = ct_run(monkeypatch, "CT", cap, block, [57, 120], sink=sink, q_cap=12)
    (_, _, _, i0), (_, tr, n, i1) = got
    assert i0["compacted"] == [] and i0["drops"] == 0                      # 5 sealed blocks: nothing evicted at the first cut
    assert i1["blocks"] == 12 and i1["resident"] == list(range(12 - len(i1["resident"]), 12))
    comp = i1["compacted"]
    assert comp and comp == list(range(len(comp))) and len(comp) + len(i1["resident"]) + i1["drops"] == 12
    # one compaction per surviving evicted block, never repeated (frozen), selection only
    assert len(R.calls) == len(ct.log) == len(comp) + i1["drops"] and len({r["block"] for r in ct.log}) == len(ct.log)
    assert all(c["selection"] for c in R.calls)
    for r, c in zip(ct.log, R.calls):
        j, up = r["block"], r["upto"]
        # delayed queries = the token ids of the blocks sealed after j and before the block being appended (j+1 .. up-1), subsampled evenly to q_cap
        toks = [t for m in range(j + 1, up) for t in range(m * block + 1, (m + 1) * block + 1)]
        want = [toks[i] for i in a4.subsample_idx(len(toks), 12)]
        assert r["n_avail"] == len(toks) and r["n_used"] == min(len(toks), 12) and toks[0] == (j + 1) * block + 1 and toks[-1] == up * block
        assert c["queries"].shape == (QL, 2, r["n_used"], DD) and c["queries"][0, 0, :, 0].tolist() == [float(t) for t in want]
        assert c["target"] == (block - (sink if j == 0 else 0)) // 5 and c["n"] == block - (sink if j == 0 else 0)
    assert any(r["n_avail"] > r["n_used"] for r in ct.log) and a4.q_summary(ct.log)["subsampled"] >= 1
    # the compacted entries are the selection of the exact host copy (first target rows), zero beta, and sit in the state in logical order
    for j in comp:
        cs = sink if j == 0 else 0
        for i in (0, 2):
            k, b, v = ct.comp[j][i]
            assert torch.equal(k, store.host[j][i][0][:, :, cs:cs + k.shape[2]]) and not b.any() and torch.equal(v, store.host[j][i][2][:, :, cs:cs + k.shape[2]])
    assert ct.comp[0][1] is None
    assert tr[0][0].shape[2] == a4.ct_rows(i1["resident"], comp, [], sink, i1["tail_tokens"], block)
    held = tr[0][0][0, 0, :, 0].tolist()
    assert held[:sink] == [float(t) for t in range(1, sink + 1)]            # the sink rows come first, then the compacted blocks, then the window, then the tail


def test_ct_compacted_blocks_are_frozen_across_cuts(monkeypatch):
    block, sink = 10, 5
    cap = sink + 6 * block + 2 * 2 + (block - 1)
    got, store, ct, R = ct_run(monkeypatch, "CT", cap, block, [70, 90, 105, 120], sink=sink)
    first, last = ct.snaps[1], ct.snaps[-1]
    assert first and set(first) <= set(last)
    for j, (ident, t) in first.items():
        assert last[j][0] == ident and torch.equal(last[j][1], t)           # the same entry object, the same values: never refit, never recompacted
    assert len(R.calls) == len(ct.log) == len({r["block"] for r in ct.log})  # one compaction per block over the whole run


def test_ct_generator_with_the_fit_and_with_own_queries(monkeypatch):
    block, sink = 10, 5
    cap = sink + 4 * block + 2 * 2 + (block - 1)
    _, _, ctf, Rf = ct_run(monkeypatch, "CA", cap, block, [120], sink=sink)
    assert not any(c["selection"] for c in Rf.calls) and ctf.comp[min(ctf.comp)][0][1].any()           # the fit: class not wrapped, beta nonzero
    _, _, cto, Ro = ct_run(monkeypatch, "CT0", cap, block, [120], sink=sink)
    assert len(Ro.calls) == len(cto.log) > 0
    for r, c in zip(cto.log, Ro.calls):
        j = r["block"]
        n = block - (sink if j == 0 else 0)
        assert r["n_avail"] == r["n_used"] == block == c["queries"].shape[2]                           # its own block's queries, nothing later
        assert c["queries"][0, 0, :, 0].tolist() == [float(t) for t in range(j * block + 1, (j + 1) * block + 1)] and c["n"] == n


def test_ct_generator_is_the_k0_generator_while_nothing_is_evicted(monkeypatch):
    block, sink, cap = 10, 5, 10_000
    got, store, ct, R = ct_run(monkeypatch, "CT", cap, block, [57, 120], sink=sink)
    assert R.calls == [] and all(i["compacted"] == [] and i["drops"] == 0 for *_, i in got)
    got0, store0, _ = lru_run(monkeypatch, 0, sink, cap, block, [57, 120], n_tokens=120)[:3]
    for (_, ta, na, ia), (_, tb, nb_, ib) in zip(got, got0):
        assert na == nb_ and ia["resident"] == ib["resident"] and all(torch.equal(x, y) for p, q in zip(ta, tb) for x, y in zip(p, q))


def test_ct_generator_drop_ceiling_counts_drops(monkeypatch):
    block, sink = 10, 5
    cap = sink + 2 * block + 2 * 2 + (block - 1)                           # window floor of 2 blocks and two compacted blocks
    got, store, ct, R = ct_run(monkeypatch, "CT", cap, block, [120], sink=sink)
    (_, tr, _, info), = got
    assert info["resident"] == [10, 11] and info["compacted"] == [8, 9] and info["drops"] == 12 - 2 - 2
    assert tr[0][0].shape[2] == a4.ct_rows(info["resident"], info["compacted"], [], sink, info["tail_tokens"], block)
    assert len(R.calls) == 10 and set(ct.queries) == {10, 11}              # every evicted block was compacted once; only the window keeps queries on host


def test_ct_generator_a_block_dropped_the_moment_it_is_evicted_is_never_compacted(monkeypatch):
    block, sink = 10, 5
    cap = sink + 2 * block + (block - 1)                                   # no room for any compacted block next to the window floor
    got, store, ct, R = ct_run(monkeypatch, "CT", cap, block, [120], sink=sink)
    (_, tr, _, info), = got
    assert info["compacted"] == [] and info["drops"] == 10 and R.calls == [] and info["resident"] == [10, 11]


def test_ct_pieces_in_logical_order_and_answer_rows(monkeypatch):
    import contextlib
    import r2_core as r2
    host = host_blocks(6, 8)
    comp = {1: entry(50, 1), 2: entry(60, 1)}   # block 8 rows: a fifth is 1 slot
    p = a4.ct_piece_list(host, comp, [4, 5], [2], 4, 8)
    assert [x[0][0].shape[2] for x in p] == [4, 1, 8, 8, 8] and p[1] is comp[1] and p[2] is host[2] and p[3] is host[4] and p[4] is host[5]
    p0 = a4.ct_piece_list(host, {1: comp[1]}, [0, 5], [], 4, 8)
    assert [x[0][0].shape[2] for x in p0] == [8, 1, 8]                      # block 0 whole: no separate sink piece
    assert a4.ct_rows([4, 5], [1, 2], [2], 4, 3, 8) == 8 * 3 + 4 + 3 + a4.ct_slots(1, 4, 8)
    plan = {"window": [4, 5], "comp": [1, 2], "rows_bound": 10 ** 6, "reserve": 10}
    seen = {}
    monkeypatch.setattr(a1_a3, "row_guard", lambda S_, got, cap: contextlib.nullcontext())
    monkeypatch.setattr(r2, "answer_ids", lambda S_, trip, t_len, suffix, max_new, timing=None: (timing.update(prefill_s=0, decode_s=0), seen.update(rows=trip[0][0].shape[2]), [1])[2])
    st = SimpleNamespace(ship=ship_of(host), host=host, sink=4, tail=3, t_len=0, linear={1}, full_layers=[0, 2])
    gen, tm, got = a4.answer_ct(SimpleNamespace(dev="cpu"), st, plan, comp, [2], [1], 1, 10 ** 6, 8)
    assert got == seen["rows"] == a4.ct_rows([4, 5], [1, 2], [2], 4, 3, 8)
    with pytest.raises(AssertionError):
        a4.answer_ct(SimpleNamespace(dev="cpu"), st, {**plan, "rows_bound": got - 1}, comp, [2], [1], 1, 10 ** 6, 8)


def test_ct_source_status():
    span = (3 * 8 + 1, 3 * 8 + 3)
    assert a4.ct_source_status(False, None, 10, [8, 9], [3], 4, 8) == "not_located"
    assert a4.ct_source_status(True, span, 10, [8, 9], [3], 4, 8) == "compacted"
    assert a4.ct_source_status(True, span, 10, [3, 9], [], 4, 8) == "resident"
    assert a4.ct_source_status(True, span, 10, [8, 9], [2], 4, 8) == "dropped"


def test_report_compact_on_evict_tables():
    tasks = [{"id": f"p{i}", "traj": f"t{i % 3}", "distance_tok": d} for i, d in enumerate([100, 5000, 20000, 30000, 16383, 16384, 40000, 9000, 70000])]
    ids = [t["id"] for t in tasks]
    split = dict(zip(ids, rep.pool_split(tasks, ids)))
    assert [split[i] for i in ids] == ["within", "within", "beyond", "beyond", "within", "beyond", "beyond", "within", "beyond"]
    cl = {t["id"]: t["traj"] for t in tasks}
    k0 = {i: 0 for i in ids}
    ct = {i: int(split[i] == "beyond" or j % 2 == 0) for j, i in enumerate(ids)}                       # better than K0 beyond the pool, mixed within it
    sv = rep.split_vs_k0(ct, k0, split, cl)
    assert sv["beyond"]["arm"]["correct"] == sv["beyond"]["arm"]["n"] == 5 and sv["beyond"]["A4_K0"]["correct"] == 0
    assert sv["beyond"]["paired"]["only_a"] == 5 and sv["beyond"]["paired"]["only_b"] == 0 and sv["beyond"]["difference"]["mean"] == 1.0
    assert sv["within"]["arm"]["n"] == 4 and sv["within"]["paired"]["n"] == 4
    assert rep.primary_rule(sv)["primary"] == "POSITIVE" and rep.primary_rule(sv)["bar"] == 0.756
    worse = rep.split_vs_k0({i: 0 for i in ids}, {i: 1 for i in ids}, split, cl)
    assert rep.primary_rule(worse)["primary"] == "NOT_POSITIVE" and worse["beyond"]["difference"]["hi"] < 0
    tie = rep.split_vs_k0(k0, k0, split, cl)
    assert rep.primary_rule(tie)["primary"] == "NOT_POSITIVE"                                          # a zero lower bound is not above 0
    q = lambda n, s: {"compactions": n, "subsampled": s, "q_avail_mean": 2000.0, "q_used_mean": 1000.0}
    cuts = [{"traj": "t0", "n_compacted": 0, "drops_build": 0, "drops_cut": 0, "n_cut_compact": 0, "window_cut": 24, "q": q(0, 0), "q_cut": q(0, 0)},
            {"traj": "t0", "n_compacted": 3, "drops_build": 1, "drops_cut": 1, "n_cut_compact": 1, "window_cut": 20, "q": q(5, 4), "q_cut": q(1, 1)},
            {"traj": "t1", "n_compacted": 1, "drops_build": 0, "drops_cut": 0, "n_cut_compact": 0, "window_cut": 23, "q": q(2, 1), "q_cut": q(0, 0)}]
    st = rep.ct_stats(cuts, [{"n_compacted": 0}, {"n_compacted": 3}, {"n_compacted": 1}, {"n_compacted": 3}])
    assert st["mean_compacted_blocks_per_cut"] == 4 / 3 and st["mean_compacted_blocks_per_probe"] == 7 / 4 and st["max_compacted_blocks"] == 3
    assert st["drops_build"] == 1 and st["drops_cut_total"] == 1 and st["cut_compactions"] == 1
    assert st["q_cut_compactions"] == 1 and st["q_cut_subsampled"] == 1                                    # cut-time fits are counted apart from the build's
    assert st["build_compactions"] == 7 and st["q_subsampled"] == 5 and st["q_avail_mean"] == 2000.0 and st["q_used_mean"] == 1000.0


# ---------------------------------------------------------------- review fixes: digests, guards, rule, router, logs, integrity, order

def test_state_digest_covers_every_tensor_and_shape():
    host = host_blocks(3, 8)
    ship = ship_of(host)
    base = a4.state_digest(ship)
    assert base == a4.state_digest(tuple(tuple(x.clone() for x in t) for t in ship))
    for layer in (0, 1, 2):                                                   # a full layer, the DeltaNet layer, the other full layer
        for c in range(3):
            if layer == 1 and c == 1:
                continue                                                      # the DeltaNet "beta" slot is empty (no element to change)
            mod = [list(t) for t in ship]
            x = mod[layer][c].clone()
            x.view(-1)[-1] += 1
            mod[layer][c] = x
            assert a4.state_digest(tuple(tuple(t) for t in mod)) != base, (layer, c)
    cut = tuple(tuple(x[..., :-1, :] if x.dim() == 4 else x[..., :-1] for x in t) if i != 1 else t for i, t in enumerate(ship))
    assert a4.state_digest(cut) != base                                       # one row less


def test_rows_all_checks_every_full_layer_and_the_build_guard_refuses_over_the_cap():
    host = host_blocks(2, 8)
    ship = ship_of(host, tail=2)
    assert a4.rows_all(ship, [0, 2]) == 18
    bad = list(ship)
    bad[2] = tuple(x[:, :, :-1] if x.dim() == 4 else x[:, :, :-1] for x in ship[2])                # layer 2 one row short, layer 0 untouched
    with pytest.raises(AssertionError):
        a4.rows_all(tuple(bad), [0, 2])
    Sx = SimpleNamespace(layer_types=["full_attention", "linear_attention", "full_attention"], linear={1})
    a4.build_guard(ship, Sx, 6, 24)
    a4.build_guard(None, Sx, 99, 1)                                                                # the first block has no cache yet
    with pytest.raises(AssertionError):
        a4.build_guard(ship, Sx, 7, 24)
    with pytest.raises(AssertionError):
        a4.build_guard(tuple(bad), Sx, 0, 99)


def test_digest_files_tracks_content(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1")
    b.write_text("y = 2")
    d0 = a4.digest_files([str(a), str(b)])
    assert d0 == a4.digest_files([str(b), str(a)])
    a.write_text("x = 2")
    assert a4.digest_files([str(a), str(b)]) != d0


def test_no_eviction_check_refuses_digest_and_answer_differences():
    k0c = [{"traj": "t", "cut_index": 0, "evicted": 0, "compacted": 0, "dropped": 0, "digest": "aa"}, {"traj": "t", "cut_index": 1, "evicted": 3, "compacted": 3, "dropped": 0, "digest": None}]
    k0r = [{"id": "p0", "traj": "t", "cut_index": 0, "gen_ids": [1, 2]}, {"id": "p1", "traj": "t", "cut_index": 1, "gen_ids": [3]}]
    ok_c = [{"traj": "t", "cut_index": 0, "evicted": 0, "compacted": 0, "dropped": 0, "digest": "aa"}, {"traj": "t", "cut_index": 1, "evicted": 3, "compacted": 3, "dropped": 0, "digest": None}]
    ok_r = [{"id": "p0", "traj": "t", "cut_index": 0, "gen_ids": [1, 2]}, {"id": "p1", "traj": "t", "cut_index": 1, "gen_ids": [9]}]     # after an eviction answers may differ
    assert rep.check_no_eviction("A4_CT", ok_c, ok_r, k0c, k0r) == {"cuts_compared": 1, "probes_compared": 1}
    with pytest.raises(SystemExit) as e:
        rep.check_no_eviction("A4_CT", [{**ok_c[0], "digest": "bb"}, ok_c[1]], ok_r, k0c, k0r)
    assert "('t', 0)" in str(e.value) and "digest" in str(e.value)
    with pytest.raises(SystemExit) as e:
        rep.check_no_eviction("A4_CT", ok_c, [{**ok_r[0], "gen_ids": [1, 3]}, ok_r[1]], k0c, k0r)
    assert "p0" in str(e.value) and "generated ids" in str(e.value)
    with pytest.raises(SystemExit) as e:                                                            # counts say no eviction, digest missing: fail closed
        rep.check_no_eviction("A4_CT", [{**ok_c[0], "digest": None}, ok_c[1]], ok_r, k0c, k0r)
    assert "missing" in str(e.value)
    with pytest.raises(SystemExit) as e:                                                            # K0's own digest missing at a quiet cut
        rep.check_no_eviction("A4_CT", ok_c, ok_r, [{**k0c[0], "digest": None}, k0c[1]], k0r)
    assert "A4_K0" in str(e.value) and "missing" in str(e.value)
    with pytest.raises(SystemExit):                                                                 # eviction counts say eviction, a digest is present
        rep.check_no_eviction("A4_CT", [ok_c[0], {**ok_c[1], "digest": "zz"}], ok_r, k0c, k0r)
    with pytest.raises(SystemExit):                                                                 # no counts at all
        rep.check_no_eviction("A4_CT", [{"traj": "t", "cut_index": 0, "digest": "aa"}, ok_c[1]], ok_r, k0c, k0r)
    with pytest.raises(SystemExit):                                                                 # evictionless in one only
        rep.check_no_eviction("A4_CT", [ok_c[0], {**ok_c[1], **{"evicted": 0, "compacted": 0}}], ok_r, k0c, k0r)
    with pytest.raises(SystemExit):                                                                 # a cut K0 does not have
        rep.check_no_eviction("A4_CT", ok_c + [{"traj": "t", "cut_index": 5, "evicted": 3, "compacted": 3, "dropped": 0, "digest": None}], ok_r, k0c, k0r)


def test_primary_rule_needs_the_beyond_bound_and_the_within_pool_guard():
    tasks = [{"id": f"p{i}", "traj": f"t{i % 6}", "distance_tok": 20000 if i < 18 else 1000} for i in range(118)]     # the registered 18 / 100 split
    ids = [t["id"] for t in tasks]
    split = dict(zip(ids, rep.pool_split(tasks, ids)))
    assert sum(v == "beyond" for v in split.values()) == 18 and sum(v == "within" for v in split.values()) == 100
    cl = {t["id"]: t["traj"] for t in tasks}
    k0 = {i: 1 for i in ids}
    k0.update({f"p{i}": 0 for i in range(18)})

    def arm(within_losses):
        a = dict(k0)
        a.update({f"p{i}": 1 for i in range(18)})                                                  # wins everywhere beyond the pool
        for i in range(18, 18 + within_losses):
            a[f"p{i}"] = 0
        return rep.primary_rule(rep.split_vs_k0(a, k0, split, cl))
    r2, r3_ = arm(2), arm(3)
    assert r2["beyond_lower_bound_above_0"] and r2["within_pool_guard"]["net_loss"] == 2 and r2["primary"] == "POSITIVE"
    assert r3_["beyond_lower_bound_above_0"] and r3_["within_pool_guard"]["net_loss"] == 3 and r3_["primary"] == "NOT_POSITIVE"
    # a within-pool win offsets a loss (net, not gross)
    a = dict(k0)
    a.update({f"p{i}": 1 for i in range(18)})
    for i in range(18, 22):
        a[f"p{i}"] = 0
    k1 = dict(k0)
    k1["p30"] = 0
    k1["p31"] = 0
    g = rep.primary_rule(rep.split_vs_k0(a, k1, split, cl))["within_pool_guard"]
    assert (g["k0_only"], g["ct_only"], g["net_loss"], g["ok"]) == (4, 2, 2, True)
    # no beyond-pool gain: not positive however clean the within-pool side
    assert rep.primary_rule(rep.split_vs_k0(dict(k0), k0, split, cl))["primary"] == "NOT_POSITIVE"


def test_ctk4_router_table_counts_compacted_sources_and_reports_only_x_last():
    route = [{"src_status": "compacted", "ranks": {"x_last": 1}, "distance_tok": 16384, "k_plan": 4, "n_cand": 9, "k": 4,
              "src_cand_blocks": [2], "fetched": [2, 3, 4, 5]},
             {"src_status": "compacted", "ranks": {"x_last": 7}, "distance_tok": 16383, "k_plan": 4, "n_cand": 9, "k": 4,
              "src_cand_blocks": [6], "fetched": [2, 3, 4, 5]},
             {"src_status": "resident", "ranks": None, "distance_tok": 100, "k_plan": 4, "n_cand": 9, "k": 4, "src_cand_blocks": [], "fetched": []}]
    t = rep.router_table(route, ("x_last",))
    assert t["source_candidate_with_ranks"] == 2 and set(k for k in t if k in a4.RANKERS) == {"x_last"}
    assert t["x_last"]["inside_k_optimistic"] == [1, 2] and t["x_last_fetch_coverage"] == {"none": 1, "partial": 0, "all": 1}
    assert len(t["x_last"]["over_16K"]) and t["x_last"]["over_16K"] != t["x_last"]["all"]            # distance 16,384 counts as beyond, 16,383 does not
    assert rep.router_table(route)["source_candidate_with_ranks"] == 2


def test_cut_time_fits_are_logged_apart_from_the_build(monkeypatch):
    block, sink = 10, 5
    cap = sink + 6 * block + 2 * 2 + (block - 1)
    got, store, ct, R = ct_run(monkeypatch, "CT", cap, block, [120], sink=sink)
    (_, ship, n, info), = got
    before = len(ct.log)
    rec = []
    j = info["resident"][0]
    ct.make(j, store.host[j], ship, info["blocks"], log=rec)
    assert len(ct.log) == before and len(rec) == 1 and rec[0]["block"] == j and rec[0]["upto"] == info["blocks"]
    assert j not in ct.comp                                                                          # nothing stored either
    ct.make(j, store.host[j], ship, info["blocks"])
    assert len(ct.log) == before + 1


def test_integrity_of_done_markers_and_cut_records():
    rows = {"A4_X": [{"traj": "t0", "cut_index": 0, "attempt": "a"}, {"traj": "t0", "cut_index": 1, "attempt": "a"}, {"traj": "t1", "cut_index": 0, "attempt": "a"}]}
    cuts = [{"arm": "A4_X", "traj": "t0", "cut_index": 0, "attempt": "a"}, {"arm": "A4_X", "traj": "t0", "cut_index": 1, "attempt": "a"},
            {"arm": "A4_X", "traj": "t1", "cut_index": 0, "attempt": "a"}]
    done = [{"traj": "t0", "n_rows": 2, "attempt": "a"}, {"traj": "t1", "n_rows": 1, "attempt": "a"}]
    ec = {"t0": {0, 1}, "t1": {0}}
    rep.check_integrity("B", rows, cuts, done, {"t0", "t1"}, ec)
    for bad_cuts, why in (([{**cuts[0], "attempt": "old"}, *cuts[1:]], "different attempt"),):               # a stale cut record under a good marker
        with pytest.raises(rep.r3.IncompleteRun) as e:
            rep.check_integrity("B", rows, bad_cuts, done, {"t0", "t1"}, ec)
        assert why in str(e.value)
    with pytest.raises(rep.r3.IncompleteRun) as e:                                                          # the task file has a cut the rows lack
        rep.check_integrity("B", rows, cuts, done, {"t0", "t1"}, {"t0": {0, 1, 2}, "t1": {0}})
    assert "task file" in str(e.value)
    for bad_done, bad_cuts, why in ((done[:1], cuts, "t1"), ([{**done[0], "n_rows": 3}, done[1]], cuts, "says 3"), ([{**done[0], "attempt": "b"}, done[1]], cuts, "attempts"),
                                    (done, cuts[:2], "cut records"), (done, cuts + [cuts[0]], "cut records"), (done + [done[0]], cuts, "t0")):
        with pytest.raises(rep.r3.IncompleteRun) as e:
            rep.check_integrity("B", rows, bad_cuts, bad_done, {"t0", "t1"})
        assert why in str(e.value)


def test_partial_report_marks_an_incomplete_arm_and_needs_nothing_else(tmp_path):
    import json
    (tmp_path / "A4_CT.jsonl").write_text(json.dumps({"id": "p0", "traj": "t0"}) + "\n")
    out = rep.compact_on_evict(str(tmp_path), None, [{"id": "p0", "traj": "t0", "cut_index": 0}, {"id": "p1", "traj": "t0", "cut_index": 0}], [], {"t0"}, {"p0", "p1"}, {"A4_K0": {}}, {}, partial=True)
    assert out == {"A4_CT": {"incomplete": "1 of 2 probes"}}


def test_run_pilot_is_arm_major_in_the_registered_order(monkeypatch):
    calls = []
    trajs = {f"t{i}": {} for i in range(6)}
    ctx = SimpleNamespace(H=SimpleNamespace(load_trajs=lambda: trajs, load_tasks=lambda: [{"traj": t, "kind": "probe"} for t in trajs]))
    monkeypatch.setattr(a4, "ensure_reference", lambda ref, out: "ref")
    monkeypatch.setattr(a4.core, "read_rows", lambda p: [{"traj": t} for t in trajs])
    monkeypatch.setattr(a4.a1_a3, "check_universe", lambda *a: None)
    monkeypatch.setattr(a4.a1_a3, "universe_ids", lambda *a: set())
    prepared = []
    monkeypatch.setattr(a4, "prepare_build", lambda out, b, log: prepared.append(b))
    monkeypatch.setattr(a4.core, "append_row", lambda *a: None)
    for fn in ("run_build", "run_build_ct"):
        monkeypatch.setattr(a4, fn, lambda ctx_, out, build, tid, *a, _f=fn: calls.append((_f, build, tid)))
    a4.run_pilot(ctx, "out", "ref", 6, print, "att")
    assert prepared == list(a4.BUILD_ORDER) and len(calls) == 36
    assert [c[1] for c in calls] == [b for b in a4.BUILD_ORDER for _ in range(6)]                    # all 6 trajectories of a build before the next build
    assert [c[2] for c in calls[:6]] == list(trajs) and {c[0] for c in calls if c[1] in a4.CT_BUILDS} == {"run_build_ct"}
    assert {c[0] for c in calls if c[1] not in a4.CT_BUILDS} == {"run_build"}


def test_launch_script_does_not_copy_the_reference_outside_the_atomic_helper():
    sh = open(os.path.join(HERE, "a1_launch_a4.sh")).read()
    assert "cp -n" not in sh and "cp " not in sh.replace("# the copy", "")


def test_manifest_with_the_build_order_claims_again_after_a_json_reload(tmp_path):
    fp = {"stage": "a4", "build_order": list(a4.BUILD_ORDER), "arms": a4.ALL_ARMS, "ct": {"builds": a4.CT_BUILDS}}
    path = str(tmp_path / "manifest.json")
    assert a4.claim(path, fp) == "new" and a4.claim(path, fp) == "match"
    assert a4.claim(path, {**fp, "build_order": a4.BUILD_ORDER}) == "match"                                 # a tuple is normalised, not refused
    with pytest.raises(SystemExit):
        a4.claim(path, {**fp, "build_order": ["S"]})


def test_prepare_build_refuses_a_done_trajectory_with_damaged_files(tmp_path):
    import json
    out = str(tmp_path)
    files, dpath = a4._paths(out, "S0")
    row = {"traj": "t0", "attempt": "a"}
    open(files["A4_K0"], "w").write(json.dumps(row) + "\n")
    open(files["cuts"], "w").write(json.dumps({"arm": "A4_K0", "traj": "t0", "cut_index": 0}) + "\n")
    open(dpath, "w").write(json.dumps({"traj": "t0", "attempt": "a", "n_rows": 1}) + "\n")
    a4.prepare_build(out, "S0", lambda *x: None)                                                            # intact: kept
    open(files["A4_K0"], "w").write("")                                                                      # rows lost under a done marker
    with pytest.raises(SystemExit) as e:
        a4.prepare_build(out, "S0", lambda *x: None)
    assert "t0" in str(e.value) and "done marker" in str(e.value)


def test_full_report_requires_the_ct_build_and_a_partial_report_marks_optional_builds_incomplete(tmp_path, monkeypatch):
    tasks = [{"id": "p0", "traj": "t0", "cut_index": 0}]
    with pytest.raises(SystemExit) as e:
        rep.compact_on_evict(str(tmp_path), None, tasks, [], {"t0"}, {"p0"}, {"A4_K0": {}}, {})
    assert "A4_CT" in str(e.value)
    import json
    (tmp_path / "A4_CA.jsonl").write_text(json.dumps({"id": "p0", "traj": "t0", "cut_index": 0}) + "\n")                   # complete rows, no done marker yet
    out = rep.compact_on_evict(str(tmp_path), None, tasks, [], {"t0"}, {"p0"}, {"A4_K0": {}}, {}, partial=True)
    assert "incomplete" in out["A4_CA"]
    (tmp_path / "A4_CT.jsonl").write_text("")
    out = rep.compact_on_evict(str(tmp_path), None, tasks, [], {"t0"}, {"p0"}, {"A4_K0": {}}, {}, partial=True)
    assert "incomplete" in out["A4_CT"] and "vs_A4_K0" not in out["A4_CT"]                                   # `partial()` then exits: CT is mandatory


def test_deps_closure_includes_models_cache_by_walking_sys_modules():
    src = a4.inspect.getsource(a4.compaction_deps)
    assert "sys.modules" in src and "models.cache" in src or 'os.path.join("models", "cache.py")' in src
