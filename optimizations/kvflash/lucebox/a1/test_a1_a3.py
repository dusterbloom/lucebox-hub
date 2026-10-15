"""Stub tests for A3 routed exact fetch (a1_a3.py; PREREG "A3 routed exact fetch"): row budget, router candidates, the k = 0 and
fetch-everything states, the discarded routing pass, and the pure selection/verdict helpers. No compaction repo, no weights.
The same five checks on the tiny random hybrid are in modal/am_qwen35/tests/test_a1.py (run on lucebox6)."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
import a1_a3 as a3  # noqa: E402
import a1_e  # noqa: E402
import test_a2_window as w  # noqa: E402

r2 = w.r2
BLOCK, KEEP = w.BLOCK, w.KEEP            # stub: 8-token blocks, 2 slots per compacted block
FULL_LAYERS = [1, 2]
NB, TAIL, P, WB = 12, 3, 2, 3             # 12 sealed blocks + 3 tail tokens; 2 exact prefix blocks, 3 exact window blocks
CAND = list(range(P, NB - WB))            # blocks 2..8


def q_extend(log=None):
    """fake_extend that also returns queries of shape (full layers, heads, n, d): the token ids themselves, so a routing pass is traceable."""
    inner = w.fake_extend([])

    def extend(model, layer_types, triples, orig_len, ids, want_queries=False):
        new, ln, _ = inner(model, layer_types, triples, orig_len, ids, want_queries=False)
        if log is not None:
            log.append({"want_queries": want_queries, "n": ids.shape[1], "ret": new})
        q = None
        if want_queries:
            n = ids.shape[1]
            q = (ids.float().view(1, 1, n, 1) / 10.0).expand(len(FULL_LAYERS), 2, n, 2).clone()
        return new, ln, q
    return extend


@pytest.fixture
def built(monkeypatch):
    log = []
    w.LIVE.clear()
    monkeypatch.setattr(r2, "extend", q_extend(log))
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda s, t, block_=BLOCK: real(s, t, block_))
    monkeypatch.setattr(r2, "slots_for", lambda n, ratio=0.1: KEEP)
    R = w.StubRunner()
    exact, comp = [], []
    (_, tr, ln, info), = list(r2.gen_fblk_window(R, w.segs_of(NB * BLOCK + TAIL), [0], WB, exact_blocks=P,
                                                 on_block=lambda j, a, b, ex, cp: (exact.append(ex), comp.append(cp))))
    st = SimpleNamespace(triples=tr, t_len=ln, nb=info["blocks"], tail=info["tail_tokens"], comp=comp, exact=exact, p=P, wb=WB,
                         linear=R.S.linear, full_layers=FULL_LAYERS)
    log.clear()
    return R, st, log


def rows_of(triples):
    return triples[FULL_LAYERS[0]][0].shape[2]


# ---------------------------------------------------------------- row budget (pure)

def test_row_formula_matches_the_state_the_generator_builds(built):
    _, st, _ = built
    assert st.nb == NB and st.tail == TAIL
    assert a3.rows_at(NB, TAIL, P, WB, 0, block=BLOCK, slots=KEEP) == rows_of(st.triples)
    for k in (0, 1, 4, len(CAND)):
        trip = a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, CAND[:k], st.linear)
        assert rows_of(trip) == a3.rows_at(NB, TAIL, P, WB, k, block=BLOCK, slots=KEEP), k


def test_choose_k_fills_the_budget_with_the_most_blocks_that_fit():
    # head 10 blocks (6400 rows); W = 6 blocks: 57 blocks and a 639-token tail
    full = a3.choose_k(57, 639, 10, 6)
    assert full["k"] == 5 and not full["over_budget"] and full["rows"] <= a3.CAP
    assert a3.rows_at(57, 639, 10, 6, 6) > a3.CAP                   # one more block would not fit
    assert a3.CAP - full["rows"] < a3.BLOCK - a3.SLOTS               # less than one block of room is left: the budget is filled
    mid = a3.choose_k(30, 100, 10, 6)                               # 14 candidates, room for 8
    assert mid["k"] == 8 and mid["n_cand"] == 14 and mid["rows"] == a3.rows_at(30, 100, 10, 6, 8) <= a3.CAP
    assert a3.choose_k(57, 639, 10, 2)["k"] == 9                    # W = 2 blocks leaves more room: k = 9


def test_choose_k_fetches_everything_at_short_contexts():
    short = a3.choose_k(17, 5, 10, 2)                               # 5 candidates, 12 exact blocks
    assert short["k"] == short["n_cand"] == 5 and short["rows"] == 17 * a3.BLOCK + 5      # all exact, like POOL16K
    assert not short["lowered"]


def test_choose_k_caps_at_the_number_of_candidates():
    few = a3.choose_k(12, 0, 10, 1)                                 # 1 candidate block
    assert few["n_cand"] == 1 and few["k"] == 1


def test_choose_k_flags_a_cut_that_is_over_budget_even_at_k_zero():
    over = a3.choose_k(57, 639, 10, 12)                             # 14,080 exact + 35 x 64 + 639
    assert over["k"] == 0 and over["over_budget"] and over["rows"] == over["rows_k0"] > a3.CAP


def test_the_grid_windows_are_never_over_budget_at_the_longest_probe_cut():
    for wb in a3.SEL_WBS:                                           # 36,974 tokens = 57 sealed blocks and the longest tail
        for reserve in (0, 164, 2000):                              # 2000: a 1,936-token question and 64 new tokens, far above any suffix
            got = a3.choose_k(57, 639, 10, wb, reserve)
            assert not got["over_budget"] and got["peak_rows"] <= a3.CAP


def test_overlapping_prefix_and_window_are_counted_once():
    assert a3.exact_set(5, 3, 4) == {0, 1, 2, 3, 4}
    assert a3.rows_at(5, 7, 3, 4, 0) == 5 * a3.BLOCK + 7
    assert a3.candidates(5, 3, 4) == []
    assert a3.choose_k(5, 7, 3, 4)["k"] == 0


# ---------------------------------------------------------------- router never ranks P, W or the tail

def test_candidates_exclude_the_prefix_the_window_and_the_tail():
    assert a3.candidates(NB, P, WB) == CAND
    assert max(a3.candidates(NB, P, WB)) < NB - WB and min(a3.candidates(NB, P, WB)) >= P


def test_router_ranks_only_candidate_blocks_even_when_prefix_and_window_slots_would_win(built):
    R, st, log = built
    for j in list(range(P)) + list(range(NB - WB, NB)):            # make every P and W block overwhelmingly attractive
        st.comp[j] = tuple(None if e is None else (torch.full_like(e[0], 1000.0), e[1], e[2]) for e in st.comp[j])
    suffix = [41, 42, 43]
    cand, a_last, tm = a3.route_a_last(R.S, st, suffix)
    assert cand == CAND and a_last.shape == (len(CAND),)
    assert torch.allclose(a_last.sum(), torch.tensor(1.0), atol=1e-4)           # normalised over the candidate blocks only
    ranked = a3.rank_candidates(a_last, cand)
    assert sorted(ranked) == CAND and set(ranked).isdisjoint(set(range(P)) | set(range(NB - WB, NB)))


def test_rank_candidates_orders_by_score_and_breaks_ties_by_block_index():
    assert a3.rank_candidates(torch.tensor([0.1, 0.5, 0.5, 0.2]), [7, 8, 9, 10]) == [8, 9, 10, 7]


# ---------------------------------------------------------------- k = 0 and fetch-everything states

def test_k_zero_is_the_prefix_plus_window_shipping_state(built):
    _, st, _ = built
    trip = a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, [], st.linear)
    assert all(torch.equal(x, y) for a, b in zip(trip, st.triples) for x, y in zip(a, b))


def test_fetching_every_compacted_block_is_the_e3_swap_of_those_blocks_and_all_exact(built):
    _, st, _ = built
    trip = a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, CAND, st.linear)
    e3 = a1_e.oracle_triples(st.triples, st.comp, st.exact, set(range(st.nb)), st.tail, st.linear)
    assert all(torch.equal(x, y) for a, b in zip(trip, e3) for x, y in zip(a, b))
    for i in FULL_LAYERS:                                           # explicit: every block exact in order, then the shipping tail
        for c in range(3):
            want = torch.cat([st.exact[j][i][c] for j in range(NB)] + [st.triples[i][c][:, :, -TAIL:]], 2)
            assert torch.equal(trip[i][c], want), (i, c)
    assert rows_of(trip) == NB * BLOCK + TAIL                       # no compacted row is left next to its exact twin


def test_a_fetched_block_replaces_its_compacted_entry_instead_of_joining_it(built):
    _, st, _ = built
    trip = a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, [4], st.linear)
    assert rows_of(trip) == rows_of(st.triples) + BLOCK - KEEP


# ---------------------------------------------------------------- the routing pass is discarded

def test_routing_pass_kv_is_discarded_before_the_answer_prefill(built, monkeypatch):
    R, st, log = built
    snap = tuple(tuple(x.clone() for x in e) for e in st.triples)
    answered = []

    def spy_answer(S, triples, seq_len, suffix, max_new, timing=None):
        answered.append({"rows": rows_of(triples), "seq_len": seq_len, "ids": {id(x) for e in triples for x in e}})
        if timing is not None:
            timing.update(prefill_s=0.0, decode_s=0.0)
        return [1]

    monkeypatch.setattr(r2, "answer_ids", spy_answer)
    item = SimpleNamespace(suffix=[41, 42, 43], max_new=4)
    cap = a3.rows_at(NB, TAIL, P, WB, 2, block=BLOCK, slots=KEEP) + 3 + 4          # 2 fetched blocks, the 3-token question, 4 new tokens
    out = a3.answer_item(R.S, st, item, cap=cap, block=BLOCK, slots=KEEP)
    routed = [c for c in log if c["want_queries"]]
    assert len(routed) == 1 and routed[0]["n"] == 3                                  # exactly one routing pass, over the question suffix
    routing_ids = {id(x) for e in routed[0]["ret"] for x in e}
    assert len(answered) == 1
    assert answered[0]["ids"].isdisjoint(routing_ids)                                # the answer never sees the routing pass's KV
    assert answered[0]["seq_len"] == st.t_len                                        # no question tokens are carried over
    assert answered[0]["rows"] == a3.rows_at(NB, TAIL, P, WB, 2, block=BLOCK, slots=KEEP) == out["budget"]["rows"] and out["budget"]["k"] == 2
    assert all(torch.equal(x, y) for a, b in zip(snap, st.triples) for x, y in zip(a, b))   # the cut state is untouched by routing


def test_the_fetched_blocks_are_the_top_k_of_the_router_order_and_the_budget_is_filled(built, monkeypatch):
    R, st, _ = built
    monkeypatch.setattr(r2, "answer_ids", lambda *a, **k: [1])
    cap = a3.rows_at(NB, TAIL, P, WB, 3, block=BLOCK, slots=KEEP) + 2 + 4 + BLOCK - KEEP - 1   # 3 blocks, question 2 + 4 new, a 4th one row short
    out = a3.answer_item(R.S, st, SimpleNamespace(suffix=[41, 42], max_new=4), cap=cap, block=BLOCK, slots=KEEP)
    assert out["budget"]["k"] == 3 and out["fetched"] == out["ranked"][:3] and len(out["ranked"]) == len(CAND)
    full = a3.answer_item(R.S, st, SimpleNamespace(suffix=[41, 42], max_new=4), cap=10 ** 9, block=BLOCK, slots=KEEP)
    assert full["budget"]["k"] == len(CAND) and not full["budget"]["lowered"]


def test_no_candidates_means_no_routing_pass_and_the_shipping_prefix_window_state(built, monkeypatch):
    R, st, log = built
    st.p, st.wb = NB, 0                                                              # everything is the exact prefix
    monkeypatch.setattr(r2, "answer_ids", lambda *a, **k: [1])
    out = a3.answer_item(R.S, st, SimpleNamespace(suffix=[41], max_new=2), cap=10 ** 9, block=BLOCK, slots=KEEP)
    assert not [c for c in log if c["want_queries"]] and out["budget"]["k"] == 0 and out["cand"] == []


# ---------------------------------------------------------------- source status, selection, verdict (pure)

def test_source_status_and_rank_of_the_located_block():
    assert a3.source_status(False, [], 12, P, WB) == "not_located"
    assert a3.source_status(True, [], 12, P, WB) == "tail"
    assert a3.source_status(True, [0], 12, P, WB) == "exact_resident"               # a prefix block
    assert a3.source_status(True, [10, 11], 12, P, WB) == "exact_resident"          # window blocks
    assert a3.source_status(True, [8, 9], 12, P, WB) == "candidate"
    ranked = [5, 3, 4, 2]
    assert a3.rank_of(ranked, [4, 9]) == 3 and a3.rank_of(ranked, [9]) is None


def test_select_config_prefers_retention_then_fewer_mean_rows_then_the_smaller_w():
    t = {"W2": {"correct": 50, "n": 118, "mean_rows": 15000},
         "W6": {"correct": 55, "n": 118, "mean_rows": 14000}}
    assert a3.select_config(t)["selected"] == "W6"
    t["W6"]["correct"] = 50
    assert a3.select_config(t)["selected"] == "W6"                                  # tie on retention: fewer mean rows
    t["W6"]["mean_rows"] = 15000
    assert a3.select_config(t)["selected"] == "W2"                                  # tie on both: the smaller W
    assert a3.select_config({})["selected"] is None


def test_final_verdict_follows_a1_section_6():
    v = a3.final_verdict
    assert v({"mean": 0.80, "lo": 0.77, "hi": 0.83}) == "PASS"
    assert v({"mean": 0.70, "lo": 0.66, "hi": 0.74}) == "FAIL"
    assert v({"mean": 0.78, "lo": 0.74, "hi": 0.82}) == "INCONCLUSIVE"              # contains the bar, point above
    assert v({"mean": 0.74, "lo": 0.70, "hi": 0.78}) == "FAIL"                      # contains the bar, point under: FAIL (section 6 item 3)


def test_arm_names_are_stable_file_stems():
    assert a3.arm_name(6) == "A3_W6" and a3.route_name(6) == "A3_W6_route" and a3.SEL_WBS == (2, 6)


# ---------------------------------------------------------------- reference seeding and the report tables

def test_seed_reference_copies_only_the_wanted_trajectories_unedited_and_never_overwrites(tmp_path):
    ref, out = tmp_path / "FULL.jsonl", tmp_path / "out.jsonl"
    lines = ['{"id": "a:1", "traj": "a", "x": 1.0}', '{"id": "b:1", "traj":"b"}', '{"id": "c:1", "traj": "c"}']
    ref.write_text("\n".join(lines) + "\n")
    assert a3.seed_reference(str(ref), str(out), ["a", "c"]) == 2
    assert out.read_text() == lines[0] + "\n" + lines[2] + "\n"             # raw lines, spacing preserved
    out.write_text("mine\n")
    assert a3.seed_reference(str(ref), str(out), ["a"]) == 0 and out.read_text() == "mine\n"


def _row(traj, cut, k, rows, reserve=164, n_cand=7, dist=100):
    return {"traj": traj, "cut_index": cut, "k": k, "rows": rows, "reserve": reserve, "peak_rows": rows + reserve, "n_cand": n_cand,
            "distance_tok": dist, "route_s": 1.0, "fetch_bytes": 10, "probe_wall_s": 2.0, "decode_tokens": 5}


def test_cuts_table_counts_each_cut_once_and_keeps_k_rows_reserve_and_peak_rows():
    import a1_a3_report as rep
    rows = [_row("t", 1, 7, 9000), _row("t", 1, 7, 9000), _row("t", 2, 2, 16000, reserve=300), _row("t", 3, 0, 12000, n_cand=0)]
    c = rep.cuts_table(rows)
    assert c["cuts"] == 3 and c["cuts_k_below_candidates"] == 1 and c["cuts_without_candidates"] == 1
    assert not [k for k in c if "over_cap" in k]                                   # a cut over budget aborts the run: no such count
    assert c["k_used_histogram"] == {"7": 1, "2": 1, "0": 1} and c["max_rows"] == 16000
    assert c["max_reserve"] == 300 and c["max_peak_rows"] == 16300 and c["mean_peak_rows"] == (9164 + 16300 + 12164) / 3


def test_retention_by_k_used_groups_probes_by_the_k_of_their_cut():
    import a1_a3_report as rep
    rows = [{"id": "a", "k": 9}, {"id": "b", "k": 9}, {"id": "c", "k": 3}, {"id": "d", "k": 0}]
    got = rep.by_k_used(rows, ["a", "b", "c", "d"], [1, 0, 1, 0])
    assert got == {"0": {"n": 1, "correct": 0}, "3": {"n": 1, "correct": 1}, "9": {"n": 2, "correct": 1}}


def test_fetch_coverage_reads_none_partial_or_all_of_the_candidate_source_blocks():
    assert a3.fetch_coverage([4], NB, P, WB, [4, 5]) == "all" and a3.fetch_coverage([4, 5], NB, P, WB, [4]) == "partial"
    assert a3.fetch_coverage([4, 5], NB, P, WB, [7]) == "none" and a3.fetch_coverage([0, 11], NB, P, WB, [4]) is None
    assert a3.fetch_coverage([1, 4], NB, P, WB, [4]) == "all"          # block 1 is exact already: only candidates count


def test_fetch_coverage_vs_retention_counts_each_coverage_class_without_causal_wording():
    import a1_a3_report as rep

    def rr(i, status, cov):
        return {"id": i, "src_status": status, "source_fetch_coverage": cov}

    route = [rr("a", "candidate", "all"), rr("b", "candidate", "all"), rr("c", "candidate", "none"), rr("e", "candidate", "partial"),
             rr("d", "exact_resident", None)]
    got = rep.fetch_coverage_vs_retention(route, {"a": 1, "b": 0, "c": 0, "d": 1, "e": 1})
    assert got == {"candidate_probes": 4, "all": {"n": 2, "correct": 1}, "partial": {"n": 1, "correct": 1}, "none": {"n": 1, "correct": 0}}


def test_router_tables_use_only_candidate_probes_with_a_rank():
    import a1_a3_report as rep

    def rr(status, rank, dist=100, n_cand=7):
        return {"src_status": status, "rank_a_last": rank, "distance_tok": dist, "n_cand": n_cand}

    t = rep.router_tables([rr("candidate", 1), rr("candidate", 9, dist=20000), rr("exact_resident", None), rr("tail", None),
                           rr("not_located", None)])
    assert t["source_status"] == {"candidate": 2, "exact_resident": 1, "tail": 1, "not_located": 1}
    assert t["a_last"]["candidate_probes"]["n"] == 2 and t["a_last"]["candidate_probes"]["recall"]["8"] == 0.5
    assert t["a_last"]["candidate_probes_over_16K"]["n"] == 1


# ---------------------------------------------------------------- review fixes: budget reserve, guard, selection, resume

def test_choose_k_reserves_the_question_and_the_decode_tokens():
    assert a3.choose_k(57, 64, 10, 6)["k"] == 6 and a3.choose_k(57, 64, 10, 6)["rows"] == a3.CAP     # the cut alone fills 16,384 exactly
    got = a3.choose_k(57, 64, 10, 6, reserve=1)                      # any question at all: one block fewer
    assert got["k"] == 5 and got["rows"] + got["reserve"] <= a3.CAP and not got["over_budget"]
    big = a3.choose_k(57, 639, 10, 6, reserve=150 + 64)             # 150-token question + 64 new tokens
    assert big["k"] == 4 and big["peak_rows"] == big["rows"] + 214 <= a3.CAP
    assert a3.choose_k(57, 639, 10, 6, reserve=214 + 576)["k"] == 3  # one more block of reserve takes one more block of k
    over = a3.choose_k(57, 639, 10, 12, reserve=100)
    assert over["k"] == 0 and over["over_budget"]
    edge = a3.choose_k(57, 639, 10, 6, reserve=a3.CAP - 13503 + 1)   # k = 0 fits by the cut alone but not with the reserve
    assert edge["k"] == 0 and edge["over_budget"]


def test_the_reserve_defaults_to_the_item_question_plus_max_new(built, monkeypatch):
    R, st, _ = built
    monkeypatch.setattr(r2, "answer_ids", lambda *a, **k: [1])
    base = a3.rows_at(NB, TAIL, P, WB, 0, block=BLOCK, slots=KEEP)
    item = SimpleNamespace(suffix=[1] * 5, max_new=7)
    out = a3.answer_item(R.S, st, item, cap=base + 12 + BLOCK - KEEP, block=BLOCK, slots=KEEP)          # room for 1 block after 5 + 7 rows
    assert out["budget"]["reserve"] == 12 and out["budget"]["k"] == 1
    st.reserve = 100                                                  # a cut-level reserve (longest question) wins: no room even at k = 0
    with pytest.raises(RuntimeError, match="over budget"):
        a3.answer_item(R.S, st, item, cap=base + 12 + BLOCK - KEEP, block=BLOCK, slots=KEEP)
    st.reserve = 12 + BLOCK - KEEP
    assert a3.answer_item(R.S, st, item, cap=base + 12 + BLOCK - KEEP, block=BLOCK, slots=KEEP)["budget"]["k"] == 0


def test_the_routing_pass_is_checked_against_the_cap(built):
    R, st, _ = built
    base = rows_of(st.triples)
    a3.route_a_last(R.S, st, [1, 2, 3], cap=base + 3)
    with pytest.raises(AssertionError, match="routing pass"):
        a3.route_a_last(R.S, st, [1, 2, 3], cap=base + 2)


def test_row_guard_asserts_the_cap_at_the_question_prefill_and_at_every_decode_step():
    calls = []

    class Model:
        def __call__(self, input_ids=None, **k):
            calls.append(input_ids.shape[-1])
            return "out"

        config = "cfg"

    S = SimpleNamespace(model=Model())
    real = S.model
    with a3.row_guard(S, rows=100, cap=100 + 5 + 3):                 # a 5-token question then 3 decode steps fit exactly
        assert S.model.config == "cfg"
        S.model(input_ids=torch.zeros(1, 5, dtype=torch.long))
        for _ in range(3):
            S.model(input_ids=torch.zeros(1, 1, dtype=torch.long))
        with pytest.raises(AssertionError, match="cap"):
            S.model(input_ids=torch.zeros(1, 1, dtype=torch.long))   # the 4th decode step is one row over
    assert S.model is real and calls == [5, 1, 1, 1]
    with a3.row_guard(S, rows=100, cap=104):
        with pytest.raises(AssertionError):
            S.model(input_ids=torch.zeros(1, 5, dtype=torch.long))   # the question prefill alone is over


def test_a_real_cap_trips_the_guard_in_the_answer_and_an_exact_fit_passes(built, monkeypatch):
    R, st, _ = built
    item = SimpleNamespace(suffix=[1, 2], max_new=4)

    class Fake:
        def __call__(self, input_ids=None, **k):
            return None

    def feed(S, triples, seq_len, suffix, max_new, timing=None):          # the question prefill, then max_new - 1 decode steps
        S.model(input_ids=torch.tensor([suffix]))
        for _ in range(max_new - 1):
            S.model(input_ids=torch.tensor([[1]]))
        return [1]

    R.S.model = Fake()
    monkeypatch.setattr(r2, "answer_ids", feed)
    base = a3.rows_at(NB, TAIL, P, WB, 2, block=BLOCK, slots=KEEP)
    out = a3.answer_item(R.S, st, item, cap=base + 2 + 4, block=BLOCK, slots=KEEP)       # k = 2 and the answer fits with a row to spare
    assert out["budget"]["k"] == 2 and type(R.S.model).__name__ == "Fake"
    st.reserve = 1                                                         # a reserve that forgets the answer: k = 2, but 5 rows are fed
    with pytest.raises(AssertionError, match="cap"):
        a3.answer_item(R.S, st, item, cap=base + 1, block=BLOCK, slots=KEEP)
    assert type(R.S.model).__name__ == "Fake"                              # the guard is removed even when it trips


def test_a_cut_over_budget_at_k_zero_aborts_instead_of_running_uncapped(built, monkeypatch):
    R, st, _ = built
    monkeypatch.setattr(r2, "answer_ids", lambda *a, **k: pytest.fail("must not answer"))
    base = a3.rows_at(NB, TAIL, P, WB, 0, block=BLOCK, slots=KEEP)
    with pytest.raises(RuntimeError, match="over budget"):
        a3.answer_item(R.S, st, SimpleNamespace(suffix=[1, 2], max_new=4), cap=base + 5, block=BLOCK, slots=KEEP)    # reserve 6 > 5 spare
    with pytest.raises(RuntimeError, match="over budget"):
        a3.answer_item(R.S, st, SimpleNamespace(suffix=[1, 2], max_new=4), cap=base - 1, block=BLOCK, slots=KEEP)


def test_full_fallback_binds_the_per_block_cleanup_into_the_generator_through_run_unit(monkeypatch):
    import a1_core
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(a1_core, "mem_gib", lambda: (0.0, 0.0))
    monkeypatch.setattr(r2, "extend", lambda model, lt, tr, ln, ids, **k: ("T", ln + ids.shape[1], None))
    monkeypatch.setattr(r2, "_info", lambda R, tr, ln, trace: {})
    monkeypatch.setattr(r2, "answer_ids", lambda S, tr, ln, sfx, mx, timing=None: timing.update(prefill_s=0.0, decode_s=0.0) or [1])
    S = SimpleNamespace(dev="cpu", eos={9}, layer_types=[], model=None, linear=set(), tok=SimpleNamespace(decode=lambda ids, **k: "x"))
    R = SimpleNamespace(S=S)
    item = a1_core.Item({"id": "T:a", "traj": "T", "kind": "probe"}, [1, 2], 3, 0, "sha")
    n = 2 * r2.BLOCK + 5                                              # two sealed blocks and a tail at the single cut
    plan = a1_core.Plan("T", [], list(range(n)), [a1_core.Cut(0, n, [item])])
    ctx = SimpleNamespace(S=S, R=R, parse=lambda text, tools: {"cleaned_text": text, "tool_calls": []})
    rows = []
    monkeypatch.setattr(a1_core, "append_row", lambda path, row: rows.append(row))
    a3.compute_full(ctx, plan, "unused", lambda m: None)
    assert len(rows) == 1 and len(calls) == 2 + 1                     # one empty_cache per sealed block (generator) + one per cut (run_unit)
    calls.clear()
    a1_core.run_unit(S, R, plan, r2.gen_fullchk, ctx.parse, rows.append, "FULL", log=lambda m: None, empty_cache=True)
    assert len(calls) == 1                                            # what the old call did: the per-block cleanup never ran


def test_selection_refuses_a_cap_other_than_the_registered_one():
    with pytest.raises(SystemExit):
        a3.main(["--stage", "sel", "--out", "unused", "--ref-dir", "unused", "--cap", "20000"])


def _sel_report(**kw):
    arm = lambda c, m: {"n": a3.PILOT_PROBES, "correct": c, "mean_rows": m}            # noqa: E731
    return {"stage": "sel", "complete": True, "trajectories": a3.PILOT_N, "reference_md5": a3.PILOT_REF_MD5,
            "arms": {"W2": arm(60, 15000.0), "W6": arm(55, 14000.0)}, "selection": {"selected": "W2"}, **kw}


def test_stage_2_needs_the_selection_report_and_takes_w_from_it(tmp_path):
    with pytest.raises(SystemExit):
        a3.main(["--stage", "proper", "--out", "unused", "--ref-dir", "unused"])
    ok = _sel_report()
    p = tmp_path / "a3_report.json"
    p.write_text(json.dumps(ok))
    assert a3.load_selection(str(p)) == (2, a3.canonical_sha256(str(p)))
    p.write_text(json.dumps(ok, indent=4, sort_keys=True))                             # reformatting does not change the digest
    assert a3.canonical_sha256(str(p)) == a3.load_selection(str(p))[1]
    arm = lambda n: {**ok["arms"]["W6"], "n": n}                                      # noqa: E731
    bad = [{**ok, "selection": {"selected": "W6"}},                                    # not what the rule gives
           {**ok, "selection": {"selected": None}}, {**ok, "complete": False}, {**ok, "arms": {"W6": ok["arms"]["W6"]}},
           {**ok, "stage": "proper"}, {**ok, "trajectories": 5}, {**ok, "reference_md5": "0" * 32},
           {**ok, "arms": {"W2": ok["arms"]["W2"], "W6": arm(117)}},                   # not the fixed 118
           {**ok, "arms": {"W2": {}, "W6": {}}},                                       # empty arm summaries
           {**ok, "arms": {"W2": {"n": 118}, "W6": {"n": 118}}}]                       # counts but no results
    for b in bad:
        p.write_text(json.dumps(b))
        with pytest.raises(SystemExit):
            a3.load_selection(str(p))


def test_main_reaches_run_proper_with_the_window_of_a_valid_report_and_writes_the_manifest(tmp_path, monkeypatch):
    rep = tmp_path / "a3_report.json"
    rep.write_text(json.dumps(_sel_report(selection={"selected": "W2"})))
    ref, out = tmp_path / "ref", tmp_path / "out"
    ref.mkdir()
    (ref / "FULL.jsonl").write_text("")
    data = {k: str(tmp_path / k) for k in ("traj", "tasks", "template", "tokenizer")}
    for f in data.values():
        open(f, "w").write("x")
    monkeypatch.setattr(a3.core, "default_paths", lambda: data)
    monkeypatch.setattr(a3.core, "Harness", lambda paths: "H")
    monkeypatch.setattr(a3.core, "load_model", lambda *a, **k: SimpleNamespace(S="S", R="R"))
    monkeypatch.setitem(sys.modules, "tool_parse", SimpleNamespace(parse=None))
    got = {}
    monkeypatch.setattr(a3, "run_proper", lambda ctx, o, rd, wb, early, log, attempt, cap: got.update(wb=wb, attempt=attempt) or {"ok": 1})
    argv = ["--stage", "proper", "--out", str(out), "--ref-dir", str(ref), "--sel-report", str(rep)]
    assert a3.main(argv) == 0
    man = json.load(open(out / "manifest.json"))
    assert got["wb"] == 2 and man["window_blocks"] == 2 and man["sel_report_sha256"] == a3.canonical_sha256(str(rep))
    assert man["ref_dir"] == str(ref) and set(man["ref_full_md5"]) and set(man["data_md5"]) == set(data)
    assert "r1b_core.py" in man["code_md5"] and "r2_core.py" in man["code_md5"]
    assert a3.main(argv + ["--attempt", "again"]) == 0                                # same manifest: resumes
    rep.write_text(json.dumps(_sel_report(selection={"selected": "W2"}, extra=1)))   # changed content: refuses
    with pytest.raises(SystemExit, match="different run"):
        a3.main(argv)
    runs = [f for f in os.listdir(out) if f.startswith("run.")]
    assert len(runs) == 3 and "run.again.log" in runs                                  # one log per attempt, none shared


def test_selection_refuses_a_pilot_count_other_than_the_registered_one():
    with pytest.raises(SystemExit):
        a3.main(["--stage", "sel", "--out", "unused", "--ref-dir", "unused", "--n", "5"])


def test_attempt_ids_are_unique_even_within_one_second():
    assert len({a3.new_attempt_id() for _ in range(200)}) == 200


def test_a_torn_final_line_is_dropped_and_logged_a_bad_line_elsewhere_still_raises(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"traj": "t1", "id": 1}\n{"traj": "t1", "id": 2}\n{"traj": "t2", "id"')
    logged = []
    assert a3.repair_tail(str(p), logged.append) and "torn" in logged[0]
    assert [r["id"] for r in a3.core.read_rows(str(p))] == [1, 2] and p.read_text().endswith("\n")
    assert not a3.repair_tail(str(p), logged.append) and len(logged) == 1               # a clean file is left alone
    p.write_text('{"traj": "t1", "id": 1}')                                            # whole last line, no newline: newline added
    assert not a3.repair_tail(str(p)) and p.read_text() == '{"traj": "t1", "id": 1}\n'
    p.write_text('{"traj": "t1"\n{"traj": "t1", "id": 2}\n')                           # bad line in the middle
    assert not a3.repair_tail(str(p))
    with pytest.raises(ValueError):
        a3.core.read_rows(str(p))
    assert not a3.repair_tail(str(tmp_path / "missing.jsonl"))


def test_prepare_window_repairs_then_purges_so_the_torn_trajectory_is_redone(tmp_path):
    out = str(tmp_path)
    ans, route, done = a3._paths(out, 2)
    open(done, "w").write(json.dumps({"traj": "t1"}) + "\n" + '{"traj": "t2"')              # t2's marker is torn: t2 is not done
    for p in (ans, route):
        open(p, "w").write(json.dumps({"traj": "t1", "id": "a"}) + "\n" + json.dumps({"traj": "t2", "id": "b"}) + "\n" + '{"traj": "t2", "i')
    a3.prepare_window(out, 2, lambda m: None)
    for p in (ans, route):
        assert [r["id"] for r in a3.core.read_rows(p)] == ["a"]
    assert [r["traj"] for r in a3.core.read_rows(done)] == ["t1"]


def test_manifest_is_written_once_and_a_changed_run_refuses_to_resume(tmp_path):
    path = str(tmp_path / "manifest.json")
    fp = {"stage": "proper", "window_blocks": 6, "sel_report_sha256": "abc", "code_md5": {"a1_a3.py": "1"}}
    assert a3.claim_manifest(path, fp) == "new" and a3.claim_manifest(path, fp) == "match"
    for change in ({"sel_report_sha256": "zzz"}, {"window_blocks": 2}, {"code_md5": {"a1_a3.py": "2"}}):
        with pytest.raises(SystemExit, match="different run"):
            a3.claim_manifest(path, {**fp, **change})
    assert json.load(open(path)) == fp


def test_a_trajectory_without_a_done_marker_is_purged_so_it_is_redone(tmp_path):
    a, r = str(tmp_path / "a.jsonl"), str(tmp_path / "r.jsonl")
    for p in (a, r):
        with open(p, "w") as f:
            for t, i in (("t1", 1), ("t1", 2), ("t2", 3)):
                f.write(json.dumps({"traj": t, "id": f"{t}:{i}"}) + "\n")
    a3.purge_partial([a, r], {"t1"})
    for p in (a, r):
        assert [x["id"] for x in a3.core.read_rows(p)] == ["t1:1", "t1:2"]
    a3.purge_partial([a, r], {"t1"})                                  # idempotent, nothing else touched
    assert not os.path.exists(a + ".part")


def test_seed_reference_is_atomic_an_interrupted_copy_leaves_no_destination(tmp_path):
    ref, out = tmp_path / "FULL.jsonl", tmp_path / "out.jsonl"
    ref.write_text('{"id": "a:1", "traj": "a"}\n{"id": "a:2", "traj": "a"\n')           # the second record is cut off
    with pytest.raises(ValueError):
        a3.seed_reference(str(ref), str(out), ["a"])
    assert not out.exists()                                           # a later run copies again, it never trusts a prefix
    ref.write_text('{"id": "a:1", "traj": "a"}\n{"id": "a:2", "traj": "a"}\n')
    assert a3.seed_reference(str(ref), str(out), ["a"]) == 2 and len(out.read_text().splitlines()) == 2


def _arec(i, attempt="x", **kw):
    return {"id": i, "traj": "t", "attempt": attempt, **kw}


def test_check_complete_refuses_missing_extra_duplicate_and_mixed_attempt_records():
    import a1_a3_report as rep
    exp = {"a", "b"}
    rep.check_complete("W6", [_arec("a"), _arec("b")], [_arec("a"), _arec("b")], exp)
    bad = [([_arec("a")], [_arec("a"), _arec("b")], "missing"), ([_arec("a"), _arec("b"), _arec("c")], [_arec("a"), _arec("b")], "unexpected"),
           ([_arec("a"), _arec("b"), _arec("b")], [_arec("a"), _arec("b")], "duplicate"),
           ([_arec("a"), _arec("b")], [_arec("a"), _arec("b", attempt="y")], "attempt")]
    for ans, route, word in bad:
        with pytest.raises(rep.IncompleteRun, match=word):
            rep.check_complete("W6", ans, route, exp)


def test_selection_needs_exactly_the_registered_windows():
    import a1_a3_report as rep
    rep.check_windows(["W2", "W6"])
    for found in (["W6"], ["W2"], ["W2", "W6", "W12"], []):
        with pytest.raises(rep.IncompleteRun):
            rep.check_windows(found)
