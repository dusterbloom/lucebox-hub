"""Stub tests for A3 routed exact fetch (a1_a3.py; PREREG "A3 routed exact fetch"): row budget, router candidates, the k = 0 and
fetch-everything states, the discarded routing pass, and the pure selection/verdict helpers. No compaction repo, no weights.
The same five checks on the tiny random hybrid are in modal/am_qwen35/tests/test_a1.py (run on lucebox6)."""
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
        assert not a3.choose_k(57, 639, 10, wb)["over_budget"]


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
    cap = a3.rows_at(NB, TAIL, P, WB, 2, block=BLOCK, slots=KEEP)                   # room for exactly 2 fetched blocks
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
    cap = a3.rows_at(NB, TAIL, P, WB, 3, block=BLOCK, slots=KEEP) + BLOCK - KEEP - 1      # room for 3 blocks, one row short of a 4th
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


def test_select_config_prefers_retention_then_fewer_mean_rows_and_only_eligible_ones():
    t = {"W2": {"correct": 50, "n": 118, "mean_rows": 15000, "eligible": True},
         "W6": {"correct": 55, "n": 118, "mean_rows": 14000, "eligible": True}}
    assert a3.select_config(t)["selected"] == "W6"
    assert "selected_unrestricted" not in a3.select_config(t)
    t["W6"]["correct"] = 50
    assert a3.select_config(t)["selected"] == "W6"                                  # tie on retention: fewer mean rows
    t["W6"]["eligible"] = False
    assert a3.select_config(t)["selected"] == "W2"
    assert a3.select_config({k: {**v, "eligible": False} for k, v in t.items()})["selected"] is None


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


def _row(traj, cut, k, rows, over=False, n_cand=7, dist=100):
    return {"traj": traj, "cut_index": cut, "k": k, "rows": rows, "over_budget": over, "n_cand": n_cand,
            "distance_tok": dist, "route_s": 1.0, "fetch_bytes": 10, "probe_wall_s": 2.0, "decode_tokens": 5}


def test_cuts_table_counts_each_cut_once_logs_k_and_rows_and_flags_over_budget_cuts():
    import a1_a3_report as rep
    rows = [_row("t", 1, 7, 9000), _row("t", 1, 7, 9000), _row("t", 2, 2, 16000), _row("t", 3, 0, 17000, over=True)]
    c = rep.cuts_table(rows)
    assert c["cuts"] == 3 and c["cuts_over_cap_at_k0"] == 1 and c["cuts_k_below_candidates"] == 2 and "cuts_k_lowered" not in c
    assert c["k_used_histogram"] == {"7": 1, "2": 1, "0": 1} and c["max_rows"] == 17000


def test_retention_by_k_used_groups_probes_by_the_k_of_their_cut():
    import a1_a3_report as rep
    rows = [{"id": "a", "k": 9}, {"id": "b", "k": 9}, {"id": "c", "k": 3}, {"id": "d", "k": 0}]
    got = rep.by_k_used(rows, ["a", "b", "c", "d"], [1, 0, 1, 0])
    assert got == {"0": {"n": 1, "correct": 0}, "3": {"n": 1, "correct": 1}, "9": {"n": 2, "correct": 1}}


def test_router_recall_within_the_fetched_set_splits_a_miss_into_routing_and_compaction():
    import a1_a3_report as rep

    def rr(i, status, fetched):
        return {"id": i, "src_status": status, "source_fetched": fetched}

    route = [rr("a", "candidate", True), rr("b", "candidate", True), rr("c", "candidate", False), rr("d", "exact_resident", False)]
    got = rep.fetched_vs_retention(route, {"a": 1, "b": 0, "c": 0, "d": 1})
    assert got == {"candidate_probes": 3, "source_fetched": {"n": 2, "correct": 1}, "source_not_fetched": {"n": 1, "correct": 0},
                   "router_recall_in_fetched_set": 2 / 3}


def test_router_tables_use_only_candidate_probes_with_a_rank():
    import a1_a3_report as rep

    def rr(status, rank, dist=100, n_cand=7):
        return {"src_status": status, "rank_a_last": rank, "distance_tok": dist, "n_cand": n_cand}

    t = rep.router_tables([rr("candidate", 1), rr("candidate", 9, dist=20000), rr("exact_resident", None), rr("tail", None),
                           rr("not_located", None)])
    assert t["source_status"] == {"candidate": 2, "exact_resident": 1, "tail": 1, "not_located": 1}
    assert t["a_last"]["candidate_probes"]["n"] == 2 and t["a_last"]["candidate_probes"]["recall"]["8"] == 0.5
    assert t["a_last"]["candidate_probes_over_16K"]["n"] == 1
