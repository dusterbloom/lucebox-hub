"""Stub tests for E9 (a1_e9.py): exact-mass ranking against a brute-force softmax, a planted block, the ranks and the budget row."""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
import a1_a3  # noqa: E402
import a1_e9 as e9  # noqa: E402

L, H, HKV, D, N, ROWS, NB = 2, 4, 2, 8, 3, 6, 5


def make(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(L, H, N, D, generator=g)
    ek = [[torch.randn(HKV, ROWS, D, generator=g) for _ in range(L)] for _ in range(NB)]
    return q, ek


def brute(q, ek, blocks):
    """Per layer, head and question token: softmax over every row of the listed blocks, summed per block; mean over heads then layers."""
    out = torch.zeros(N, len(blocks))
    for l in range(L):
        keys = torch.cat([ek[j][l] for j in blocks], 1)                         # (Hkv, S, d)
        for h in range(H):
            kv = keys[h // (H // HKV)]
            p = torch.softmax(q[l, h] @ kv.T / math.sqrt(D), -1)                 # (n, S)
            out += torch.stack([p[:, i * ROWS:(i + 1) * ROWS].sum(-1) for i in range(len(blocks))], -1) / (H * L)
    return out


def test_exact_mass_with_zero_beta_equals_a_brute_force_softmax_mass():
    q, ek = make()
    for blocks in ([0, 1, 2, 3, 4], [1, 3, 4]):
        assert torch.allclose(e9.exact_mass(q, ek, blocks), brute(q, ek, blocks), atol=1e-5)
    assert torch.allclose(e9.exact_mass(q, ek, [0, 1, 2]).sum(-1), torch.ones(N), atol=1e-5)    # shares of a softmax sum to 1


def test_a_planted_high_mass_block_ranks_first_for_both_aggregations():
    q, ek = make(1)
    q = q * 0.1
    ek[3][0][:, 0] = q[0].mean((0, 1)).expand(HKV, D) * 400                           # block 3 holds a key every query of layer 0 points at
    ek[3][1][:, 0] = q[1].mean((0, 1)).expand(HKV, D) * 400
    x = e9.exact_mass(q, ek, list(range(NB)))
    b = torch.rand(NB, N)
    r = e9.ranks_for(x, x, b, list(range(NB)), [3])
    assert r["x_last"] == 1 and r["x_max"] == 1 and r["a_last"] == 1
    assert e9.ranks_for(x, x, b, list(range(NB)), [0, 3])["x_last"] == 1      # the best source block counts


def test_ranks_for_restricts_to_the_listed_blocks_and_maps_source_ids():
    a = torch.tensor([[0.1, 0.7, 0.2]])                     # (n=1, 3 candidates) blocks 4, 6, 9
    x = torch.tensor([[0.5, 0.2, 0.3]])
    b = torch.tensor([[0.0], [1.0], [0.5]])
    r = e9.ranks_for(a, x, b, [4, 6, 9], [9, 2])             # block 2 is not a candidate and is ignored
    assert (r["a_last"], r["x_last"], r["b_last"]) == (2, 2, 2)
    assert e9.ranks_for(a, x, b, [4, 6, 9], [2]) is None


def test_budget_row_takes_k_from_a3_choose_k_and_reports_inside_k():
    nb, tail, p = 57, 64, 10
    bud = a1_a3.choose_k(nb, tail, p, 6, 300)
    ranks = {r: bud["k"] for r in e9.BUDGET_ROUTERS}
    ranks["x_last"] = bud["k"] + 1
    row = e9.budget_row(nb, tail, p, 6, 300, ranks)
    assert row["k"] == bud["k"] == 5
    assert row["in_k"]["a_last"] is True and row["in_k"]["x_last"] is False and row["src_cand"] is True
    none = e9.budget_row(nb, tail, p, 6, 300, None)
    assert none["src_cand"] is False and "in_k" not in none


def test_report_tables_count_better_equal_worse_and_inside_k():
    import a1_e_report as rep
    rows = [{"id": "a", "eligible": True, "distance_tok": 20000, "ranks": {"a_last": 3, "a_max": 2, "x_last": 1, "x_max": 1, "b_last": 5, "b_max": 4},
             "budget": {"2": {"k": 7, "src_cand": True, "in_k": {"a_last": True, "x_last": True, "x_max": True, "b_last": False}},
                        "6": {"k": 3, "src_cand": False}}},
            {"id": "b", "eligible": True, "distance_tok": 100, "ranks": {"a_last": 1, "a_max": 1, "x_last": 2, "x_max": 1, "b_last": 1, "b_max": 1},
             "budget": {"2": {"k": 7, "src_cand": False}, "6": {"k": 3, "src_cand": False}}},
            {"id": "c", "eligible": False}]
    t = rep.e9_tables(rows)
    assert t["eligible"] == 2 and t["far_eligible"] == 1 and t["x_last"]["far_over_16K"]["n"] == 1
    assert t["x_vs_a_last"] == {"x_better": 1, "equal": 0, "x_worse": 1} and t["x_vs_a_max"] == {"x_better": 1, "equal": 1, "x_worse": 0}
    assert t["budget"]["W2"]["inside_k"]["x_last"] == [1, 1] and t["budget"]["W2"]["inside_k"]["b_last"] == [0, 1]
    assert t["budget"]["W6"]["inside_k"] is None and t["budget"]["W6"]["source_in_candidates"] == 0
