"""Pure/stub tests for the E2/E3 helpers (a1_e.py): oracle swap, router math, kvflash_qk-style cosine, ranks, span location."""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
import a1_e as e  # noqa: E402

NB, SLOTS, BLOCK, TAIL = 4, 2, 8, 3        # 4 sealed blocks (8 tokens -> 2 slots), open tail of 3
LINEAR = {0}                              # layer 0 linear, layers 1 and 2 full attention


def block_pieces(tag, rows):
    """(K, beta, V) of one block for one layer: rows filled with `tag`."""
    return (torch.full((1, 1, rows, 2), float(tag)), torch.full((1, 1, rows), -1.0 if rows == SLOTS else 0.0),
            torch.full((1, 1, rows, 2), float(tag) + 0.5))


def make_state():
    comp, exact = [], []
    for j in range(NB):
        comp.append(tuple(None if i in LINEAR else block_pieces(10 + j, SLOTS) for i in range(3)))
        exact.append(tuple(None if i in LINEAR else block_pieces(100 + j, BLOCK) for i in range(3)))
    tail = {i: block_pieces(900, TAIL) for i in (1, 2)}
    ship = []
    for i in range(3):
        if i in LINEAR:
            ship.append((torch.ones(1, 1, 1), torch.zeros(1, 1, 0), torch.ones(1, 1, 1)))
        else:
            ship.append(tuple(torch.cat([comp[j][i][c] for j in range(NB)] + [tail[i][c]], 2) for c in range(3)))
    return tuple(ship), comp, exact


def test_overlap_blocks_marks_exactly_the_blocks_the_span_touches():
    assert e.overlap_blocks(0, 0, 4, 8) == [0]
    assert e.overlap_blocks(7, 8, 4, 8) == [0, 1]              # span touches the last token of block 0 and the first of block 1
    assert e.overlap_blocks(8, 15, 4, 8) == [1]
    assert e.overlap_blocks(30, 40, 4, 8) == [3]               # beyond the sealed blocks is the open tail: dropped
    assert e.overlap_blocks(33, 40, 4, 8) == []


def test_oracle_swap_replaces_exactly_the_overlapping_blocks_and_nothing_else():
    ship, comp, exact = make_state()
    out = e.oracle_triples(ship, comp, exact, [1, 3], TAIL, LINEAR)
    for i in (1, 2):
        K, B, V = out[i]
        assert K.shape[2] == 2 * SLOTS + 2 * BLOCK + TAIL                      # swapped blocks counted once, as exact
        want = [comp[0][i], exact[1][i], comp[2][i], exact[3][i]]
        pos = 0
        for piece in want:
            n = piece[0].shape[2]
            assert all(torch.equal(out[i][c][:, :, pos:pos + n], piece[c]) for c in range(3))
            pos += n
        assert torch.equal(K[:, :, pos:], ship[i][0][:, :, -TAIL:])            # open tail untouched
        assert not B[:, :, SLOTS:SLOTS + BLOCK].any()                          # exact rows carry beta 0
    assert out[0] is ship[0]                                                   # DeltaNet state exactly as in the shipping arm


def test_oracle_swap_with_no_blocks_is_the_shipping_state_and_with_all_blocks_is_all_exact():
    ship, comp, exact = make_state()
    none = e.oracle_triples(ship, comp, exact, [], TAIL, LINEAR)
    assert all(torch.equal(a, b) for i in (1, 2) for a, b in zip(none[i], ship[i]))
    allx = e.oracle_triples(ship, comp, exact, [0, 1, 2, 3], TAIL, LINEAR)
    assert allx[1][0].shape[2] == NB * BLOCK + TAIL


def test_router_mass_matches_a_direct_computation():
    # one layer, one query head over one KV head, d = 1 (scale 1): block A has two slots at k = 1, block B two slots at k = -1, beta ln 4
    q = torch.tensor([[[[2.0], [0.5]]]])                      # (L=1, H=1, n=2 tokens, d=1)
    keys = [torch.tensor([[[1.0], [1.0], [-1.0], [-1.0]]])]   # (Hkv=1, S=4, d=1)
    beta = [torch.tensor([[0.0, 0.0, math.log(4), math.log(4)]])]
    share = e.router_mass(q, keys, beta, [(0, 2), (2, 4)])    # (n, B)
    for t, qt in enumerate((2.0, 0.5)):
        a = 2 * math.exp(qt * 1.0)
        b = 2 * math.exp(qt * -1.0 + math.log(4))
        assert share[t].tolist() == pytest.approx([a / (a + b), b / (a + b)], abs=1e-6)
    assert torch.allclose(share.sum(-1), torch.ones(2))


def test_router_mass_averages_over_query_heads_in_a_group_and_layers():
    # two query heads share one KV head (group 2), two layers; head 0 prefers block A, head 1 block B
    q = torch.zeros(2, 2, 1, 1)
    q[:, 0, 0, 0], q[:, 1, 0, 0] = 5.0, -5.0
    keys = [torch.tensor([[[1.0], [-1.0]]])] * 2
    beta = [torch.zeros(1, 2)] * 2
    share = e.router_mass(q, keys, beta, [(0, 1), (1, 2)])
    s = 1 / (1 + math.exp(-10))
    assert share[0].tolist() == pytest.approx([(s + 1 - s) / 2, (1 - s + s) / 2], abs=1e-6)       # symmetric heads average to 0.5


def test_pooled_chunk_keys_and_qk_cosine_match_a_hand_computed_example():
    # layer 0: chunk 0 keys (1,0),(1,0) -> sum (2,0) -> normalised (1,0); chunk 1 keys (0,3),(0,1) -> (0,1)
    K = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 3.0], [0.0, 1.0]]]])          # (Hkv=1, 4 tokens, d=2) per layer
    pooled = e.pooled_chunk_keys([K[0]], chunk=2)                                   # (C=2, L=1, Hkv=1, d=2)
    assert pooled.shape == (2, 1, 1, 2)
    assert pooled[0, 0, 0].tolist() == pytest.approx([1.0, 0.0], abs=1e-5) and pooled[1, 0, 0].tolist() == pytest.approx([0.0, 1.0], abs=1e-5)
    # two query heads (group 2 on the one KV head); one token each; head 0 = (3,0), head 1 = (1,1)
    q = torch.tensor([[[[3.0, 0.0]], [[1.0, 1.0]]]])                                # (L=1, H=2, n=1, d=2)
    sc = e.qk_chunk_scores(pooled, q)                                               # (C, n)
    # chunk 0: max(cos((3,0),(1,0)) = 1, cos((1,1),(1,0)) = 0.7071) = 1 ; chunk 1: max(0, 0.7071) = 0.7071
    assert sc[:, 0].tolist() == pytest.approx([1.0, 1 / math.sqrt(2)], abs=1e-4)


def test_qk_cosine_is_the_mean_over_layers_of_the_max_over_heads():
    pooled = torch.tensor([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]]).view(1, 2, 1, 2)      # C=1, L=2, Hkv=1, d=2
    q = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])                              # (L=2, H=1, n=1, d=2): both layers query (1,0)
    assert e.qk_chunk_scores(pooled, q)[0, 0].item() == pytest.approx((1.0 + 0.0) / 2, abs=1e-4)


def test_block_scores_take_the_max_over_a_blocks_chunks_and_ranks_are_one_based():
    chunk_scores = torch.tensor([[0.1], [0.9], [0.3], [0.2]])                       # 4 chunks, 1 token; 2 chunks per block
    blocks = e.block_scores(chunk_scores, 2)
    assert blocks[:, 0].tolist() == pytest.approx([0.9, 0.3])
    s = torch.tensor([0.2, 0.9, 0.5, 0.9])
    assert e.best_rank(s, [0]) == 4 and e.best_rank(s, [2]) == 3 and e.best_rank(s, [0, 2]) == 3
    assert e.best_rank(s, [1]) == 1 and e.best_rank(s, [3]) == 1                    # ties do not push the source down


def test_recall_table_and_aggregate_over_tokens():
    t = e.recall_table([1, 3, 5, 20], (1, 2, 4, 8, 16))
    assert t["n"] == 4 and t["recall"] == {"1": 0.25, "2": 0.25, "4": 0.5, "8": 0.75, "16": 0.75}
    assert t["median_rank"] == 4.0
    s = torch.tensor([[0.1, 0.8], [0.9, 0.2]])                                      # (B=2, n=2)
    assert e.aggregate_tokens(s)["max"].tolist() == pytest.approx([0.8, 0.9])
    assert e.aggregate_tokens(s)["last"].tolist() == pytest.approx([0.8, 0.2])


def test_locate_span_maps_the_first_gold_occurrence_to_token_positions():
    toks = ["ab", "cd", "foo", "bar", "baz", "foo", "bar"]
    ids = list(range(100, 100 + len(toks)))
    dec = lambda sel: "".join(toks[i - 100] for i in sel)                           # noqa: E731
    assert e.locate_span(dec, ids, 0, len(ids), "foobar") == (2, 3)
    assert e.locate_span(dec, ids, 4, len(ids), "foobar") == (5, 6)                 # search starts at lo
    assert e.locate_span(dec, ids, 0, len(ids), "nothere") is None
    assert e.locate_span(dec, ids, 0, len(ids), "dfo") == (1, 2)                    # a gold string may start mid-token


def test_on_block_hook_feeds_oracle_triples_so_that_no_swap_is_the_shipping_state(monkeypatch):
    """Generator + hook + oracle_triples on the stub model: swap [] reproduces the yielded shipping triples; a swap puts the
    block's exact rows (beta 0) where its compacted slots were."""
    import test_a2_window as w
    r2 = w.r2
    monkeypatch.setattr(r2, "extend", w.fake_extend([]))
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda s, t, block_=w.BLOCK: real(s, t, block_))
    monkeypatch.setattr(r2, "slots_for", lambda n, ratio=0.1: w.KEEP)
    comp, exact, seen = [], [], []

    def hook(j, a, b, ex, cp):
        seen.append((j, a, b))
        exact.append(ex)
        comp.append(cp)

    (_, ship, _, info), = r2.gen_fblk_window(w.StubRunner(), w.segs_of(8 * 4 + 3), [0], 0, on_block=hook)
    assert seen == [(0, 0, 8), (1, 8, 16), (2, 16, 24), (3, 24, 32)] and info["tail_tokens"] == 3
    same = e.oracle_triples(ship, comp, exact, [], 3, {0})
    assert all(torch.equal(x, y) for i in (1, 2) for x, y in zip(same[i], ship[i]))
    sw = e.oracle_triples(ship, comp, exact, [2], 3, {0})
    assert sw[1][0].shape[2] == ship[1][0].shape[2] + (8 - w.KEEP)
    at = w.KEEP * 2                                             # blocks 0 and 1 compacted, then block 2 exact
    assert torch.equal(sw[1][0][:, :, at:at + 8], exact[2][1][0]) and not sw[1][1][:, :, at:at + 8].any()
