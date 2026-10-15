"""A2 window logic with a stub model (no compaction repo, no weights): which rows the cache holds, what each block is prefilled behind,
and the row arithmetic. The real-model checks (a) to (c) of PREREG A2 are in modal/am_qwen35/tests/test_a1.py."""
import gc
import os
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("A2_R2_DIR") or os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
r2 = pytest.importorskip("r2_core")
sys.path.insert(0, HERE)

BLOCK, KEEP = 8, 2                      # stub: a compacted block keeps 2 of 8 rows (ratio 0.25)
LAYERS = 3                              # layer 0 is "linear" (state), layers 1 and 2 are full attention


LIVE, QS = [], []                       # weak refs to every linear-state tensor made; the query tensor returned per extend call


def fake_extend(prefix_log):
    """Rows are a deterministic function of the token id; beta 0 for appended rows; the linear 'state' is the running token sum."""
    def extend(model, layer_types, triples, orig_len, ids, want_queries=False):
        prefix_log.append(0 if triples is None else triples[1][0].shape[2])
        n = ids.shape[1]
        rows = ids.float().view(1, 1, n, 1).expand(1, 1, n, 2).clone()
        out = []
        for i in range(LAYERS):
            if i == 0:
                prev = 0.0 if triples is None else float(triples[0][0].sum())
                st = torch.full((1, 1, 1), prev + float(ids.sum()))
                st2 = st.clone()
                LIVE.extend([weakref.ref(st), weakref.ref(st2)])
                out.append((st, torch.zeros(1, 1, 0), st2))
            elif triples is None:
                out.append((rows + i, torch.zeros(1, 1, n), rows * 2 + i))
            else:
                out.append((torch.cat([triples[i][0], rows + i], 2), torch.cat([triples[i][1], torch.zeros(1, 1, n)], 2),
                            torch.cat([triples[i][2], rows * 2 + i], 2)))
        q = torch.zeros(1) if want_queries else None
        if want_queries:
            QS.append(q)
        return tuple(out), (orig_len or 0) + n, q
    return extend


class StubRunner(r2.Runner):
    def __init__(self):
        self.S = SimpleNamespace(linear={0}, layer_types=["linear_attention", "full_attention", "full_attention"], model=None,
                                 dev="cpu", tok=SimpleNamespace(decode=lambda ids: "x"))
        self.ratio, self.fits, self.queries = 0.25, [], []

    def _compact(self, kv, target_size, indices, text, queries=None):
        self.fits.append(tuple(None if i == 0 else e for i, e in enumerate(kv)))   # keep no linear state here: this spy must not hold it alive
        self.queries.append(queries)
        out = []
        for i, (k, v) in enumerate(kv):
            if i == 0:
                out.append((k, torch.zeros(1, 1, 0), v))
            else:
                out.append((k[:, :, ::4] * 10, torch.full((1, 1, KEEP), -1.0), v[:, :, ::4] * 10))
        return tuple(out), {}


def segs_of(n):
    ids = torch.arange(1, n + 1).view(1, -1)
    return [SimpleNamespace(ids=ids, n=n)]


@pytest.fixture
def stub(monkeypatch):
    log = []
    LIVE.clear()
    QS.clear()
    monkeypatch.setattr(r2, "extend", fake_extend(log))
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda s, t, block_=BLOCK: real(s, t, block_))
    monkeypatch.setattr(r2, "slots_for", lambda n, ratio=0.1: KEEP)
    return StubRunner(), log


def run(R, n, window, ks=(0,)):
    return {k: (tr, ln, info) for k, tr, ln, info in r2.gen_fblk(R, segs_of(n), None, list(ks), window=window)}


@pytest.mark.parametrize("window", [1, 2, 4])
def test_rows_older_compacted_window_exact_tail_exact(stub, window):
    R, log = stub
    n = 8 * 6 + 3                                           # 6 sealed blocks, tail 3
    (tr, ln, info), = run(R, n, window).values()
    older = 6 - window
    assert ln == n and info["tail_tokens"] == 3 and len(R.fits) == 6        # each block fitted exactly once
    for i in (1, 2):
        K, B, V = tr[i]
        assert K.shape[2] == KEEP * older + BLOCK * window + 3
        pos = 0
        for j, kv in enumerate(R.fits):
            if j < older:
                assert torch.equal(K[:, :, pos:pos + KEEP], kv[i][0][:, :, ::4] * 10) and torch.equal(B[:, :, pos:pos + KEEP], torch.full((1, 1, KEEP), -1.0))
                pos += KEEP
            else:
                assert torch.equal(K[:, :, pos:pos + BLOCK], kv[i][0]) and torch.equal(V[:, :, pos:pos + BLOCK], kv[i][1])
                assert not B[:, :, pos:pos + BLOCK].any()
                pos += BLOCK
        assert not B[:, :, pos:].any()
    assert float(tr[0][0].sum()) == float(sum(range(1, n + 1)))             # the linear state is carried through every block


def test_blocks_are_prefilled_behind_compacted_older_plus_exact_window(stub):
    R, log = stub
    run(R, 8 * 5 + 3, 2)
    # five block prefills (prefix rows seen), then the tail prefill behind the final cache
    assert log == [0, 8, 16, 18, 20, 22]


def test_window_at_least_the_blocks_is_all_exact_and_equals_the_plain_chain(stub, monkeypatch):
    R, log = stub
    n = 8 * 4 + 5
    (tr, _, _), = run(R, n, 4).values()
    (ref, _, _), = run(R, n, 40).values()
    ids = torch.arange(1, n + 1).float().view(1, 1, n, 1).expand(1, 1, n, 2)
    for i in (1, 2):
        assert torch.equal(tr[i][0], ids + i) and torch.equal(ref[i][0], ids + i)


def test_each_checkpoint_does_not_change_the_kept_state(stub):
    R, _ = stub
    # two checkpoints on one stream: the intermediate yield (tail only) leaves later blocks unchanged
    segs = [SimpleNamespace(ids=torch.arange(1, 21).view(1, -1), n=20), SimpleNamespace(ids=torch.arange(21, 61).view(1, -1), n=40)]
    both = {k: tr for k, tr, _, _ in r2.gen_fblk(R, segs, None, [0, 1], window=2)}
    R2 = StubRunner()
    one = {k: tr for k, tr, _, _ in r2.gen_fblk(R2, [SimpleNamespace(ids=torch.arange(1, 61).view(1, -1), n=60)], None, [0], window=2)}
    assert all(torch.equal(a, b) for x, y in zip(both[1], one[0]) for a, b in zip(x, y))


def test_row_arithmetic_at_52224_tokens():
    assert r2.window_rows(52224, 18) == 18 * 640 + 63 * 64 + 384 == 15936 <= 16384
    assert r2.window_rows(52224, 25) == 25 * 640 + 56 * 64 + 384 == 19968
    assert r2.window_rows(52224, 19) == 16512 > 16384
    assert r2.window_rows(52224, 0) == 81 * 64 + 384
    assert r2.window_rows(1000, 100) == 1000


def test_each_block_fit_gets_the_blocks_own_captured_prefill_queries(stub):
    R, _ = stub
    run(R, 8 * 4 + 3, 2)
    assert len(R.queries) == 4 and all(q is not None for q in R.queries)
    assert all(q is QS[i] for i, q in enumerate(R.queries))      # the object extend(want_queries=True) returned for that block


def test_window_path_with_window_0_equals_the_untouched_a1_path_at_every_checkpoint(stub):
    R, _ = stub
    segs = [SimpleNamespace(ids=torch.arange(1, 21).view(1, -1), n=20), SimpleNamespace(ids=torch.arange(21, 41).view(1, -1), n=20),
            SimpleNamespace(ids=torch.arange(41, 62).view(1, -1), n=21)]
    ref = [(k, tr, n) for k, tr, n, _ in r2.gen_fblk(R, segs, None, [0, 1, 2], window=0)]
    got = [(k, tr, n) for k, tr, n, _ in r2.gen_fblk_window(StubRunner(), segs, [0, 1, 2], 0)]
    assert [r[0] for r in ref] == [g[0] for g in got] == [0, 1, 2]
    for (_, rt, rn), (_, gt, gn) in zip(ref, got):
        assert rn == gn and all(torch.equal(a, b) for x, y in zip(rt, gt) for a, b in zip(x, y))


def test_live_linear_states_do_not_grow_with_the_number_of_blocks(stub):
    R, _ = stub
    counts = []
    for nb in (2, 6, 14):
        LIVE.clear()
        for _k, _tr, _n, _i in r2.gen_fblk(StubRunner(), segs_of(8 * nb + 3), None, [0], window=4):
            gc.collect()
            counts.append(sum(r() is not None for r in LIVE))   # inside the generator, checkpoint reached
    assert counts[0] == counts[1] == counts[2], counts


def test_primary_arm_defaults_to_the_first_non_full_arm_and_must_be_selected():
    import a1_core
    assert a1_core.resolve_primary(("FULL", "WIN18", "WIN25"), None) == "WIN18"
    assert a1_core.resolve_primary(("FULL", "SHIP"), None) == "SHIP"
    assert a1_core.resolve_primary(("FULL", "WIN18", "WIN25"), "WIN25") == "WIN25"
    for arms, p in ((("FULL", "WIN18", "WIN25"), "SHIP"), (("FULL",), None), (("FULL", "WIN18"), "FULL")):
        with pytest.raises(ValueError):
            a1_core.resolve_primary(arms, p)
