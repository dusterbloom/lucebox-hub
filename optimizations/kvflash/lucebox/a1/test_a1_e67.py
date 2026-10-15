"""Stub tests for diagnostics E6 (repeat-prefill queries) and E7 (system and tool prefix exact): r2_core.gen_fblk_window's
`repeat_prompt` / `exact_blocks` options and the pure helpers of a1_e67.py. Real-model checks are in modal/am_qwen35/tests/test_a1.py."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "modal", "am_qwen35"))
import a1_e67 as e67  # noqa: E402
import test_a2_window as w  # noqa: E402

r2 = w.r2
PROMPT = torch.tensor([[100, 101]])            # stub repeat prompt (ids >= 100 mark the repeat pass)


def tagged_extend(calls, mutate_on_repeat=False):
    """fake_extend whose queries are the token ids themselves, so a fit can be traced to the pass that produced them; records
    (orig_len, ids) of every call that asks for queries; on a repeat pass it scribbles on its input triples (a real forward may
    update recurrent state in place)."""
    inner = w.fake_extend([])

    def extend(model, layer_types, triples, orig_len, ids, want_queries=False):
        repeat = want_queries and int(ids[0, 0]) >= 100
        if want_queries:
            calls.append((orig_len, ids.clone().view(-1).tolist()))
        if repeat and mutate_on_repeat:
            for e in triples:
                for x in e:
                    x.add_(1000.0)
        new, ln, _ = inner(model, layer_types, triples, orig_len, ids, want_queries=False)
        q = ids.float().view(1, 1, -1, 1) if want_queries else None
        return new, ln, q
    return extend


@pytest.fixture
def stub(monkeypatch):
    calls = []
    w.LIVE.clear()
    monkeypatch.setattr(r2, "extend", tagged_extend(calls))
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda s, t, block_=w.BLOCK: real(s, t, block_))
    monkeypatch.setattr(r2, "slots_for", lambda n, ratio=0.1: w.KEEP)
    return w.StubRunner(), calls


def build(R, n, **kw):
    (_, tr, ln, info), = list(r2.gen_fblk_window(R, w.segs_of(n), [0], 0, **kw))
    return tr, ln, info


# ---------------------------------------------------------------- E6

def test_e6_fit_receives_the_repeat_queries_not_the_context_ones(stub):
    R, calls = stub
    build(R, 8 * 3 + 2, repeat_prompt=PROMPT)
    assert len(R.queries) == 3
    for j, q in enumerate(R.queries):
        block = list(range(1 + 8 * j, 9 + 8 * j))
        assert q.view(-1).tolist() == [100.0, 101.0] + [float(x) for x in block]          # prompt then the repeated block


def test_e6_default_fit_still_gets_the_context_queries(stub):
    R, _ = stub
    build(R, 8 * 2)
    assert [q.view(-1).tolist() for q in R.queries] == [[float(x) for x in range(1, 9)], [float(x) for x in range(9, 17)]]


def test_e6_repeated_tokens_sit_after_the_block_at_the_true_positions(stub):
    R, calls = stub
    build(R, 8 * 3, repeat_prompt=PROMPT)
    reps = [c for c in calls if c[1][0] >= 100]
    assert [c[0] for c in reps] == [8, 16, 24]                    # logical length after the block just prefilled = first repeat position
    assert reps[1][1] == [100, 101] + list(range(9, 17))


def test_e6_clone_never_reaches_the_generators_cache_or_state(monkeypatch):
    calls = []
    w.LIVE.clear()
    monkeypatch.setattr(r2, "extend", tagged_extend(calls, mutate_on_repeat=True))
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda s, t, block_=w.BLOCK: real(s, t, block_))
    monkeypatch.setattr(r2, "slots_for", lambda n, ratio=0.1: w.KEEP)
    got, _, _ = build(w.StubRunner(), 8 * 4 + 3, repeat_prompt=PROMPT)
    calls.clear()
    monkeypatch.setattr(r2, "extend", tagged_extend(calls))
    ref, _, _ = build(w.StubRunner(), 8 * 4 + 3)
    # the stub's compaction ignores queries, so with the clone isolated the final state equals the shipping arm's
    assert all(torch.equal(a, b) for x, y in zip(got, ref) for a, b in zip(x, y))


# ---------------------------------------------------------------- E7

def test_e7_prefix_blocks_stay_exact_and_the_rest_are_compacted(stub):
    R, _ = stub
    tr, ln, info = build(R, 8 * 5 + 3, exact_blocks=2)
    assert len(R.fits) == 3                                       # blocks 0 and 1 are never fitted
    for i in (1, 2):
        K, B, V = tr[i]
        assert K.shape[2] == 2 * w.BLOCK + 3 * w.KEEP + 3
        ids = torch.arange(1, 17).float().view(1, 1, 16, 1).expand(1, 1, 16, 2)
        assert torch.equal(K[:, :, :16], ids + i) and not B[:, :, :16].any()                  # exact rows, beta 0
        assert torch.equal(B[:, :, 16:16 + 3 * w.KEEP], torch.full((1, 1, 3 * w.KEEP), -1.0))  # compacted afterwards
    assert float(tr[0][0].sum()) == float(sum(range(1, 8 * 5 + 3 + 1)))        # linear state carried through all blocks


def test_e7_zero_exact_blocks_is_the_shipping_state(stub):
    R, _ = stub
    got, _, _ = build(R, 8 * 4 + 3, exact_blocks=0)
    ref, _, _ = build(w.StubRunner(), 8 * 4 + 3)
    assert all(torch.equal(a, b) for x, y in zip(got, ref) for a, b in zip(x, y))


def test_e7_more_exact_blocks_than_sealed_blocks_is_all_exact(stub):
    R, _ = stub
    tr, _, _ = build(R, 8 * 3 + 1, exact_blocks=9)
    assert not R.fits and tr[1][0].shape[2] == 8 * 3 + 1


def test_e7_later_blocks_are_prefilled_behind_the_exact_prefix(monkeypatch):
    log = []
    monkeypatch.setattr(r2, "extend", w.fake_extend(log))
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda s, t, block_=w.BLOCK: real(s, t, block_))
    monkeypatch.setattr(r2, "slots_for", lambda n, ratio=0.1: w.KEEP)
    build(w.StubRunner(), 8 * 3, exact_blocks=1)
    assert log == [0, 8, 8 + w.KEEP]                              # prefix rows seen by blocks 0, 1, 2


# ---------------------------------------------------------------- pure helpers

def test_system_end_is_one_past_the_first_im_end():
    assert e67.system_end([5, 6, 7, 9, 8, 9], im_end=9) == 4
    with pytest.raises(ValueError):
        e67.system_end([1, 2, 3], im_end=9)


def test_prefix_blocks_rounds_up_to_block_granularity():
    assert [e67.prefix_blocks(n, 640) for n in (0, 1, 640, 641, 7000)] == [0, 1, 1, 2, 11]


def test_repeat_prompt_text_is_the_repos_repeat_starter_as_a_chat_turn():
    seen = {}

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            seen.update(msgs=msgs, kw=kw)
            return "T"
    assert e67.repeat_prompt_text(Tok()) == "T"
    assert seen["msgs"] == [{"role": "user", "content": "Repeat the previous context verbatim."}]
    assert seen["kw"] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
