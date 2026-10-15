"""CPU tests for the A1 driver changes (PREREG "Rung A1" section 2, items 1-3) on the tiny random Qwen3.5-style hybrid of test_r2b.

Run on lucebox6 like the R2b tests (pinned compaction checkout + overlay, modal/am_qwen35, lucebox/r1e and lucebox/a1 on
PYTHONPATH, HIP_VISIBLE_DEVICES= empty), from ~/r1e/code/repo:
  pytest -q ~/r1e/code/am_qwen35/tests/test_a1.py
Checkpoints are not multiples of the 24-token test block: inside the first block (10), exactly on a block boundary (48), two inside
the same block (55, 70) and the stream end (643 = 26 blocks + 19).
(a) FULL-CHK at every checkpoint = one-pass prefill of stream[:cut] (keys, values 1e-4; logits of a following suffix 1e-3, fp32).
(b) F-BLK-UNI at every checkpoint = the state of a fresh run with that single checkpoint; at one checkpoint = R1(e)'s local_triples.
(c) answer_ids = model.generate greedy on the one-pass cache of stream[:cut] + suffix, stop token and max_new honoured.
(d) block=512 seals at 512; empty_cache is called once per block and only when asked.
Real data (skipped when absent): on all 60 trajectories the cut-prefix ids are a prefix of the stream and of every request.
"""
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

sys.modules["fla"] = None  # CPU: torch fallback of the gated delta rule

import pytest
import torch

a1_core = pytest.importorskip("a1_core")

import evaluation  # noqa: F401,E402  (import order: evaluation <-> compaction circular import)
from models.qwen3_5 import dynamic_cache_to_triples, restore_model  # noqa: E402

import r1b_core  # noqa: E402
import r1e_run as r1e  # noqa: E402
import r2_core as r2  # noqa: E402
from test_r2b import ART, BLOCK, _small_blocks, setup  # noqa: E402,F401  (fixture `setup`: tiny model, tokenizer, ART segments)

ENDS = [10, 48, 55, 70, 643]
QUESTION = "Question: who rested?"


@contextmanager
def session(S):
    """r1b Session on the tiny model (adapts it, as the real driver does); restored on exit so tests do not stack adapters."""
    sess = r1b_core.Session(S.model, S.tok)
    try:
        yield sess
    finally:
        restore_model(S.model)


def stream_of(segs):
    return torch.cat([s.ids for s in segs], 1)[0].tolist()


def one_pass(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1).past_key_values


def suffix_logits(sess, triples, n, chunk):
    with torch.no_grad():
        return sess.model(input_ids=chunk, past_key_values=sess._cache(triples, n), use_cache=True).logits


def question_ids(S):
    return S.tok(QUESTION, return_tensors="pt", add_special_tokens=False)["input_ids"]


def test_segs_from_checkpoints_cut_the_stream_at_the_ends(setup):
    S, segs = setup
    stream = stream_of(segs)
    out = a1_core.segs_from_checkpoints(stream, ENDS)
    assert [s.n for s in out] == [10, 38, 7, 15, 573]
    assert torch.equal(torch.cat([s.ids for s in out], 1)[0], torch.tensor(stream))
    with pytest.raises(AssertionError):
        a1_core.segs_from_checkpoints(stream, [10, 10, 643])
    with pytest.raises(AssertionError):
        a1_core.segs_from_checkpoints(stream, [10, 48])          # the stream must end at the last checkpoint


def test_a_full_chk_equals_one_pass_at_every_checkpoint(setup, monkeypatch):
    S, segs = setup
    _small_blocks(monkeypatch)
    stream = stream_of(segs)
    cs = a1_core.segs_from_checkpoints(stream, ENDS)
    chunk = question_ids(S)
    with session(S) as sess:
        R = SimpleNamespace(S=sess, ratio=0.1, cell="FUNI", budget=None)
        got = {k: (tr, n, info) for k, tr, n, info in r2.gen_fullchk(R, cs, None, list(range(len(cs))))}
        assert sorted(got) == list(range(5))
        for k, end in enumerate(ENDS):
            tr, n, info = got[k]
            assert n == end and info["blocks"] == end // BLOCK and info["tail_tokens"] == end % BLOCK
            ref, _ = dynamic_cache_to_triples(one_pass(S.model, stream[:end]), sess.layer_types)
            for i, (r, g) in enumerate(zip(ref, tr)):
                if i in sess.linear:
                    assert torch.allclose(g[0], r[0], atol=1e-4) and torch.allclose(g[2], r[2], atol=1e-3), f"state {k}/{i}"
                else:
                    assert g[0].shape == r[0].shape
                    assert torch.allclose(g[0], r[0], atol=1e-4) and torch.allclose(g[2], r[2], atol=1e-4), f"kv {k}/{i}"
            a, b = suffix_logits(sess, ref, end, chunk), suffix_logits(sess, tr, end, chunk)
            assert torch.allclose(a, b, atol=1e-3), (k, float((a - b).abs().max()))


def test_b_f_blk_uni_intermediate_yields_change_nothing(setup, monkeypatch):
    S, segs = setup
    _small_blocks(monkeypatch)
    stream = stream_of(segs)[:300]
    ends = [10, 48, 55, 70, 300]
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        multi = {k: (tr, n) for k, tr, n, _ in r2.gen_fblk(R, a1_core.segs_from_checkpoints(stream, ends), None, list(range(5)))}
        for k, end in enumerate(ends):
            (tr, n), = [(t, m) for _, t, m, _ in r2.gen_fblk(R, a1_core.segs_from_checkpoints(stream[:end], [end]), None, [0])]
            assert n == multi[k][1] == end
            for i, (x, y) in enumerate(zip(multi[k][0], tr)):
                assert all(torch.equal(p, q) for p, q in zip(x, y)), f"checkpoint {end} layer {i}"


def test_b_f_blk_uni_equals_r1e_local_triples_on_the_same_ids(setup, monkeypatch):
    S, _ = setup
    _small_blocks(monkeypatch)
    segs0, _ = r2.build_segments(S.tok, [ART[0]])
    ids = stream_of(segs0)
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        ref, ref_n, _ = r1e.local_triples(R, sess, ART[0])
        (_, got, got_n, info), = list(r2.gen_fblk(R, a1_core.segs_from_checkpoints(ids, [len(ids)]), None, [0]))
    assert got_n == ref_n == len(ids) and info["blocks"] >= 2
    for i, (r, g) in enumerate(zip(ref, got)):
        assert all(torch.equal(x, y) for x, y in zip(r, g)), f"layer {i}"


def _reference_tokens(S, stream, cut, suffix, n, eos=None):
    ids = torch.tensor([stream[:cut] + suffix])
    kw = {"eos_token_id": eos} if eos is not None else {}
    with torch.no_grad():
        out = S.model.generate(input_ids=ids, max_new_tokens=n, do_sample=False, pad_token_id=0, **kw)
    return out[0, ids.shape[1]:].tolist()


def test_c_answer_ids_equal_generate_greedy_with_stop_and_limit(setup, monkeypatch):
    S, segs = setup
    _small_blocks(monkeypatch)
    stream = stream_of(segs)
    suffix = question_ids(S)[0].tolist()
    cut = 55                                                   # inside a block, with an open tail
    ref = _reference_tokens(S, stream, cut, suffix, 8, eos=[])
    assert len(ref) == 8
    with session(S) as sess:
        R = SimpleNamespace(S=sess, ratio=0.1, cell="FUNI", budget=None)
        cs = a1_core.segs_from_checkpoints(stream[:cut], [cut])
        (_, tr, n, _), = list(r2.gen_fullchk(R, cs, None, [0]))
        sess.eos = {-1}                                        # no stop token: the limit decides
        tm = {}
        assert r2.answer_ids(sess, tr, n, suffix, 8, timing=tm) == ref
        assert r2.answer_ids(sess, tr, n, suffix, 5) == ref[:5]
        assert set(tm) >= {"prefill_s", "decode_s"}
        stop = ref[2]
        sess.eos = {stop}                                      # stop token is returned, nothing after it
        want = _reference_tokens(S, stream, cut, suffix, 8, eos=[stop])
        assert want[-1] == stop and want[:3] == ref[:3]
        assert r2.answer_ids(sess, tr, n, suffix, 8) == want[: want.index(stop) + 1]


def test_d_block_argument_and_empty_cache_flag(setup, monkeypatch):
    S, segs = setup
    stream = stream_of(segs)
    cs = a1_core.segs_from_checkpoints(stream, [len(stream)])
    with session(S) as sess:
        R = SimpleNamespace(S=sess, ratio=0.1, cell="FUNI", budget=None)
        (_, tr, n, info), = list(r2.gen_fullchk(R, cs, None, [0], block=512))
        assert info["blocks"] == 1 and info["tail_tokens"] == 643 - 512 and n == 643      # seals at 512, not 640
        (_, _, _, dflt), = list(r2.gen_fullchk(R, cs, None, [0]))
        assert dflt["blocks"] == 1 and dflt["tail_tokens"] == 3                           # default stays 640 (R1(e), R2b)
        ref, _ = dynamic_cache_to_triples(one_pass(S.model, stream), sess.layer_types)
        assert all(torch.allclose(g[0], r[0], atol=1e-4) for g, r in zip(tr, ref))
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
        list(r2.gen_fullchk(R, cs, None, [0]))
        assert calls == []                                                                # off by default
        list(r2.gen_fullchk(R, cs, None, [0], block=BLOCK * 8, empty_cache=True))
        assert len(calls) == 643 // (BLOCK * 8)                                           # once per sealed block


def test_run_unit_rows_are_harness_rows_and_match_generate(setup, monkeypatch):
    S, segs = setup
    _small_blocks(monkeypatch)
    stream = stream_of(segs)
    sfx = question_ids(S)[0].tolist()
    mk = lambda i, kind="probe": a1_core.Item({"id": f"T:{i}", "traj": "T", "kind": kind}, sfx, 6, 0, "sha")   # noqa: E731
    plan = a1_core.Plan("T", [], stream, [a1_core.Cut(2, 48, [mk("a"), mk("b")]), a1_core.Cut(6, 70, [mk("c")]),
                                          a1_core.Cut(9, 643, [mk("d")])])
    rows = []
    parse = lambda text, tools: {"cleaned_text": text, "tool_calls": []}                  # noqa: E731
    with session(S) as sess:
        sess.eos = {-1}
        R = SimpleNamespace(S=sess, ratio=0.1, cell="FUNI", budget=None)
        st = a1_core.run_unit(sess, R, plan, r2.gen_fullchk, parse, rows.append, "FULL", log=lambda s: None, det_repeat=2)
    assert [r["id"] for r in rows] == ["T:a", "T:b", "T:c", "T:d"] and st["n_rows"] == 4 and len(st["checkpoints"]) == 3
    for r, cut in zip(rows, [48, 48, 70, 643]):
        want = _reference_tokens(S, stream, cut, sfx, 6, eos=[])
        assert r["gen_ids"] == want and r["ctx_len"] == cut and r["completion_tokens"] == 6 and r["finish"] == "length"
        assert r["content"] == r["text"] == S.tok.decode(want) and r["arm"] == "FULL" and r["tool_calls"] is None
        assert r["timings"]["prefilled_tokens"] == len(sfx) and r["timings"]["cached_prefix_tokens"] == cut
    assert [r.get("repeat_equal") for r in rows] == [True, True, None, None]               # determinism check on the first two


# ---------------------------------------------------------------- real data, CPU

_P = a1_core.default_paths()
HAVE_DATA = all(os.path.exists(_P[k]) for k in ("traj", "tasks", "template", "tokenizer"))


@pytest.mark.skipif(not HAVE_DATA, reason="A1 data not present")
def test_cut_prefix_ids_are_a_prefix_of_every_request_all_60_trajectories():
    import a1_hash
    H = a1_core.Harness(_P)
    lines, cuts, n_traj = a1_hash.request_hashes(H, log=lambda s: None)       # plan_trajectory raises PrefixError on a violation
    assert n_traj == 60 and len(lines) == 3292 and len(cuts) == 1527
    expect = os.environ.get("A1_HASH_FILE")
    if expect:
        assert a1_hash.read_hash_file(expect) == a1_hash.hash_file_lines(H, lines, cuts)
