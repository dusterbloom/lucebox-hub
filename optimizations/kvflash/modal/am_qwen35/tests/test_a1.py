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


def test_d1_ratio_option_default_is_the_shipping_ratio_and_ratio_1_keeps_every_slot(setup, monkeypatch):
    import inspect
    assert inspect.signature(a1_core.load_model).parameters["ratio"].default == a1_core.SHIP_RATIO == r2.RATIO == 0.10
    assert inspect.signature(r2.Runner).parameters["ratio"].default == r2.RATIO
    S, _ = setup
    _small_blocks(monkeypatch)
    segs0, _ = r2.build_segments(S.tok, [ART[0]])
    ids = stream_of(segs0)
    cs = a1_core.segs_from_checkpoints(ids, [len(ids)])
    with session(S) as sess:
        dflt, = [t for _, t, _, _ in r2.gen_fblk(r2.Runner(sess, None, cell="FUNI"), cs, None, [0])]
        same, = [t for _, t, _, _ in r2.gen_fblk(r2.Runner(sess, None, cell="FUNI", ratio=r2.RATIO), cs, None, [0])]
        for x, y in zip(dflt, same):
            assert all(torch.equal(p, q) for p, q in zip(x, y))                    # default path == explicit 0.10
        full, = [t for _, t, _, _ in r2.gen_fullchk(r2.Runner(sess, None, cell="FUNI"), cs, None, [0])]
        ident, = [t for _, t, _, _ in r2.gen_fblk(r2.Runner(sess, None, cell="FUNI", ratio=1.0), cs, None, [0])]
        for i, (f, d, c) in enumerate(zip(full, ident, dflt)):
            if i not in sess.linear:
                assert d[0].shape == f[0].shape and c[0].shape[2] < f[0].shape[2], f"layer {i}"   # every slot kept vs 0.10
                print(f"D1 layer {i} max|dK| {float((d[0] - f[0]).abs().max()):.3g}")


# ---------------------------------------------------------------- A2: window of exact recent blocks

W_ENDS = [10, 48, 55, 70, 300]          # block 24: 12 sealed blocks at 300, tail 12
BLOCK_W = 24


def _blocks24(monkeypatch):
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda start, total, block_=BLOCK_W: real(start, total, block_))


def _snapshot(gen):
    return {k: (tuple(tuple(x.clone() for x in t) for t in tr), n) for k, tr, n, _ in gen}


def test_a2_window_0_is_torch_equal_to_the_a1_gen_fblk_at_every_checkpoint(setup, monkeypatch):
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:300], W_ENDS)
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        ref = _snapshot(r2.gen_fblk(R, cs, None, list(range(5))))
        got = _snapshot(r2.gen_fblk(R, cs, None, list(range(5)), window=0))
        via_window_code = _snapshot(r2.gen_fblk_window(R, cs, list(range(5)), 0))      # the new path, independent of the A1 path
    for k in ref:
        assert ref[k][1] == got[k][1] == via_window_code[k][1]
        assert all(torch.equal(x, y) for a, b in zip(ref[k][0], got[k][0]) for x, y in zip(a, b)), k
        assert all(torch.equal(x, y) for a, b in zip(ref[k][0], via_window_code[k][0]) for x, y in zip(a, b)), k


def test_a2_window_at_least_the_number_of_blocks_equals_full_chk(setup, monkeypatch):
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:300], W_ENDS)
    chunk = question_ids(S)
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        ref = {k: (tr, n) for k, tr, n, _ in r2.gen_fullchk(R, cs, None, list(range(5)), block=BLOCK_W)}
        for W in (12, 40):
            got = {k: (tr, n) for k, tr, n, _ in r2.gen_fblk(R, cs, None, list(range(5)), window=W)}
            for k in ref:
                (rt, rn), (gt, gn) = ref[k], got[k]
                assert rn == gn
                for i, (r, g) in enumerate(zip(rt, gt)):
                    if i in sess.linear:
                        assert torch.allclose(g[0], r[0], atol=1e-4) and torch.allclose(g[2], r[2], atol=1e-3), (W, k, i)
                    else:
                        assert g[0].shape == r[0].shape
                        assert torch.allclose(g[0], r[0], atol=1e-4) and torch.allclose(g[2], r[2], atol=1e-4), (W, k, i)
                a, b = suffix_logits(sess, rt, rn, chunk), suffix_logits(sess, gt, gn, chunk)
                assert torch.allclose(a, b, atol=1e-3), (W, k, float((a - b).abs().max()))


@pytest.mark.parametrize("W", [1, 3, 5])
def test_a2_older_blocks_hold_their_seal_time_fit_window_blocks_their_exact_kv(setup, monkeypatch, W):
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:300], [300])
    calls, captured, fit_queries = [], [], []
    real = r2.Runner._compact
    real_extend = r2.extend

    def extend_spy(*a, **kw):
        out = real_extend(*a, **kw)
        if kw.get("want_queries"):
            captured.append(out[2])
        return out

    def spy(self, kv, target_size, indices, text, queries=None):
        fit_queries.append(queries)
        out = real(self, kv, target_size, indices, text, queries=queries)
        calls.append((tuple(tuple(x.clone() for x in e) for e in kv), out[0]))   # kv: per layer (K, V) of the block, as handed to the fit
        return out

    monkeypatch.setattr(r2.Runner, "_compact", spy)
    monkeypatch.setattr(r2, "extend", extend_spy)
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        (_, tr, n, info), = list(r2.gen_fblk(R, cs, None, [0], window=W))
        assert len(captured) == len(fit_queries) == 300 // BLOCK_W
        assert all(q is not None and q is c for q, c in zip(fit_queries, captured))   # each fit gets its own block's prefill queries
        nb, tail, slots = 300 // BLOCK_W, 300 % BLOCK_W, r2.slots_for(BLOCK_W, 0.25)
        assert len(calls) == nb and info["tail_tokens"] == tail and n == 300       # every block fitted once, at seal time
        older = nb - W
        for i in range(len(tr)):
            if i in sess.linear:
                continue
            K, B, V = tr[i]
            assert K.shape[2] == slots * older + BLOCK_W * W + tail, i
            pos = 0
            for j, (kv, seg) in enumerate(calls):
                if j < older:                                    # left the window: exactly the seal-time fit
                    sl = slice(pos, pos + slots)
                    want = (seg[i][0], seg[i][1], seg[i][2])
                    pos += slots
                else:                                            # in the window: the block's own exact keys and values, beta 0
                    sl = slice(pos, pos + BLOCK_W)
                    want = (kv[i][0], torch.zeros_like(B[:, :, sl]), kv[i][1])
                    pos += BLOCK_W
                assert torch.equal(K[:, :, sl], want[0]) and torch.equal(V[:, :, sl], want[2]), (W, i, j)
                assert torch.equal(B[:, :, sl], want[1]), (W, i, j)


def test_a2_row_arithmetic_at_52224_tokens():
    assert r2.window_rows(52224, 18) == 15936 <= 16384
    assert r2.window_rows(52224, 25) == 19968
    assert r2.window_rows(52224, 19) == 16512 > 16384
    assert r2.window_rows(52224, 0) == 81 * 64 + 384
    assert r2.window_rows(1000, 100) == 1000                  # a window larger than the stream keeps everything exact


# ---------------------------------------------------------------- E3 / E2 on the tiny hybrid (lucebox/a1/a1_e.py)

def _collect(R, cs, ks):
    comp, exact = [], []
    out = list(r2.gen_fblk_window(R, cs, ks, 0, on_block=lambda j, a, b, ex, cp: (exact.append(ex), comp.append(cp))))
    return out, comp, exact


def test_e3_swapping_the_only_block_restores_its_exact_kv_and_keeps_the_shipping_tail_and_swapping_nothing_is_the_shipping_state(setup, monkeypatch):
    import a1_e
    S, segs = setup
    _blocks24(monkeypatch)
    stream = stream_of(segs)[:30]                                   # one sealed block of 24 and a tail of 6
    cs = a1_core.segs_from_checkpoints(stream, [30])
    chunk = question_ids(S)
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        ref = _snapshot(r2.gen_fblk(R, cs, None, [0], window=0))
        got, comp, exact = _collect(R, cs, [0])
        (_, ship, n, info), = got
        tail = info["tail_tokens"]
        assert tail == 6 and len(comp) == len(exact) == 1
        none = a1_e.oracle_triples(ship, comp, exact, [], tail, sess.linear)
        assert all(torch.equal(x, y) for a, b in zip(ref[0][0], none) for x, y in zip(a, b))        # no swap == the A1 shipping state
        swapped = a1_e.oracle_triples(ship, comp, exact, [0], tail, sess.linear)
        full, = [(tr, nn) for _, tr, nn, _ in r2.gen_fullchk(R, cs, None, [0], block=BLOCK_W)]
        for i, (r, g) in enumerate(zip(full[0], swapped)):
            if i in sess.linear:
                assert torch.equal(g[0], ship[i][0])                                            # DeltaNet state as in the shipping arm
            else:
                assert g[0].shape == r[0].shape
                b = r[0].shape[2] - tail                                                            # the swapped block is the exact KV
                assert torch.allclose(g[0][:, :, :b], r[0][:, :, :b], atol=1e-4) and torch.allclose(g[2][:, :, :b], r[2][:, :, :b], atol=1e-4), i
                # the open tail stays as in the shipping arm (PREREG E3): past the first full-attention layer its KV was computed
                # under the compacted block, so it is not FULL-CHK's tail
                assert torch.equal(g[0][:, :, b:], ship[i][0][:, :, -tail:]) and torch.equal(g[2][:, :, b:], ship[i][2][:, :, -tail:]), i
        assert swapped[sess.layer_types.index("full_attention")][0].shape[2] == 30


def test_e2_router_inputs_on_the_real_model_have_the_expected_shapes(setup, monkeypatch):
    import a1_e
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:100], [100])     # 4 blocks + tail of 4
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        got, comp, exact = _collect(R, cs, [0])
        (_, ship, n, info), = got
        full = [i for i in range(len(ship)) if i not in sess.linear]
        slots = r2.slots_for(BLOCK_W, 0.25)
        spans = [(j * slots, (j + 1) * slots) for j in range(4)]
        _, _, q = r2.extend(sess.model, sess.layer_types, ship, n, question_ids(S), want_queries=True)
        assert q.shape[0] == len(full) and q.shape[2] == question_ids(S).shape[1]
        share = a1_e.router_mass(q, [ship[i][0][0, :, :4 * slots] for i in full], [ship[i][1][0, :, :4 * slots] for i in full], spans)
        assert share.shape == (q.shape[2], 4) and torch.allclose(share.sum(-1), torch.ones(q.shape[2]), atol=1e-4)
        pooled = torch.cat([a1_e.pooled_chunk_keys([exact[j][i][0][0] for i in full], chunk=12) for j in range(4)])
        sc = a1_e.block_scores(a1_e.qk_chunk_scores(pooled, q), 2)
        assert sc.shape == (4, q.shape[2]) and float(sc.abs().max()) <= 1.0 + 1e-4


# ---------------------------------------------------------------- E6 / E7 on the tiny hybrid (PREREG "Diagnostics E6, E7")

def test_e6_repeat_queries_change_the_fitted_tensors_and_nothing_else_about_the_state_shape(setup, monkeypatch):
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:100], [100])      # 4 sealed blocks + tail of 4
    prompt = torch.tensor([S.tok("Repeat the previous context verbatim.", add_special_tokens=False)["input_ids"]])
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        (_, ship, n0, _), = list(r2.gen_fblk_window(R, cs, [0], 0))
        (_, e6, n1, info), = list(r2.gen_fblk_window(R, cs, [0], 0, repeat_prompt=prompt))
        assert n0 == n1 == 100 and info["tail_tokens"] == 4
        full = [i for i in range(len(ship)) if i not in sess.linear]
        assert all(ship[i][0].shape == e6[i][0].shape for i in full)                  # same rows per layer as the shipping arm
        assert all(torch.isfinite(e6[i][c]).all() for i in full for c in (0, 2))
        slots = r2.slots_for(BLOCK_W, 0.25)
        differ = [(ship[i][c][:, :, :4 * slots] != e6[i][c][:, :, :4 * slots]).any().item() for i in full for c in range(3)]
        assert any(differ)                                                            # the fit saw other queries


def test_e7_zero_prefix_equals_the_shipping_state_and_a_prefix_holds_the_exact_rows(setup, monkeypatch):
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:100], [100])
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        (_, ship, _, _), = list(r2.gen_fblk_window(R, cs, [0], 0))
        (_, zero, _, _), = list(r2.gen_fblk_window(R, cs, [0], 0, exact_blocks=0))
        assert all(torch.equal(x, y) for a, b in zip(ship, zero) for x, y in zip(a, b))        # a zero-length prefix is SHIP
        (_, e7, n, info), = list(r2.gen_fblk_window(R, cs, [0], 0, exact_blocks=2))
        full_chk, = [tr for _, tr, _, _ in r2.gen_fullchk(R, cs, None, [0], block=BLOCK_W)]
        slots = r2.slots_for(BLOCK_W, 0.25)
        for i in range(len(e7)):
            if i in sess.linear:
                continue
            assert e7[i][0].shape[2] == 2 * BLOCK_W + 2 * slots + 4
            assert torch.allclose(e7[i][0][:, :, :2 * BLOCK_W], full_chk[i][0][:, :, :2 * BLOCK_W], atol=1e-4)    # exact rows, as FULL-CHK
            assert not e7[i][1][:, :, :2 * BLOCK_W].any()
        assert any((e7[i][1][:, :, 2 * BLOCK_W:2 * BLOCK_W + 2 * slots] != 0).any() for i in range(len(e7)) if i not in sess.linear)   # then compacted blocks


# ---------------------------------------------------------------- A3 routed exact fetch on the tiny hybrid (lucebox/a1/a1_a3.py)

A3_P, A3_WB = 2, 3          # block 24: stream of 300 = 12 sealed blocks + tail 12; exact prefix blocks 0-1, window blocks 9-11, candidates 2-8


def _a3_build(S, segs, sess, a1_a3, ratio=0.25):
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:300], [300])
    R = r2.Runner(sess, None, cell="FUNI", ratio=ratio)
    comp, exact = [], []
    (_, trip, t_len, info), = list(r2.gen_fblk_window(R, cs, [0], A3_WB, exact_blocks=A3_P,
                                                      on_block=lambda j, a, b, ex, cp: (exact.append(ex), comp.append(cp))))
    st = SimpleNamespace(triples=trip, t_len=t_len, nb=info["blocks"], tail=info["tail_tokens"], comp=comp, exact=exact, p=A3_P,
                         wb=A3_WB, linear=sess.linear, full_layers=[i for i in range(len(sess.layer_types)) if i not in sess.linear])
    assert st.nb == 12 and st.tail == 12
    return R, st


def test_a3_row_budget_fills_k_and_matches_the_rows_of_the_real_state(setup, monkeypatch):
    import a1_a3
    S, segs = setup
    _blocks24(monkeypatch)
    with session(S) as sess:
        R, st = _a3_build(S, segs, sess, a1_a3)
        slots = r2.slots_for(BLOCK_W, 0.25)
        for k in range(0, 4):
            trip = a1_a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, a1_a3.candidates(12, A3_P, A3_WB)[:k], sess.linear)
            assert trip[st.full_layers[0]][0].shape[2] == a1_a3.rows_at(12, 12, A3_P, A3_WB, k, block=BLOCK_W, slots=slots), k
        cap = a1_a3.rows_at(12, 12, A3_P, A3_WB, 2, block=BLOCK_W, slots=slots)
        got = a1_a3.choose_k(12, 12, A3_P, A3_WB, cap=cap, block=BLOCK_W, slots=slots)
        assert got["k"] == 2 and got["lowered"] and got["rows"] <= cap and not got["over_budget"]
        assert a1_a3.choose_k(12, 12, A3_P, A3_WB, cap=cap - 1, block=BLOCK_W, slots=slots)["k"] == 1
        assert a1_a3.choose_k(12, 12, A3_P, A3_WB, cap=10 ** 9, block=BLOCK_W, slots=slots)["k"] == 7     # candidates 2-8, all fetched
        assert a1_a3.choose_k(12, 12, A3_P, A3_WB, reserve=1, cap=cap, block=BLOCK_W, slots=slots)["k"] == 1   # the question takes a block


def test_a3_router_never_ranks_the_prefix_the_window_or_the_tail(setup, monkeypatch):
    import a1_a3
    S, segs = setup
    _blocks24(monkeypatch)
    with session(S) as sess:
        R, st = _a3_build(S, segs, sess, a1_a3)
        cand, a_last, tm = a1_a3.route_a_last(sess, st, question_ids(S)[0].tolist())
        assert cand == list(range(A3_P, 12 - A3_WB)) and a_last.shape == (len(cand),)
        assert torch.allclose(a_last.sum(), torch.tensor(1.0), atol=1e-4)
        ranked = a1_a3.rank_candidates(a_last, cand)
        assert sorted(ranked) == cand and not set(ranked) & (set(range(A3_P)) | set(range(12 - A3_WB, 12)))


def test_a3_k_zero_is_the_prefix_plus_window_shipping_state(setup, monkeypatch):
    import a1_a3
    S, segs = setup
    _blocks24(monkeypatch)
    with session(S) as sess:
        R, st = _a3_build(S, segs, sess, a1_a3)
        trip = a1_a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, [], sess.linear)
        assert all(torch.equal(x, y) for a, b in zip(trip, st.triples) for x, y in zip(a, b))


def test_a3_fetching_every_compacted_block_is_the_e3_swap_and_all_exact(setup, monkeypatch):
    import a1_a3
    import a1_e
    S, segs = setup
    _blocks24(monkeypatch)
    with session(S) as sess:
        R, st = _a3_build(S, segs, sess, a1_a3)
        cand = a1_a3.candidates(12, A3_P, A3_WB)
        trip = a1_a3.a3_state(st.triples, st.comp, st.exact, st.nb, st.tail, st.p, st.wb, cand, sess.linear)
        e3 = a1_e.oracle_triples(st.triples, st.comp, st.exact, set(range(12)), st.tail, sess.linear)
        assert all(torch.equal(x, y) for a, b in zip(trip, e3) for x, y in zip(a, b))
        for i in st.full_layers:
            for c in range(3):
                want = torch.cat([st.exact[j][i][c] for j in range(12)] + [st.triples[i][c][:, :, -st.tail:]], 2)
                assert torch.equal(trip[i][c], want), (i, c)
            assert trip[i][0].shape[2] == 12 * BLOCK_W + st.tail
        for i in sess.linear:                                       # the DeltaNet state is the generator's, untouched by the swap
            assert all(torch.equal(x, y) for x, y in zip(trip[i], st.triples[i]))


def test_a3_routing_pass_kv_is_discarded_before_the_answer_prefill(setup, monkeypatch):
    import a1_a3
    S, segs = setup
    _blocks24(monkeypatch)
    with session(S) as sess:
        R, st = _a3_build(S, segs, sess, a1_a3)
        snap = tuple(tuple(x.clone() for x in e) for e in st.triples)
        suffix = question_ids(S)[0].tolist()
        seen, real_answer, real_extend = [], r2.answer_ids, r2.extend
        routing = []

        def spy_extend(*a, **k):
            out = real_extend(*a, **k)
            if k.get("want_queries"):
                routing.append([x for e in out[0] for x in e])                     # strong references: no id reuse
            return out

        def spy_answer(S_, triples, seq_len, sfx, max_new, timing=None):
            seen.append((seq_len, [x for e in triples for x in e], triples[st.full_layers[0]][0].shape[2],
                         tuple(tuple(x.clone() for x in e) for e in triples)))
            return real_answer(S_, triples, seq_len, sfx, max_new, timing=timing)

        monkeypatch.setattr(r2, "extend", spy_extend)
        monkeypatch.setattr(r2, "answer_ids", spy_answer)
        slots = r2.slots_for(BLOCK_W, 0.25)
        cap = a1_a3.rows_at(12, 12, A3_P, A3_WB, 2, block=BLOCK_W, slots=slots) + len(suffix) + 4   # 2 fetched blocks, question, 4 new
        out = a1_a3.answer_item(sess, st, SimpleNamespace(suffix=suffix, max_new=4), cap=cap, block=BLOCK_W, slots=slots)
        assert out["budget"]["k"] == 2
        assert len(routing) == 1 and len(seen) == 1
        assert seen[0][0] == st.t_len                                               # logical length without question tokens
        for i in sess.linear:                                                       # the answer starts from the cut's DeltaNet state
            assert all(torch.equal(x, y) for x, y in zip(seen[0][3][i], snap[i]))
        assert seen[0][2] == a1_a3.rows_at(12, 12, A3_P, A3_WB, 2, block=BLOCK_W, slots=slots)
        assert all(torch.equal(x, y) for a, b in zip(snap, st.triples) for x, y in zip(a, b))      # the cut state survived the routing pass
        monkeypatch.setattr(r2, "extend", real_extend)
        monkeypatch.setattr(r2, "answer_ids", real_answer)
        clean = r2.answer_ids(sess, a1_a3.a3_state(st.triples, st.comp, st.exact, 12, 12, A3_P, A3_WB, out["fetched"], sess.linear),
                              st.t_len, suffix, 4)
        assert out["gen"] == clean                                      # the same tokens as an answer that never routed
        for r in routing[0]:                                            # storage independence: changing a routing tensor leaves the answer's alone
            if not r.is_floating_point():
                continue
            orig = r.clone()
            for a in seen[0][1]:
                if a.shape == r.shape and a.dtype == r.dtype:
                    before = a.clone()
                    r.add_(1)
                    assert torch.equal(a, before), "routing-pass tensor shares storage with the answer state"
                    r.copy_(orig)


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
