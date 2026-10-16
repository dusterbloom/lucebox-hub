"""A4 (lucebox/a1/a1_a4.py) on the tiny random hybrid of test_r2b: the LRU generator, the host store, the cut state with fetched blocks, the
routing pass and the answer. Run on lucebox6 like test_a1.py (pinned compaction checkout + overlay, modal/am_qwen35, lucebox/r1e and lucebox/a1 on
PYTHONPATH, HIP_VISIBLE_DEVICES= empty). Stream of 300 tokens at 24-token blocks: 12 sealed blocks and a tail of 12; chunk 12 (2 per block), sink 12.
The harness layer (tasks, plan, locate_span) is not on the tiny model. Reuses test_a1's helpers by import; test_a1.py is not modified.
(a) the generator evicts oldest blocks under the build inequality, keeps the sink rows, and every resident row equals its host copy;
(b) the state with fetched blocks has the planned shapes and the fetched blocks' K/V are byte-equal to the host copies, in logical order;
(c) with nothing evicted, the A4 state and its answer equal the FULL-CHK state and answer (so no-candidate cuts reproduce the reference);
(d) the routing pass returns the brute-force exact mass over the candidates, discards its KV, and the answer under the row guard runs."""
import math
import sys
from types import SimpleNamespace

sys.modules["fla"] = None  # CPU: torch fallback of the gated delta rule

import pytest
import torch

a1_core = pytest.importorskip("a1_core")

import evaluation  # noqa: F401,E402
import a1_a3  # noqa: E402
import a1_a4 as a4  # noqa: E402
import r2_core as r2  # noqa: E402
from test_a1 import BLOCK_W, question_ids, session, stream_of  # noqa: E402
from test_r2b import setup  # noqa: E402,F401  (fixture `setup`)

CHUNK, SINK, N, NB, TAIL = 12, 12, 300, 12, 12
CAP = 24 * 4 + SINK + (BLOCK_W - 1)              # build window of 4 whole blocks before the last append is the most that fits: 131 rows


def build(sess, stream, cap, pinned, sink):
    R = SimpleNamespace(S=sess)
    full = [i for i in range(len(sess.layer_types)) if i not in sess.linear]
    store = a4.HostStore(full, chunk=CHUNK)
    cs = a1_core.segs_from_checkpoints(stream[:N], [N])
    (_, ship, n, info), = list(a4.gen_lru(R, cs, [0], pinned, sink, cap, BLOCK_W, store.on_block))
    return store, ship, n, info, full


def brute(q, keys, blocks):
    L, H, n, d = q.shape
    out = torch.zeros(n, len(blocks))
    for l in range(L):
        ks = torch.cat([keys[j][l] for j in blocks], 1).float()
        g = H // ks.shape[0]
        for h in range(H):
            p = torch.softmax(q[l, h].float() @ ks[h // g].T / math.sqrt(d), -1)
            pos = 0
            cols = []
            for j in blocks:
                m = keys[j][l].shape[1]
                cols.append(p[:, pos:pos + m].sum(-1))
                pos += m
            out += torch.stack(cols, -1) / (H * L)
    return out


def test_a_generator_evicts_under_the_build_inequality_and_every_resident_row_equals_its_host_copy(setup):
    S, segs = setup
    stream = stream_of(segs)
    with session(S) as sess:
        store, ship, n, info, full = build(sess, stream, CAP, 0, SINK)
        res = info["resident"]
        assert n == N and info["blocks"] == NB and info["tail_tokens"] == TAIL and len(store.host) == NB
        assert res == [8, 9, 10, 11] and a4.state_rows(res, SINK, BLOCK_W - 1, BLOCK_W) <= CAP          # block 0 evicted; (BUILD) holds
        cfg = sess.model.config.get_text_config()
        hkv, d = cfg.num_key_value_heads, cfg.head_dim
        assert all(store.host[j][i][0].shape == (1, hkv, BLOCK_W, d) for j in range(NB) for i in full)
        rows = a4.state_rows(res, SINK, TAIL, BLOCK_W)
        for i in full:
            K, B_, V = ship[i]
            assert K.shape == (1, hkv, rows, d) and B_.shape == (1, hkv, rows) and V.shape == K.shape
            for c in range(3):
                assert torch.equal(ship[i][c][:, :, :SINK], store.host[0][i][c][:, :, :SINK])         # the sink rows are block 0's first rows
                for pos, j in enumerate(res):
                    a = SINK + pos * BLOCK_W
                    assert torch.equal(ship[i][c][:, :, a:a + BLOCK_W], store.host[j][i][c])           # resident rows = the host copy
        assert float(store.host[0][full[0]][1].abs().sum()) == 0.0                                      # exact blocks carry beta 0


def test_b_fetched_blocks_are_byte_equal_to_the_host_copies_in_logical_order(setup):
    S, segs = setup
    stream = stream_of(segs)
    with session(S) as sess:
        store, ship, n, info, full = build(sess, stream, CAP, 0, SINK)
        res = info["resident"]
        # the base state rebuilt from host pieces is the generator's own state, bit for bit
        again = a4.cut_state(ship, store.host, res, SINK, TAIL, sess.linear, sess.dev)
        for i, (a, b) in enumerate(zip(ship, again)):
            assert all(torch.equal(x, y) for x, y in zip(a, b)), i
        reserve = 10
        plan = a4.cut_plan(NB, TAIL, 0, len(res), reserve, 2, SINK, CAP + reserve, BLOCK_W)
        assert plan["base"] == [10, 11] and plan["k"] == 2 and plan["cands"] == list(range(10)) and plan["peak_bound"] <= CAP + reserve
        fetched = [3, 7]
        trip = a4.cut_state(ship, store.host, sorted(set(plan["base"]) | set(fetched)), SINK, TAIL, sess.linear, sess.dev)
        assert trip[full[0]][0].shape[2] == a4.final_rows(plan, fetched, SINK, TAIL, BLOCK_W) == plan["rows_bound"]
        for i in full:
            for c in range(3):
                want = torch.cat([store.host[0][i][c][:, :, :SINK]] + [store.host[j][i][c] for j in (3, 7, 10, 11)] + [ship[i][c][:, :, -TAIL:]], 2)
                assert torch.equal(trip[i][c], want)
        for i in sess.linear:
            assert all(torch.equal(x, y) for x, y in zip(trip[i], ship[i]))
        # block 0 fetched: its remaining rows come back, the sink rows are not duplicated
        t0 = a4.cut_state(ship, store.host, [0, 10, 11], SINK, TAIL, sess.linear, sess.dev)
        assert t0[full[0]][0].shape[2] == 3 * BLOCK_W + TAIL and torch.equal(t0[full[0]][0][:, :, :BLOCK_W], store.host[0][full[0]][0])


def test_c_nothing_evicted_equals_the_full_chk_state_and_answer(setup):
    S, segs = setup
    stream = stream_of(segs)
    with session(S) as sess:
        store, ship, n, info, full = build(sess, stream, 10 ** 6, 0, SINK)
        assert info["resident"] == list(range(NB))
        R = SimpleNamespace(S=sess, ratio=0.1, cell="FUNI", budget=None)
        cs = a1_core.segs_from_checkpoints(stream[:N], [N])
        (_, ref, rn, _), = list(r2.gen_fullchk(R, cs, None, [0], block=BLOCK_W))
        assert rn == n
        for i, (a, b) in enumerate(zip(ship, ref)):
            assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b)), i
        plan = a4.cut_plan(NB, TAIL, 0, NB, 10, 2, SINK, 10 ** 6, BLOCK_W)
        assert plan["cands"] == [] and plan["k"] == 0 and plan["base"] == list(range(NB))
        suffix = question_ids(S)[0].tolist()
        st = SimpleNamespace(ship=ship, host=store.host, sink=SINK, tail=TAIL, t_len=n, linear=sess.linear, full_layers=full)
        gen, _, got = a4.answer_state(sess, st, plan, [], suffix, 6, 10 ** 6, BLOCK_W)
        assert got == NB * BLOCK_W + TAIL and gen == r2.answer_ids(sess, ref, rn, suffix, 6)


def test_d_routing_pass_matches_the_brute_force_exact_mass_and_the_answer_runs_under_the_guard(setup):
    S, segs = setup
    stream = stream_of(segs)
    with session(S) as sess:
        store, ship, n, info, full = build(sess, stream, CAP, 0, SINK)
        suffix = question_ids(S)[0].tolist()
        max_new = 4
        reserve = len(suffix) + max_new
        plan = a4.cut_plan(NB, TAIL, 0, len(info["resident"]), reserve, 2, SINK, CAP + reserve, BLOCK_W)
        cand = plan["cands"]
        base = a4.cut_state(ship, store.host, plan["base"], SINK, TAIL, sess.linear, sess.dev)
        keys = store.cand_keys(cand, SINK, sess.dev)
        pooled, counts = store.cand_pooled(cand, SINK, sess.dev)
        assert keys[0][0].shape[1] == BLOCK_W - SINK and counts[0] == 1 and sum(counts) == pooled.shape[0]      # block 0 without its sink rows
        before = [t.clone() for tr in base for t in tr]
        x, b, rt = a4.route_probe(sess, base, n, suffix, cand, keys, pooled, counts, 1, 2)
        _, _, q = r2.extend(sess.model, sess.layer_types, base, n, torch.tensor([suffix]), want_queries=True)
        assert x.shape == (len(suffix), len(cand)) and b.shape == (len(cand), len(suffix))
        assert torch.allclose(x, brute(q, keys, cand), atol=1e-5) and torch.allclose(x.sum(-1), torch.ones(len(suffix)), atol=1e-5)
        assert rt["imend"] is None and rt["ucontent"] is None and rt["lit5"] == max(0, len(suffix) - 5)
        assert all(torch.equal(u, v) for u, v in zip(before, [t for tr in base for t in tr]))                    # the routing pass left the base state alone
        ranked = a1_a3.rank_candidates(x[-1], cand)
        assert ranked[0] == cand[int(x[-1].argmax())]
        fetched = ranked[:plan["k"]]
        st = SimpleNamespace(ship=ship, host=store.host, sink=SINK, tail=TAIL, t_len=n, linear=sess.linear, full_layers=full)
        gen, tm, got = a4.answer_state(sess, st, plan, fetched, suffix, max_new, CAP + reserve, BLOCK_W)
        assert 1 <= len(gen) <= max_new and got + reserve <= CAP + reserve
        with pytest.raises(AssertionError):                                                                     # the guard refuses a state over the cap
            a4.answer_state(sess, st, plan, fetched, suffix, max_new, got + reserve - 1, BLOCK_W)


def test_e_compact_on_evict_state_rows_frozen_blocks_selection_values_and_the_fetch_swap(setup, monkeypatch):
    """Amendment 1 on the tiny hybrid with the real compactor (selection only, delayed queries): 12 sealed blocks at cap 139 rows give a window of
    4 blocks and a compacted region; the state has the planned rows, every kept row is a row of the block's exact host copy with beta 0, the
    compacted entries are the generator's own, the cut plan holds the cap, and a swapped fetch is byte-equal to the host copy. Not run locally (needs the pinned
    compaction checkout and the model tokenizer); run on lucebox6 through a1_pytest.sh before the launch."""
    S, segs = setup
    stream = stream_of(segs)
    monkeypatch.setattr(a4, "CT_SLOTS", BLOCK_W // 5)
    cap = SINK + 4 * BLOCK_W + 2 * (BLOCK_W // 5) + (BLOCK_W - 1)
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        full = [i for i in range(len(sess.layer_types)) if i not in sess.linear]
        store = a4.HostStore(full, chunk=CHUNK)
        cs = a1_core.segs_from_checkpoints(stream[:N], [N])
        ct = a4.CtCompactor(R, torch.cat([s.ids for s in cs], 1), "CT", SINK, BLOCK_W)
        (_, ship, n, info), = list(a4.gen_lru(R, cs, [0], 0, SINK, cap, BLOCK_W, store.on_block, ct=ct))
        res, comp = info["resident"], info["compacted"]
        assert n == N and comp and len(res) >= 2 and len(comp) + len(res) + info["drops"] == NB
        rows = a4.ct_rows(res, comp, [], SINK, TAIL, BLOCK_W)
        assert ship[full[0]][0].shape[2] == rows
        for j in comp:
            assert ct.comp[j][next(iter(sess.linear))] is None
            for i in full:
                k, b, v = ct.comp[j][i]
                cs_ = SINK if j == 0 else 0
                assert k.shape[2] == a4.ct_slots(j, SINK, BLOCK_W) and not b.any()                   # selection only: zero beta
                for h in range(k.shape[1]):                                                         # each kept row is one of the block's exact rows
                    src_k, src_v = store.host[j][i][0][0, h, cs_:], store.host[j][i][2][0, h, cs_:]
                    pos = [(src_k - r).abs().sum(-1).argmin().item() for r in k[0, h]]
                    assert torch.allclose(k[0, h], src_k[pos], atol=1e-4) and torch.allclose(v[0, h], src_v[pos], atol=1e-4)
        # the cut plan holds the cap, with a fetch swapping a compacted block for its exact host copy at logical order
        reserve = 10
        plan = a4.ct_cut_plan(NB, TAIL, res, comp, reserve, 2, SINK, cap + reserve, BLOCK_W)
        assert plan["peak_bound"] <= cap + reserve and plan["k"] == min(2, len(plan["comp"]))
        comps = {j: ct.comp[j] if j in ct.comp else ct.make(j, a4.to_device(store.host[j], sess.dev), ship, NB) for j in plan["comp"]}
        fetched = plan["comp"][:plan["k"]]
        trip = a4.pieces_state(ship, a4.ct_piece_list(store.host, comps, plan["window"], fetched, SINK, BLOCK_W), TAIL, sess.linear, sess.dev)
        assert trip[full[0]][0].shape[2] == a4.ct_rows(plan["window"], plan["comp"], fetched, SINK, TAIL, BLOCK_W) <= plan["rows_bound"]
        for i in full:
            sink_k = [store.host[0][i][0][:, :, :SINK]] if 0 not in set(plan["window"]) | set(fetched) else []
            want = torch.cat(sink_k + [(store.host[j][i][0] if (j in plan["window"] or j in fetched) else comps[j][i][0]) for j in sorted(set(plan["window"]) | set(plan["comp"]))]
                             + [ship[i][0][:, :, -TAIL:]], 2)
            assert torch.equal(trip[i][0], want)                                                     # K in logical order: sink, then blocks (exact or compacted), then the tail
