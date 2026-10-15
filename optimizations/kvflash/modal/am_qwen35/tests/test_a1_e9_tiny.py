"""E9 (lucebox/a1/a1_e9.py) on the tiny random hybrid of test_r2b: the driver's block hook and per-probe routing on one cut.
Run on lucebox6 like test_a1.py (pinned compaction checkout + overlay, modal/am_qwen35, lucebox/r1e and lucebox/a1 on PYTHONPATH,
HIP_VISIBLE_DEVICES= empty). Stream of 100 tokens at 24-token blocks: 4 sealed blocks and a tail of 4; chunk 12 (2 per block).
It drives the same pieces run_trajectory does (gen_fblk_window with BlockStore.on_block, r2.extend queries, probe_rows); the harness layer
(tasks, plan, locate_span) is not on the tiny model. Reuses test_a1's helpers by import; test_a1.py is not modified."""
import math
import sys

sys.modules["fla"] = None  # CPU: torch fallback of the gated delta rule

import pytest
import torch

a1_core = pytest.importorskip("a1_core")

import evaluation  # noqa: F401,E402
import a1_a3  # noqa: E402
import a1_e  # noqa: E402
import a1_e9 as e9  # noqa: E402
import r2_core as r2  # noqa: E402
from test_a1 import BLOCK_W, _blocks24, question_ids, session, stream_of  # noqa: E402
from test_r2b import setup  # noqa: E402,F401  (fixture `setup`)

CHUNK, NB, P, WB = 12, 4, 1, 1            # candidates of nb = 4 with one exact prefix block and a window of one: blocks 1 and 2


def brute(q, exact_k, blocks):
    """Softmax mass per block over the exact keys of `blocks`, mean over query heads then layers: (n, len(blocks))."""
    L, H, n, d = q.shape
    out = torch.zeros(n, len(blocks))
    for l in range(L):
        keys = torch.cat([exact_k[j][l] for j in blocks], 1).float()
        g = H // keys.shape[0]
        for h in range(H):
            p = torch.softmax(q[l, h].float() @ keys[h // g].T / math.sqrt(d), -1)
            rows = exact_k[blocks[0]][0].shape[1]
            out += torch.stack([p[:, i * rows:(i + 1) * rows].sum(-1) for i in range(len(blocks))], -1) / (H * L)
    return out


def test_e9_driver_pass_on_the_tiny_hybrid(setup, monkeypatch):
    S, segs = setup
    _blocks24(monkeypatch)
    cs = a1_core.segs_from_checkpoints(stream_of(segs)[:100], [100])
    with session(S) as sess:
        R = r2.Runner(sess, None, cell="FUNI", ratio=0.25)
        full = [i for i in range(len(sess.layer_types)) if i not in sess.linear]
        store = e9.BlockStore(full, chunk=CHUNK)
        (_, ship, n, info), = list(r2.gen_fblk_window(R, cs, [0], 0, on_block=store.on_block))
        assert info["blocks"] == NB and len(store.exact_k) == NB
        cfg = sess.model.config.get_text_config()
        hkv, d = cfg.num_key_value_heads, cfg.head_dim
        # the on_block assumption: per block and full-attention layer the exact keys are (Hkv, block rows, d)
        assert all(len(ks) == len(full) and all(k.shape == (hkv, BLOCK_W, d) for k in ks) for ks in store.exact_k)
        slots = r2.slots_for(BLOCK_W, 0.25)
        assert all(k.shape == (hkv, slots, d) for ks in store.comp_k for k in ks) and all(b.shape == (hkv, slots) for bs in store.comp_b for b in bs)

        _, _, q = r2.extend(sess.model, sess.layer_types, ship, n, question_ids(S), want_queries=True)
        allb, cand = list(range(NB)), a1_a3.candidates(NB, P, WB)
        assert cand == [1, 2]

        def e2_a(blocks):             # E2's code on the shipping triples: router_mass over the blocks' compacted slots
            sl = [(j * slots, (j + 1) * slots) for j in blocks]
            return a1_e.router_mass(q, [torch.cat([ship[i][0][0, :, a:b] for a, b in sl], 1) for i in full],
                                    [torch.cat([ship[i][1][0, :, a:b] for a, b in sl], 1) for i in full],
                                    [(m * slots, (m + 1) * slots) for m in range(len(blocks))])

        sc = a1_e.block_scores(a1_e.qk_chunk_scores(torch.cat([a1_e.pooled_chunk_keys(ks, chunk=CHUNK) for ks in store.exact_k]), q), 2)   # (B, n)
        assert torch.allclose(e9.exact_mass(q, store.exact_k, allb), brute(q, store.exact_k, allb), atol=1e-5)    # x_last is the brute-force mass

        src = [2]
        ranks, budget = e9.probe_rows(q, store, NB, info["tail_tokens"], P, 14372, src, block=BLOCK_W, chunk=CHUNK, wbs=(WB,))
        xs, a_all = brute(q, store.exact_k, allb), e2_a(allb)
        assert ranks["x_last"] == a1_e.best_rank(xs[-1], src) and ranks["x_max"] == a1_e.best_rank(xs.max(0).values, src)
        assert ranks["a_last"] == a1_e.best_rank(a_all[-1], src) and ranks["a_max"] == a1_e.best_rank(a_all.max(0).values, src)
        assert ranks["b_last"] == a1_e.best_rank(sc[:, -1], src) and ranks["b_max"] == a1_e.best_rank(sc.max(1).values, src)

        # pass 2: A3's candidate-only routers and k from choose_k (reserve chosen so exactly one block fits: k = 1 of 2)
        bud = a1_a3.choose_k(NB, info["tail_tokens"], P, WB, 14372)
        assert bud["k"] == 1 and bud["n_cand"] == 2
        row = budget[str(WB)]
        xc, ac = brute(q, store.exact_k, cand), e2_a(cand)
        pos = [cand.index(2)]
        want = {"a_last": a1_e.best_rank(ac[-1], pos), "x_last": a1_e.best_rank(xc[-1], pos), "x_max": a1_e.best_rank(xc.max(0).values, pos),
                "b_last": a1_e.best_rank(sc[cand][:, -1], pos)}
        assert row["k"] == 1 and row["src_cand"] and row["rank"] == want
        assert row["in_k"] == {r: want[r] <= 1 for r in want}                                  # inside-k flag follows rank <= k
        # a source block outside the candidates (the exact prefix) is recorded as not routable
        assert e9.probe_rows(q, store, NB, info["tail_tokens"], P, 14372, [0], block=BLOCK_W, chunk=CHUNK, wbs=(WB,))[1][str(WB)]["src_cand"] is False
