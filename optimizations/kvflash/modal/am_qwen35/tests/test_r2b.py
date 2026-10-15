"""CPU tests for the R2b arms (PREREG "Rung R2b", driver changes item 1) on a tiny random Qwen3.5-style hybrid.

Run on lucebox6 like the R1(e) tests (pinned compaction checkout + overlay, modal/am_qwen35, lucebox/r1e and lucebox/r2b on
PYTHONPATH, HIP_VISIBLE_DEVICES= empty):
  pytest -q tests/test_r2b.py
FULL-CHK must equal gen_full (block-by-block uncompacted prefill = one-shot) on keys, values and following-question logits;
F-BLK-UNI at k = 0 over one article must equal R1(e)'s local_triples.
"""
import sys
from types import SimpleNamespace

sys.modules["fla"] = None  # CPU: torch fallback of the gated delta rule

import pytest
import torch

import evaluation  # noqa: F401  (import order: evaluation <-> compaction circular import)
from models.qwen3_5 import adapt_model, compacted_cache_from_triples, restore_model

import r1e_run as r1e
import r2_core as r2

NAME = "Qwen/Qwen3.8-27B"
ART = ["The quick brown fox jumps over the lazy dog near the river bank, and the old mill turns slowly in the wind. " * 12
       + "Then it rested.",
       "A second article about boats and harbours and long voyages across cold grey seas at dawn. " * 8
       + "The end of the second.",
       "Third article: mountains, snow, and a small hut at the top of the pass where the wind never stops. " * 7
       + "Finally they went home."]
BLOCK = 24


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    tok = AutoTokenizer.from_pretrained(NAME)
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=248320, hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    cfg._name_or_path = NAME
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    model.config._attn_implementation = "eager"
    lt = model.config.layer_types
    S = SimpleNamespace(model=model, tok=tok, layer_types=lt, dev="cpu", shim=None,
                        linear={i for i, t in enumerate(lt) if t == "linear_attention"})
    segs, additive = r2.build_segments(tok, ART)
    assert additive
    return S, segs


def _small_blocks(monkeypatch):
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda start, total, block_=BLOCK: real(start, total, block_))


def _logits(S, triples, n, chunk):
    adapt_model(S.model)
    try:
        with torch.no_grad():
            return S.model(input_ids=chunk, past_key_values=compacted_cache_from_triples(triples, S.linear, n),
                           use_cache=True).logits
    finally:
        restore_model(S.model)


def test_arms_registered():
    assert "FULL-CHK" in r2.ARMS and "F-BLK-UNI" in r2.ARMS
    assert r2.ARM_GENS["F-BLK-UNI"] is r2.gen_fblk and r2.ARM_CELL["F-BLK-UNI"] == "FUNI"
    assert r2.ARM_GENS["FULL-CHK"] is r2.gen_fullchk and r2.ARM_CELL["FULL-CHK"] == "FUNI"


def test_full_chk_equals_gen_full(setup, monkeypatch):
    S, segs = setup
    _small_blocks(monkeypatch)
    ks = [1, 2]
    R = SimpleNamespace(S=S, ratio=0.1, cell="F", budget=None)
    chunk = S.tok("Question: who rested?", return_tensors="pt", add_special_tokens=False)["input_ids"]
    ref = {k: (tr, n) for k, tr, n, _ in r2.gen_full(R, segs, ART, ks)}
    adapt_model(S.model)
    try:
        got = {k: (tr, n, info) for k, tr, n, info in r2.ARM_GENS["FULL-CHK"](R, segs, ART, ks)}
    finally:
        restore_model(S.model)
    assert set(got) == set(ref) == set(ks)
    for k in ks:
        (rt, rn), (gt, gn, info) = ref[k], got[k]
        assert rn == gn == sum(s.n for s in segs[:k + 1])
        assert info["blocks"] == gn // BLOCK and info["tail_tokens"] == gn % BLOCK and info["blocks"] > 2
        for i, (r, g) in enumerate(zip(rt, gt)):
            if i in S.linear:
                assert torch.allclose(g[0], r[0], atol=1e-4) and torch.allclose(g[2], r[2], atol=1e-3), f"state layer {i}"
            else:
                assert g[0].shape == r[0].shape
                assert torch.allclose(g[0], r[0], atol=1e-4) and torch.allclose(g[2], r[2], atol=1e-4), f"kv layer {i}"
        a, b = _logits(S, rt, rn, chunk), _logits(S, gt, gn, chunk)
        assert torch.allclose(a, b, atol=1e-3), float((a - b).abs().max())


def test_f_blk_uni_k0_equals_r1e_local_triples(setup, monkeypatch):
    S, _ = setup
    _small_blocks(monkeypatch)
    R = r2.Runner(S, None, cell="FUNI", ratio=0.25)
    adapt_model(S.model)
    try:
        ref, ref_n, _ = r1e.local_triples(R, S, ART[0])
        segs0, _ = r2.build_segments(S.tok, [ART[0]])
        (k, got, got_n, info), = list(r2.ARM_GENS["F-BLK-UNI"](R, segs0, [ART[0]], [0]))
    finally:
        restore_model(S.model)
    assert k == 0 and got_n == ref_n and info["blocks"] >= 2
    for i, (r, g) in enumerate(zip(ref, got)):
        assert all(torch.equal(x, y) for x, y in zip(r, g)), f"layer {i}"
