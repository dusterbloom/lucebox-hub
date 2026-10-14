"""CPU tests for the R1(e) driver on a tiny random Qwen3.5-style hybrid (fp32, eager, real tokenizer).

Run in the pinned compaction checkout (patches 0001-0003 + overlay) with modal/am_qwen35 and this directory on PYTHONPATH:
  PYTHONPATH=$R1E/code/repo:$R1E/code/am_qwen35:$R1E/code/r1e HIP_VISIBLE_DEVICES= pytest -q test_r1e.py
The LOCAL arm is r2_core.gen_fblk at k = 0 over one segment; these tests pin that its block cutting equals `block_plan`,
that short articles stay exact, and that the uniform FUNI compactor gives every head the same slots per sealed block.
"""
import sys
from types import SimpleNamespace

sys.modules["fla"] = None  # CPU: torch fallback of the gated delta rule

import pytest
import torch

import evaluation  # noqa: F401  (import order: evaluation <-> compaction circular import)
from compaction.compaction_methods.per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from models.qwen3_5 import adapt_model, restore_model

import r1e_run as r1e
import r2_core as r2

NAME = "Qwen/Qwen3.8-27B"
ARTICLE = ("The quick brown fox jumps over the lazy dog near the river bank, and the old mill turns slowly in the wind. " * 12
           + "Then it rested.")


class IdentityAlgo:
    def __init__(self, **kw):
        pass

    def name(self):
        return "identity"

    def compute_compacted_cache(self, K, V, queries, target):
        return K, K.new_zeros(K.shape[0]), V, list(range(K.shape[0]))


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
    return S


def _runner(S, method, ratio):
    R = r2.Runner.__new__(r2.Runner)
    R.S, R.ratio, R.method, R.cell, R.budget = S, ratio, method, "FUNI", None
    return R


def _small_blocks(monkeypatch, block):
    real = r2.sealed_blocks
    monkeypatch.setattr(r2, "sealed_blocks", lambda start, total, block_=block: real(start, total, block_))


def test_block_plan_pure():
    assert r1e.block_plan(1500) == ([(0, 640), (640, 1280)], 220)
    assert r1e.block_plan(640) == ([(0, 640)], 0)
    assert r1e.block_plan(639) == ([], 639)


def test_local_cuts_blocks_like_block_plan_and_keeps_tail_exact(setup, monkeypatch):
    S = setup
    _small_blocks(monkeypatch, 24)
    segs, _ = r2.build_segments(S.tok, [ARTICLE])
    n = segs[0].n
    blocks, tail = r1e.block_plan(n, 24)
    assert len(blocks) >= 3 and tail > 0
    adapt_model(S.model)
    try:
        R = _runner(S, PerLayerHeadOnPolicyCompaction(algorithm_class=IdentityAlgo, algorithm_kwargs={},
                                                     config_name="identity"), 0.25)
        triples, orig_len, info = r1e.local_triples(R, S, ARTICLE)
        ref, ref_n, _ = r2.extend(S.model, S.layer_types, None, 0, segs[0].ids)
    finally:
        restore_model(S.model)
    assert info["blocks"] == len(blocks) and info["tail_tokens"] == tail and orig_len == ref_n == n
    for i, (c1, _, c2) in enumerate(triples):  # identity compaction: the cache equals the uncompacted one
        if i not in S.linear:
            assert torch.allclose(c1, ref[i][0], atol=1e-4) and torch.allclose(c2, ref[i][2], atol=1e-4)


def test_local_short_article_is_uncompacted(setup):
    S = setup
    segs, _ = r2.build_segments(S.tok, [ARTICLE])
    assert segs[0].n < 640
    adapt_model(S.model)
    try:
        R = _runner(S, PerLayerHeadOnPolicyCompaction(algorithm_class=IdentityAlgo, algorithm_kwargs={},
                                                     config_name="identity"), 0.1)
        _, orig_len, info = r1e.local_triples(R, S, ARTICLE)
    finally:
        restore_model(S.model)
    assert info["blocks"] == 0 and info["tail_tokens"] == segs[0].n and orig_len == segs[0].n


def test_local_uniform_funi_gives_every_head_the_same_slots_per_block(setup, monkeypatch):
    S = setup
    _small_blocks(monkeypatch, 24)
    segs, _ = r2.build_segments(S.tok, [ARTICLE])
    blocks, tail = r1e.block_plan(segs[0].n, 24)
    adapt_model(S.model)
    try:
        R = r2.Runner(S, None, cell="FUNI", ratio=0.25)
        triples, orig_len, info = r1e.local_triples(R, S, ARTICLE)
    finally:
        restore_model(S.model)
    slots = 6 * len(blocks) + tail  # 0.25 * 24 slots per sealed block, exact tail
    lens = {t[0].shape[2] for i, t in enumerate(triples) if i not in S.linear}
    assert lens == {slots}, (lens, slots)
    assert info["slots_mean"] == slots and not info["nan_layers"]
