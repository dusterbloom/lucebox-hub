"""CPU tests for the Qwen3.5 AM adapter: pure mask helpers and a tiny random hybrid model (fp32, eager).
Run inside the r1a image: modal run am_qwen35/r1a.py::unit
"""
import math
import sys

sys.modules["fla"] = None  # CPU test: force the torch fallback of the gated delta rule (triton kernels need a GPU)

import pytest
import torch

from models.qwen3_5.masks import add_beta_to_mask, causal_additive_mask, repeat_kv_for_beta


def test_causal_mask_prefix_visible_and_causal_inside():
    m = causal_additive_mask(3, 2, torch.float32, "cpu")
    assert m.shape == (1, 1, 2, 5)
    assert (m[0, 0, :, :3] == 0).all()
    assert m[0, 0, 0, 4] < -1e30 and m[0, 0, 0, 3] == 0 and (m[0, 0, 1, 3:] == 0).all()


def test_causal_mask_padding_2d():
    pad = torch.tensor([[1, 1, 0, 1, 1]])
    m = causal_additive_mask(3, 2, torch.float32, "cpu", attention_mask_2d=pad)
    assert m[0, 0, 0, 2] < -1e30 and m[0, 0, 1, 2] < -1e30 and m[0, 0, 1, 1] == 0


def test_repeat_kv_for_beta_head_order():
    b = torch.arange(4.0).view(1, 2, 2)  # kv head 0: [0,1], kv head 1: [2,3]
    r = repeat_kv_for_beta(b, 3)
    assert r.shape == (1, 6, 2)
    assert torch.equal(r[0, :3], b[0, 0].expand(3, 2)) and torch.equal(r[0, 3:], b[0, 1].expand(3, 2))


def test_add_beta_only_prefix_columns_and_no_mutation():
    mask = causal_additive_mask(3, 1, torch.float32, "cpu")
    before = mask.clone()
    out = add_beta_to_mask(mask, torch.full((1, 2, 3), -0.5), 2, 4)
    assert out.shape == (1, 4, 1, 4)
    assert torch.equal(mask, before)
    assert (out[..., :3] == -0.5).all() and (out[..., 3] == 0).all()


def test_add_beta_slices_last_kv_len_columns():
    mask = causal_additive_mask(5, 1, torch.float32, "cpu")  # 6 columns, this layer holds 4
    out = add_beta_to_mask(mask, torch.zeros(1, 1, 2), 1, 4)
    assert out.shape == (1, 1, 1, 4)


# ---------------------------------------------------------------- tiny model
@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    )
    m = Qwen3_5ForCausalLM(cfg).float().eval()
    m.config._attn_implementation = "eager"
    return m


def _split(model, ids, k):
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=ids[:, :-k], past_key_values=cache, use_cache=True)
    return cache


def test_uncompacted_adapter_is_exact(tiny):
    from models.qwen3_5 import (adapt_model, compacted_cache_from_triples, dynamic_cache_to_triples,
                                greedy_decode, restore_model)
    ids = torch.randint(0, 128, (1, 40))
    lt = tiny.config.layer_types
    ref_cache = _split(tiny, ids, 8)
    triples, lin = dynamic_cache_to_triples(ref_cache, lt)
    ac = compacted_cache_from_triples(triples, lin, 32)
    with torch.no_grad():
        o = tiny(input_ids=ids[:, -8:], past_key_values=ref_cache, use_cache=True)
        t_ref, l_ref = greedy_decode(tiny, ref_cache, o.logits, 12, keep_logits=True)
    adapt_model(tiny)
    try:
        with torch.no_grad():
            o2 = tiny(input_ids=ids[:, -8:], past_key_values=ac, use_cache=True)
            t_ad, l_ad = greedy_decode(tiny, ac, o2.logits, 12, keep_logits=True)
    finally:
        restore_model(tiny)
    assert torch.equal(o.logits, o2.logits)
    assert t_ref == t_ad and torch.equal(l_ref, l_ad)


def test_duplicated_keys_with_minus_ln2_beta_are_invariant(tiny):
    from models.qwen3_5 import adapt_model, compacted_cache_from_triples, dynamic_cache_to_triples, restore_model
    ids = torch.randint(0, 128, (1, 40))
    cache = _split(tiny, ids, 8)
    triples, lin = dynamic_cache_to_triples(cache, tiny.config.layer_types)
    dup = []
    for i, (c1, b, c2) in enumerate(triples):
        if i in lin:
            dup.append((c1, b, c2))
        else:
            dup.append((torch.cat([c1, c1], 2), torch.cat([b, b], 2) - math.log(2), torch.cat([c2, c2], 2)))
    adapt_model(tiny)
    try:
        with torch.no_grad():
            a = tiny(input_ids=ids[:, -8:], past_key_values=compacted_cache_from_triples(triples, lin, 32),
                     use_cache=True).logits
            b = tiny(input_ids=ids[:, -8:], past_key_values=compacted_cache_from_triples(tuple(dup), lin, 32),
                     use_cache=True).logits
    finally:
        restore_model(tiny)
    assert (a - b).abs().max() < 1e-4


def test_real_compaction_stays_finite(tiny):
    from models.qwen3_5 import adapt_model, compacted_cache_from_triples, dynamic_cache_to_triples, restore_model
    ids = torch.randint(0, 128, (1, 40))
    cache = _split(tiny, ids, 8)
    triples, lin = dynamic_cache_to_triples(cache, tiny.config.layer_types)
    half = tuple((c1, b, c2) if i in lin else (c1[:, :, ::2], b[:, :, ::2], c2[:, :, ::2])
                 for i, (c1, b, c2) in enumerate(triples))
    adapt_model(tiny)
    try:
        with torch.no_grad():
            out = tiny(input_ids=ids[:, -8:], past_key_values=compacted_cache_from_triples(half, lin, 32),
                       use_cache=True).logits
    finally:
        restore_model(tiny)
    assert torch.isfinite(out).all()
