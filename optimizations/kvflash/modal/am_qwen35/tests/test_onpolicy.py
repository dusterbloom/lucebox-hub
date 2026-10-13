"""CPU test of the repo's on-policy per-layer-head compactor on a tiny random Qwen3.5-style hybrid (fp32).

An identity "compaction" (keep every key, beta = 0, values unchanged) must hand every layer exactly the queries the
original cache produces, so layer 7's on-policy queries (computed through a CompactedPrefixCache that carries the
DeltaNet state and the already-compacted attention layers) equal the original ones. That checks the whole plumbing:
query rows of a hybrid, DeltaNet state carry-over, RoPE offset, hooks, early exit at `max_layer`.
Run inside the r1a image: modal run am_qwen35/r1a.py::unit
"""
import sys
from types import SimpleNamespace

sys.modules["fla"] = None  # CPU: torch fallback of the gated delta rule

import pytest
import torch

import evaluation  # noqa: F401  (import order: evaluation <-> compaction circular import)
from compaction.compaction_methods.per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from compaction.query_generation import QueryGenerator
from evaluation.configs.utils import load_query_config
from evaluation.utils import format_context
from models.generate import get_sliding_layer_info
from models.qwen3_5 import (HFGenerateShim, adapt_model, dynamic_cache_to_pseudo_kv, inject_linear_state,
                            restore_model)

NAME = "Qwen/Qwen3.8-27B"


class IdentityAlgo:
    seen = []

    def __init__(self, **kw):
        pass

    def name(self):
        return "identity"

    def compute_compacted_cache(self, K, V, queries, target):
        IdentityAlgo.seen.append(queries.detach().clone())
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
    return model, tok


def _prefill(model, tok):
    from transformers import DynamicCache
    article = "The quick brown fox jumps over the lazy dog. " * 12 + "Then it rested by the river bank for a long while."
    formatted = format_context(tok, article, None, NAME)
    ids = tok(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"]
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    from compaction.compaction_methods.base import FullCacheCompactionAlgorithm  # noqa: F401
    from evaluation.utils import compute_article_indices
    return formatted, ids, cache, compute_article_indices(tok, formatted, article)


def test_sliding_info_includes_recurrent_layers(setup):
    model, _ = setup
    sliding, window = get_sliding_layer_info(model)
    assert sliding == {0, 1, 2, 4, 5, 6} and window is None


def test_identity_onpolicy_queries_equal_original(setup):
    model, tok = setup
    formatted, ids, cache, article_idx = _prefill(model, tok)
    layer_types = model.config.layer_types
    kv = dynamic_cache_to_pseudo_kv(cache, layer_types)
    sliding, _ = get_sliding_layer_info(model)
    qcfg = load_query_config("repeat")
    adapt_model(model)
    try:
        gen = QueryGenerator(model=model, tokenizer=tok, config=qcfg, device="cpu", dtype=torch.float32)
        orig, _stats, _seq = gen.generate_queries(formatted_context=formatted, past_key_values=cache, indices=article_idx)
        assert orig.shape[0] == 2  # one row per full-attention layer
        IdentityAlgo.seen = []
        method = PerLayerHeadOnPolicyCompaction(algorithm_class=IdentityAlgo, algorithm_kwargs={}, config_name="identity")
        compacted, stats = method.compact_kv_cache(
            past_key_values=kv, target_size=ids.shape[1], indices=article_idx, query_config=qcfg, model=model,
            tokenizer=tok, formatted_context=formatted, compute_stats=False, sliding_layer_indices=sliding,
            past_key_values_for_queries=cache)
    finally:
        restore_model(model)
    assert len(IdentityAlgo.seen) == 4  # 2 attention layers x 2 KV heads
    for h in range(2):
        assert torch.allclose(IdentityAlgo.seen[h], orig[0, h], atol=1e-5)          # first attention layer: original
        assert torch.allclose(IdentityAlgo.seen[2 + h], orig[1, h], atol=1e-4), (    # second: through the cache
            float((IdentityAlgo.seen[2 + h] - orig[1, h]).abs().max()))
    # identity result equals the original K/V in the attention layers
    for i in (3, 7):
        assert torch.equal(compacted[i][0], cache.layers[i].keys) and torch.equal(compacted[i][2], cache.layers[i].values)
    full = inject_linear_state(compacted, cache, layer_types)
    assert full[0][0].ndim == 3


def test_hf_generate_shim_runs_adapted_and_plain(setup):
    model, tok = setup
    sp = SimpleNamespace(temperature=0.7, top_p=0.8, top_k=20, max_tokens=4)
    shim = HFGenerateShim(model, tok, seed=1, batch_size=2)
    plain = shim.generate(["Hello there", "A longer prompt to force left padding in the batch"], sp)
    adapt_model(model)
    try:
        adapted = HFGenerateShim(model, tok, seed=1, batch_size=2).generate(
            ["Hello there", "A longer prompt to force left padding in the batch"], sp)
    finally:
        restore_model(model)
    assert len(plain) == len(adapted) == 2
    assert [o.outputs[0].text for o in plain] == [o.outputs[0].text for o in adapted]  # same seed, same weights
