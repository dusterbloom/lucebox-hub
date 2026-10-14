"""CPU tests for the R2 multi-segment plumbing on a tiny random Qwen3.5-style hybrid (fp32, eager, real tokenizer).

The load-bearing check (PREREG R2): a 2-segment and a 3-segment UNCOMPACTED prefix built by `extend` is exact against a
one-shot forward over the whole sequence (K/V of the full-attention layers, DeltaNet conv/recurrent state, logits of the
next chunk). With an identity "compaction" the F-SEG step (`Runner.segment_step`) must reproduce the same cache and hand
the repo's compactor exactly the queries a one-shot forward produces for the second segment.
Run inside the r1b image: modal run am_qwen35/r2.py::unit
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
from models.qwen3_5 import adapt_model, compacted_cache_from_triples, dynamic_cache_to_triples, restore_model

import r2_core as r2

NAME = "Qwen/Qwen3.8-27B"
ART = ["The quick brown fox jumps over the lazy dog. " * 8 + "Then it rested by the river.",
       "A second article about boats and harbours and long voyages. " * 6 + "The end of the second.",
       "Third article: mountains, snow, and a small hut at the top. " * 5 + "Finally they went home."]


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
    segs, additive = r2.build_segments(tok, ART)
    assert additive
    return model, tok, segs


def _linear(model):
    return {i for i, t in enumerate(model.config.layer_types) if t == "linear_attention"}


def _chain(model, segs, upto):
    """Uncompacted prefix of segs[:upto+1] built segment by segment through the adapter."""
    lt = model.config.layer_types
    adapt_model(model)
    try:
        tr, n = None, 0
        for s in segs[:upto + 1]:
            tr, n, _ = r2.extend(model, lt, tr, n, s.ids)
    finally:
        restore_model(model)
    return tr, n


@torch.no_grad()
def _oneshot(model, ids):
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return cache, out


@pytest.mark.parametrize("upto", [1, 2])
def test_multisegment_uncompacted_prefix_is_exact(setup, upto):
    model, tok, segs = setup
    lin = _linear(model)
    ids = torch.cat([s.ids for s in segs[:upto + 1]], 1)
    ref, _ = _oneshot(model, ids)
    tr, n = _chain(model, segs, upto)
    assert n == ids.shape[1]
    for i, (c1, beta, c2) in enumerate(tr):
        if i in lin:
            assert torch.allclose(c1, ref.layers[i].conv_states[0], atol=1e-5), f"conv state layer {i}"
            assert torch.allclose(c2, ref.layers[i].recurrent_states[0], atol=1e-4), f"recurrent state layer {i}"
        else:
            assert torch.allclose(c1, ref.layers[i].keys, atol=1e-4) and torch.allclose(c2, ref.layers[i].values, atol=1e-4)
            assert (beta == 0).all() and beta.shape == c1.shape[:3]
    # logits of a following chunk: against the one-shot forward over (prefix + chunk), not against a split reference
    chunk = tok("Question: who rested?", return_tensors="pt", add_special_tokens=False)["input_ids"]
    _, full = _oneshot(model, torch.cat([ids, chunk], 1))
    adapt_model(model)
    try:
        with torch.no_grad():
            got = model(input_ids=chunk, past_key_values=compacted_cache_from_triples(tr, lin, n), use_cache=True).logits
    finally:
        restore_model(model)
    assert torch.allclose(got, full.logits[:, -chunk.shape[1]:], atol=1e-3), float((got - full.logits[:, -chunk.shape[1]:]).abs().max())


def test_extend_does_not_mutate_its_prefix(setup):
    model, tok, segs = setup
    lt = model.config.layer_types
    tr, n = _chain(model, segs, 0)
    before = [tuple(x.clone() for x in t) for t in tr]
    adapt_model(model)
    try:
        r2.extend(model, lt, tr, n, segs[1].ids)
    finally:
        restore_model(model)
    assert all(torch.equal(a, b) for t, u in zip(tr, before) for a, b in zip(t, u))


def test_captured_queries_equal_oneshot_queries(setup):
    model, tok, segs = setup
    lt = model.config.layer_types
    ids = torch.cat([s.ids for s in segs[:2]], 1)
    adapt_model(model)
    try:
        _, _, q_full = r2.extend(model, lt, None, 0, ids, want_queries=True)
        tr, n, _ = r2.extend(model, lt, None, 0, segs[0].ids)
        _, _, q_seg = r2.extend(model, lt, tr, n, segs[1].ids, want_queries=True)
    finally:
        restore_model(model)
    assert q_full.shape[0] == 2 and q_seg.shape == (*q_full.shape[:2], segs[1].n, q_full.shape[3])
    assert torch.allclose(q_seg, q_full[:, :, segs[0].n:], atol=1e-4), float((q_seg - q_full[:, :, segs[0].n:]).abs().max())


def test_fseg_step_with_identity_algo_reproduces_uncompacted_cache_and_queries(setup):
    model, tok, segs = setup
    lt, lin = model.config.layer_types, _linear(model)
    ref_tr, ref_n = _chain(model, segs, 1)
    adapt_model(model)
    try:
        S = SimpleNamespace(model=model, tok=tok, layer_types=lt, linear=lin, dev="cpu", shim=None)
        R = r2.Runner.__new__(r2.Runner)
        R.S, R.ratio = S, 0.1
        R.method = PerLayerHeadOnPolicyCompaction(algorithm_class=IdentityAlgo, algorithm_kwargs={}, config_name="identity")
        tr0, n0, _ = r2.extend(model, lt, None, 0, segs[0].ids)
        IdentityAlgo.seen = []
        out, n1, info = R.segment_step(tr0, n0, segs[1].ids, segs[1].art, segs[1].text)
        # reference queries: the repo's own generator on the whole text, article range shifted behind segment 0
        shifted = range(segs[1].art.start + segs[0].n, segs[1].art.stop + segs[0].n)
        gen = QueryGenerator(model=model, tokenizer=tok, config=load_query_config("context-prefill"), device="cpu",
                             dtype=torch.float32)
        orig, _, _ = gen.generate_queries(formatted_context=segs[0].text + segs[1].text, past_key_values=None, indices=shifted)
    finally:
        restore_model(model)
    assert n1 == ref_n
    for i, (c1, b, c2) in enumerate(out):
        if i in lin:
            assert torch.allclose(c2, ref_tr[i][2], atol=1e-4)
            continue
        assert torch.allclose(c1, ref_tr[i][0], atol=1e-4) and torch.allclose(c2, ref_tr[i][2], atol=1e-4)
        assert (b == 0).all()
    assert len(IdentityAlgo.seen) == 4  # 2 attention layers x 2 KV heads
    for h in range(2):
        assert torch.allclose(IdentityAlgo.seen[h], orig[0, h], atol=1e-4)  # first attention layer = original queries


# ---------------------------------------------------------------- pure helpers

def test_pick_distractors_fixed_nested_disjoint_from_probe_set():
    a = r2.pick_distractors(3, 8, seed=0)
    assert a == r2.pick_distractors(3, 8, seed=0) and a != r2.pick_distractors(4, 8, seed=0)
    assert len(set(a)) == 8 and all(50 <= d < 90 for d in a)


def test_sealed_blocks():
    b, end = r2.sealed_blocks(0, 2000)
    assert b == [(0, 640), (640, 1280), (1280, 1920)] and end == 1920
    b2, end2 = r2.sealed_blocks(end, 2600)
    assert b2 == [(1920, 2560)] and end2 == 2560
    assert r2.sealed_blocks(end2, 2600) == ([], 2560)


def test_pad_layers_and_splice():
    lin = {0}
    mk = lambda n: (torch.ones(1, 2, n, 4), torch.zeros(1, 2, n), torch.ones(1, 2, n, 4))
    conv = (torch.ones(1, 3, 4), torch.zeros(1, 1, 0), torch.ones(1, 2, 4, 4))
    tr = (conv, mk(3), mk(5))
    p, L = r2.pad_layers(tr, lin)
    assert L == 5 and p[1][0].shape[2] == 5 and torch.isinf(p[1][1][..., 3:]).all() and (p[1][2][:, :, 3:] == 0).all()
    s = r2.splice(tr, tr, lin)
    assert s[1][0].shape[2] == 6 and s[2][1].shape[2] == 10 and torch.equal(s[0][0], conv[0])
    assert r2.splice(None, tr, lin) is tr


def test_slot_stats_counts_zero_and_one_slot_heads():
    st = r2.slot_stats({(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 7})
    assert st["zero"] == 1 and st["one"] == 2 and st["sum"] == 9 and st["max"] == 7
