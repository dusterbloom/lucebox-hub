"""CPU test: R1(c) plumbing (reference answers, influence curves, solver) on a tiny random Qwen3.5-style hybrid."""
import json
import sys

sys.modules["fla"] = None

import pytest
import torch

import evaluation  # noqa: F401
import r1b_core as core

NAME = "Qwen/Qwen3.8-27B"


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
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    cfg._name_or_path = NAME
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    model.config._attn_implementation = "eager"
    return model, tok


def art():
    qs = [{"question": f"Who jumped number {i}?", "options": ["fox", "dog", "cat", "cow"], "gold_label": 1 + i % 4,
           "question_unique_id": f"q{i}"} for i in range(2)]
    return {"article_id": "t1", "title": "t", "questions": qs,
            "article": "The quick brown fox jumps over the lazy dog. " * 10 + "Then it rested by the river."}


def test_curves_and_solver(setup, tmp_path):
    model, tok = setup
    core.run_curves(model, tok, str(tmp_path / "r1c"), [0], target_ratio=0.1, n_eval=3, max_ratio=0.5,
                    log=lambda *_: None, data=[art()])
    c = json.load(open(tmp_path / "r1c/per_article_curves/article_0.json"))
    heads = c.get("head_curves", c)
    want = {f"L{l}H{h}" for l in (3, 7) for h in range(2)}
    assert {k for k in heads if k.startswith("L")} == want
    props, _ = core.solve_budgets(str(tmp_path / "r1c/per_article_curves"), str(tmp_path / "r1c"),
                                  solve_ratios=(0.1,), num_layers=8, num_heads=2)
    assert set(props) == want
    assert abs(sum(props.values()) - 1.0) < 1e-2


def test_run_cells_all_nonselfstudy_cells(setup, tmp_path):
    model, tok = setup
    budget = tmp_path / "b.json"
    budget.write_text(json.dumps({f"L{l}H{h}": 0.25 for l in (3, 7) for h in range(2)}))
    cells = ["orig", "F", "O1", "O2", "UAM", "RND", "TRN"]
    run = lambda: core.run_cells(model, tok, None, str(tmp_path), cells, "sel", 0.2, str(budget), [0],
                                 log=lambda *_: None, data=[art()])
    run()
    for c in cells:
        rows = [json.loads(l) for l in open(core.cell_path(str(tmp_path), c, 0.2, "sel"))]
        assert len(rows) == 2 and all(r["finite"] for r in rows), c
        if c != "orig":
            assert rows[0]["nan_layers"] == [] and rows[0]["full_layer_len_mean"] < rows[0]["seq_len"], c
    run()  # resume: nothing appended
    assert len(open(core.cell_path(str(tmp_path), "O1", 0.2, "sel")).readlines()) == 2
