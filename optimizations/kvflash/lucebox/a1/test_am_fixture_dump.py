"""CPU stub tests for am_fixture_dump.py: the files the dump writes have the shapes, dtypes and element order the manifest claims
(read back with nothing but the manifest), and the capture glue records what the reference fit sees."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am_fixture_dump as fx  # noqa: E402

D, N, T, HKV, G, L = 8, 16, 4, 2, 3, 2     # head_dim, block tokens, slots, kv heads, group, layers (stub sizes)


def read_back(case_dir, entry):
    """Rebuild a tensor from the raw file and the manifest entry alone: ggml order lists the fastest dim first."""
    dt = {"float32": torch.float32, "int32": torch.int32}[entry["dtype"]]
    raw = torch.frombuffer(bytearray(open(os.path.join(case_dir, entry["file"]), "rb").read()), dtype=dt)
    assert raw.numel() == int(torch.tensor(entry["shape_ggml"]).prod())
    return raw.view(*reversed(entry["shape_ggml"]))


def fake_layer(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {"q": r(HKV * G, N, D), "k": r(HKV, N, D), "v": r(HKV, N, D), "scores": r(HKV, N).abs(),
            "idx": torch.stack([torch.randperm(N, generator=g)[:T] for _ in range(HKV)]).int(), "beta": r(HKV, T), "c2": r(HKV, T, D)}


def test_ggml_shapes_are_the_torch_shapes_reversed():
    assert fx.ggml_shape((24, 640, 256)) == [256, 640, 24]
    assert fx.ggml_shape((4, 64)) == [64, 4]


def test_write_case_files_match_the_manifest_shapes_dtypes_and_order(tmp_path):
    layers = {1: fake_layer(1), 5: fake_layer(2)}
    tokens, pos = torch.arange(100, 100 + N), torch.arange(30, 30 + N)
    fx.write_case(str(tmp_path), layers, tokens, pos, {"case": "stub", "model": "m"})
    m = json.load(open(tmp_path / "manifest.json"))
    assert m["case"] == "stub" and m["layers"] == [1, 5] and m["element_order"].startswith("ggml")
    by = {(e["layer"], e["name"]): e for e in m["files"]}
    want = {"q": [D, N, HKV * G], "k": [D, N, HKV], "v": [D, N, HKV], "scores": [N, HKV], "idx": [T, HKV], "beta": [T, HKV], "c2": [D, T, HKV]}
    for layer, data in layers.items():
        for name, shape in want.items():
            e = by[(layer, name)]
            assert e["shape_ggml"] == shape and e["dtype"] == ("int32" if name == "idx" else "float32")
            got = read_back(str(tmp_path), e)
            assert got.dtype == (torch.int32 if name == "idx" else torch.float32) and torch.equal(got, data[name].to(got.dtype))
            assert os.path.getsize(tmp_path / e["file"]) == 4 * int(torch.tensor(shape).prod())
    tok, ps = by[(None, "tokens")], by[(None, "positions")]
    assert read_back(str(tmp_path), tok).tolist() == list(range(100, 100 + N)) and read_back(str(tmp_path), ps).tolist() == list(range(30, 30 + N))


def test_element_order_is_fastest_dim_first(tmp_path):
    t = torch.arange(2 * 3 * 4).view(2, 3, 4).float()           # torch (H=2, n=3, d=4): d fastest in memory
    e = fx.write_tensor(str(tmp_path), "x.f32", t, "float32")
    raw = torch.frombuffer(bytearray(open(tmp_path / "x.f32", "rb").read()), dtype=torch.float32)
    assert e["shape_ggml"] == [4, 3, 2] and raw[1].item() == t[0, 0, 1].item() and raw[4].item() == t[0, 1, 0].item()


def test_key_scores_match_a_hand_computed_example_for_max_and_rms():
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    q = torch.tensor([[2.0, 0.0], [0.0, 2.0]]) * (2.0 ** 0.5)       # d = 2: scale 1/sqrt(2) gives logits 2 on the matching key
    s = torch.softmax(torch.tensor([[2.0, 0.0, -2.0], [0.0, 2.0, 0.0]]), -1)
    assert torch.allclose(fx.key_scores(K, q, "max"), s.max(0).values, atol=1e-6)
    assert torch.allclose(fx.key_scores(K, q, "rms"), (s ** 2).mean(0).sqrt(), atol=1e-6)
    assert torch.allclose(fx.key_scores(K, q, "mean"), s.mean(0), atol=1e-6)


def test_group_queries_concatenates_the_groups_heads_along_the_query_axis():
    q = torch.arange(HKV * G * 2 * 3).float().view(HKV * G, 2, 3)
    g = fx.group_queries(q, HKV)
    assert g.shape == (HKV, G * 2, 3) and torch.equal(g[1], torch.cat([q[3], q[4], q[5]]))


class FakeAlgo:
    """Stands in for HighestAttentionKeysCompaction: rank order from topk of key_scores, beta and values made up."""
    score_method, pooling = "rms", None

    def __init__(self, **kw):
        pass

    def compute_compacted_cache(self, K, V, queries, t):
        idx = torch.topk(fx.key_scores(K, queries, "rms"), t).indices
        return K[idx], torch.zeros(t) - 1, V[idx] * 2, idx.tolist()


def test_spy_records_what_the_fit_saw_and_that_the_scores_reproduce_the_selection():
    rec = []
    spy = fx.make_spy(FakeAlgo, rec)()
    K, V, q = torch.randn(N, D), torch.randn(N, D), torch.randn(G * N, D)
    out = spy.compute_compacted_cache(K, V, q, T)
    assert len(out) == 4 and len(rec) == 1
    r = rec[0]
    assert r["scores"].shape == (N,) and r["idx"].dtype == torch.int32 and r["idx"].tolist() == out[3]
    assert r["reproduces"] is True and torch.equal(r["k"], K) and torch.equal(r["queries"], q)
    assert torch.equal(r["c2"], V[torch.tensor(out[3])] * 2)


def test_assemble_layers_orders_calls_by_layer_then_head_and_checks_the_query_grouping():
    q_all = torch.randn(L, HKV * G, N, D)                            # (layers, heads, n, d) as captured
    rec, layer_ids = [], [3, 7]
    for li in range(L):
        grouped = fx.group_queries(q_all[li], HKV)
        for h in range(HKV):
            K, V = torch.randn(N, D), torch.randn(N, D)
            idx = torch.randperm(N)[:T].int()
            rec.append({"k": K, "v": V, "queries": grouped[h], "scores": torch.rand(N), "idx": idx, "beta": torch.randn(T),
                        "c2": torch.randn(T, D), "reproduces": True})
    layers, info = fx.assemble_layers(rec, q_all, layer_ids)
    assert sorted(layers) == [3, 7] and info == {"q_group_check": True, "scores_reproduce_indices": True}
    assert layers[7]["k"].shape == (HKV, N, D) and torch.equal(layers[7]["idx"][1], rec[3]["idx"]) and layers[3]["q"].shape == (HKV * G, N, D)
    rec[0]["queries"] = rec[0]["queries"] + 1
    assert fx.assemble_layers(rec, q_all, layer_ids)[1]["q_group_check"] is False


def test_target_blocks_are_the_first_after_the_system_prefix_and_the_one_near_12k():
    assert fx.trajectory_blocks(system_end=7200, block=640) == [12, 18]        # ceil(7200/640) = 12; 12000 is inside block 18
    assert fx.trajectory_blocks(system_end=0, block=640) == [0, 18]
