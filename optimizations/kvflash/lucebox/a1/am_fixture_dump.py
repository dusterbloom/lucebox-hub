"""Reference fixtures for the engine's AM gates (M3 query capture, M4 selection, M5 beta and value fit): the shipping fit's inputs and
outputs on real blocks, as raw little-endian files a C++ test reads with the manifest alone (no numpy, no pickle).

Cases (shipping arm: gen_fblk_window with window 0, uniform 64 slots of a 640-token block, context-prefill queries, ratio 0.10):
  traj1_first   the first sealed block after the system prefix of pilot trajectory 1
  traj1_12k     the sealed block that holds token 12,000 of pilot trajectory 1
  article_r1e   block 3 of the first article of the R1(e) selection split (formatted context as R1(e) builds it)
Per case and per full-attention layer (16), in ggml element order (fastest dimension first):
  q [d, 640, n_head] post-RoPE context-prefill queries the fit used     k, v [d, 640, n_kv] the block's post-RoPE keys and values
  scores [640, n_kv] raw HighestAttnKeys key scores (the fit's own method)   idx [64, n_kv] int32 selected positions in rank order
  beta [64, n_kv]     c2 [d, 64, n_kv] fitted values       and tokens [640] / positions [640] int32 (once per case).
  python3 am_fixture_dump.py --out ~/r1e/out/am_fixtures"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as core  # noqa: E402

BLOCK, SLOTS_NOTE, NEAR_12K = 640, 64, 12000
DTYPES = {"float32": torch.float32, "int32": torch.int32}


# ---------------------------------------------------------------- pure helpers (tested on CPU)

def ggml_shape(torch_shape):
    return [int(x) for x in reversed(tuple(torch_shape))]


def trajectory_blocks(system_end, block=BLOCK):
    """Block indices of the two trajectory cases: the first block after the system prefix (rounded up to block granularity), and the
    block that holds token NEAR_12K."""
    return [math.ceil(system_end / block), NEAR_12K // block]


def write_tensor(directory, fname, t, dtype):
    """Raw little-endian file of `t` cast to `dtype`; the file is the row-major torch tensor, i.e. ggml order with the shape reversed."""
    t = t.detach().to("cpu", DTYPES[dtype]).contiguous().clone()
    data = bytes(t.untyped_storage())[: t.numel() * t.element_size()]
    assert sys.byteorder == "little"
    with open(os.path.join(directory, fname), "wb") as f:
        f.write(data)
    return {"file": fname, "dtype": dtype, "shape_ggml": ggml_shape(t.shape), "shape_torch": list(t.shape), "bytes": len(data)}


def key_scores(K, queries, score_method="max", pooling=None, kernel_size=7):
    """The reference's raw per-key score (HighestAttentionKeysCompaction._select_keys_highest_attention, same operations and
    precision): K (T, d), queries (n, d) -> (T,) fp32."""
    import torch.nn.functional as F
    d = queries.shape[1]
    s = (queries @ K.T).to(torch.float32) * ((1.0 / d) ** 0.5)
    e = torch.exp(s - s.max(dim=1, keepdim=True)[0])
    w = e / e.sum(dim=1, keepdim=True)
    if score_method == "rms":
        sc = torch.sqrt((w ** 2).mean(dim=0))
    elif score_method == "max":
        sc = w.max(dim=0)[0]
    else:
        sc = w.mean(dim=0)
    if pooling is not None:
        p = sc.unsqueeze(0).unsqueeze(0)
        p = (F.avg_pool1d if pooling == "avgpool" else F.max_pool1d)(p, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        sc = p.squeeze(0).squeeze(0)
    return sc


def group_queries(q_layer, n_kv):
    """(H, n, d) -> (n_kv, (H / n_kv) * n, d): the heads of a GQA group concatenated along the query axis (query_generator._attention_to_kv)."""
    H, n, d = q_layer.shape
    return q_layer.reshape(n_kv, (H // n_kv) * n, d)


def make_spy(algo_cls, records):
    """Subclass of the reference algorithm class that records, per (layer, kv head) call, everything the fit saw and produced (fp32 cpu)."""
    class Spy(algo_cls):
        def compute_compacted_cache(self, K, V, queries, t, *a, **kw):
            out = super().compute_compacted_cache(K, V, queries, t, *a, **kw)
            c1, beta, c2, idx = out
            sc = key_scores(K, queries, getattr(self, "score_method", "max"), getattr(self, "pooling", None),
                            getattr(self, "kernel_size", 7))
            top = torch.topk(sc, t, largest=True).indices.tolist()
            f = lambda x: x.detach().float().cpu().clone()  # noqa: E731
            records.append({"k": f(K), "v": f(V), "queries": f(queries), "scores": f(sc), "idx": torch.tensor(idx, dtype=torch.int32),
                            "beta": f(beta), "c2": f(c2), "reproduces": top == list(idx)})
            return out
    return Spy


def assemble_layers(records, q_all, layer_ids):
    """Group the spy records (called layer by layer, kv head by kv head) into per-layer tensors with a leading kv-head axis; check that the
    queries the fit received are the captured ones grouped per kv head. q_all: (L, H, n, d). Returns ({layer: tensors}, checks)."""
    L, H = q_all.shape[0], q_all.shape[1]
    assert len(layer_ids) == L and len(records) % L == 0
    n_kv = len(records) // L
    layers, q_ok, rep_ok = {}, True, True
    for li, layer in enumerate(layer_ids):
        rs = records[li * n_kv:(li + 1) * n_kv]
        grouped = group_queries(q_all[li].float().cpu(), n_kv)
        q_ok &= all(torch.equal(r["queries"], grouped[h]) for h, r in enumerate(rs))
        rep_ok &= all(r["reproduces"] for r in rs)
        layers[layer] = {"q": q_all[li].float().cpu(), **{k: torch.stack([r[k] for r in rs]) for k in ("k", "v", "scores", "idx", "beta", "c2")}}
    return layers, {"q_group_check": bool(q_ok), "scores_reproduce_indices": bool(rep_ok)}


def write_case(directory, layers, tokens, positions, meta):
    """Write the raw files and manifest.json of one case. layers: {layer: {q,k,v,scores,idx,beta,c2}} (torch layouts: q (H,n,d), k/v (n_kv,n,d),
    scores (n_kv,n), idx/beta (n_kv,slots), c2 (n_kv,slots,d))."""
    os.makedirs(directory, exist_ok=True)
    files = [{"layer": None, "name": "tokens", **write_tensor(directory, "tokens.i32", tokens, "int32")},
             {"layer": None, "name": "positions", **write_tensor(directory, "positions.i32", positions, "int32")}]
    for layer in sorted(layers):
        for name, t in layers[layer].items():
            dt = "int32" if name == "idx" else "float32"
            files.append({"layer": layer, "name": name, **write_tensor(directory, f"L{layer:02d}_{name}.{'i32' if dt == 'int32' else 'f32'}", t, dt)})
    manifest = {**meta, "layers": sorted(layers), "element_order": "ggml: shape_ggml lists the fastest-varying dimension first; the file is "
                "the row-major torch tensor of shape_torch (little-endian, no header)", "files": files}
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def code_md5(paths):
    return {os.path.basename(p): hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths}


# ---------------------------------------------------------------- capture (real model)

class _Stop(Exception):
    pass


def capture_blocks(R, S, ids, targets, log=print):
    """Run the shipping build (gen_fblk_window, window 0) over one stream of ids (1, n) and capture the fit of the target blocks. Returns
    {block index: (records, q_all)}; stops after the last target."""
    import r2_core as r2
    cls = type(R)
    real = cls._compact
    state = {"j": 0, "out": {}}
    last = max(targets)

    def wrapped(self, kv, target_size, indices, text, queries=None):
        j = state["j"]
        state["j"] += 1
        if j not in targets:
            return real(self, kv, target_size, indices, text, queries=queries)
        assert target_size == SLOTS_NOTE and len(indices) == BLOCK, (target_size, len(indices))
        records, algo = [], self.method.algorithm_class
        self.method.algorithm_class = make_spy(algo, records)
        try:
            t0 = time.time()
            out = real(self, kv, target_size, indices, text, queries=queries)
        finally:
            self.method.algorithm_class = algo
        state["out"][j] = (records, queries.detach().cpu())
        log(f"[fixtures] captured block {j}: {len(records)} fits in {time.time() - t0:.0f}s")
        if j == last:
            raise _Stop
        return out

    cls._compact = wrapped
    seg = SimpleNamespace(ids=ids, n=ids.shape[1])
    try:
        for _ in r2.gen_fblk_window(R, [seg], [0], 0):
            pass
    except _Stop:
        pass
    finally:
        cls._compact = real
    return state["out"]


def dump_case(out, label, stream_ids, block, captured, layer_ids, meta, S, R):
    records, q_all = captured
    layers, checks = assemble_layers(records, q_all, layer_ids)
    a, b = block * BLOCK, (block + 1) * BLOCK
    algo = R.method.algorithm_class
    manifest = write_case(os.path.join(out, label), layers, stream_ids[0, a:b], torch.arange(a, b), {
        "case": label, "block_index": block, "start_token": a, "end_token": b, "n_tokens": BLOCK, "slots": SLOTS_NOTE,
        "head_dim": q_all.shape[-1], "n_head": q_all.shape[1], "n_kv": len(records) // q_all.shape[0], "checks": checks,
        "precision": "model bf16; tensors dumped as float32 (exact upcast of the bf16 values); scores computed in fp32 as the reference does",
        "fit": {"arm": "A1 shipping (gen_fblk_window window 0), uniform slots per head, context-prefill queries", "cell": R.cell,
                "ratio": R.ratio, "slots": SLOTS_NOTE, "algorithm_class": f"{algo.__module__}.{algo.__name__}",
                "algorithm_kwargs": {k: repr(v) for k, v in R.method.algorithm_kwargs.items()}, "seed": 0,
                "queries_per_kv_head": "the group's query heads concatenated along the query axis (head-major)"},
        **meta})
    log_checks = f"q_group_check={checks['q_group_check']} scores_reproduce_indices={checks['scores_reproduce_indices']}"
    return manifest, log_checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    import a1_e67
    import r1b_core
    import r2_core as r2
    H = core.Harness(core.default_paths())
    m = core.load_model(a.model, a.mem_cap_gib, log)
    S, R = m.S, m.R
    layer_ids = [i for i in range(len(S.layer_types)) if i not in S.linear]
    here = os.path.dirname(os.path.abspath(__file__))
    md5 = code_md5([os.path.join(here, "am_fixture_dump.py"), os.path.join(here, "a1_core.py"), r2.__file__])
    common = {"model": a.model, "code_md5": md5, "layer_types_full_attention": layer_ids}

    trajs, tasks = H.load_trajs(), H.load_tasks()
    tid = list(trajs)[0]
    plan = core.plan_trajectory(H, trajs[tid], [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"])
    stream = torch.tensor([plan.stream], dtype=torch.long)
    end = a1_e67.system_end(plan.stream, S.tok.convert_tokens_to_ids("<|im_end|>"))
    blocks = trajectory_blocks(end)
    assert stream.shape[1] >= (blocks[1] + 1) * BLOCK, stream.shape
    log(f"[fixtures] trajectory 1 ({tid}): system end token {end}, blocks {blocks}")
    t0 = time.time()
    got = capture_blocks(R, S, stream[:, : (blocks[1] + 1) * BLOCK].to(S.dev), set(blocks), log)
    for label, b in zip(("traj1_first", "traj1_12k"), blocks):
        _, chk = dump_case(a.out, label, stream, b, got[b], layer_ids, {**common, "source": {"trajectory": tid, "system_end_token": end}}, S, R)
        log(f"[fixtures] {label} block {b} written; {chk}")
    del got
    torch.cuda.empty_cache()

    from evaluation.datasets import load_dataset
    ai = r1b_core.split_range("sel", 0, 1)[0]
    art = load_dataset("quality")[ai]
    segs, _ = r2.build_segments(S.tok, [art["article"]])
    ids = segs[0].ids
    block = 3
    assert ids.shape[1] >= (block + 1) * BLOCK, ids.shape
    got = capture_blocks(R, S, ids[:, : (block + 1) * BLOCK].to(S.dev), {block}, log)
    _, chk = dump_case(a.out, "article_r1e", ids, block, got[block], layer_ids,
                       {**common, "source": {"split": "sel", "article_index": ai, "article_id": str(art["article_id"])}}, S, R)
    log(f"[fixtures] article_r1e block {block} written; {chk}; total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
