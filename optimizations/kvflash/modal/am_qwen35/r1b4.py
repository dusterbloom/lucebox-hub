"""R1b matched-protocol Qwen3-4B reference cells (PREREG R1b Amendment 1, item 1).

Same driver (`r1b_core.run_cells`), prompts, splits, query configs, key selection, non-thinking greedy 256 tokens as the
27B cells; the model is Qwen3-4B on the repo's own (unpatched) modeling in the transformers 4.57.1 image, per-head
budgets are the repo's shipped `Qwen3-4B/optimized_agnostic.json` with `--max-ratio-per-head 0.75`.

  modal run am_qwen35/r1b4.py::cells --cells orig,O1,F,O2 --split sel --target 0.1 --start 0 --n 20
Results: /results/qwen3-4b/r1b/ ; costs appended to /results/qwen3-4b/cost_log.jsonl.
"""
import json
import os
import sys
import time
import types

import modal

SHA = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
MODEL = "Qwen/Qwen3-4B"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/results/qwen3-4b"
BUDGET = "/repo/head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate==1.12.0", "datasets==4.4.1", "numpy", "requests",
                 "scipy", "huggingface_hub")
    .run_commands("git clone https://github.com/adamzweiger/compaction /repo",
                  f"cd /repo && git checkout {SHA}")
    .add_local_dir(HERE, "/am", copy=True, ignore=["__pycache__", "*.pyc", "*.log"])
)
app = modal.App("kvflash-am-r1b4", image=image)
results = modal.Volume.from_name("kvflash-am-results", create_if_missing=True)
hfcache = modal.Volume.from_name("kvflash-am-hfcache", create_if_missing=True)
VOLS = {"/results": results, "/root/.cache/huggingface": hfcache}
GPU = ["H200", "H100"]
USD_PER_S = {"H200": 4.54 / 3600, "H100": 3.95 / 3600}


def _boot():
    import importlib.util
    import torch
    sys.path[:0] = ["/repo", "/am"]
    os.chdir("/repo")
    # The repo's self-study imports vLLM only for SamplingParams; the HF shim reads the same four fields.
    stub = types.ModuleType("vllm")
    stub.SamplingParams = types.SimpleNamespace
    sys.modules.setdefault("vllm", stub)
    spec = importlib.util.spec_from_file_location("hf_generate", "/am/overlay/models/qwen3_5/hf_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import evaluation  # noqa: F401
    from evaluation.utils import load_model_and_tokenizer
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MODEL, "cuda")
    return model, tok, mod.HFGenerateShim(model, tok, seed=1), t0, torch.cuda.get_device_name(0)


def _cost(name, t0, gpu, **extra):
    wall = time.time() - t0
    rec = {"name": name, "gpu": gpu, "wall_s": round(wall),
           "usd_est": round(wall * USD_PER_S["H200" if "H200" in gpu else "H100"], 2), **extra}
    os.makedirs(f"{ROOT}/cost", exist_ok=True)  # one file per job: parallel jobs must not write the same volume file
    json.dump(rec, open(f"{ROOT}/cost/{int(time.time())}_{os.getpid()}.json", "w"))
    results.commit()
    return rec


@app.function(gpu=GPU, volumes=VOLS, timeout=8 * 3600)
def cells(cells: str = "orig,O1", split: str = "sel", target: float = 0.10, start: int = 0, n: int = 2,
          out: str = "r1b"):
    model, tok, shim, t0, gpu = _boot()
    import r1b_core as core
    arts = core.split_range(split, start, n)
    core.run_cells(model, tok, shim, f"{ROOT}/{out}", cells.split(","), split, target, BUDGET, arts, name=MODEL)
    return _cost(f"cells4b {cells} {split} t{target} [{arts.start},{arts.stop})", t0, gpu)
