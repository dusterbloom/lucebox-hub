"""R1b/R1c on Modal (PREREG sections R1b/R1c). Results on volume kvflash-am-results under /qwen3.8-27b/r1b/ and /r1c/.

  modal run am_qwen35/r1b.py::curves   --articles 90,91,92,93,94     # R1(c) influence curves (GPU)
  modal run am_qwen35/r1b.py::solve                                  # R1(c) budget file (CPU)
  modal run am_qwen35/r1b.py::validate --articles 95,96,97,98,99     # R1(c) check (GPU)
  modal run am_qwen35/r1b.py::cells --cells orig,O1 --split sel --target 0.1 --start 0 --n 2
Every GPU function appends {name, wall_s, usd_est} to /results/qwen3.8-27b/cost_log.jsonl (running cost total).
"""
import json
import os
import sys
import time

import modal

SHA = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
MODEL = "Qwen/Qwen3.8-27B"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/results/qwen3.8-27b"
BUDGET = f"{ROOT}/r1c/optimized_agnostic.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==5.19.0", "accelerate", "datasets", "pytest", "huggingface_hub",
                 "jinja2", "numpy", "requests", "scipy", "flash-linear-attention")
    .run_commands("git clone https://github.com/adamzweiger/compaction /repo",
                  f"cd /repo && git checkout {SHA}")
    .add_local_dir(HERE, "/am", copy=True, ignore=["__pycache__", "*.pyc", "*.log"])
    .run_commands("cd /repo && git apply /am/patches/0001-hybrid-qwen3_5-cache-and-compaction.patch "
                  "/am/patches/0002-hybrid-onpolicy-and-self-study.patch /am/patches/0003-influence-hybrid.patch",
                  "cp -r /am/overlay/models/qwen3_5 /repo/models/qwen3_5",
                  "mkdir -p /repo/am_tests && cp /am/tests/test_*.py /repo/am_tests/")
)
app = modal.App("kvflash-am-r1b", image=image)
results = modal.Volume.from_name("kvflash-am-results", create_if_missing=True)
hfcache = modal.Volume.from_name("kvflash-am-hfcache", create_if_missing=True)
VOLS = {"/results": results, "/root/.cache/huggingface": hfcache}
GPU = ["H200", "H100"]
USD_PER_S = {"H200": 4.54 / 3600, "H100": 3.95 / 3600}


def _boot():
    import torch
    sys.path.insert(0, "/repo")
    sys.path.insert(0, "/am")
    os.chdir("/repo")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    import r1a_core
    r1a_core.set_attn(model, "sdpa")
    return tok, model, t0, torch.cuda.get_device_name(0)


def _cost(name, t0, gpu, **extra):
    wall = time.time() - t0
    rec = {"name": name, "gpu": gpu, "wall_s": round(wall),
           "usd_est": round(wall * USD_PER_S["H200" if "H200" in gpu else "H100"], 2), **extra}
    os.makedirs(ROOT, exist_ok=True)
    with open(f"{ROOT}/cost_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    results.commit()
    return rec


def _ints(s):
    return [int(x) for x in str(s).split(",") if x != ""]


@app.function(timeout=1800, cpu=4, memory=16384)
def unit():
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "/repo/am_tests"],
                       cwd="/repo", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/repo:/am"})
    print(r.stdout[-6000:], r.stderr[-3000:])
    return r.returncode


@app.function(gpu=GPU, volumes=VOLS, timeout=6 * 3600)
def curves(articles: str = "90"):
    tok, model, t0, gpu = _boot()
    import r1b_core as core
    core.run_curves(model, tok, f"{ROOT}/r1c", _ints(articles))
    return _cost(f"curves {articles}", t0, gpu)


@app.function(volumes=VOLS, timeout=3600, cpu=4, memory=16384)
def solve():
    sys.path[:0] = ["/repo", "/am"]
    os.chdir("/repo")
    import r1b_core as core
    props, stats = core.solve_budgets(f"{ROOT}/r1c/per_article_curves", f"{ROOT}/r1c")
    results.commit()
    v = sorted(props.values())
    return {"n_heads": len(v), "min": v[0], "max": v[-1], "sum": sum(v)}


@app.function(gpu=GPU, volumes=VOLS, timeout=6 * 3600)
def validate(articles: str = "95,96,97,98,99", target: float = 0.10):
    tok, model, t0, gpu = _boot()
    import r1b_core as core
    summ = core.run_validation(model, tok, f"{ROOT}/r1c", BUDGET, _ints(articles), target)
    return {"summary": {k: v for k, v in summ.items() if k != "rows"}, "cost": _cost("validate", t0, gpu)}


@app.function(gpu=GPU, volumes=VOLS, timeout=8 * 3600)
def cells(cells: str = "orig,O1", split: str = "sel", target: float = 0.10, start: int = 0, n: int = 2):
    tok, model, t0, gpu = _boot()
    import r1b_core as core
    from models.qwen3_5 import HFGenerateShim
    out = f"{ROOT}/r1b"
    arts = core.split_range(split, start, n)
    core.run_cells(model, tok, HFGenerateShim(model, tok, seed=1), out, cells.split(","), split, target, BUDGET, arts)
    return _cost(f"cells {cells} {split} t{target} [{arts.start},{arts.stop})", t0, gpu)
