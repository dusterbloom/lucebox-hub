"""R2 multi-round on Modal (PREREG section "R2 multi-round"). Results on volume kvflash-am-results under /qwen3.8-27b/.

  modal run am_qwen35/r2.py::unit                                   # CPU tests (tiny hybrid, incl. 2/3-segment exactness)
  modal run am_qwen35/r2.py::budgets                                # CPU: heads with 0/1 slots at 64 and ~660 slots
  KVFLASH_GPU=A100-80GB modal run am_qwen35/r2.py::pilot --probes 0 --ks 1,8 --arms FULL,F-SEG,F-BLK,F-RE --out r2_pilot
Every GPU function writes one cost record per job to /results/qwen3.8-27b/cost/ (name, gpu, wall_s, usd_est).
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
    .add_local_dir(HERE, "/am", copy=True, ignore=["__pycache__", "*.pyc", "*.log", "peek.py"])
    .run_commands("cd /repo && git apply /am/patches/0001-hybrid-qwen3_5-cache-and-compaction.patch "
                  "/am/patches/0002-hybrid-onpolicy-and-self-study.patch /am/patches/0003-influence-hybrid.patch",
                  "cp -r /am/overlay/models/qwen3_5 /repo/models/qwen3_5",
                  "mkdir -p /repo/am_tests && cp /am/tests/test_*.py /repo/am_tests/")
)
app = modal.App("kvflash-am-r2", image=image)
results = modal.Volume.from_name("kvflash-am-results", create_if_missing=True)
hfcache = modal.Volume.from_name("kvflash-am-hfcache", create_if_missing=True)
VOLS = {"/results": results, "/root/.cache/huggingface": hfcache}
GPU = os.environ.get("KVFLASH_GPU", "A100-80GB").split(",")  # "H100!" pins H100; "A100-80GB" first (PREREG R2)
USD_PER_S = {"A100": 2.50 / 3600, "H100": 3.95 / 3600, "H200": 4.54 / 3600}


def _usd(gpu):
    return next((v for k, v in USD_PER_S.items() if k in gpu), USD_PER_S["H200"])


def _ints(s):
    return [int(x) for x in str(s).split(",") if x != ""]


@app.function(timeout=1800, cpu=4, memory=16384)
def unit(k: str = ""):
    import subprocess
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "/repo/am_tests"] + (["-k", k] if k else [])
    r = subprocess.run(cmd, cwd="/repo", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/repo:/am"})
    print(r.stdout[-8000:], r.stderr[-3000:])
    return r.returncode


@app.function(volumes=VOLS, timeout=900, cpu=2, memory=8192)
def budgets():
    sys.path[:0] = ["/repo", "/am"]
    os.chdir("/repo")
    import evaluation  # noqa: F401
    import r2_core
    out = {}
    for slots, seg in ((64, 640), (662, 6626)):  # block of 640 tokens at 0.10; a typical 6626-token article at 0.10
        out[f"{slots}_slots_over_{seg}_tokens"] = r2_core.budget_slot_stats(BUDGET, slots, seg, 0.75)
    return out


@app.function(gpu=GPU, volumes=VOLS, timeout=4 * 3600)
def pilot(probes: str = "0", ks: str = "1,8", arms: str = "FULL,F-SEG,F-BLK,F-RE", out: str = "r2_pilot",
          n_questions: int = 0):
    import torch
    sys.path[:0] = ["/repo", "/am"]
    os.chdir("/repo")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    import r1a_core
    r1a_core.set_attn(model, "sdpa")
    gpu = torch.cuda.get_device_name(0)
    load_s = round(time.time() - t0)
    from models.qwen3_5 import HFGenerateShim
    import r2_core
    r2_core.run_r2(model, tok, HFGenerateShim(model, tok, seed=1), f"{ROOT}/{out}", _ints(probes), _ints(ks),
                   arms.split(","), BUDGET, n_questions=n_questions or None, gpu=gpu,
                   log=lambda s: (print(s, flush=True), results.commit()))
    wall = time.time() - t0
    rec = {"name": f"r2 {out} probes {probes} ks {ks} arms {arms}", "gpu": gpu, "wall_s": round(wall), "load_s": load_s,
           "usd_est": round(wall * _usd(gpu), 2)}
    os.makedirs(f"{ROOT}/cost", exist_ok=True)
    json.dump(rec, open(f"{ROOT}/cost/{int(time.time())}_{os.getpid()}.json", "w"))
    results.commit()
    return rec


@app.local_entrypoint()
def budgets_main():
    print(json.dumps(budgets.remote(), indent=1))
