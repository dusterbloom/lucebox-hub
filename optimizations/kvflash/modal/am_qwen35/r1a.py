"""R1a: Qwen3.8-27B adapter for Attention Matching, exact without compaction (see evidence/am/PREREG.md).

  modal run optimizations/kvflash/modal/am_qwen35/r1a.py::unit       # CPU: mask tests + tiny hybrid model
  modal run optimizations/kvflash/modal/am_qwen35/r1a.py::prefetch   # CPU: weights into the hf cache volume
  modal run optimizations/kvflash/modal/am_qwen35/r1a.py::run        # 1 GPU: R1a gating run + smoke
Results: volume kvflash-am-results under /qwen3.8-27b/r1a/.
"""
import json
import os
import subprocess
import sys
import time

import modal

SHA = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
MODEL = "Qwen/Qwen3.8-27B"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/results/qwen3.8-27b/r1a"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==5.19.0", "accelerate", "datasets", "pytest", "huggingface_hub",
                 "jinja2", "numpy", "requests", "flash-linear-attention")
    .run_commands("git clone https://github.com/adamzweiger/compaction /repo",
                  f"cd /repo && git checkout {SHA}")
    .add_local_dir(HERE, "/am", copy=True, ignore=["__pycache__", "*.pyc", "*.log"])
    .run_commands("cd /repo && git apply /am/patches/0001-hybrid-qwen3_5-cache-and-compaction.patch",
                  "cp -r /am/overlay/models/qwen3_5 /repo/models/qwen3_5",
                  "mkdir -p /repo/am_tests && cp /am/tests/test_adapter.py /repo/am_tests/")
)
app = modal.App("kvflash-am-r1a", image=image)
results = modal.Volume.from_name("kvflash-am-results", create_if_missing=True)
hfcache = modal.Volume.from_name("kvflash-am-hfcache", create_if_missing=True)
VOLS = {"/results": results, "/root/.cache/huggingface": hfcache}
GPU = ["H200", "H100"]
USD_PER_S = {"H200": 4.54 / 3600, "H100": 3.95 / 3600}


@app.function(timeout=1800, cpu=4, memory=16384)
def unit():
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                        "/repo/am_tests/test_adapter.py"], cwd="/repo", capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "/repo"})
    print(r.stdout[-6000:], r.stderr[-3000:])
    return r.returncode


@app.function(volumes=VOLS, timeout=3600, cpu=4)
def prefetch():
    from huggingface_hub import snapshot_download
    t0 = time.time()
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    hfcache.commit()
    return {"path": p, "wall_s": round(time.time() - t0)}


def _boot():
    import torch
    sys.path.insert(0, "/repo")
    sys.path.insert(0, "/am")
    os.chdir("/repo")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import r1a_core as core

    os.makedirs(OUT, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    tcfg = model.config.get_text_config()
    meta = {"model": MODEL, "gpu": gpu, "torch": torch.__version__, "load_s": round(time.time() - t0),
            "class": type(model).__name__, "text_cfg_output_gate_type": getattr(tcfg, "output_gate_type", None),
            "layer_types_counts": {k: tcfg.layer_types.count(k) for k in set(tcfg.layer_types)}}
    print("[r1a] loaded", meta, flush=True)
    return core, tok, model, meta, t0


def _cost(t0, gpu):
    wall = time.time() - t0
    return {"wall_s": round(wall), "usd_est": round(wall * USD_PER_S["H200" if "H200" in gpu else "H100"], 2), "gpu": gpu}


@app.function(gpu=GPU, volumes=VOLS, timeout=3 * 3600)
def run():
    core, tok, model, meta, t0 = _boot()
    out = {"meta": meta, "r1a": core.run_r1a(model, tok, backends=("eager", "sdpa"), n_new=256,
                                              out_path=f"{OUT}/r1a_partial.json")}
    out["cost"] = _cost(t0, meta["gpu"])
    json.dump(out, open(f"{OUT}/r1a.json", "w"), indent=1)
    results.commit()
    return {b: v["verdict"] for b, v in out["r1a"].items()} | {"cost": out["cost"]}


@app.function(gpu=GPU, volumes=VOLS, timeout=3600)
def smoke():
    core, tok, model, meta, t0 = _boot()
    out = {"meta": meta, "smoke": core.run_smoke(model, tok)}
    out["cost"] = _cost(t0, meta["gpu"])
    json.dump(out, open(f"{OUT}/smoke.json", "w"), indent=1)
    results.commit()
    return out
