"""Reproduce Attention Matching KV compaction (arXiv 2602.16284) on Modal, unmodified paper code.

Usage (from repo worktree root):
  modal run optimizations/kvflash/modal/am_repro.py::smoke
  modal run --detach optimizations/kvflash/modal/am_repro.py::aime     # 3 compaction runs + baselines
  modal run --detach optimizations/kvflash/modal/am_repro.py::quality_run
Results: Volume kvflash-am-results under /qwen3-4b/<run-name>/ (JSON + stdout.log).
"""
import subprocess
import sys
import time

import modal

SHA = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
MODEL = "Qwen/Qwen3-4B"
BUDGET = "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "vllm==0.11.0",
                 "accelerate==1.12.0", "datasets==4.4.1")
    .run_commands(
        "git clone https://github.com/adamzweiger/compaction /repo",
        f"cd /repo && git checkout {SHA}",
    )
)
app = modal.App("kvflash-am-repro", image=image)
results = modal.Volume.from_name("kvflash-am-results", create_if_missing=True)
hfcache = modal.Volume.from_name("kvflash-am-hfcache", create_if_missing=True)
VOLS = {"/results": results, "/root/.cache/huggingface": hfcache}
GPU = ["H100", "A100-80GB"]


def _run(name: str, argv: list[str], log_dir: bool = True) -> dict:
    import os
    out = f"/results/qwen3-4b/{name}"
    os.makedirs(out, exist_ok=True)
    cmd = ["python", "-u", "-m", *argv]
    if log_dir:
        cmd += ["--log-dir", out, "--name", name]
    t0 = time.time()
    with open(f"{out}/stdout.log", "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n")
        lf.write(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout)
        lf.flush()
        p = subprocess.Popen(cmd, cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        n = 0
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
            n += 1
            if n % 20 == 0:
                lf.flush()
        rc = p.wait()
        lf.write(f"\nEXIT {rc} wall {time.time()-t0:.0f}s\n")
    results.commit()
    hfcache.commit()
    return {"name": name, "rc": rc, "wall_s": round(time.time() - t0)}


@app.function(gpu=GPU, volumes=VOLS, timeout=3600)
def smoke():
    return _run("smoke", ["examples.qa_demo", "--model", MODEL, "--target-size", "0.1"], log_dir=False)


@app.function(gpu=GPU, volumes=VOLS, timeout=6 * 3600, max_containers=6)
def reasoning(name: str, mode: str, max_seq_len: int, max_compactions: int = 0,
              start_problem: int = 0, n_problems: int = -1, seed: int = 1, target: float = 0.2):
    argv = ["evaluation.run_reasoning_evaluation", "--mode", mode, "--max-seq-len", str(max_seq_len),
            "--seed", str(seed), "--model-name", MODEL, "--start-problem", str(start_problem),
            "--n-problems", str(n_problems)]
    if mode == "compaction":
        argv += ["--target-size", str(target), "--max-compactions", str(max_compactions),
                 "--method", "AM-HighestAttnKeys-basic", "--query-config", "repeat",
                 "--algorithm-config", "default"]
    return _run(name, argv)


@app.function(gpu=GPU, volumes=VOLS, timeout=8 * 3600, max_containers=4)
def quality(name: str, target: float, budget: bool, query_config: str, n_articles: int = 10,
            methods: list[str] = ("original", "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy"),
            algorithm_config: str = "best", start_article: int = 0, compute_stats: int = 1):
    argv = ["evaluation.run_qa_evaluation", "--dataset-name", "quality", "--n-articles", str(n_articles),
            "--start-article", str(start_article), "--compute-stats", str(compute_stats), "--target-size", str(target),
            "--methods", *methods,
            "--query-config", query_config, "--algorithm-config", algorithm_config, "--model-name", MODEL]
    if budget:
        argv += ["--precomputed-budget-path", BUDGET, "--max-ratio-per-head", "0.75"]
    return _run(name, argv)


H100_USD_PER_S = 0.001097  # Modal list price, approx $3.95/h


@app.function(volumes=VOLS, timeout=1800)
def manifests():
    """Write manifest.json into every run dir on the results volume (provenance)."""
    import glob, json, os
    from huggingface_hub import HfApi
    api = HfApi()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    model_rev = api.model_info(MODEL).sha
    ds_rev = {c: api.dataset_info("opencompass/AIME2025").sha for c in ("AIME2025-I", "AIME2025-II")}
    results.reload()
    for d in sorted(glob.glob("/results/qwen3-4b/*/")):
        log = os.path.join(d, "stdout.log")
        if not os.path.exists(log):
            continue
        txt = open(log).read().splitlines()
        cmd = txt[0].removeprefix("CMD: ") if txt else ""
        gpu = txt[1] if len(txt) > 1 else ""
        wall = None
        for ln in reversed(txt):
            if ln.startswith("EXIT"):
                wall = int(ln.split("wall")[1].strip().rstrip("s"))
                break
        js = sorted(glob.glob(os.path.join(d, "*.json")))
        js = [j for j in js if not j.endswith("manifest.json")]
        cfg = {}
        if js:
            cfg = json.load(open(js[0])).get("config", {})
        man = {
            "repo": "https://github.com/adamzweiger/compaction", "repo_sha": SHA,
            "pip_freeze": freeze, "model": MODEL, "model_hf_revision": model_rev,
            "datasets": {"aime2025": {"hf": "opencompass/AIME2025", "split": "test", "revisions": ds_rev},
                         "quality": "repo-bundled data/QuALITY.v1.0.1.htmlstripped.dev"},
            "seed": cfg.get("seed"), "gpu": gpu, "command": cmd,
            "result_files": [os.path.basename(j) for j in js],
            "start_time": (cfg and json.load(open(js[0])).get("timestamp")),
            "wall_s": wall,
            "est_cost_usd": None if wall is None else round(wall * H100_USD_PER_S, 2),
        }
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=2)
    results.commit()


@app.local_entrypoint()
def aime(which: str = "all", split: int = 1):
    """which: all|comp|base. split: number of parallel problem-shards per run (30 problems)."""
    jobs = []
    # compaction: scripts/qwen-reasoning.sh (max_seq_len 2048, t0.2, c in 1,3,8)
    if which in ("all", "comp"):
        for c in (1, 3, 8):
            jobs.append(("compact_2048_t0.2_c%d_s1" % c, "compaction", 2048, c))
    # baselines with comparable total max_tokens (2048 + c*~1640): 4096, 8192, 16384; plus 2048
    if which in ("all", "base"):
        for L in (2048, 4096, 8192, 16384):
            jobs.append(("baseline_%d_s1" % L, "baseline", L, 0))
    calls = []
    for n, m, L, c in jobs:
        step = 30 // split
        for i in range(split):
            nm = n if split == 1 else f"{n}_p{i}"
            calls.append(reasoning.spawn(nm, m, L, c, i * step, -1 if split == 1 else step))
    for k in calls:
        print(k.get())


@app.local_entrypoint()
def quality_run(query_config: str = "ss-plus-repeat"):
    calls = []
    for t in (0.05, 0.1):
        for b in (False, True):
            calls.append(quality.spawn(f"quality_t{t}_{'optbud' if b else 'uniform'}_{query_config}", t, b, query_config))
    for k in calls:
        print(k.get())


@app.local_entrypoint()
def quality_ctrl(query_config: str = "ss-plus-repeat"):
    """Negative controls (repo's naive compactors) at the same target sizes, uniform budgets."""
    calls = []
    for t in (0.05, 0.1):
        calls.append(quality.spawn(
            f"quality_t{t}_uniform_ctrl_{query_config}", t, False, query_config, 10,
            ["random_subset_keys_nnls2_-3_3_lsq_on-policy", "truncate_nnls2_-3_3_lsq_on-policy"],
            "key-selection"))
    for k in calls:
        print(k.get())


@app.local_entrypoint()
def aime_table9(seeds: str = "1,2,3,4", split: int = 5, skip_seed1_baselines: bool = True):
    """Paper Table 9: phys 2048, 50% per compaction, <=2 and <=6 compactions; baselines 2048/4096/8192."""
    jobs = []
    for s in [int(x) for x in seeds.split(",")]:
        for c in (2, 6):
            jobs.append((f"t9_compact_2048_t0.5_c{c}_s{s}", "compaction", 2048, c, s, 0.5))
        if not (s == 1 and skip_seed1_baselines):
            for L in (2048, 4096, 8192):
                jobs.append((f"baseline_{L}_s{s}", "baseline", L, 0, s, 0.2))
    calls = []
    step = 30 // split
    for n, m, L, c, s, t in jobs:
        for i in range(split):
            calls.append(reasoning.spawn(f"{n}_p{i}", m, L, c, i * step, step, s, t))
    for k in calls:
        print(k.get())


HA = "highest_attn_keys_rms_nnls2_-3_3_lsq"
OMP = "omp_nnls0_-inf_7_drop-7_lsq"
OMPF = "omp_nnls0_-inf_7_drop-7_lsq_k4int2"


@app.local_entrypoint()
def pilot():
    """Timing-only pilot on selection articles 0-1 (PREREG Amendment 1)."""
    ms = [m + s for m in (HA, OMP, OMPF) for s in ("", "_on-policy")]
    calls = [quality.spawn(f"pilot_{qc}", 0.1, False, qc, 2, ["original", *ms], "best", 0, 0)
             for qc in ("context-prefill", "repeat", "ss-plus-repeat")]
    for k in calls:
        print(k.get())
