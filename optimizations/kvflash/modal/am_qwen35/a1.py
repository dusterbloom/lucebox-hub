"""A1 on Modal: a thin entry over lucebox/a1/a1_run.py (NOT RUN; the A1 machine is lucebox6, Amendment 2).
Stage the inputs on volume kvflash-am-results under /qwen3.8-27b/a1_in/ first: trajectories.jsonl, tasks_all60.jsonl,
qwen36_template.jinja, qwen3.8-tokenizer.gguf, bin/tool_parser_cli, bin/tokenize_cli (+ libggml-base.so beside it), agentic_eval/*.py.
  modal run am_qwen35/a1.py::stage1 --out a1_stage1 --extra "--control-n 8"
"""
import os
import subprocess
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
A1_DIR = os.path.join(HERE, "..", "..", "lucebox", "a1")
ROOT = "/results/qwen3.8-27b"
IN = f"{ROOT}/a1_in"
MODEL = "Qwen/Qwen3.8-27B"

import r2 as base  # noqa: E402  (same image, volumes and pinned compaction repo as R2)

image = base.image.add_local_dir(A1_DIR, "/a1", copy=True, ignore=["__pycache__", "*.pyc", "build"])
app = modal.App("kvflash-am-a1", image=image)


@app.function(gpu=base.GPU, volumes=base.VOLS, timeout=24 * 3600)
def stage1(out: str = "a1_stage1", extra: str = ""):
    env = {**os.environ, "PYTHONPATH": "/repo:/am:/a1", "A1_TRAJ": f"{IN}/trajectories.jsonl", "A1_TASKS": f"{IN}/tasks_all60.jsonl",
           "A1_TEMPLATE": f"{IN}/qwen36_template.jinja", "A1_TOKENIZER": f"{IN}/qwen3.8-tokenizer.gguf",
           "A1_HARNESS": f"{IN}/agentic_eval", "A1_TOOL_PARSER": f"{IN}/bin/tool_parser_cli",
           "A1_TOKENIZE_CLI": f"{IN}/bin/tokenize_cli"}
    r = subprocess.run([sys.executable, "/a1/a1_run.py", "--out", f"{ROOT}/{out}", "--model", MODEL, "--mem-cap-gib", "0"]
                       + extra.split(), cwd="/repo", env=env)
    base.results.commit()
    return r.returncode
