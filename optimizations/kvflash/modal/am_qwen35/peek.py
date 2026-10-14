"""Read-only inspection of the untrusted AM repo inside the r1b image (never cloned locally).
  modal run am_qwen35/peek.py --cmd 'grep -n max_compactions -r . --include=*.py'"""
import os
import modal

SHA = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
HERE = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("git clone https://github.com/adamzweiger/compaction /repo", f"cd /repo && git checkout {SHA}")
    .add_local_dir(HERE, "/am", copy=True, ignore=["__pycache__", "*.pyc", "*.log", "recon"])
    .run_commands("cd /repo && git apply /am/patches/0001-hybrid-qwen3_5-cache-and-compaction.patch "
                  "/am/patches/0002-hybrid-onpolicy-and-self-study.patch /am/patches/0003-influence-hybrid.patch",
                  "cp -r /am/overlay/models/qwen3_5 /repo/models/qwen3_5")
)
app = modal.App("kvflash-am-peek", image=image)


@app.function(cpu=1, timeout=300)
def sh(cmd: str):
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/repo")
    return r.stdout + r.stderr


@app.local_entrypoint()
def main(cmd: str, out: str = ""):
    r = sh.remote(cmd)
    if out:
        open(out, "w").write(r)
    else:
        print(r)
