"""Copy the transformers 5.x sources we need to the results volume (CPU only, cheap).
modal run am_qwen35/inspect_tf.py ; then modal volume get kvflash-am-results am_qwen35/src <local dir>
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "transformers==5.19.0", "huggingface_hub")
app = modal.App("kvflash-am-inspect", image=image)
results = modal.Volume.from_name("kvflash-am-results", create_if_missing=True)


@app.function(timeout=900, volumes={"/results": results})
def main():
    import os
    import shutil
    import transformers
    d = os.path.dirname(transformers.__file__)
    out = "/results/am_qwen35/src"
    os.makedirs(out, exist_ok=True)
    for rel in ["models/qwen3_5/modeling_qwen3_5.py", "models/qwen3_5/configuration_qwen3_5.py",
                "masking_utils.py", "cache_utils.py", "generation/utils.py"]:
        shutil.copy(f"{d}/{rel}", f"{out}/{rel.replace('/', '__')}")
    print("transformers", transformers.__version__, "->", os.listdir(out))
    results.commit()


@app.local_entrypoint()
def run():
    main.remote()
