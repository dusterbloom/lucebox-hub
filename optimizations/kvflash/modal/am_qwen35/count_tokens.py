"""Token counts of the R1a candidate prompts (CPU, tokenizer only). Run before PREREG is finalised.
modal run am_qwen35/count_tokens.py
"""
import modal

SHA = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("transformers==5.19.0", "huggingface_hub", "jinja2")
    .run_commands("git clone https://github.com/adamzweiger/compaction /repo", f"cd /repo && git checkout {SHA}")
)
app = modal.App("kvflash-am-count", image=image)


@app.function(timeout=900)
def main():
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
    arts = {}
    for line in open("/repo/data/QuALITY.v1.0.1.htmlstripped.dev"):
        if line.strip():
            d = json.loads(line)
            arts.setdefault(str(d["article_id"]), d["article"].strip())
    print("n articles", len(arts))
    for aid in ["20041", "51350", "22876"]:
        a = arts[aid]
        msgs = [{"role": "user", "content": a + "\n\nQuestion: Summarize the story in five sentences."}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        print(aid, "words", len(a.split()), "chat tokens", n)
    both = arts["51350"] + "\n\n" + arts["22876"]
    msgs = [{"role": "user", "content": both + "\n\nQuestion: Summarize the two stories in five sentences."}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("51350+22876 chat tokens", len(tok(s, add_special_tokens=False)["input_ids"]))
    print(repr(s[:200])); print(repr(s[-200:]))


@app.local_entrypoint()
def run():
    main.remote()
