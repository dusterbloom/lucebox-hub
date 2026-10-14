"""R1(e) smoke timing: synced prefill (warm, 2nd call), decode tok/s on the uncompacted and on the LOCAL-compacted cache."""
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import r1a_core
import r1b_core as core
import r2_core as r2
import r1e_run
from evaluation.datasets import load_dataset
from models.qwen3_5 import HFGenerateShim

path = os.path.expanduser("~/r1e/models/Qwen3.8-27B")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
r1a_core.set_attn(model, "sdpa")
S = core.Session(model, tok, HFGenerateShim(model, tok, seed=1))
R = r2.Runner(S, None, cell="FUNI")
art = load_dataset("quality")[int(sys.argv[1]) if len(sys.argv) > 1 else 0]


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t


def decode(triples, seq_len, n=128):
    cache = S._cache(triples, seq_len)
    ids = tok("Summarize the story.", return_tensors="pt", add_special_tokens=False)["input_ids"].to(S.dev)
    with torch.no_grad():
        o = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    S.eos = set()  # never stop early: measure n tokens
    _, dt = sync_time(lambda: S._greedy(cache, o.logits, n))
    return n / dt


for i in range(2):
    pf, dt = sync_time(lambda: S.prefill(art["article"]))
    print(f"[bench] prefill call {i}: {pf['seq_len']} tokens in {dt:.1f}s = {pf['seq_len'] / dt:.0f} tok/s", flush=True)
print(f"[bench] decode uncompacted: {decode(S.original_triples(pf), pf['seq_len']):.2f} tok/s", flush=True)
(tr, n, info), dt = sync_time(lambda: r1e_run.local_triples(R, S, art["article"]))
print(f"[bench] LOCAL build {dt:.1f}s {info}", flush=True)
print(f"[bench] decode LOCAL-compacted: {decode(tr, n):.2f} tok/s", flush=True)
print(f"[bench] peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
