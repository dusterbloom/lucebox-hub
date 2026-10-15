#!/usr/bin/env python3
"""Print the last k forced tokens of a LUCE_TEACHER_LOGITS dump as vocab strings (to pick LUCE_TEACHER_DROP: the
chat-template suffix tokens that must not be teacher-forced). usage: tokcheck.py <dump.bin> <gguf shard 1> [k=12]"""
import sys
import numpy as np
import gguf

raw = np.fromfile(sys.argv[1], dtype=np.uint8)
n_vocab = int(raw[:4].view(np.int32)[0])
rec = 4 + 2 * n_vocab
body = raw[4:]
n = len(body) // rec
toks = body[: n * rec].reshape(n, rec)[:, :4].copy().view(np.int32).reshape(n)
r = gguf.GGUFReader(sys.argv[2])
vocab = [bytes(t).decode("utf-8", "replace") if not isinstance(t, str) else t
         for t in r.fields["tokenizer.ggml.tokens"].contents()]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 12
print(f"{n} steps; last {k} forced tokens:")
for i in range(max(0, n - k), n):
    print(f"  step {i}: id {toks[i]} {vocab[toks[i]]!r}")
