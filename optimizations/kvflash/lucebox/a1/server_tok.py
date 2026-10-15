"""The server's tokenizer as a Python object: a persistent tokenize_cli process (tool_parser/tokenize_cli.cpp over the server's
tokenizer.cpp, unchanged) reading a GGUF's tokenizer metadata. encode(text) -> ids, decode(ids) -> text.

Why (A1 render parity): the server's hand-written pre-tokenizer splits a whitespace-only line ("\\n   \\n") into "\\n" + "   \\n",
the HF tokenizers pre-tokenizer keeps it as one piece, so the HF count is 1 to 44 tokens short of usage.prompt_tokens in 1399 of
3292 B0 requests. With this tokenizer every one of the 3292 counts matches (test_a1_render.py).
Binary: $A1_TOKENIZE_CLI, else $A1_TP_BUILD/tokenize_cli, else tool_parser/build/tokenize_cli next to this file.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def binary_path():
    if os.environ.get("A1_TOKENIZE_CLI"):
        return os.environ["A1_TOKENIZE_CLI"]
    return os.path.join(os.environ.get("A1_TP_BUILD") or os.path.join(HERE, "tool_parser", "build"), "tokenize_cli")


class ServerTokenizer:
    def __init__(self, gguf, binary=None):
        self.p = subprocess.Popen([binary or binary_path(), gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)

    def _call(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("tokenize_cli died")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def encode(self, text):
        return self._call({"text": text})["ids"]

    def decode(self, ids):
        return self._call({"decode": list(ids)})["text"]

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            self.p.wait(timeout=30)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
