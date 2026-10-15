"""Render parity (A1 task 4): the token count of the A1 render of every B0 request equals the engine's usage.prompt_tokens.
  python3 -m pytest -q lucebox/a1/test_a1_render.py        (A1_TP_BUILD points at the built tokenize_cli; data paths as a1_core)
The B0 FULL rows (committed copy evidence/agentic/B0_3090/FULL.jsonl, or $A1_B0_FULL, or ~/r1e/data/a1/B0_FULL.jsonl) carry the
server's prompt_tokens for all 3292 requests."""
import collections
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a1_core as core  # noqa: E402
import server_tok  # noqa: E402

PATHS = core.default_paths()
B0 = next((p for p in (os.environ.get("A1_B0_FULL"), os.path.expanduser("~/r1e/data/a1/B0_FULL.jsonl"),
                       os.path.join(HERE, "../../evidence/agentic/B0_3090/FULL.jsonl")) if p and os.path.exists(p)), None)
HAVE_GGUF = PATHS["tokenizer"].endswith(".gguf") and os.path.exists(PATHS["tokenizer"]) and os.path.exists(server_tok.binary_path())
HAVE_DATA = all(os.path.exists(PATHS[k]) for k in ("traj", "tasks", "template")) and B0 is not None


@pytest.fixture(scope="module")
def tok():
    if not HAVE_GGUF:
        pytest.skip("server tokenizer program or tokenizer GGUF missing")
    return server_tok.ServerTokenizer(PATHS["tokenizer"])


def test_whitespace_only_line_is_split_like_the_server(tok):
    # HF tokenizers keep "\n   \n" as one piece; the server's pre-tokenizer yields "\n" then "   \n" (the cause of the count gap)
    assert tok.encode("x\n   \ny") == [87, 198, 5690, 88]
    assert tok.decode([87, 198, 5690]) == "x\n   \n"


def test_special_tokens_and_roundtrip(tok):
    ids = tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\nhé 😀")
    assert ids[0] == 248045 and tok.decode(ids) == "<|im_start|>assistant\n<think>\n\n</think>\n\nhé 😀"


def test_default_paths_use_the_server_tokenizer_when_present(tmp_path):
    g = tmp_path / "r1e" / "data" / "a1"
    g.mkdir(parents=True)
    (g / "qwen3.8-tokenizer.gguf").write_bytes(b"x")
    assert core.default_paths(env={}, home=str(tmp_path))["tokenizer"] == str(g / "qwen3.8-tokenizer.gguf")
    assert core.default_paths(env={"A1_TOKENIZER": "/x/tokenizer.json"}, home=str(tmp_path))["tokenizer"] == "/x/tokenizer.json"


@pytest.mark.skipif(not (HAVE_GGUF and HAVE_DATA), reason="server tokenizer or A1 data missing")
def test_token_count_equals_prompt_tokens_for_all_3292_b0_requests():
    H = core.Harness(PATHS)
    assert isinstance(H.tk, server_tok.ServerTokenizer)
    trajs = H.load_trajs()
    server = {}
    for l in open(B0):
        r = json.loads(l)
        server[r["id"]] = r["prompt_tokens"]
    res, bad = collections.Counter(), []
    for t in H.load_tasks():
        body = H.run_arm.build_request(t, trajs[t["traj"]])
        n = len(H.ids(H.convert.render(body["messages"], body["tools"], add_generation_prompt=True)))
        res["exact" if n == server[t["id"]] else "off"] += 1
        if n != server[t["id"]]:
            bad.append((t["id"], n - server[t["id"]]))
    print(dict(res))
    assert res["exact"] == 3292 and not bad, bad[:5]
