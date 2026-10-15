"""Tests for the A1 tool-parser program. python3 -m pytest -q lucebox/a1/test_tool_parser.py
Builds the binary on demand (make). The round trip needs the A1 data (trajectories, tasks, template): for every one of the
decision-point gold calls, render the assistant turn with the harness template and parse it back; names and arguments must
equal the source call (PREREG A1 section 1, item 1)."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tool_parse  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "execute_bash", "description": "run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"]}}}]


@pytest.fixture(scope="module", autouse=True)
def built():
    p = tool_parse.binary_path()
    if not os.path.exists(p):
        args = ["make", "-C", os.path.join(HERE, "tool_parser"), f"BUILD={os.path.dirname(p)}"]
        for env, var in (("A1_TP_SRC", "TP_SRC"), ("A1_JSON_INC", "JSON_INC")):
            if os.environ.get(env):
                args.append(f"{var}={os.environ[env]}")
        subprocess.run(args, check=True)
    assert os.path.exists(p)


def test_qwen_xml_call():
    r = tool_parse.parse("Let me look.\n\n<tool_call>\n<function=execute_bash>\n<parameter=command>\nls -la\n</parameter>\n"
                         "</function>\n</tool_call>", TOOLS)
    assert [c["name"] for c in r["tool_calls"]] == ["execute_bash"]
    assert json.loads(r["tool_calls"][0]["arguments"]) == {"command": "ls -la"}
    assert "tool_call" not in r["cleaned_text"] and r["cleaned_text"].strip() == "Let me look."


def test_plain_text_has_no_call():
    r = tool_parse.parse("src/app/main.py", TOOLS)
    assert r["tool_calls"] == [] and r["cleaned_text"] == "src/app/main.py"


def test_schema_coercion_and_message_shape():
    r = tool_parse.parse("<tool_call>\n<function=execute_bash>\n<parameter=command>\nsleep 1\n</parameter>\n"
                         "<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call>", TOOLS)
    assert json.loads(r["tool_calls"][0]["arguments"]) == {"command": "sleep 1", "timeout": 5}
    m = tool_parse.to_message(r)
    assert m["tool_calls"][0]["function"]["name"] == "execute_bash" and m["tool_calls"][0]["type"] == "function"


def test_batch_keeps_order_and_survives_odd_bytes():
    out = tool_parse.parse_batch([("a", TOOLS), ("bé\U0001f600", TOOLS), ("c", None)])
    assert [o["cleaned_text"] for o in out] == ["a", "bé\U0001f600", "c"]


def _have_data():
    import a1_core
    p = a1_core.default_paths()
    return all(os.path.exists(p[k]) for k in ("traj", "tasks", "template"))


@pytest.mark.skipif(not _have_data(), reason="A1 data not present")
def test_round_trip_all_gold_calls():
    import a1_core
    H = a1_core.Harness(a1_core.default_paths())
    trajs = H.load_trajs()
    cases, expect, skipped = [], [], 0
    for t in H.load_tasks():
        if t["kind"] != "decision":
            continue
        r = trajs[t["traj"]]
        j = t["cut_index"]
        gold = r["messages"][j] if j < len(r["messages"]) else {}
        if not gold.get("tool_calls"):
            skipped += 1  # the trajectory ends here or the gold turn made no call: nothing to round-trip
            continue
        cases.append((a1_core.generated_text(H, r["messages"][:j], gold, r["tools"]), r["tools"]))
        expect.append([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in gold["tool_calls"]])
    assert len(cases) + skipped == 1438 and len(cases) > 1000, (len(cases), skipped)
    print(f"round trip: {len(cases)} gold calls, {skipped} decision points without a gold call")
    bad = []
    for i, (res, exp) in enumerate(zip(tool_parse.parse_batch(cases), expect)):
        got = [(c["name"], json.loads(c["arguments"])) for c in res["tool_calls"]]
        if got != exp:
            bad.append((i, exp, got))
    assert not bad, f"{len(bad)} of {len(cases)} gold calls do not round-trip, first: {bad[0]}"
