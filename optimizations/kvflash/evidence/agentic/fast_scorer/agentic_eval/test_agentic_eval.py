"""Unit check: python3 -I test_agentic_eval.py   (unittest; no GPU, no server)
Round-trip: 3 real converted trajectories render without errors and the tool calls appear in Qwen format
(<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>) and parse back to the source calls."""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert  # noqa: E402
import probes  # noqa: E402

RAW = "/mnt/kimi-k3/datasets/swe-rebench-openhands-raw/trajectories.parquet"
CALL_RE = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n", re.S)


def parse_rendered_calls(text):
    calls = []
    text = text.split("<|im_end|>\n", 1)[1] if text.startswith("<|im_start|>system") else text  # skip tool preamble example
    for name, body in CALL_RE.findall(text):
        calls.append((name, {k: v for k, v in PARAM_RE.findall(body)}))
    return calls


def expected_calls(messages):
    out = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            args = json.loads(tc["function"]["arguments"])
            out.append((tc["function"]["name"],
                        {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in args.items()}))
    return out


class RoundTrip(unittest.TestCase):
    def test_synthetic(self):
        raw = [{"role": "system", "content": "sys", "name": None, "tool_call_id": None, "tool_calls": None},
               {"role": "user", "content": "fix it", "name": None, "tool_call_id": None, "tool_calls": None},
               {"role": "assistant", "content": "looking", "name": None, "tool_call_id": None, "tool_calls": [
                   {"id": "c1", "type": "function", "function": {"name": "execute_bash", "arguments": '{"command": "ls -la\\nx"}'}}]},
               {"role": "tool", "content": "out", "name": "execute_bash", "tool_call_id": "c1", "tool_calls": None}]
        ms = convert.convert_messages(raw)
        self.assertEqual([m["role"] for m in ms], ["system", "user", "assistant", "tool"])
        text = convert.render(ms, [], add_generation_prompt=True)
        self.assertEqual(parse_rendered_calls(text), [("execute_bash", {"command": "ls -la\nx"})])
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_bad_arguments_rejected(self):
        raw = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "x", "arguments": "{not json"}}]}]
        self.assertIsNone(convert.convert_messages(raw))

    def test_history_stable_when_user_turn_appended(self):
        ms = [{"role": "user", "content": "q"},
              {"role": "assistant", "content": "a", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c", "content": "r"}]
        base = convert.render(ms, [])
        ext = convert.render(ms + [{"role": "user", "content": "probe"}], [])
        self.assertTrue(ext.startswith(base))  # preserve_thinking keeps history rendering identical

    def test_three_real_trajectories(self):
        if not os.path.exists(RAW):
            self.skipTest("raw parquet not present")
        import pyarrow.parquet as pq
        t = pq.ParquetFile(RAW).read_row_group(0, columns=["trajectory", "tools"]).slice(0, 3).to_pylist()
        for d in t:
            ms = convert.convert_messages(d["trajectory"])
            self.assertIsNotNone(ms)
            tools = convert.convert_tools(d["tools"])
            text = convert.render(ms, tools, add_generation_prompt=True)
            exp = expected_calls(ms)
            self.assertGreater(len(exp), 10)
            self.assertEqual(parse_rendered_calls(text), exp)
            self.assertEqual(text.split("<|im_end|>\n", 1)[1].count("<tool_call>"), len(exp))
            self.assertEqual(text.count("<tool_response>"), sum(1 for m in ms if m["role"] == "tool"))
            tot, pos = convert.boundary_positions(ms, tools)
            self.assertGreater(tot, 10000)
            self.assertEqual(tot, convert.count_tokens(convert.render(ms, tools)))
            self.assertTrue(all(pos[j] <= tot for j in pos))


class Pure(unittest.TestCase):
    def test_normalize_and_score(self):
        self.assertTrue(probes.score_exact("  `src/a.py`. ", "src/a.py"))
        self.assertTrue(probes.score_exact('"foo   bar"', "foo bar"))
        self.assertFalse(probes.score_exact("Foo bar", "foo bar"))  # case-sensitive
        self.assertFalse(probes.score_exact("src/a.py and more", "src/a.py"))

    def test_call_keys(self):
        self.assertEqual(probes.call_key("execute_bash", {"command": "python -m pytest tests/x.py -v"})[1],
                         "execute_bash|python|pytest")
        n, k = probes.call_key("str_replace_editor", {"command": "view", "path": "/a/b.py"})
        self.assertEqual((n, k), ("str_replace_editor", "str_replace_editor|view|/a/b.py|"))
        self.assertEqual(probes.call_key("think", {"thought": "x"}), ("think", "think"))
        self.assertIsNone(probes.parse_response_call({"tool_calls": [{"function": {"name": "t", "arguments": "{bad"}}]}))
        self.assertIsNone(probes.parse_response_call({"content": "hi"}))

    def test_extract_only_here(self):
        def call(i, name, args):
            import json as j
            return {"role": "assistant", "content": "", "tool_calls": [{"id": i, "type": "function", "function": {"name": name, "arguments": j.dumps(args)}}]}
        view = "Here's the result of running `cat -n` on /w/m.py:\n     1\timport os\n     2\tdef zork_unique_fn(a):\n     3\t    return value_computed_here + 1\n"
        ms = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
              call("c0", "think", {"thought": "x"}), {"role": "tool", "tool_call_id": "c0", "content": "ok"},
              call("c1", "str_replace_editor", {"command": "view", "path": "/w/m.py"}),
              {"role": "tool", "tool_call_id": "c1", "content": view},
              call("c2", "execute_bash", {"command": "echo hi"}), {"role": "tool", "tool_call_id": "c2", "content": "hi\n"},
              call("c3", "execute_bash", {"command": "echo more"}), {"role": "tool", "tool_call_id": "c3", "content": "more"}]
        bnd = {"2": 100, "4": 5000, "6": 5500, "8": 6000}
        c = probes.extract_candidates(ms, [], bnd, 9000)
        self.assertIn(("defname", "zork_unique_fn"), {(x["type"], x["answer"]) for x in c})
        self.assertIn(("line", "return value_computed_here + 1"), {(x["type"], x["answer"]) for x in c})
        ms[6] = call("c2", "execute_bash", {"command": "echo zork_unique_fn"})  # now repeated elsewhere -> not 'only here'
        c = probes.extract_candidates(ms, [], bnd, 9000)
        self.assertNotIn("zork_unique_fn", {x["answer"] for x in c})


if __name__ == "__main__":
    unittest.main(verbosity=2)
