#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_kimi_k3_m1a_parity as analyzer
import run_kimi_k3_m1a_parity as runner


class M1aParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, value: bytes) -> Path:
        path = self.root / name
        path.write_bytes(value)
        return path

    def trace(self, response_id: str, stream: bool = False) -> dict:
        return {"schema": analyzer.TRACE_SCHEMA, "response_id": response_id,
                "stream": stream, "ok": True, "prompt_tokens": [1],
                "output_tokens": [2]}

    def test_json_response_id(self) -> None:
        self.assertEqual(runner.observed_response_id(
            b'{"id":"r1","choices":[]}', False), "r1")

    def test_sse_response_id(self) -> None:
        raw = (b'data: {"id":"r2","choices":[]}\n\n'
               b'data: {"id":"r2","choices":[]}\n\ndata: [DONE]\n\n')
        self.assertEqual(runner.observed_response_id(raw, True), "r2")

    def test_sse_requires_done(self) -> None:
        with self.assertRaisesRegex(ValueError, "one DONE"):
            runner.observed_response_id(b'data: {"id":"r"}\n', True)

    def test_sse_rejects_inconsistent_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            runner.observed_response_id(
                b'data: {"id":"a"}\ndata: {"id":"b"}\ndata: [DONE]\n', True)

    def test_atomic_json_is_fresh_only(self) -> None:
        path = self.root / "out.json"
        runner.atomic_json(path, {"a": 1})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})
        with self.assertRaises(FileExistsError):
            runner.atomic_json(path, {"a": 2})

    def test_loads_full_frozen_p55_shape(self) -> None:
        rows = [{"id": task, "text": task, "prompt_tokens": [1],
                 "output_tokens": [2]} for task in runner.TASK_IDS]
        value = {"schema": "kimi-k3-h16-suite-v1",
                 "provider": "all-layers-calibrated96",
                 "chat_template": "gguf-jinja-thinking-off",
                 "thinking_enabled": False, "max_context": 256,
                 "n_gen": 24, "paired": False, "sequences": rows}
        path = self.write("p55.json", json.dumps(value).encode())
        with mock.patch.object(runner, "REFERENCE_SHA256", runner.sha256(path)):
            observed = runner.load_references(path)
        self.assertEqual([row["id"] for row in observed], list(runner.TASK_IDS))
        self.assertEqual(len(observed), 12)

    def test_parse_traces_exact_set(self) -> None:
        path = self.write("stderr", ("noise\n" + analyzer.TRACE_PREFIX +
                          json.dumps(self.trace("a")) + "\n").encode())
        self.assertEqual(set(analyzer.parse_traces(path, {"a"})), {"a"})

    def test_parse_traces_rejects_duplicate(self) -> None:
        line = analyzer.TRACE_PREFIX + json.dumps(self.trace("a")) + "\n"
        path = self.write("stderr", (line + line).encode())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyzer.parse_traces(path, {"a"})

    def test_parse_traces_rejects_foreign(self) -> None:
        path = self.write("stderr", (analyzer.TRACE_PREFIX +
                          json.dumps(self.trace("foreign")) + "\n").encode())
        with self.assertRaisesRegex(ValueError, "foreign"):
            analyzer.parse_traces(path, {"a"})

    def test_parse_traces_rejects_missing(self) -> None:
        path = self.write("stderr", b"ordinary server output\n")
        with self.assertRaisesRegex(ValueError, "missing"):
            analyzer.parse_traces(path, {"a"})

    def test_parse_traces_rejects_malformed_prefixed_row(self) -> None:
        path = self.write("stderr", (analyzer.TRACE_PREFIX + "{bad\n").encode())
        with self.assertRaisesRegex(ValueError, "malformed"):
            analyzer.parse_traces(path, {"a"})

    def test_bound_file_rejects_symlink(self) -> None:
        target = self.write("target", b"x")
        link = self.root / "link"
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular"):
            analyzer.bound_file(str(link), runner.sha256(target))

    def test_launch_required_environment(self) -> None:
        identity = {"path": "/x", "sha256": "a" * 64}
        repo = "/repo"
        aux = {"path": "/aux/all_layers_calibrated96_manifest.json",
               "sha256": "b" * 64}
        sidecar = {"path": "/sidecars/all_layers_manifest.json",
                   "sha256": "c" * 64}
        policy = {"path": "/policy", "sha256":
                  "b73d203e9ae7d4f382baf3f30cf0381387596edda7a6fb16c6cf5c88626ad97a"}
        value = {"schema": runner.LAUNCH_SCHEMA, "repo": repo,
                 "bind": "127.0.0.1:8080", "binary": identity,
                 "cmake_cache": identity,
                 "chat_template": {"path": "/t",
                                   "sha256": runner.CHAT_TEMPLATE_SHA256},
                 "model": {"path": "/model/model-00001-of-00014.gguf",
                           "frozen_p55_manifest": "/manifest",
                           "frozen_p55_manifest_sha256": "d" * 64,
                           "shard_count": 14, "total_bytes": 585690490336,
                           "shards": [{"name": f"model-{i:05d}-of-00014.gguf",
                                       "size": 1} for i in range(1, 15)]},
                 "g2_contract": {"model_name": "kimi-k3", "max_context": 256,
                                 "prefix_cache_slots": 0,
                                 "prefill_cache_slots": 0,
                                 "disk_prefix_cache": "off"},
                 "server_argv": ["/x", "--target-device", "hip:1",
                                 "--moe-storage", "ssd"],
                 "source_hashes": {"share/model_cards/kimi-k3.json": "e" * 64},
                 "aux_manifest": aux, "sidecar_manifest": sidecar,
                 "h23_policy": policy,
                 "environment": {**runner.REQUIRED_ENVIRONMENT,
                    "DFLASH_KIMI_CALIBRATED96_AUX_DIR": "/aux",
                    "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR": "/sidecars",
                    "DFLASH_KIMI_H22_LAYER_BUDGETS": "/policy"}}
        path = self.write("launch.json", json.dumps(value).encode())
        self.assertEqual(runner.validate_launch(path)["model_name"], "kimi-k3")
        value["environment"]["DFLASH_KIMI_P52_PERSISTENT_ROUTED_JOIN"] = "1"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "contract"):
            runner.validate_launch(path)

    def test_trace_parity_rejects_wrong_stream_and_tokens(self) -> None:
        references = [{"id": task, "prompt": task,
                       "prompt_tokens": [1], "output_tokens": [2]}
                      for task in runner.TASK_IDS]
        rows = []
        traces = {}
        for index, (task, stream) in enumerate(
                (x for task in runner.TASK_IDS for x in ((task, False), (task, True)))):
            response_id = f"r{index}"
            rows.append({"task_id": task, "response_id": response_id})
            traces[response_id] = self.trace(response_id, stream)
        analyzer.validate_trace_parity(rows, references, traces)
        traces["r1"]["stream"] = False
        with self.assertRaisesRegex(ValueError, "parity"):
            analyzer.validate_trace_parity(rows, references, traces)
        traces["r1"]["stream"] = True
        traces["r4"]["output_tokens"] = [3]
        with self.assertRaisesRegex(ValueError, "parity"):
            analyzer.validate_trace_parity(rows, references, traces)


if __name__ == "__main__":
    unittest.main()
