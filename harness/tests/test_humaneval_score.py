#!/usr/bin/env python3
"""Deterministic unit tests for isolated HumanEval report scoring."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client_test_runner as harness


class HumanEvalScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prompts = self.root / "he.jsonl"
        self.prompts.write_text(json.dumps({
            "id": "he_01", "entry_point": "answer", "gold_test": "",
        }) + "\n", encoding="utf-8")
        self.report = self.root / "generation.json"
        self.report.write_text(json.dumps({"cases": [{
            "id": "he_01", "text": "def answer():\n    return 42\n",
        }]}), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_score_records_require_exact_fixture_ids(self) -> None:
        rows = harness._load_score_records(self.report.read_bytes(), self.prompts.read_bytes())
        self.assertEqual(rows[0]["id"], "he_01")
        self.report.write_text(json.dumps({"cases": []}), encoding="utf-8")
        with self.assertRaisesRegex(harness.HarnessError, "case IDs differ"):
            harness._load_score_records(self.report.read_bytes(), self.prompts.read_bytes())

    def test_score_records_reject_malformed_generated_case(self) -> None:
        self.report.write_text(json.dumps({"cases": [{"id": "he_01"}]}),
                               encoding="utf-8")
        with self.assertRaisesRegex(harness.HarnessError, "malformed generation report case"):
            harness._load_score_records(self.report.read_bytes(), self.prompts.read_bytes())

    def test_generated_source_is_stdin_to_bwrap_only(self) -> None:
        with mock.patch.object(harness.shutil, "which", return_value="/usr/bin/bwrap"):
            command = harness._bwrap_command("bwrap", 10, 256, 32)
        self.assertEqual(command[0], "bwrap")
        self.assertIn("--unshare-all", command)
        self.assertIn("--clearenv", command)
        runner = command[command.index("-c") + 1]
        self.assertIn("os.dup2(null_fd, 1)", runner)
        self.assertIn("for fd in range(4, 256)", runner)
        self.assertIn("A<->B pipes", runner)
        self.assertIn("B<->C pipes never carry it", runner)
        self.assertIn('os.execve("/usr/bin/python3"', runner)
        self.assertIn('trusted["check"](candidate_proxy)', runner)
        self.assertIn('report("passed" if', runner)
        self.assertNotIn("HE_SANDBOX_SUCCESS_EXIT", runner)
        self.assertNotIn("print('generated')", command)

    def test_forged_output_and_os_exit_have_no_trusted_completion(self) -> None:
        for forged_source in (
                "import os; os.write(3, b'done'); os._exit(0)",
                "import os; os._exit(0)", "import os; os._exit(1)",
                "import os; os._exit(73)",
                "import os, sys; os.write(3, b'forged'); os._exit(0)"):
            self.assertNotIn(forged_source, harness._bwrap_command("bwrap", 10, 256, 32))
        runner = harness._bwrap_command("bwrap", 10, 256, 32)[-4]
        self.assertIn('raise RuntimeError("candidate exited")', runner)
        self.assertIn("len(c_buffer) >= 65536", runner)
        self.assertIn('"op":"stop_ack"', runner)
        self.assertIn('trusted["check"](candidate_proxy)', runner)
        self.assertEqual(
            harness._verify_sandbox_status(0, b'{"ok":true}\n')[0], False)
        self.assertEqual(harness._verify_sandbox_status(0, b"")[0], False)
        self.assertEqual(harness._verify_sandbox_status(1, b"")[0], False)
        good = json.dumps({"schema": harness.HE_SANDBOX_STATUS_SCHEMA,
                           "status": "passed"}).encode()
        self.assertTrue(harness._verify_sandbox_status(0, good)[0])

    def test_rpc_challenge_rejects_prefill_and_accepts_current_reply(self) -> None:
        current = "fresh-supervisor-challenge"
        candidate_nonce = "broker-only-challenge"
        prefilled = {"op": "result", "nonce": "stale", "ok": True, "value": True}
        with self.assertRaisesRegex(harness.HarnessError, "authentication"):
            harness._validate_rpc_reply(prefilled, current, "result")
        # A candidate that learns its B<->C nonce through sys._getframe() or
        # writes directly to fd3 still cannot satisfy A's distinct nonce.
        with self.assertRaisesRegex(harness.HarnessError, "authentication"):
            harness._validate_rpc_reply(
                {"op": "result", "nonce": candidate_nonce, "ok": True, "value": True},
                current, "result")
        with self.assertRaisesRegex(harness.HarnessError, "authentication"):
            harness._validate_rpc_reply({"op": "stop_ack", "nonce": current}, current, "result")
        self.assertEqual(
            harness._validate_rpc_reply(
                {"op": "result", "nonce": current, "ok": True, "value": True},
                current, "result")["value"], True)

    def test_missing_bwrap_fails_closed(self) -> None:
        with mock.patch.object(harness.shutil, "which", return_value=None):
            ok, detail = harness._run_he_in_bwrap("raise SystemExit", "answer", "")
        self.assertFalse(ok)
        self.assertIn("bwrap is unavailable", detail)

    def test_timeout_and_resource_verdict_fail(self) -> None:
        for status in ("failed:timeout", "failed:MemoryError"):
            output = json.dumps({"schema": harness.HE_SANDBOX_STATUS_SCHEMA,
                                 "status": status}).encode()
            ok, detail = harness._verify_sandbox_status(0, output)
            self.assertFalse(ok)
            self.assertEqual(detail, "wrong: sandbox test did not complete")

    def test_input_size_limits_fail_before_sandbox_launch(self) -> None:
        oversized = "x" * (harness.HE_SANDBOX_PAYLOAD_MAX_BYTES + 1)
        ok, detail = harness._run_he_in_bwrap(oversized, "answer", "")
        self.assertFalse(ok)
        self.assertIn("payload exceeds", detail)
        args = Namespace(generation_report=self.report, prompts=self.prompts,
                         bwrap="bwrap", timeout_seconds=10, memory_mib=256,
                         pid_limit=32, json_out=self.root / "score.json")
        with mock.patch.object(harness, "HE_SCORE_REPORT_MAX_BYTES", 1):
            with self.assertRaisesRegex(harness.HarnessError, "generation report exceeds"):
                harness.cmd_score_he(args)

    def test_legacy_bench_scorer_remains_direct_and_explicitly_unsafe(self) -> None:
        completed = mock.Mock(returncode=0, stderr="")
        with mock.patch.object(harness.subprocess, "run", return_value=completed) as run:
            self.assertTrue(harness._score_he_response(
                "def answer():\n    return 42", "answer", "def check(fn): assert fn() == 42")[0])
        self.assertEqual(run.call_args.args[0][0:2], ["python3", "-c"])

    def test_score_command_reports_pass_and_failure(self) -> None:
        output = self.root / "score.json"
        args = Namespace(generation_report=self.report, prompts=self.prompts,
                         bwrap="bwrap", timeout_seconds=10, memory_mib=256,
                         pid_limit=32, json_out=output)
        with mock.patch.object(harness, "_score_he_response_isolated",
                               return_value=(True, "correct: tests passed")):
            self.assertEqual(harness.cmd_score_he(args), 0)
        payload = json.loads(output.read_text())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"], {"passed": 1, "total": 1})

        with mock.patch.object(harness, "_score_he_response_isolated",
                               return_value=(False, "wrong: AssertionError")):
            self.assertEqual(harness.cmd_score_he(args), 1)
        payload = json.loads(output.read_text())
        self.assertFalse(payload["ok"])

    def test_score_command_hashes_input_snapshots_and_publishes_atomically(self) -> None:
        original_report = self.report.read_bytes()
        original_prompts = self.prompts.read_bytes()
        output = self.root / "score.json"
        args = Namespace(generation_report=self.report, prompts=self.prompts,
                         bwrap="bwrap", timeout_seconds=10, memory_mib=256,
                         pid_limit=32, json_out=output)

        def mutate_after_snapshot(*_args, **_kwargs):
            self.report.write_text('{"cases":[]}', encoding="utf-8")
            return True, "correct: tests passed"

        with mock.patch.object(harness, "_score_he_response_isolated",
                               side_effect=mutate_after_snapshot):
            self.assertEqual(harness.cmd_score_he(args), 0)
        payload = json.loads(output.read_text())
        self.assertEqual(payload["generation_report_sha256"], hashlib.sha256(original_report).hexdigest())
        self.assertEqual(payload["prompts_sha256"], hashlib.sha256(original_prompts).hexdigest())

        output.write_text("old-score\n", encoding="utf-8")
        self.report.write_bytes(original_report)
        with mock.patch.object(harness, "_score_he_response_isolated",
                               return_value=(True, "correct: tests passed")), \
                mock.patch.object(harness.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                harness.cmd_score_he(args)
        self.assertEqual(output.read_text(encoding="utf-8"), "old-score\n")

    def test_command_has_all_required_limits(self) -> None:
        with mock.patch.object(harness.shutil, "which", return_value="/usr/bin/bwrap"):
            command = harness._bwrap_command("bwrap", 7, 128, 9)
        self.assertIn("--unshare-all", command)
        self.assertIn("--cap-drop", command)
        self.assertIn("--tmpfs", command)
        self.assertIn("--ro-bind", command)
        self.assertIn("7", command)
        self.assertIn(str(128 * 1024 * 1024), command)
        self.assertIn("9", command)


if __name__ == "__main__":
    unittest.main()
