from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "analyze_kimi_k3_p56_prefill.py"
SPEC = importlib.util.spec_from_file_location("p56", SCRIPT)
assert SPEC and SPEC.loader
P56 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P56)


class P56PrefillAnalysisTest(unittest.TestCase):
    def test_phase_split_and_macro_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "suite").mkdir()
            manifest = {
                "environment": {"KIMI_H16_REPOSITORY_COMMIT": "abc123"},
                "sequences": [
                    {"prompt_token_count": 4, "output_tokens": [1, 2]},
                    {"prompt_token_count": 2, "output_tokens": [3, 4]},
                ],
            }
            (root / "suite" / "suite-manifest.json").write_text(
                json.dumps(manifest))
            stage_fields = (
                "embedding_ms=1 dense_ms=2 routed_prep_ms=3 "
                "offload_prep_ms=0 experts_ms=4 join_ms=1 output_ms=1 other_ms=0")
            stderr = "\n".join([
                f"[kimi-k3-stage] position=0 tokens=2 total_ms=12 {stage_fields}",
                f"[kimi-k3-stage] position=2 tokens=2 total_ms=10 {stage_fields}",
                f"[kimi-k3-stage] position=4 tokens=1 total_ms=8 {stage_fields}",
                "[kimi-k3-p56] phase=prefill positions=4 forwards=2 seconds=2.0 "
                "positions-per-second=2 process-read-bytes=400 " + self._counters(40),
                "[kimi-k3-p56] phase=decode positions=1 forwards=1 seconds=1.0 "
                "positions-per-second=1 process-read-bytes=100 " + self._counters(10),
                f"[kimi-k3-stage] position=0 tokens=2 total_ms=9 {stage_fields}",
                f"[kimi-k3-stage] position=2 tokens=1 total_ms=7 {stage_fields}",
                "[kimi-k3-p56] phase=prefill positions=2 forwards=1 seconds=1.0 "
                "positions-per-second=2 process-read-bytes=200 " + self._counters(20),
                "[kimi-k3-p56] phase=decode positions=1 forwards=1 seconds=1.0 "
                "positions-per-second=1 process-read-bytes=100 " + self._counters(10),
            ]) + "\n"
            (root / "stderr.log").write_text(stderr)
            header = (
                "request_id\tprompt_id\tbase_pos\ttoken_index\tmodel_layer\t"
                "expert_id\tregion\tqtype\tprefix_depth\texact_fallback\t"
                "file_path\tfile_offset\tlogical_length\taligned_offset\t"
                "aligned_length\tdestination_kind\tdestination_offset\t"
                "explicit_read_bytes\n")
            rows = [
                self._trace_row(0, 0, 0, 1, 7, 0, 4096),
                self._trace_row(1, 0, 1, 1, 8, 4096, 4096),
                self._trace_row(2, 2, 0, 1, 7, 0, 4096),
                self._trace_row(3, 4, 0, 1, 9, 8192, 4096),
                self._trace_row(4, 0, 0, 1, 7, 0, 4096),
                self._trace_row(5, 2, 0, 1, 9, 8192, 4096),
            ]
            trace = root / "io_trace.tsv"
            trace.write_text(header + "".join(rows))

            census = P56.sum_census(P56.parse_lines(root / "stderr.log", P56.P56), "prefill")
            self.assertEqual(census["positions"], 6)
            self.assertEqual(census["forwards"], 3)
            self.assertEqual(census["physical-direct-read-bytes"], 60)
            stages = P56.stage_summary(
                P56.parse_lines(root / "stderr.log", P56.STAGE), manifest)
            self.assertEqual(stages["prefill"]["positions"], 6)
            self.assertEqual(stages["decode"]["positions"], 2)
            geometry = P56.prefill_io_geometry(trace, [4, 2])
            self.assertEqual(geometry["physical_read_events"], 4)
            self.assertEqual(geometry["compact_jobs"], 4)
            self.assertLess(
                geometry["macro_widths"]["4"]["deduplicated_physical_bytes"],
                geometry["macro_widths"]["1"]["deduplicated_physical_bytes"])

            output = root / "p56.json"
            subprocess.run([
                sys.executable, str(SCRIPT), "--root", str(root),
                "--output", str(output), "--io-trace", str(trace),
            ], check=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["scope"], "NON_SPECULATIVE_PREFILL_AND_AR_DECODE")
            self.assertEqual(report["prefill"]["positions"], 6)
            self.assertEqual(report["decode"]["positions"], 2)
            self.assertEqual(report["io_geometry"]["status"],
                             "TRACE_GEOMETRY_NOT_TIMED_REPLAY")

            manifest["draft_path"] = "/tmp/draft.gguf"
            (root / "suite" / "suite-manifest.json").write_text(
                json.dumps(manifest))
            rejected = subprocess.run([
                sys.executable, str(SCRIPT), "--root", str(root),
                "--output", str(output), "--io-trace", str(trace),
            ], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("requires speculative decoding off", rejected.stderr)

    @staticmethod
    def _counters(value: int) -> str:
        names = (
            "logical-provider-bytes explicit-read-bytes physical-direct-read-bytes "
            "direct-io-ns payload-h2d-bytes metadata-h2d-bytes compact-pack-ns "
            "expert-graph-ns expert-readback-ns compact-attempted compact-completed "
            "compact-fallbacks compact-invalid async-begins async-jobs async-h2d-calls "
            "async-h2d-bytes async-input-d2d-copies async-input-d2d-bytes "
            "async-graph-enqueues async-layer-flushes async-abort-syncs "
            "ordered-expert-d2d-copies ordered-expert-d2d-bytes ordered-join-launches "
            "ordered-output-d2d-copies ordered-output-d2d-bytes")
        return " ".join(f"{name}={value}" for name in names.split())

    @staticmethod
    def _trace_row(
        request: int, base: int, token: int, layer: int, expert: int,
        offset: int, length: int,
    ) -> str:
        return (
            f"{request}\t0\t{base}\t{token}\t{layer}\t{expert}\tgate\tIQ1_S\t1\t0\t"
            f"/tmp/sidecar\t{offset}\t{length}\t{offset}\t{length}\thost\t0\t{length}\n")


if __name__ == "__main__":
    unittest.main()
