from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (Path(__file__).parents[2] / "scripts" /
          "build_kimi_k3_compact_schedule.py")


class CompactScheduleTest(unittest.TestCase):
    def test_exact_specs_depth_masks_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "environment": {"KIMI_H16_REPOSITORY_COMMIT": "fixture"},
                "sequences": [{"prompt_token_count": 2, "output_tokens": [1]}],
            }))
            trace = root / "io_trace.tsv"
            self._write_trace(trace)
            output = root / "schedule"
            subprocess.run([
                sys.executable, str(SCRIPT), "--trace", str(trace),
                "--manifest", str(manifest), "--output-dir", str(output),
                "--expect-jobs", "4", "--expect-positions", "2",
            ], check=True, capture_output=True, text=True)
            report = json.loads((output / "k3_compact_schedule.json").read_text())
            self.assertEqual(report["totals"]["jobs"], 4)
            self.assertEqual(report["totals"]["layer_groups"], 2)
            self.assertEqual(report["totals"]["depth_histogram"], {"2": 4})
            with (output / "k3_compact_jobs.tsv").open(newline="") as handle:
                jobs = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([int(row["expert_id"]) for row in jobs], [3, 7, 3, 7])
            self.assertEqual([int(row["job_rank"]) for row in jobs], [0, 1, 0, 1])
            self.assertEqual(jobs[0]["natural_ids"], "0,11")
            self.assertEqual(int(jobs[0]["natural_mask"]), (1 << 0) | (1 << 11))
            mapping = [int(value) for value in jobs[0]["natural_to_compact"].split(",")]
            self.assertEqual(mapping[0], 0)
            self.assertEqual(mapping[11], 1)
            self.assertEqual(mapping[1:11], [-1] * 10)
            self.assertEqual(report["release_benchmark_contract"]
                             ["roofline_gate_positions_per_second"]["minimum"], 20.0)

    def test_ambiguous_natural_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sequences": [{"prompt_token_count": 1, "output_tokens": [1]}],
            }))
            trace = root / "io_trace.tsv"
            header = self._header()
            rows = self._job_rows(0, 0, 1, 7, [5])
            trace.write_text(header + "".join(rows))
            failed = subprocess.run([
                sys.executable, str(SCRIPT), "--trace", str(trace),
                "--manifest", str(manifest), "--output-dir", str(root / "out"),
            ], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("natural IDs are ambiguous", failed.stderr)

    def test_summary_only_needs_no_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sequences": [{"prompt_token_count": 1, "output_tokens": [1]}],
            }))
            trace = root / "io_trace.tsv"
            trace.write_text(
                self._header() +
                "".join(self._job_rows(0, 0, 1, 3, [0, 11])) +
                "".join(self._job_rows(0, 0, 1, 7, [0, 11])))
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--trace", str(trace),
                "--manifest", str(manifest), "--summary-only",
                "--expect-jobs", "2", "--expect-positions", "1",
            ], check=True, capture_output=True, text=True)
            report = json.loads(completed.stdout)
            self.assertEqual(report["totals"]["jobs"], 2)
            self.assertIsNone(report["outputs"]["jobs"])

    def test_sidecar_exact_route_is_a_full_compact_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sequences": [{"prompt_token_count": 1, "output_tokens": [1]}],
            }))
            trace = root / "io_trace.tsv"
            trace.write_text(
                self._header() +
                "".join(self._job_rows(
                    0, 0, 1, 7, list(range(12)), exact=True)))
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--trace", str(trace),
                "--manifest", str(manifest), "--summary-only",
                "--expect-jobs", "1", "--expect-positions", "1",
            ], check=True, capture_output=True, text=True)
            report = json.loads(completed.stdout)
            self.assertEqual(report["totals"]["sidecar_exact_jobs"], 1)
            self.assertEqual(report["totals"]["depth_histogram"], {"12": 1})

    @classmethod
    def _write_trace(cls, path: Path) -> None:
        body: list[str] = []
        # Trace order deliberately differs from canonical expert order.
        for position in range(2):
            body.extend(cls._job_rows(position, 0, 1, 7, [0, 11]))
            body.extend(cls._job_rows(position, 0, 1, 3, [0, 11]))
        path.write_text(cls._header() + "".join(body))

    @staticmethod
    def _header() -> str:
        return (
            "request_id\tprompt_id\tbase_pos\ttoken_index\tmodel_layer\t"
            "expert_id\tregion\tqtype\tprefix_depth\texact_fallback\t"
            "file_path\tfile_offset\tlogical_length\taligned_offset\t"
            "aligned_length\tdestination_kind\tdestination_offset\t"
            "explicit_read_bytes\n")

    @staticmethod
    def _job_rows(
        base: int, token: int, layer: int, expert: int, naturals: list[int],
        exact: bool = False,
    ) -> list[str]:
        payload = 4096
        gate_bytes, up_bytes, down_bytes = 100, 100, 50
        record = gate_bytes + up_bytes + down_bytes
        depth = len(naturals)
        result: list[str] = []
        for slot, natural in enumerate(naturals):
            record_offset = payload + (expert * 12 + natural) * record
            components = (
                ("gate", "IQ2_XXS", gate_bytes, 32 + slot * gate_bytes),
                ("up", "IQ2_XXS", up_bytes,
                 32 + depth * gate_bytes + slot * up_bytes),
                ("down", "IQ1_S", down_bytes,
                 32 + depth * (gate_bytes + up_bytes) + slot * down_bytes),
            )
            component_offset = 0
            for region, qtype, length, destination in components:
                result.append(
                    f"0\t0\t{base}\t{token}\t{layer}\t{expert}\t{region}\t"
                    f"{qtype}\t{depth}\t{int(exact)}\t/tmp/layer1.sidecar\t"
                    f"{record_offset + component_offset}\t{length}\t0\t4096\t"
                    f"host-compact-slab\t{destination}\t0\n")
                component_offset += length
        return result


if __name__ == "__main__":
    unittest.main()
