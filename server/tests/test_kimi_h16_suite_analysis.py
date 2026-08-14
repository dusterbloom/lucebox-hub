#!/usr/bin/env python3
"""Synthetic regression for the sequence-aware Kimi H16 analyzer."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


LOGIT_HEADER = struct.Struct("<8sIIQQQII")
INTERVENTION_HEADER = struct.Struct("<8s6I4Q")


def write_logits(path: Path, values: np.ndarray) -> None:
    rows, vocabulary = values.shape
    path.write_bytes(
        LOGIT_HEADER.pack(
            b"K3LOG001", 1, vocabulary, rows, rows, 1, 0, 0
        ) + values.astype("<f4").tobytes()
    )


def write_interventions(path: Path) -> None:
    dimension = 3584
    top_k = 16
    records = 2
    record_bytes = 8 + top_k * 8 + 3 * dimension * 4
    chunks = [INTERVENTION_HEADER.pack(
        b"K3INT001", 1, 1, 96, dimension, top_k, 1,
        records, record_bytes, 0, 0,
    )]
    for position in range(records):
        exact = np.linspace(-1.0, 1.0, dimension, dtype="<f4")
        approximate = exact.copy()
        approximate[position] += np.float32(0.25 + position * 0.1)
        delta = approximate - exact
        chunks.extend([
            struct.pack("<ii", position, 0),
            np.arange(top_k, dtype="<i4").tobytes(),
            np.full(top_k, 1.0 / top_k, dtype="<f4").tobytes(),
            exact.tobytes(),
            approximate.tobytes(),
            delta.tobytes(),
        ])
    path.write_bytes(b"".join(chunks))


class KimiH16SuiteAnalysisTest(unittest.TestCase):
    def test_sequence_join_and_exact_reference(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        analyzer = repository / "scripts/analyze_kimi_h16_suite.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            teacher = np.asarray([
                [2.0, 1.0, 0.0, -1.0],
                [0.0, 1.0, 2.0, -1.0],
            ], dtype=np.float32)
            intervention_logits = teacher.copy()
            intervention_logits[0, 0] -= 0.2
            intervention_logits[0, 1] += 0.2
            intervention_logits[1, 2] -= 0.3
            intervention_logits[1, 3] += 0.3
            write_logits(reference / "short.teacher.logits.f32", teacher)
            write_logits(candidate / "short.teacher.logits.f32", teacher)
            write_logits(
                candidate / "short.candidate.logits.f32",
                intervention_logits,
            )
            sequence = {
                "id": "short",
                "split": "validation",
                "text": "synthetic",
                "prompt_tokens": [0, 1],
                "prompt_token_count": 2,
                "output_tokens": [2],
                "teacher_logits": "short.teacher.logits.f32",
                "candidate_logits": "",
                "intervention_record_start": 0,
                "intervention_record_count": 0,
            }
            reference_manifest = {
                "schema": "kimi-k3-h16-suite-v1",
                "paired": False,
                "provider": "exact",
                "sequences": [sequence],
            }
            (reference / "suite-manifest.json").write_text(
                json.dumps(reference_manifest)
            )
            paired_sequence = dict(sequence)
            paired_sequence.update({
                "candidate_logits": "short.candidate.logits.f32",
                "intervention_record_count": 2,
            })
            paired_manifest = {
                "schema": "kimi-k3-h16-suite-v1",
                "paired": True,
                "provider": "slabs",
                "sequences": [paired_sequence],
            }
            (candidate / "suite-manifest.json").write_text(
                json.dumps(paired_manifest)
            )
            write_interventions(candidate / "interventions.f32")
            output_json = root / "analysis.json"
            output_csv = root / "rows.csv"
            subprocess.run([
                sys.executable,
                str(analyzer),
                str(candidate),
                "--reference-suite", str(reference),
                "--output-json", str(output_json),
                "--output-csv", str(output_csv),
            ], check=True, capture_output=True, text=True)
            result = json.loads(output_json.read_text())
            self.assertTrue(result["exact_reference"]["byte_identical"])
            self.assertEqual(result["intervention_header"]["records"], 2)
            self.assertGreater(
                result["terminal_teacher_to_intervention_kl"]["mean"], 0.0
            )
            self.assertEqual(result["by_split"]["validation"]["rows"], 2)
            self.assertEqual(len(output_csv.read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
