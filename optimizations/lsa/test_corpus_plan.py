from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from corpus_plan import (
    BLOCK_SIZE,
    LOOKAHEAD_HORIZON,
    aligned_boundary_positions,
    build_corpus_plan,
    load_corpus_plan,
    write_boundary_files,
    write_corpus_plan,
)


class CorpusPlanTest(unittest.TestCase):
    def test_pilot_plan_matches_expected_budget(self) -> None:
        plan = build_corpus_plan("pilot")
        self.assertEqual(plan.total_documents, 64)
        self.assertEqual(plan.total_source_tokens, 2_621_440)
        self.assertEqual(plan.total_boundaries, 512)
        splits = {
            split: sum(1 for document in plan.documents if document.split == split)
            for split in ("train", "validation", "test")
        }
        self.assertEqual(splits, {"train": 48, "validation": 8, "test": 8})
        counts = {
            source["name"]: source["document_count"]
            for source in plan.source_mix
        }
        self.assertEqual(
            counts,
            {
                "agent_coding_trace": 26,
                "synthetic_retrieval": 19,
                "technical_document": 13,
                "local_or_no_context_control": 6,
            },
        )

    def test_scaled_plan_matches_expected_budget(self) -> None:
        plan = build_corpus_plan("scaled")
        self.assertEqual(plan.total_documents, 384)
        self.assertEqual(plan.total_source_tokens, 16_777_216)
        self.assertEqual(plan.total_boundaries, 3_072)

    def test_boundaries_are_aligned_and_leave_future_window(self) -> None:
        boundaries = aligned_boundary_positions(16_384)
        self.assertEqual(len(boundaries), 8)
        self.assertEqual(boundaries, tuple(sorted(boundaries)))
        for position in boundaries:
            self.assertEqual(position % BLOCK_SIZE, 0)
            self.assertLessEqual(position + LOOKAHEAD_HORIZON, 16_384)

    def test_manifest_round_trip_and_boundary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pilot.json"
            plan = build_corpus_plan("pilot")
            write_corpus_plan(path, plan)
            payload = load_corpus_plan(path)
            self.assertEqual(payload["schema"], "luce.lsa.qwen35.corpus_plan.v1")
            self.assertEqual(payload["summary"]["total_documents"], 64)

            write_boundary_files(root, plan)
            first = plan.documents[0]
            contents = (root / first.boundary_path).read_text().splitlines()
            self.assertEqual(
                [int(value) for value in contents],
                list(first.boundary_positions),
            )

    def test_rejects_too_short_document(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            aligned_boundary_positions(128, count=8)


if __name__ == "__main__":
    unittest.main()
