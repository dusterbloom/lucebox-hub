from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from online_qk import (
    aggregate_layer_scores,
    cold_block_geometry,
    evaluate_raw_capture,
    pooled_cold_keys,
    qk_block_scores,
    top_budget_indices,
)
from raw_dataset import load_raw_capture


class OnlineQkTest(unittest.TestCase):
    def test_cold_block_geometry_matches_oracle_window(self) -> None:
        first, count = cold_block_geometry(
            boundary_position=4096,
            block_size=64,
            sink_tokens=64,
            recent_tokens=1024,
        )
        self.assertEqual(first, 1)
        self.assertEqual(count, 47)

    def test_qk_scores_select_matching_cold_block(self) -> None:
        query = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
        keys = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.5, 0.5]],
                [[0.5, 0.5]],
            ]
        )
        pooled = pooled_cold_keys(
            keys,
            boundary_position=4,
            block_size=2,
            sink_tokens=0,
            recent_tokens=0,
        )
        self.assertEqual(tuple(pooled.shape), (2, 1, 2))
        scores = qk_block_scores(
            query,
            keys,
            boundary_position=4,
            block_size=2,
            sink_tokens=0,
            recent_tokens=0,
        )
        selected = top_budget_indices(scores, budget=1)
        self.assertEqual(selected.tolist(), [1])

    def test_aggregate_layer_scores_validates_shapes(self) -> None:
        merged = aggregate_layer_scores(
            [torch.tensor([0.0, 1.0]), torch.tensor([2.0, 0.5])],
            mode="max",
        )
        self.assertEqual(merged.tolist(), [2.0, 1.0])
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            aggregate_layer_scores([torch.ones(2), torch.ones(3)])

    def test_evaluate_raw_capture_beats_recency_on_matching_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            manifest = {
                "schema": "luce.lsa.qwen35.raw.v1",
                "endianness": "little",
                "model_fingerprint": "unit-test",
                "tokens": 6,
                "examples": 1,
                "hidden_size": 4,
                "kv_heads": 1,
                "query_heads": 1,
                "head_dim": 2,
                "block_size": 2,
                "lookahead_horizon": 2,
                "key_layer": 0,
                "oracle_layers": [3, 7, 11],
            }
            (raw / "manifest.json").write_text(json.dumps(manifest))
            np.asarray([2, 2, 2], dtype="<i4").tofile(raw / "chunk_tokens.i32")
            np.asarray([4], dtype="<i4").tofile(raw / "boundary_pos.i32")
            np.zeros((1, 4), dtype="<u2").tofile(raw / "query_hidden.bf16")
            key_pre = np.asarray(
                [[[1, 0]], [[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]], [[1, 0]]],
                dtype="<f2",
            )
            key_pre.tofile(raw / "key_pre.f16")
            query = np.asarray([[[[1, 0]], [[1, 0]]]], dtype="<f2")
            for layer in manifest["oracle_layers"]:
                key_pre.tofile(raw / f"layer_{layer:02d}.key_post.f16")
                query.tofile(raw / f"layer_{layer:02d}.query_post.f16")

            report = evaluate_raw_capture(
                load_raw_capture(raw),
                keep_ratios=[0.5],
                sink_tokens=0,
                recent_tokens=0,
            )
            metrics = report["metrics"]
            self.assertEqual(report["schema"], "luce.lsa.qwen35.online_qk_report.v1")
            self.assertAlmostEqual(metrics["qk_recall@0.500"], 1.0)
            self.assertLess(metrics["recent_recall@0.500"], metrics["qk_recall@0.500"])


if __name__ == "__main__":
    unittest.main()
