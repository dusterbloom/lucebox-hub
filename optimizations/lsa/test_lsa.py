from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset import LsaExampleDataset, float_to_bf16_bits, load_shard
from evaluate import mass_recall, top_indices
from make_synthetic import make_shard
from model import CompactQwen35Encoder, focal_mass_loss
from oracle import cross_layer_oracle, layer_block_attention_mass
from raw_dataset import convert_raw_capture, load_raw_capture


class LsaToolingTest(unittest.TestCase):
    def test_shard_round_trip_and_training_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            make_shard(
                path,
                seed=3,
                examples=5,
                blocks=8,
                hidden_size=64,
                kv_heads=4,
                head_dim=16,
            )
            shard = load_shard(path)
            self.assertEqual(shard.example_count(), 5)
            dataset = LsaExampleDataset([path])
            example = dataset[0]
            model = CompactQwen35Encoder(
                hidden_size=64,
                rank=16,
                kv_heads=4,
                head_dim=16,
            )
            logits = model(example["hidden"], example["keys"])
            loss = focal_mass_loss(logits, example["target"])
            loss.backward()
            self.assertEqual(logits.shape, example["target"].shape)
            self.assertTrue(torch.isfinite(loss))

    def test_rejects_corrupt_label_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            make_shard(
                path,
                seed=4,
                examples=2,
                blocks=4,
                hidden_size=32,
                kv_heads=2,
                head_dim=8,
            )
            with np.load(path, allow_pickle=False) as source:
                arrays = {name: source[name].copy() for name in source.files}
            arrays["label_offsets"][-1] -= 1
            np.savez(path, **arrays)
            load_shard.cache_clear()
            with self.assertRaisesRegex(ValueError, "offsets"):
                load_shard(path)

    def test_qwen_parameter_budget(self) -> None:
        model = CompactQwen35Encoder()
        self.assertEqual(model.parameter_count(), 1_572_864)

    def test_mass_recall_and_top_budget(self) -> None:
        target = torch.tensor([0.1, 0.2, 0.7])
        selected = top_indices(torch.tensor([0.0, 0.5, 1.0]), 0.34)
        self.assertEqual(set(selected.tolist()), {1, 2})
        self.assertAlmostEqual(mass_recall(target, selected), 0.9)

    def test_attention_mass_uses_full_denominator_and_cold_bins(self) -> None:
        query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        key = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        mass = layer_block_attention_mass(
            query,
            key,
            torch.tensor([3]),
            torch.tensor([0, 1, 2, 3]),
            block_size=1,
            sink_tokens=1,
            recent_tokens=1,
            boundary_position=4,
        )
        self.assertEqual(tuple(mass.shape), (1, 2))
        self.assertGreater(float(mass[0, 1]), float(mass[0, 0]))
        self.assertLess(float(mass.sum()), 1.0)

    def test_cross_layer_voting_and_abstention(self) -> None:
        mass = torch.tensor(
            [
                [[0.01, 0.30, 0.02]],
                [[0.01, 0.40, 0.02]],
                [[0.01, 0.50, 0.02]],
                [[0.001, 0.001, 0.001]],
            ]
        )
        result = cross_layer_oracle(
            mass,
            top_p=0.6,
            minimum_cold_mass=0.02,
            minimum_layer_votes=3,
        )
        self.assertEqual(result.positive.tolist(), [False, True, False])
        self.assertGreater(float(result.label_mass[1]), 0)
        self.assertEqual(float(result.label_mass[0]), 0)

    def test_raw_capture_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            manifest = {
                "schema": "luce.lsa.qwen35.raw.v1",
                "endianness": "little",
                "model_fingerprint": "unit-test",
                "tokens": 4,
                "examples": 1,
                "hidden_size": 4,
                "kv_heads": 1,
                "query_heads": 1,
                "head_dim": 2,
                "block_size": 2,
                "lookahead_horizon": 2,
                "key_layer": 0,
                "oracle_layers": [0, 1, 2],
            }
            (raw / "manifest.json").write_text(json.dumps(manifest))
            np.asarray([2, 2], dtype="<i4").tofile(raw / "chunk_tokens.i32")
            np.asarray([2], dtype="<i4").tofile(raw / "boundary_pos.i32")
            float_to_bf16_bits(
                np.asarray([[1, 2, 3, 4]], dtype=np.float32)
            ).astype("<u2").tofile(raw / "query_hidden.bf16")
            keys = np.asarray(
                [[[1, 0]], [[1, 0]], [[1, 0]], [[1, 0]]],
                dtype="<f2",
            )
            keys.tofile(raw / "key_pre.f16")
            queries = np.asarray(
                [[[[1, 0]], [[1, 0]]]], dtype="<f2"
            )
            for layer in manifest["oracle_layers"]:
                keys.tofile(raw / f"layer_{layer:02d}.key_post.f16")
                queries.tofile(raw / f"layer_{layer:02d}.query_post.f16")

            output = Path(directory) / "converted.npz"
            convert_raw_capture(
                load_raw_capture(raw),
                output,
                sink_tokens=0,
                recent_tokens=0,
            )
            shard = load_shard(output)
            self.assertEqual(shard.metadata.model_fingerprint, "unit-test")
            self.assertEqual(shard.block_keys.shape, (2, 1, 2))
            self.assertEqual(shard.visible_blocks.tolist(), [1])
            self.assertGreater(float(shard.label_mass[0]), 0)


if __name__ == "__main__":
    unittest.main()
