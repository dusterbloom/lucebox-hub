import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "materialize_kimi_k3_slab_bank.py"
SPEC = importlib.util.spec_from_file_location("materialize_kimi_k3_slab_bank", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StreamedSlabMaterializerTest(unittest.TestCase):
    def test_reference_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "references"
            references.mkdir()
            for layer in range(1, 93):
                shard = f"Kimi-K3-UD-IQ1_S-{2 + (layer // 8):05d}-of-00014.gguf"
                record = {
                    "model_layer": layer,
                    "ordering": "natural neuron order (all-192 numerical control only)",
                    "source_shards": {
                        component: {"path": f"/source/{shard}", "bytes": 123456}
                        for component in ("gate", "up", "down")
                    },
                    "output_bytes": 1000 + layer,
                    "output_sha256": f"{layer:064x}",
                }
                (references / f"kimi_layer{layer:02d}_natural_slabs.json").write_text(
                    json.dumps(record)
                )
            layers, sources = MODULE.load_reference(references)
            self.assertEqual(len(layers), 92)
            self.assertEqual(layers[0].layer, 1)
            self.assertEqual(layers[-1].layer, 92)
            self.assertGreater(len(sources), 1)

    def test_refuses_unmarked_nonempty_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = root / "deployment"
            deployment.mkdir()
            (deployment / "foreign.txt").write_text("owned elsewhere")
            with self.assertRaises(FileExistsError):
                MODULE.ensure_root(deployment, root)

    def test_punch_ranges_reclaims_and_zeroes(self) -> None:
        if not hasattr(os, "SEEK_DATA"):
            self.skipTest("filesystem hole queries are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"A" * (256 << 10))
            before = source.stat().st_blocks * 512
            ranges = [
                MODULE.TensorRange(
                    component="gate",
                    tensor_name="blk.1.ffn_gate_exps.weight",
                    source=source,
                    offset=16 << 10,
                    length=64 << 10,
                ),
                MODULE.TensorRange(
                    component="up",
                    tensor_name="blk.1.ffn_up_exps.weight",
                    source=source,
                    offset=128 << 10,
                    length=64 << 10,
                ),
            ]
            measured_before, measured_after = MODULE.punch_ranges(ranges)
            self.assertEqual(measured_before, before)
            self.assertLess(measured_after, measured_before)
            repeated_before, repeated_after = MODULE.punch_ranges(ranges)
            self.assertEqual(repeated_before, measured_after)
            self.assertEqual(repeated_after, measured_after)
            with source.open("rb") as value:
                value.seek(16 << 10)
                self.assertEqual(value.read(64 << 10), bytes(64 << 10))
                value.seek(128 << 10)
                self.assertEqual(value.read(64 << 10), bytes(64 << 10))
                value.seek(224 << 10)
                self.assertEqual(value.read(4096), b"A" * 4096)


if __name__ == "__main__":
    unittest.main()
