import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "export_kimi_all_layer_calibrated_runtime.py"
SPEC = importlib.util.spec_from_file_location("calibrated96_export", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class Calibrated96ExportTests(unittest.TestCase):
    def test_hit_threshold_preserves_exact_fallback(self):
        hits = np.zeros(exporter.EXPERT_COUNT, dtype="<u4")
        hits[:4] = [1, 7, 8, 40]
        source = (hits != 0).astype(np.uint8)
        mask = exporter.runtime_calibrated_mask(source, hits, 8)
        self.assertEqual(mask[:5].tolist(), [0, 0, 1, 1, 0])
        source[4] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            exporter.runtime_calibrated_mask(source, hits, 8)

    def test_capture_counts_calibration_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            sequences = [
                (b"cal", 0, [1, 1, 2] + [3] * 13),
                (b"val", 1, [7] * 16),
            ]
            with path.open("wb") as output:
                output.write(exporter.CAPTURE_HEADER.pack(
                    exporter.CAPTURE_MAGIC, 1, 12, exporter.DIMENSION, 16,
                    len(sequences), len(sequences), 1, 0, 0, 0, 0, 0,
                ))
                for identifier, split, ids in sequences:
                    output.write(exporter.CAPTURE_RECORD.pack(
                        len(identifier), split, b"\0\0\0", 1
                    ))
                    output.write(identifier)
                    output.write(struct.pack("<i", 42))
                    output.write(bytes(exporter.DIMENSION * 2))
                    output.write(np.asarray(ids, dtype="<i4").tobytes())
                    output.write(np.full(16, 1 / 16, dtype="<f4").tobytes())
            hits, info = exporter.capture_calibration_hits(path, 12)
            self.assertEqual(int(hits[1]), 2)
            self.assertEqual(int(hits[2]), 1)
            self.assertEqual(int(hits[3]), 13)
            self.assertEqual(int(hits[7]), 0)
            self.assertEqual(info["calibration_token_count"], 1)

    def test_v2_mixed_layout_is_not_forced_to_v1_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.k3slab"
            gate = up = 179_200
            down = 236_544
            slab = gate + up + down
            record = slab * exporter.SLAB_COUNT
            order_offset = exporter.ALIGNMENT
            order_bytes = exporter.EXPERT_COUNT * exporter.SLAB_COUNT * 2
            payload_offset = exporter.align(order_offset + order_bytes)
            file_bytes = payload_offset + exporter.EXPERT_COUNT * record
            header = exporter.SIDECAR_HEADER_V2.pack(
                exporter.SIDECAR_MAGIC, 2, 92, exporter.EXPERT_COUNT,
                exporter.DIMENSION, exporter.SLAB_SIZE * exporter.SLAB_COUNT,
                exporter.SLAB_SIZE, exporter.SLAB_COUNT, exporter.ALIGNMENT,
                order_offset, order_bytes, payload_offset, slab, record,
                gate, up, down,
            )
            with path.open("wb") as output:
                output.write(header)
                output.truncate(file_bytes)  # sparse fixture, no large write
            layout = exporter.sidecar_layout(path, 92)
            self.assertEqual(layout["header_version"], 2)
            self.assertEqual(layout["down_slab_bytes"], down)
            self.assertEqual(layout["record_bytes"], record)


if __name__ == "__main__":
    unittest.main()
