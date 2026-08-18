import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "pack_kimi_k3_slim_core.py"
SPEC = importlib.util.spec_from_file_location("pack_kimi_k3_slim_core", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SlimCoreSparseCopyTest(unittest.TestCase):
    def test_coalesce_and_sparse_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "destination.bin"
            payload = bytearray(1 << 20)
            payload[:4096] = b"H" * 4096
            payload[256 << 10 : (256 << 10) + 8192] = b"A" * 8192
            payload[768 << 10 : (768 << 10) + 4096] = b"B" * 4096
            source.write_bytes(payload)
            ranges = MODULE.coalesce_ranges([
                MODULE.ByteRange(0, 4096),
                MODULE.ByteRange(256 << 10, 4096),
                MODULE.ByteRange((256 << 10) + 4096, 4096),
                MODULE.ByteRange(768 << 10, 4096),
            ])
            self.assertEqual(len(ranges), 3)
            result = MODULE.copy_sparse_ranges(
                source, destination, len(payload), ranges, verify=True
            )
            self.assertEqual(destination.stat().st_size, len(payload))
            self.assertEqual(result["copied_bytes"], 4096 + 8192 + 4096)
            copied = destination.read_bytes()
            self.assertEqual(copied[:4096], payload[:4096])
            self.assertEqual(
                copied[256 << 10 : (256 << 10) + 8192],
                payload[256 << 10 : (256 << 10) + 8192],
            )
            self.assertEqual(
                copied[768 << 10 : (768 << 10) + 4096],
                payload[768 << 10 : (768 << 10) + 4096],
            )
            self.assertEqual(copied[512 << 10 : (512 << 10) + 4096], bytes(4096))


if __name__ == "__main__":
    unittest.main()
