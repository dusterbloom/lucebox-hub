import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "kimi_h23_capture_chunks", ROOT / "scripts/kimi_h23_capture_chunks.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_chunk(root: Path, plan: dict, index: int) -> None:
    chunk = plan["chunks"][index]
    directory = root / "chunks" / f"chunk-{index:04d}"
    directory.mkdir(parents=True)
    captures = []
    for layer in range(1, 93):
        path = directory / f"kimi_layer{layer:02d}_4.bin"
        records = []
        with path.open("wb") as output:
            output.write(
                MODULE.CAPTURE_HEADER.pack(
                    MODULE.CAPTURE_MAGIC,
                    1,
                    layer,
                    MODULE.DIMENSION,
                    MODULE.TOP_K,
                    len(chunk["ids"]),
                    len(chunk["ids"]),
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            )
            for identifier, split_name in zip(chunk["ids"], chunk["splits"]):
                identifier_raw = identifier.encode()
                split = 0 if split_name == "calibration" else 1
                output.write(MODULE.CAPTURE_RECORD.pack(len(identifier_raw), split, b"\0\0\0", 1))
                output.write(identifier_raw)
                output.write(struct.pack("<i", 7))
                output.write(bytes(MODULE.DIMENSION * 2))
                output.write(bytes(MODULE.TOP_K * 4))
                output.write(bytes(MODULE.TOP_K * 4))
                records.append({"id": identifier, "split": split_name, "tokens": 1})
        index_value = {
            "schema": "kimi-k3-panel-capture-v1",
            "model_layer": layer,
            "latent_dimension": MODULE.DIMENSION,
            "top_k": MODULE.TOP_K,
            "sequence_count": len(records),
            "token_count": len(records),
            "capture_path": str(path),
            "sequences": records,
        }
        Path(str(path) + ".json").write_text(json.dumps(index_value) + "\n")
        captures.append({"model_layer": layer, "path": str(path)})
    manifest = {
        "schema": "kimi-k3-panel-multi-layer-capture-v1",
        "model_path": "synthetic.gguf",
        "sequence_count": len(chunk["ids"]),
        "token_count": len(chunk["ids"]),
        "first_routed_layer": 1,
        "last_routed_layer": 92,
        "layer_count": 92,
        "captures": captures,
    }
    (directory / "all_layers_capture_manifest.json").write_text(json.dumps(manifest) + "\n")


class H23ChunkedCaptureTest(unittest.TestCase):
    def test_chunk_resume_and_merge_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            corpus = Path(temporary) / "corpus.jsonl"
            rows = [
                {"id": "cal-a", "split": "calibration", "text": "a"},
                {"id": "val-a", "split": "validation", "text": "b"},
                {"id": "cal-b", "split": "calibration", "text": "c"},
                {"id": "val-b", "split": "validation", "text": "d"},
            ]
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
            plan = MODULE.prepare(corpus, root, total_tokens=4, rows_per_chunk=2)
            initial = MODULE.status(root)
            self.assertEqual(initial["completed_tokens"], 0)
            self.assertEqual(initial["next_chunk"]["index"], 0)

            write_chunk(root, plan, 0)
            partial = MODULE.status(root)
            self.assertEqual(partial["completed_tokens"], 2)
            self.assertEqual(partial["next_chunk"]["index"], 1)

            invalid = root / "chunks/chunk-0001"
            invalid.mkdir()
            (invalid / "partial").write_text("crash")
            interrupted = MODULE.status(root)
            self.assertEqual(interrupted["invalid_next"]["index"], 1)
            for path in invalid.iterdir():
                path.unlink()
            invalid.rmdir()

            write_chunk(root, plan, 1)
            complete = MODULE.status(root)
            self.assertTrue(complete["complete"])
            self.assertEqual(complete["completed_tokens"], 4)

            merged_root = root / "merged"
            first_manifest = MODULE.merge(root, merged_root)
            (merged_root / "all_layers_capture_manifest.json").unlink()
            second_manifest = MODULE.merge(root, merged_root)
            self.assertEqual(first_manifest, second_manifest)
            info = MODULE.inspect_capture(merged_root / "kimi_layer01_4.bin", 1)
            self.assertEqual(info["token_count"], 4)
            self.assertEqual(info["sequence_count"], 4)


if __name__ == "__main__":
    unittest.main()
