import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "k3_progressive_io_replay.py"
SPEC = importlib.util.spec_from_file_location("k3_progressive_io_replay", SCRIPT)
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


MODEL_SHA = "1" * 64
SIDECAR_SHA = "2" * 64
FIELDS = [
    "request_id", "prompt_id", "base_pos", "token_index", "model_layer",
    "expert_id", "region", "qtype", "prefix_depth", "exact_fallback",
    "file_path", "file_offset", "logical_length", "aligned_offset",
    "aligned_length", "destination_kind", "destination_offset",
    "explicit_read_bytes",
]


def row(identifier: int, path: Path, region: str, offset: int, logical: int,
        aligned_offset: int, aligned_length: int, explicit: int,
        fallback: int = 0, base_pos: int | None = None) -> dict[str, object]:
    return {
        "request_id": identifier,
        "prompt_id": "frozen",
        "base_pos": identifier // 4 if base_pos is None else base_pos,
        "token_index": 0,
        "model_layer": 1,
        "expert_id": 7,
        "region": region,
        "qtype": "iq1_s" if region != "slab-mean" else "f32",
        "prefix_depth": 12,
        "exact_fallback": fallback,
        "file_path": str(path),
        "file_offset": offset,
        "logical_length": logical,
        "aligned_offset": aligned_offset,
        "aligned_length": aligned_length,
        "destination_kind": "host-compact-slab",
        "destination_offset": 0,
        "explicit_read_bytes": explicit,
    }


class K3ProgressiveCacheSimulationTests(unittest.TestCase):
    def write_fixture(self, directory: str, rows: list[dict[str, object]]):
        root = Path(directory)
        sidecar = root / "layer01.k3slab"
        sidecar.write_bytes(bytes(4096))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "kimi-k3-all-layer-calibrated-slab-runtime-v2",
            "layers": [{
                "model_layer": 1,
                "output": str(root / "layer01.k3aux"),
                "output_bytes": 4096,
                "output_sha256": "5" * 64,
                "provenance": {
                    "model_checksum_registry_sha256": MODEL_SHA,
                    "sidecar": str(sidecar),
                    "sidecar_bytes": 4096,
                    "sidecar_sha256": SIDECAR_SHA,
                },
            }],
        }))
        (root / "layer01.k3aux").write_bytes(bytes(4096))
        trace = root / "trace.tsv"
        with trace.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return sidecar, manifest, trace

    def test_exact_identity_and_range_define_hits(self):
        common = dict(
            model_identity=MODEL_SHA,
            sidecar_identity=SIDECAR_SHA,
            path="sidecar", layer=1,
        )
        requests = [
            replay.CacheRequest(offset=0, length=16, **common),
            replay.CacheRequest(offset=0, length=16, **common),
            replay.CacheRequest(offset=0, length=17, **common),
            replay.CacheRequest(
                offset=0, length=16, **{
                    **common, "sidecar_identity": "3" * 64}),
            replay.CacheRequest(
                offset=0, length=16, **{
                    **common, "model_identity": "4" * 64}),
        ]
        result = replay.simulate_bounded_lru(requests, 128)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["misses"], 4)
        self.assertEqual(result["hit_bytes"], 16)
        self.assertTrue(result["capacity_respected"])

    def test_lru_is_bounded_and_evicts_oldest_exact_range(self):
        common = dict(
            model_identity=MODEL_SHA,
            sidecar_identity=SIDECAR_SHA,
            path="sidecar", layer=1, length=16,
        )
        requests = [
            replay.CacheRequest(offset=0, **common),
            replay.CacheRequest(offset=16, **common),
            replay.CacheRequest(offset=0, **common),
            replay.CacheRequest(offset=32, **common),
            replay.CacheRequest(offset=16, **common),
        ]
        result = replay.simulate_bounded_lru(requests, 32)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["misses"], 4)
        self.assertEqual(result["evictions"], 2)
        self.assertEqual(result["peak_resident_bytes"], 32)
        self.assertTrue(result["capacity_respected"])

    def test_p27_trace_accounts_uncached_semantic_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "layer01.k3slab"
            rows = [
                row(0, sidecar, "gate", 100, 100, 0, 512, 512),
                row(1, sidecar, "up", 200, 100, 0, 512, 0),
                row(2, sidecar, "down", 300, 100, 0, 512, 0),
                row(3, sidecar, "gate", 100, 100, 0, 512, 512),
                row(4, sidecar, "up", 200, 100, 0, 512, 0),
                row(5, sidecar, "down", 300, 100, 0, 512, 0),
                row(6, root / "layer01.k3aux",
                    "slab-mean", 600, 64, 512, 512, 512),
                row(7, Path("<native-model-shards>"),
                    "native-exact-expert", -1, 900, -1, -1, 0, fallback=1),
            ]
            _, manifest, trace = self.write_fixture(directory, rows)
            result = replay.simulate_cache_trace(trace, manifest, [2])
            accounting = result["accounting"]
            policy = result["policies"][0]
            self.assertEqual(accounting["selected_component_rows"], 6)
            self.assertEqual(accounting["selected_logical_bytes"], 600)
            self.assertEqual(accounting["slab_page_requests"], 2)
            self.assertEqual(accounting["global_unique_slab_pages"], 1)
            self.assertEqual(accounting["global_unlimited_avoidable_bytes"], 512)
            self.assertEqual(accounting["mean_tail_logical_bytes"], 64)
            self.assertEqual(accounting["exact_fallback_bytes_uncached"], 900)
            self.assertEqual(policy["hits"], 1)
            self.assertEqual(policy["misses"], 1)
            self.assertFalse(result["runtime_mutation"])
            self.assertFalse(result["default_behavior_changed"])

    def test_decode_is_prefill_warm_with_sequence_resets_and_unified_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "layer01.k3slab"
            mean = root / "layer01.k3aux"
            rows = [
                row(0, sidecar, "gate", 0, 16, 0, 512, 512, base_pos=0),
                row(1, mean, "slab-mean", 512, 16, 512, 512, 512,
                    base_pos=0),
                row(2, sidecar, "gate", 0, 16, 0, 512, 512, base_pos=1),
                row(3, mean, "slab-mean", 512, 16, 512, 512, 512,
                    base_pos=1),
                row(4, sidecar, "gate", 0, 16, 0, 512, 512, base_pos=0),
                row(5, mean, "slab-mean", 512, 16, 512, 512, 512,
                    base_pos=0),
                row(6, sidecar, "gate", 0, 16, 0, 512, 512, base_pos=1),
                row(7, mean, "slab-mean", 512, 16, 512, 512, 512,
                    base_pos=1),
            ]
            _, manifest, trace = self.write_fixture(directory, rows)
            suite = root / "suite.json"
            suite.write_text(json.dumps({
                "schema": "kimi-k3-h16-suite-v1",
                "sequences": [
                    {"id": "one", "prompt_token_count": 1},
                    {"id": "two", "prompt_token_count": 1},
                ],
            }))
            result = replay.simulate_cache_trace(
                trace, manifest, [2], cache_scopes=["slab-only", "unified"],
                phase="decode", reset_policy="sequence",
                sequence_manifest=suite)
            slab, unified = result["policies"]
            self.assertEqual(result["sequence_count"], 2)
            self.assertEqual(slab["hits"], 2)
            self.assertEqual(slab["misses"], 0)
            self.assertEqual(slab["warmup_misses"], 2)
            self.assertEqual(slab["sequence_resets"], 1)
            self.assertEqual(unified["hits"], 4)
            self.assertEqual(unified["misses"], 0)
            self.assertEqual(unified["warmup_misses"], 4)

    def test_unregistered_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered = root / "layer01.k3slab"
            unregistered = root / "other.k3slab"
            rows = [row(0, unregistered, "gate", 0, 16, 0, 512, 512)]
            _, manifest, trace = self.write_fixture(directory, rows)
            registered.write_bytes(bytes(4096))
            with self.assertRaisesRegex(ValueError, "unregistered sidecar"):
                replay.simulate_cache_trace(trace, manifest, [2])

    def test_registered_revision40_result_matches_independent_audit(self):
        result_path = (
            Path(__file__).parents[2] /
            "results/k3_revision40_slab_page_cache.json")
        result = json.loads(result_path.read_text())
        self.assertEqual(result["measured_phase"], "decode")
        self.assertTrue(result["prefill_warms_decode"])
        self.assertEqual(result["reset_policy"], "sequence")
        self.assertEqual(result["sequence_count"], 12)
        policies = {
            (row["cache_scope"], int(row["capacity_gib"])): row
            for row in result["policies"]
        }
        expected = {
            ("slab-only", 2): 0.3570354262933768,
            ("slab-only", 4): 0.5107276557364897,
            ("slab-only", 8): 0.6449031975092046,
            ("slab-only", 16): 0.749701469025674,
            ("unified", 2): 0.31277187784468685,
            ("unified", 4): 0.49164275581197164,
            ("unified", 8): 0.6237551084222168,
            ("unified", 16): 0.7383071755631279,
        }
        self.assertEqual(set(policies), set(expected))
        for key, fraction in expected.items():
            self.assertAlmostEqual(policies[key]["byte_hit_fraction"], fraction)
            self.assertEqual(policies[key]["sequence_resets"], 11)
            self.assertTrue(policies[key]["capacity_respected"])


if __name__ == "__main__":
    unittest.main()
