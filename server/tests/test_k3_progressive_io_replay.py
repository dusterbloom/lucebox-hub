import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def coupled_request(self, offset: int, sequence: int = 0,
                        base_pos: int = 0, length: int = 16,
                        observed: int | None = None):
        if observed is None:
            observed = length
        return replay.CacheRequest(
            MODEL_SHA, SIDECAR_SHA, "sidecar", offset, length, 1,
            sequence=sequence, base_pos=base_pos,
            observed_physical_bytes=observed)

    def control(self, requests, host_capacity=64):
        prompt_lengths = [1] * (max(row.sequence for row in requests) + 1)
        expected = {"prefill": 0, "decode": 0}
        for request in requests:
            expected[replay.request_phase(request, prompt_lengths)] += (
                request.observed_physical_bytes)
        return replay.CoupledControl(
            host_capacity, expected, prompt_lengths,
            {"prefill": 0, "decode": 0})

    def test_coupled_residency_resets_both_tiers_between_prompts(self):
        requests = [
            self.coupled_request(0, sequence=0, base_pos=0),
            self.coupled_request(0, sequence=0, base_pos=1),
            self.coupled_request(0, sequence=1, base_pos=0),
            self.coupled_request(0, sequence=1, base_pos=1),
        ]
        result = replay.simulate_coupled_residency(
            requests, self.control(requests), 32, "lru")
        self.assertEqual(result["phase"]["prefill"]["gpu_hits"], 0)
        self.assertEqual(result["phase"]["decode"]["gpu_hits"], 2)
        self.assertEqual(result["required_physical_bytes"]["prefill"], 32)
        self.assertEqual(result["required_physical_bytes"]["decode"], 0)

    def test_second_touch_admission_and_lru_eviction(self):
        requests = [self.coupled_request(offset) for offset in (0, 16, 0, 32, 16)]
        control = self.control(requests, host_capacity=16)
        second = replay.simulate_coupled_residency(
            requests, control, 16, "second-touch")
        lru = replay.simulate_coupled_residency(requests, control, 16, "lru")
        self.assertEqual(second["phase"]["prefill"]["promotion_bytes"], 32)
        self.assertGreaterEqual(second["gpu_evictions"], 1)
        self.assertGreaterEqual(lru["gpu_evictions"], 1)

    def test_belady_oracle_is_not_worse_than_lru_on_equal_ranges(self):
        requests = [self.coupled_request(offset) for offset in
                    (0, 16, 32, 0, 16, 32, 0)]
        control = self.control(requests, host_capacity=16)
        lru = replay.simulate_coupled_residency(requests, control, 32, "lru")
        belady = replay.simulate_coupled_residency(
            requests, control, 32, "belady")
        self.assertLessEqual(
            belady["required_physical_bytes"]["prefill"],
            lru["required_physical_bytes"]["prefill"])

    def test_frozen_concurrent_p30_outcomes_override_serial_lru_order(self):
        # Three frozen misses are possible when concurrent workers perform all
        # gets before their puts. A serialized 32-byte host LRU would call the
        # third A a hit, so reconstructing P30 from row order is invalid.
        requests = [
            self.coupled_request(0, observed=16),
            self.coupled_request(16, observed=16),
            self.coupled_request(0, observed=16),
        ]
        control = self.control(requests, host_capacity=32)
        result = replay.simulate_coupled_residency(
            requests, control, 32, "lru")
        self.assertEqual(control.expected_physical_bytes["prefill"], 48)
        self.assertEqual(result["phase"]["prefill"]["gpu_hits"], 1)
        self.assertEqual(result["required_physical_bytes"]["prefill"], 32)
        self.assertEqual(result["p30_outcome_source"], "FROZEN_CONCURRENT_TRACE")

    def test_frozen_p30_hits_bypass_gpu_and_claim_no_saving(self):
        requests = [
            self.coupled_request(0, observed=0),
            self.coupled_request(0, observed=0),
        ]
        result = replay.simulate_coupled_residency(
            requests, self.control(requests), 32, "lru")
        stats = result["phase"]["prefill"]
        self.assertEqual(stats["host_hits"], 2)
        self.assertEqual(stats["gpu_hits"], 0)
        self.assertEqual(stats["promotion_bytes"], 0)
        self.assertEqual(stats["nvme_bytes"], 0)

    def test_projected_service_formula_is_labeled_and_has_exact_boundary(self):
        requests = [self.coupled_request(0), self.coupled_request(0)]
        control = self.control(requests)
        curve = {"nvme_gib_s": 1.0, "peer_gib_s": 2.0,
                 "promotion_gib_s": 4.0, "peer_latency_us": 10.0}
        result = replay.simulate_coupled_residency(
            requests, control, 32, "lru", service_curve=curve)
        projection = result["service_projection"]["prefill"]
        expected_candidate = 16 / replay.GIB + 16 / replay.GIB / 2 + \
            16 / replay.GIB / 4 + 10e-6
        self.assertEqual(
            projection["classification"],
            "PROJECTED_FROM_EXTERNAL_SERVICE_CURVE")
        self.assertAlmostEqual(projection["candidate_seconds"], expected_candidate)
        self.assertFalse(result["performance_measured"])

    def test_zero_explicit_gate_is_reconstructed_and_control_reconciles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "layer01.k3slab"
            rows = [
                row(0, sidecar, "gate", 0, 16, 0, 512, 512, base_pos=0),
                row(1, sidecar, "up", 16, 16, 0, 512, 0, base_pos=0),
                row(2, sidecar, "down", 32, 16, 0, 512, 0, base_pos=0),
                row(3, sidecar, "gate", 0, 16, 0, 512, 0, base_pos=1),
                row(4, sidecar, "up", 16, 16, 0, 512, 0, base_pos=1),
                row(5, sidecar, "down", 32, 16, 0, 512, 0, base_pos=1),
            ]
            _, manifest, trace = self.write_fixture(directory, rows)
            suite = root / "suite.json"
            suite.write_text(json.dumps({
                "schema": "kimi-k3-h16-suite-v1",
                "sequences": [{"id": "one", "prompt_token_count": 1}],
            }))
            pinned = root / "pinned.json"
            pinned.write_text(json.dumps([{
                "model_identity": MODEL_SHA,
                "sidecar_identity": SIDECAR_SHA,
                "offset": 0,
                "length": 512,
            }]))
            result = replay.simulate_coupled_trace(
                trace, manifest, suite, 1, 1, 512, 0, pinned)
            self.assertTrue(result["control"]["exact_match"])
            self.assertEqual(result["zero_explicit_read_demands"], 1)
            self.assertEqual(result["logical_cacheable_requests"], 2)
            self.assertEqual(result["sequence_resets"], 0)
            without_pinned = replay.simulate_coupled_trace(
                trace, manifest, suite, 1, 1, 512, 0)
            self.assertEqual(
                [item["policy"] for item in without_pinned["policies"]],
                ["lru", "second-touch", "belady"])
            self.assertEqual(without_pinned["pinned_hot_sha256"], None)
            self.assertEqual(without_pinned["policy_skipped"], [{
                "policy": "heldout-pinned-hot",
                "reason": (
                    "no independently frozen --pinned-hot-keys artifact "
                    "supplied"),
            }])
            self.assertTrue(without_pinned["belady_dominates_practical"])
            with self.assertRaisesRegex(ValueError, "trace physical mismatch"):
                replay.simulate_coupled_trace(
                    trace, manifest, suite, 1, 1, 511, 0, pinned)

    def test_atomic_publication_removes_temporary_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with mock.patch.object(
                    replay.os, "replace", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    replay.atomic_write_json(output, {"complete": True})
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(".result.json.tmp.*")), [])

    def test_atomic_publication_replaces_only_with_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            replay.atomic_write_json(output, {"complete": True})
            self.assertEqual(json.loads(output.read_text()), {"complete": True})
            self.assertEqual(list(output.parent.glob(".result.json.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
