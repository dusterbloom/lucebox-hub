#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_kimi_k3_m1a_telemetry as analyzer
import run_kimi_k3_m1a_telemetry as wrapper


def sample(at: int, *, swap: int = 0, residency: int = 0,
           competitors: str = "") -> dict[str, int | str]:
    return {
        "monotonic_ns": at, "vm_swap_kib": swap,
        "vm_rss_kib": 1024, "fd_count": 10,
        "temperature_edge_c": 55, "average_socket_power_w": 80,
        "average_gfxclk_mhz": 1800, "current_gfxclk_mhz": 1800,
        "average_gfx_activity_percent": 90,
        **{field: residency for field in wrapper.THROTTLE_RESIDENCY_FIELDS},
        "read_bytes": at // 1000, "write_bytes": at // 2000,
        "kfd_competitor_count":
            0 if not competitors else len(competitors.split(";")),
        "kfd_competitor_pids": competitors,
    }


def classify(rows: list[dict[str, int | str]], **updates):
    arguments = {
        "child_exit_code": 0, "monitor_error": None, "rows": rows,
        "start_ns": 900_000_000, "end_ns": 2_100_000_000,
        "interval_ms": 250,
        "vm_before": {"pswpin": 10, "pswpout": 20},
        "vm_after": {"pswpin": 10, "pswpout": 20},
        "cgroup_before": {"oom": 2, "oom_kill": 1},
        "cgroup_after": {"oom": 2, "oom_kill": 1},
        "kfd_before": [], "kfd_after": [], "kfd_seen": set(),
    }
    arguments.update(updates)
    return wrapper.classify(**arguments)


class ClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [sample(1_000_000_000 + i * 250_000_000)
                     for i in range(5)]

    def test_clean_is_both_valid(self) -> None:
        value = classify(self.rows)
        self.assertTrue(value["operational_soak_valid"])
        self.assertTrue(value["performance_timing_valid"])

    def test_throttle_at_20_percent_is_soak_only(self) -> None:
        rows = [sample(1_000_000_000 + i * 250_000_000) for i in range(6)]
        rows[-1]["throttle_residency_prochot"] = 1
        value = classify(rows, end_ns=2_350_000_000)
        self.assertTrue(value["operational_soak_valid"])
        self.assertFalse(value["performance_timing_valid"])
        self.assertTrue(value["throttle_annotation"])

    def test_throttle_over_20_percent_invalidates_both(self) -> None:
        self.rows[-2]["throttle_residency_prochot"] = 1
        self.rows[-1]["throttle_residency_prochot"] = 2
        value = classify(self.rows)
        self.assertFalse(value["operational_soak_valid"])
        self.assertFalse(value["performance_timing_valid"])

    def test_historical_nonzero_constant_residency_is_clean(self) -> None:
        rows = [sample(1_000_000_000 + i * 250_000_000, residency=41)
                for i in range(5)]
        value = classify(rows)
        self.assertTrue(value["operational_soak_valid"])
        self.assertEqual(value["throttled_samples"], 0)

    def test_residency_decrease_or_reset_invalidates(self) -> None:
        self.rows[0]["throttle_residency_thm_soc"] = 9
        for row in self.rows[1:]:
            row["throttle_residency_thm_soc"] = 8
        value = classify(self.rows)
        self.assertFalse(value["operational_soak_valid"])
        self.assertEqual(value["throttle_counter_resets"], 1)
        self.assertIn("throttle_counter_decrease_or_reset",
                      value["operational_invalid_reasons"])

    def test_host_paging_is_annotated_soak_only(self) -> None:
        value = classify(
            self.rows,
            vm_after={"pswpin": 300, "pswpout": 21})
        self.assertTrue(value["operational_soak_valid"])
        self.assertFalse(value["performance_timing_valid"])
        self.assertTrue(value["host_paging_annotation"])
        self.assertTrue(value["host_page_in_over_1_mib"])

    def test_target_swap_is_not_tolerated(self) -> None:
        self.rows[2]["vm_swap_kib"] = 4
        value = classify(self.rows)
        self.assertFalse(value["operational_soak_valid"])
        self.assertIn("target_process_swap",
                      value["operational_invalid_reasons"])

    def test_oom_and_kfd_are_never_annotations(self) -> None:
        oom = classify(
            self.rows, cgroup_after={"oom": 3, "oom_kill": 1})
        kfd = classify(self.rows, kfd_seen={999})
        self.assertFalse(oom["operational_soak_valid"])
        self.assertFalse(kfd["operational_soak_valid"])

    def test_gap_and_child_failure_are_invalid(self) -> None:
        sparse = [sample(1_000_000_000), sample(4_000_000_001)]
        self.assertFalse(classify(
            sparse, end_ns=4_100_000_000)["operational_soak_valid"])
        self.assertFalse(classify(
            self.rows, child_exit_code=7)["operational_soak_valid"])


class GpuMetricsTests(unittest.TestCase):
    def blob(self) -> bytearray:
        raw = bytearray(wrapper.GPU_METRICS_SIZE_V3_0)
        struct.pack_into("<HBB", raw, 0, len(raw), 3, 0)
        struct.pack_into("<H", raw, 4, 5500)
        struct.pack_into("<H", raw, 42, 87)
        struct.pack_into("<I", raw, 112, 81_000)
        struct.pack_into("<H", raw, 174, 1800)
        struct.pack_into("<H", raw, 224, 2400)
        return raw

    def test_v3_0_metrics_are_decoded_with_unit_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu_metrics"
            path.write_bytes(self.blob())
            value = wrapper.gpu_metrics(path)
        self.assertEqual(value["temperature_edge_c"], 55)
        self.assertEqual(value["average_socket_power_w"], 81)
        self.assertEqual(value["average_gfx_activity_percent"], 87)
        self.assertEqual(value["average_gfxclk_mhz"], 1800)
        self.assertEqual(value["current_gfxclk_mhz"], 2400)

    def test_wrong_revision_content_and_size_fail_closed(self) -> None:
        for mutation in ("format", "content", "header_size", "file_size"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                raw = self.blob()
                if mutation == "format": raw[2] = 1
                if mutation == "content": raw[3] = 1
                if mutation == "header_size": struct.pack_into("<H", raw, 0, 120)
                if mutation == "file_size": raw.pop()
                path = Path(tmp) / "gpu_metrics"
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "gpu_metrics"):
                    wrapper.gpu_metrics(path)

    def test_v3_throttle_residency_is_emitted_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = self.blob()
            struct.pack_into("<I", raw, 228 + 3 * 4, 1)
            path = Path(tmp) / "gpu_metrics"
            path.write_bytes(raw)
            self.assertEqual(
                wrapper.gpu_metrics(path)["throttle_residency_sppt"], 1)


class ExitRaceTests(unittest.TestCase):
    def test_terminal_missing_status_is_clean_terminator(self) -> None:
        class Exited:
            pid = 111

            @staticmethod
            def poll():
                return 0

        with mock.patch.object(wrapper, "child_status", return_value=None):
            self.assertEqual(
                wrapper.wait_for_status_or_exit(Exited()), (None, 0))

    def test_missing_status_while_live_remains_failure(self) -> None:
        class Live:
            pid = 110

            @staticmethod
            def poll():
                return None

        with mock.patch.object(wrapper, "child_status", return_value=None):
            self.assertEqual(
                wrapper.wait_for_status_or_exit(Live(), timeout_s=0),
                (None, None))

    def test_stale_status_after_exit_is_discarded(self) -> None:
        class Exited:
            pid = 112
            returncode = 0

            @staticmethod
            def poll():
                return 0

        with mock.patch.object(wrapper, "gpu_metrics") as metrics:
            row, code, competitors, error = wrapper.sample_live_row(
                Exited(), {"VmSwap": 0, "VmRSS": 1}, Path("unused"), set())
        self.assertIsNone(row)
        self.assertEqual(code, 0)
        self.assertEqual(competitors, set())
        self.assertIsNone(error)
        metrics.assert_not_called()

    def test_live_missing_io_is_failure(self) -> None:
        class Live:
            pid = 113
            returncode = None

            @staticmethod
            def poll():
                return None

        with mock.patch.object(wrapper, "gpu_metrics", return_value={
                "temperature_edge_c": 50, "average_socket_power_w": 1,
                "average_gfxclk_mhz": 1, "current_gfxclk_mhz": 1,
                "average_gfx_activity_percent": 1, "throttle_status": 0,
                "independent_throttle_status": 0}), \
                mock.patch.object(wrapper, "process_io", return_value=None), \
                mock.patch.object(Path, "iterdir", return_value=[]), \
                mock.patch.object(wrapper, "kfd_processes", return_value=[]):
            row, code, _, error = wrapper.sample_live_row(
                Live(), {"VmSwap": 0, "VmRSS": 1}, Path("unused"), set())
        self.assertIsNone(row)
        self.assertIsNone(code)
        self.assertEqual(error, "live child lacks process I/O")

    def test_exit_during_io_discards_partial_row(self) -> None:
        class ExitsDuringIo:
            pid = 114
            returncode = None

            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                if self.polls >= 2:
                    self.returncode = 0
                    return 0
                return None

        process = ExitsDuringIo()
        metrics = {
            "temperature_edge_c": 50, "average_socket_power_w": 1,
            "average_gfxclk_mhz": 1, "current_gfxclk_mhz": 1,
            "average_gfx_activity_percent": 1, "throttle_status": 0,
            "independent_throttle_status": 0,
        }
        with mock.patch.object(wrapper, "gpu_metrics", return_value=metrics), \
                mock.patch.object(
                    wrapper, "process_io",
                    return_value={"read_bytes": 1, "write_bytes": 2}), \
                mock.patch.object(Path, "iterdir", return_value=[]), \
                mock.patch.object(wrapper, "kfd_processes", return_value=[]):
            row, code, competitors, error = wrapper.sample_live_row(
                process, {"VmSwap": 0, "VmRSS": 1}, Path("unused"), set())
        self.assertIsNone(row)
        self.assertEqual(code, 0)
        self.assertEqual(competitors, set())
        self.assertIsNone(error)


class AnalyzerTests(unittest.TestCase):
    def fixture(self, directory: Path):
        samples = directory / "samples.csv"
        stdout = directory / "stdout.log"
        stderr = directory / "stderr.log"
        manifest_path = directory / "manifest.json"
        rows = [sample(1_000_000_000 + i * 250_000_000) for i in range(5)]
        wrapper.atomic_csv(samples, rows)
        stdout.write_text("ok\n")
        stderr.write_text("")
        validity = classify(rows)
        manifest = {
            "schema": wrapper.SCHEMA, "scope": "prospective_m1a_only",
            "historical_evidence_reclassified": False,
            "wrapper_complete": True, "command": ["/bin/true"],
            "effective_control_environment": {"DFLASH_TEST": "1"},
            "implementation_hashes": {
                relative: wrapper.sha256(
                    Path(__file__).resolve().parent.parent / relative)
                for relative in wrapper.IMPLEMENTATION_PATHS},
            "implementation_hashes_after": {
                relative: wrapper.sha256(
                    Path(__file__).resolve().parent.parent / relative)
                for relative in wrapper.IMPLEMENTATION_PATHS},
            "implementation_hashes_stable": True,
            "monitor_error": None, "child_exit_code": 0,
            "start_monotonic_ns": 900_000_000,
            "end_monotonic_ns": 2_100_000_000, "interval_ms": 250,
            "sample_count": len(rows), "samples": str(samples.resolve()),
            "samples_sha256": wrapper.sha256(samples),
            "stdout": str(stdout.resolve()),
            "stdout_sha256": wrapper.sha256(stdout),
            "stderr": str(stderr.resolve()),
            "stderr_sha256": wrapper.sha256(stderr),
            "vmstat_before": {"pswpin": 10, "pswpout": 20},
            "vmstat_after": {"pswpin": 10, "pswpout": 20},
            "cgroup_path": "/sys/fs/cgroup/test",
            "wrapper_cgroup_path": "/sys/fs/cgroup/test",
            "cgroup_before": {"oom": 2, "oom_kill": 1},
            "cgroup_after": {"oom": 2, "oom_kill": 1},
            "kfd_competitors_before": [], "kfd_competitors_after": [],
            "kfd_competitor_pids_seen": [], "validity": validity,
            "gpu_metrics_path": str(wrapper.EXPECTED_GPU_METRICS_PATH),
            "gpu_metrics_identity": {
                "class_path": str(wrapper.EXPECTED_GPU_METRICS_PATH),
                "resolved_device":
                    "/sys/devices/pci0000:00/0000:00:08.1/0000:c5:00.0",
                "pci_bdf": wrapper.EXPECTED_GPU_BDF,
                "pci_device": wrapper.EXPECTED_GPU_DEVICE,
            },
            "gpu_metrics_identity_after": {
                "class_path": str(wrapper.EXPECTED_GPU_METRICS_PATH),
                "resolved_device":
                    "/sys/devices/pci0000:00/0000:00:08.1/0000:c5:00.0",
                "pci_bdf": wrapper.EXPECTED_GPU_BDF,
                "pci_device": wrapper.EXPECTED_GPU_DEVICE,
            },
            "gpu_metrics_abi": {"structure_size": 264,
                                "format_revision": 3,
                                "content_revision": 0},
        }
        wrapper.atomic_json(manifest_path, manifest)
        return manifest, manifest_path, samples

    def test_clean_manifest_preserves_two_validity_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, _ = self.fixture(Path(tmp))
            result, code = analyzer.analyze(manifest, path)
        self.assertEqual(code, 0)
        self.assertTrue(result["operational_soak_valid"])
        self.assertTrue(result["performance_timing_valid"])
        self.assertIsNone(result["speedup_or_historical_verdict"])

    def test_manifest_validity_forgery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, _ = self.fixture(Path(tmp))
            manifest["validity"]["performance_timing_valid"] = False
            with self.assertRaisesRegex(ValueError, "does not recompute"):
                analyzer.analyze(manifest, path)

    def test_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, samples = self.fixture(Path(tmp))
            with samples.open("a") as handle:
                handle.write("corrupt\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyzer.analyze(manifest, path)

    def test_atomic_publish_rejects_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "value.json"
            target.write_text("existing")
            with self.assertRaises(FileExistsError):
                wrapper.atomic_json(target, {"new": True})
            self.assertEqual(target.read_text(), "existing")

    def test_gpu_identity_and_abi_forgery_are_rejected(self) -> None:
        for key, value in (("pci_bdf", "0000:01:00.0"),
                           ("pci_device", "0x0000")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                manifest, path, _ = self.fixture(Path(tmp))
                manifest["gpu_metrics_identity"][key] = value
                with self.assertRaisesRegex(ValueError, "physical identity"):
                    analyzer.analyze(manifest, path)
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, _ = self.fixture(Path(tmp))
            manifest["gpu_metrics_path"] = "/sys/class/drm/card1/device/gpu_metrics"
            with self.assertRaisesRegex(ValueError, "physical identity"):
                analyzer.analyze(manifest, path)
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, _ = self.fixture(Path(tmp))
            manifest["gpu_metrics_identity_after"]["pci_bdf"] = "0000:01:00.0"
            with self.assertRaisesRegex(ValueError, "physical identity"):
                analyzer.analyze(manifest, path)
        with tempfile.TemporaryDirectory() as tmp:
            manifest, path, _ = self.fixture(Path(tmp))
            manifest["gpu_metrics_abi"]["content_revision"] = 1
            with self.assertRaisesRegex(ValueError, "physical identity"):
                analyzer.analyze(manifest, path)


if __name__ == "__main__":
    unittest.main()
