#!/usr/bin/env python3
"""Validate prospective M1a telemetry without making a speed claim."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import run_kimi_k3_m1a_telemetry as wrapper


RESULT_SCHEMA = "kimi_k3_m1a_telemetry_result_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(row: dict[str, str], key: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sample {key}") from error
    require(value >= 0, f"negative sample {key}")
    return value


def load_samples(path: Path, expected_sha: str) -> list[dict[str, int | str]]:
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            "samples artifact is absent/non-regular")
    require(wrapper.sha256(path) == expected_sha,
            "samples artifact hash mismatch")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == wrapper.SAMPLE_FIELDS,
                "sample header mismatch")
        raw_rows = list(reader)
    rows: list[dict[str, int | str]] = []
    last_time = -1
    last_read = -1
    last_write = -1
    for raw in raw_rows:
        row: dict[str, int | str] = {}
        for key in wrapper.SAMPLE_FIELDS:
            if key == "kfd_competitor_pids":
                value = raw.get(key)
                require(isinstance(value, str), "invalid competitor PID cell")
                if value:
                    require(all(part.isdigit() for part in value.split(";")),
                            "invalid competitor PID list")
                row[key] = value
            else:
                row[key] = integer(raw, key)
        now = int(row["monotonic_ns"])
        read_bytes = int(row["read_bytes"])
        write_bytes = int(row["write_bytes"])
        require(now > last_time, "sample timestamps are not increasing")
        require(read_bytes >= last_read and write_bytes >= last_write,
                "process I/O counters regressed")
        require(int(row["kfd_competitor_count"]) ==
                (0 if row["kfd_competitor_pids"] == "" else
                 len(str(row["kfd_competitor_pids"]).split(";"))),
                "KFD count/PID mismatch")
        last_time, last_read, last_write = now, read_bytes, write_bytes
        rows.append(row)
    return rows


def analyze(manifest: dict[str, Any], manifest_path: Path) -> tuple[dict[str, Any], int]:
    require(manifest.get("schema") == wrapper.SCHEMA and
            manifest.get("scope") == "prospective_m1a_only" and
            manifest.get("historical_evidence_reclassified") is False and
            manifest.get("wrapper_complete") is True,
            "manifest scope/schema mismatch")
    require(manifest_path.is_file() and not manifest_path.is_symlink(),
            "manifest input is absent/non-regular")
    start = manifest.get("start_monotonic_ns")
    end = manifest.get("end_monotonic_ns")
    interval = manifest.get("interval_ms")
    require(isinstance(start, int) and isinstance(end, int) and 0 < start < end,
            "invalid telemetry time bracket")
    require(isinstance(interval, int) and 50 <= interval <= 30_000,
            "invalid telemetry interval")
    require(isinstance(manifest.get("command"), list) and
            manifest["command"] and
            all(isinstance(part, str) and part for part in manifest["command"]),
            "invalid recorded command")
    controls = manifest.get("effective_control_environment")
    require(isinstance(controls, dict) and
            all(isinstance(key, str) and isinstance(value, str) and
                key.startswith(wrapper.CONTROL_PREFIXES)
                for key, value in controls.items()),
            "invalid effective controls")
    implementation = manifest.get("implementation_hashes")
    implementation_after = manifest.get("implementation_hashes_after")
    require(isinstance(implementation, dict) and
            set(implementation) == set(wrapper.IMPLEMENTATION_PATHS) and
            implementation_after == implementation and
            manifest.get("implementation_hashes_stable") is True and
            all(isinstance(value, str) and len(value) == 64 and
                all(character in "0123456789abcdef" for character in value)
                for value in implementation.values()),
            "implementation hash lock mismatch")
    repo = Path(__file__).resolve().parent.parent
    require(all((repo / relative).is_file() and
                wrapper.sha256(repo / relative) == digest
                for relative, digest in implementation.items()),
            "analyzed implementation differs from wrapper lock")
    metrics_identity = manifest.get("gpu_metrics_identity")
    require(isinstance(metrics_identity, dict) and
            manifest.get("gpu_metrics_path") ==
                str(wrapper.EXPECTED_GPU_METRICS_PATH) and
            metrics_identity.get("class_path") ==
                str(wrapper.EXPECTED_GPU_METRICS_PATH) and
            isinstance(metrics_identity.get("resolved_device"), str) and
            Path(metrics_identity["resolved_device"]).is_absolute() and
            Path(metrics_identity["resolved_device"]).name.lower() ==
                wrapper.EXPECTED_GPU_BDF and
            metrics_identity.get("pci_bdf") == wrapper.EXPECTED_GPU_BDF and
            metrics_identity.get("pci_device") == wrapper.EXPECTED_GPU_DEVICE and
            set(metrics_identity) == {"class_path", "resolved_device",
                                      "pci_bdf", "pci_device"} and
            manifest.get("gpu_metrics_identity_after") == metrics_identity and
            manifest.get("gpu_metrics_abi") == {
                "structure_size": wrapper.GPU_METRICS_SIZE_V3_0,
                "format_revision": 3, "content_revision": 0},
            "gpu_metrics physical identity/ABI mismatch")

    sample_path = Path(manifest.get("samples", ""))
    rows = load_samples(sample_path, manifest.get("samples_sha256", ""))
    require(len(rows) == manifest.get("sample_count"),
            "sample cardinality mismatch")
    for label in ("stdout", "stderr"):
        path = Path(manifest.get(label, ""))
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and
                wrapper.sha256(path) == manifest.get(label + "_sha256"),
                f"{label} artifact mismatch")
    require(not manifest_path.with_name(manifest_path.name + ".partial").exists() and
            not sample_path.with_name(sample_path.name + ".partial").exists(),
            "partial evidence artifact remains")

    before_vm = manifest.get("vmstat_before")
    after_vm = manifest.get("vmstat_after")
    c_before = manifest.get("cgroup_before")
    c_after = manifest.get("cgroup_after")
    require(all(isinstance(value, dict) for value in
                (before_vm, after_vm, c_before, c_after)),
            "telemetry counter bracket absent")
    require(all(isinstance(mapping.get(key), int)
                for mapping, keys in (
                    (before_vm, ("pswpin", "pswpout")),
                    (after_vm, ("pswpin", "pswpout")),
                    (c_before, ("oom", "oom_kill")),
                    (c_after, ("oom", "oom_kill")))
                for key in keys), "telemetry counter is missing")
    require(after_vm["pswpin"] >= before_vm["pswpin"] and
            after_vm["pswpout"] >= before_vm["pswpout"] and
            c_after["oom"] >= c_before["oom"] and
            c_after["oom_kill"] >= c_before["oom_kill"],
            "monotonic telemetry counter regressed")
    require(manifest.get("cgroup_path") ==
            manifest.get("wrapper_cgroup_path") and
            isinstance(manifest.get("cgroup_path"), str) and
            manifest["cgroup_path"].startswith("/"),
            "wrapper/child cgroup mismatch")
    for key in ("kfd_competitors_before", "kfd_competitors_after",
                "kfd_competitor_pids_seen"):
        values = manifest.get(key)
        require(isinstance(values, list) and
                values == sorted(set(values)) and
                all(isinstance(value, int) and value > 0 for value in values),
                f"invalid {key}")

    recomputed = wrapper.classify(
        child_exit_code=manifest.get("child_exit_code"),
        monitor_error=manifest.get("monitor_error"), rows=rows,
        start_ns=start, end_ns=end, interval_ms=interval,
        vm_before=before_vm, vm_after=after_vm,
        cgroup_before=c_before, cgroup_after=c_after,
        kfd_before=manifest["kfd_competitors_before"],
        kfd_after=manifest["kfd_competitors_after"],
        kfd_seen=set(manifest["kfd_competitor_pids_seen"]))
    require(manifest.get("validity") == recomputed,
            "manifest validity does not recompute")
    require(recomputed.get("throttle_measured_intervals") ==
            max(0, len(rows) - 1) and
            isinstance(recomputed.get("throttle_counter_resets"), int) and
            recomputed["throttle_counter_resets"] >= 0,
            "v3 cumulative throttle accounting mismatch")
    require(isinstance(recomputed["throttle_fraction"], float) and
            math.isfinite(recomputed["throttle_fraction"]),
            "non-finite throttle fraction")
    result = {
        "schema": RESULT_SCHEMA,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": wrapper.sha256(manifest_path),
        "scope": "prospective_m1a_only",
        "historical_evidence_reclassified": False,
        "operational_soak_valid": recomputed["operational_soak_valid"],
        "performance_timing_valid": recomputed["performance_timing_valid"],
        "validity": recomputed,
        "sample_count": len(rows),
        "throttle_counter_semantics": "v3_cumulative_interval_delta",
        "timing_claim_allowed": recomputed["performance_timing_valid"],
        "speedup_or_historical_verdict": None,
    }
    return result, 0 if recomputed["operational_soak_valid"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(
            args.output.name + ".partial").exists():
        parser.error("output must be fresh")
    try:
        manifest = json.loads(args.manifest.read_text())
        result, code = analyze(manifest, args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    wrapper.atomic_json(args.output, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
