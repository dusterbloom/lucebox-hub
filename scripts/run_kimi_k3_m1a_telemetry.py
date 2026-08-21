#!/usr/bin/env python3
"""Run one prospective M1a command under exit-race-safe telemetry.

This wrapper classifies operational soak evidence separately from evidence that
is clean enough for performance comparisons.  It does not reinterpret any
P58/P62/P63 artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "kimi_k3_m1a_telemetry_v1"
SAMPLE_FIELDS = (
    "monotonic_ns", "vm_swap_kib", "vm_rss_kib", "fd_count",
    "temperature_edge_c", "average_socket_power_w", "average_gfxclk_mhz",
    "current_gfxclk_mhz", "average_gfx_activity_percent",
    "throttle_residency_prochot", "throttle_residency_spl",
    "throttle_residency_fppt", "throttle_residency_sppt",
    "throttle_residency_thm_core", "throttle_residency_thm_gfx",
    "throttle_residency_thm_soc", "read_bytes",
    "write_bytes", "kfd_competitor_count", "kfd_competitor_pids",
)
CONTROL_PREFIXES = (
    "DFLASH_", "GGML_", "HIP_", "ROCR_", "HSA_", "AMD_", "ROC_",
    "ROCBLAS_", "TENSILE_", "OMP_", "CUDA_", "GPU_",
)
IMPLEMENTATION_PATHS = (
    "scripts/run_kimi_k3_m1a_telemetry.py",
    "scripts/analyze_kimi_k3_m1a_telemetry.py",
)
EXPECTED_GPU_METRICS_PATH = Path("/sys/class/drm/card2/device/gpu_metrics")
EXPECTED_GPU_BDF = "0000:c5:00.0"
EXPECTED_GPU_DEVICE = "0x1586"
GPU_METRICS_SIZE_V3_0 = 264
THROTTLE_RESIDENCY_FIELDS = (
    "throttle_residency_prochot", "throttle_residency_spl",
    "throttle_residency_fppt", "throttle_residency_sppt",
    "throttle_residency_thm_core", "throttle_residency_thm_gfx",
    "throttle_residency_thm_soc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_values(path: Path, separator: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.replace(":", separator, 1).split(separator, 1)
        if len(fields) != 2:
            continue
        try:
            values[fields[0].strip()] = int(fields[1].strip().split()[0])
        except (ValueError, IndexError):
            continue
    return values


def vmstat() -> dict[str, int]:
    values = read_values(Path("/proc/vmstat"), " ")
    return {key: values[key] for key in ("pswpin", "pswpout")}


def child_status(pid: int) -> dict[str, int] | None:
    try:
        return read_values(Path(f"/proc/{pid}/status"), ":")
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_io(pid: int) -> dict[str, int] | None:
    try:
        values = read_values(Path(f"/proc/{pid}/io"), ":")
    except (FileNotFoundError, ProcessLookupError):
        return None
    if not all(key in values for key in ("read_bytes", "write_bytes")):
        return None
    return {key: values[key] for key in ("read_bytes", "write_bytes")}


def process_group_pids(pgid: int) -> set[int]:
    result: set[int] = set()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            # /proc/PID/stat field 5 is process group.  The command field may
            # contain spaces and ')' characters, so split only after the last ).
            suffix = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if int(suffix[2]) == pgid:
                result.add(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                ValueError, IndexError):
            continue
    return result


def kfd_processes(excluded: set[int]) -> list[int]:
    result: list[int] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) in excluded:
            continue
        try:
            if any(os.readlink(handle) == "/dev/kfd"
                   for handle in (process / "fd").iterdir()):
                result.append(int(process.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(result)


def cgroup_path(pid: int) -> Path:
    relative: str | None = None
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        hierarchy, controllers, value = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            relative = value
            break
    if relative is None:
        raise RuntimeError("process is not in cgroup v2")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        if right.split()[0] != "cgroup2":
            continue
        fields = left.split()
        mount_root = fields[3].rstrip("/") or "/"
        mount_point = Path(fields[4])
        if relative == mount_root:
            return mount_point
        prefix = "/" if mount_root == "/" else mount_root + "/"
        if relative.startswith(prefix):
            return mount_point / relative[len(prefix):]
    raise RuntimeError("cgroup v2 mount is absent")


def gpu_metrics_identity(path: Path) -> dict[str, str]:
    if path != EXPECTED_GPU_METRICS_PATH or path.is_symlink() or not path.is_file():
        raise ValueError("gpu_metrics must be physical GPU1 card2 regular node")
    device_dir = path.parent
    resolved_device = device_dir.resolve(strict=True)
    bdf = resolved_device.name.lower()
    device_id = (device_dir / "device").read_text().strip().lower()
    if bdf != EXPECTED_GPU_BDF or device_id != EXPECTED_GPU_DEVICE:
        raise ValueError("gpu_metrics node GPU BDF/device mismatch")
    if path.resolve(strict=True) != resolved_device / "gpu_metrics":
        raise ValueError("gpu_metrics node does not belong to expected device")
    return {"class_path": str(path), "resolved_device": str(resolved_device),
            "pci_bdf": bdf, "pci_device": device_id}


def gpu_metrics(path: Path) -> dict[str, int]:
    raw = path.read_bytes()
    if len(raw) != GPU_METRICS_SIZE_V3_0:
        raise ValueError("unexpected gpu_metrics size")
    size, format_revision, content_revision = struct.unpack_from("<HBB", raw)
    if (size, format_revision, content_revision) != (
            GPU_METRICS_SIZE_V3_0, 3, 0):
        raise ValueError("unexpected gpu_metrics ABI")
    throttle_residencies = struct.unpack_from("<7I", raw, 228)
    result = {
        # v3.0 reports temperature in centi-C and power in mW.  Preserve the
        # v1 harness CSV units with conservative integer conversion.
        "temperature_edge_c": struct.unpack_from("<H", raw, 4)[0] // 100,
        "average_socket_power_w": struct.unpack_from("<I", raw, 112)[0] // 1000,
        "average_gfxclk_mhz": struct.unpack_from("<H", raw, 174)[0],
        # v3.0 exposes the current enforced GFX maximum, not an instantaneous
        # GFX clock.  This field remains descriptive and is named compatibly
        # with the frozen CSV schema.
        "current_gfxclk_mhz": struct.unpack_from("<H", raw, 224)[0],
        "average_gfx_activity_percent": struct.unpack_from("<H", raw, 42)[0],
    }
    result.update(zip(THROTTLE_RESIDENCY_FIELDS, throttle_residencies))
    return result


def wait_for_status_or_exit(
        process: subprocess.Popen[bytes], timeout_s: float = 1.0,
) -> tuple[dict[str, int] | None, int | None]:
    """Return a live complete status, a reaped exit, or (None, None)."""
    deadline = time.monotonic() + timeout_s
    while True:
        before = process.poll()
        status = child_status(process.pid)
        after = process.poll()
        if status is not None and "VmSwap" in status and "VmRSS" in status:
            if after is not None:
                return None, after
            return status, None
        if before is not None or after is not None:
            return None, after if after is not None else before
        if time.monotonic() >= deadline:
            return None, None
        time.sleep(0.01)


def sample_live_row(
        process: subprocess.Popen[bytes], status: dict[str, int],
        metrics_path: Path, excluded: set[int],
) -> tuple[dict[str, int | str] | None, int | None, set[int], str | None]:
    # Poll after every fallible /proc/device read.  A row that overlaps exit is
    # discarded; a missing field while the child remains live is fatal.
    if process.poll() is not None:
        return None, process.returncode, set(), None
    now = time.monotonic_ns()
    try:
        metrics = gpu_metrics(metrics_path)
        io_values = process_io(process.pid)
        fd_count = len(list(Path(f"/proc/{process.pid}/fd").iterdir()))
        competitors = set(kfd_processes(excluded))
    except (FileNotFoundError, ProcessLookupError):
        exit_code = process.poll()
        if exit_code is not None:
            return None, exit_code, set(), None
        return None, None, set(), "live child telemetry path disappeared"
    exit_code = process.poll()
    if exit_code is not None:
        return None, exit_code, competitors, None
    if io_values is None:
        return None, None, competitors, "live child lacks process I/O"
    return ({
        "monotonic_ns": now,
        "vm_swap_kib": status["VmSwap"],
        "vm_rss_kib": status["VmRSS"],
        "fd_count": fd_count,
        **metrics,
        **io_values,
        "kfd_competitor_count": len(competitors),
        "kfd_competitor_pids": ";".join(str(pid) for pid in competitors),
    }, None, competitors, None)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(partial, path)
    partial.unlink()


def atomic_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(partial, path)
    partial.unlink()


def publish_partial(partial: Path, final: Path) -> None:
    with partial.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.link(partial, final)
    partial.unlink()


def classify(
        *, child_exit_code: int | None, monitor_error: str | None,
        rows: list[dict[str, int | str]], start_ns: int, end_ns: int,
        interval_ms: int, vm_before: dict[str, int], vm_after: dict[str, int],
        cgroup_before: dict[str, int], cgroup_after: dict[str, int],
        kfd_before: list[int], kfd_after: list[int], kfd_seen: set[int],
) -> dict[str, Any]:
    pswpin = vm_after.get("pswpin", -1) - vm_before.get("pswpin", 0)
    pswpout = vm_after.get("pswpout", -1) - vm_before.get("pswpout", 0)
    oom = cgroup_after.get("oom", -1) - cgroup_before.get("oom", 0)
    oom_kill = cgroup_after.get("oom_kill", -1) - cgroup_before.get("oom_kill", 0)
    throttle_count = 0
    throttle_counter_resets = 0
    for previous, current in zip(rows, rows[1:]):
        deltas = [int(current[field]) - int(previous[field])
                  for field in THROTTLE_RESIDENCY_FIELDS]
        throttle_counter_resets += int(any(delta < 0 for delta in deltas))
        throttle_count += int(any(delta > 0 for delta in deltas))
    measured_intervals = max(0, len(rows) - 1)
    throttle_fraction = (throttle_count / measured_intervals
                         if measured_intervals else 1.0)
    times = [int(row["monotonic_ns"]) for row in rows]
    gaps = ([times[0] - start_ns, end_ns - times[-1]] +
            [b - a for a, b in zip(times, times[1:])]) if times else []
    gap_limit_ns = max(2_000_000_000, 2 * interval_ms * 1_000_000)
    max_gap_ns = max(gaps) if gaps else None
    process_swap_clean = bool(rows) and all(
        int(row["vm_swap_kib"]) == 0 for row in rows)
    kfd_clean = not (kfd_before or kfd_after or kfd_seen) and bool(rows) and all(
        int(row["kfd_competitor_count"]) == 0 for row in rows)
    common_reasons: list[str] = []
    if child_exit_code != 0:
        common_reasons.append("child_exit_nonzero")
    if monitor_error is not None:
        common_reasons.append("monitor_error")
    if not rows:
        common_reasons.append("no_complete_live_samples")
    if max_gap_ns is None or max_gap_ns > gap_limit_ns:
        common_reasons.append("sampling_gap")
    if pswpin < 0 or pswpout < 0 or oom < 0 or oom_kill < 0:
        common_reasons.append("counter_regression_or_missing")
    if not process_swap_clean:
        common_reasons.append("target_process_swap")
    if oom != 0 or oom_kill != 0:
        common_reasons.append("cgroup_oom")
    if not kfd_clean:
        common_reasons.append("foreign_kfd_activity")
    if throttle_counter_resets:
        common_reasons.append("throttle_counter_decrease_or_reset")
    if throttle_fraction > 0.20:
        common_reasons.append("throttle_fraction_over_20_percent")
    operational = not common_reasons
    timing_reasons = list(common_reasons)
    if pswpin != 0 or pswpout != 0:
        timing_reasons.append("host_paging_annotation")
    if throttle_count != 0:
        timing_reasons.append("throttle_annotation")
    timing = not timing_reasons
    return {
        "operational_soak_valid": operational,
        "performance_timing_valid": timing,
        "operational_invalid_reasons": common_reasons,
        "performance_invalid_reasons": timing_reasons,
        "throttled_samples": throttle_count,
        "throttle_measured_intervals": measured_intervals,
        "throttle_counter_resets": throttle_counter_resets,
        "throttle_fraction": throttle_fraction,
        "throttle_annotation": 0 < throttle_fraction <= 0.20,
        "swap_in_pages": pswpin,
        "swap_out_pages": pswpout,
        "host_paging_annotation": pswpin != 0 or pswpout != 0,
        "host_page_in_bytes": max(0, pswpin) * os.sysconf("SC_PAGE_SIZE"),
        "host_page_in_over_1_mib":
            max(0, pswpin) * os.sysconf("SC_PAGE_SIZE") > 1024 ** 2,
        "oom_delta": oom,
        "oom_kill_delta": oom_kill,
        "max_sample_gap_ns": max_gap_ns,
        "sample_gap_limit_ns": gap_limit_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=30_000)
    parser.add_argument(
        "--gpu-metrics", type=Path,
        default=EXPECTED_GPU_METRICS_PATH)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if not 50 <= args.interval_ms <= 30_000:
        parser.error("interval-ms must be in [50,30000]")
    outputs = [args.manifest, args.samples, args.stdout, args.stderr]
    partials = [path.with_name(path.name + ".partial") for path in outputs]
    if len({str(path.resolve()) for path in outputs}) != len(outputs) or any(
            path.exists() or partial.exists()
            for path, partial in zip(outputs, partials)):
        parser.error("output paths must be distinct and fresh")

    controls = {key: value for key, value in os.environ.items()
                if key.startswith(CONTROL_PREFIXES)}
    repo = Path(__file__).resolve().parent.parent
    try:
        metrics_identity_before = gpu_metrics_identity(args.gpu_metrics)
        # Fail before launching the child if the physical node exposes any
        # other ABI.  Later samples repeat the same strict parse.
        gpu_metrics(args.gpu_metrics)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    implementation_before = {
        relative: sha256(repo / relative) for relative in IMPLEMENTATION_PATHS
    }
    parent_cgroup = cgroup_path(os.getpid())
    before_events = read_values(parent_cgroup / "memory.events", " ")
    before_vm = vmstat()
    competitors_before = kfd_processes({os.getpid(), os.getppid()})
    start_ns = time.monotonic_ns()
    rows: list[dict[str, int | str]] = []
    competitors_seen: set[int] = set()
    monitor_error: str | None = None
    child_exit: int | None = None
    child_cgroup: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    stdout_partial = args.stdout.with_name(args.stdout.name + ".partial")
    stderr_partial = args.stderr.with_name(args.stderr.name + ".partial")

    with stdout_partial.open("xb") as stdout, stderr_partial.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                args.command, stdout=stdout, stderr=stderr,
                start_new_session=True)
            child_cgroup = cgroup_path(process.pid)
            if child_cgroup != parent_cgroup:
                raise RuntimeError("wrapper and child cgroups differ")
            while True:
                status, exit_code = wait_for_status_or_exit(process)
                if exit_code is not None:
                    child_exit = exit_code
                    break
                if status is None:
                    monitor_error = "live child lacks VmSwap/VmRSS"
                    os.killpg(process.pid, signal.SIGTERM)
                    break
                excluded = {os.getpid(), os.getppid()} | \
                    process_group_pids(process.pid)
                row, sample_exit, competitors, error = sample_live_row(
                    process, status, args.gpu_metrics, excluded)
                competitors_seen.update(competitors)
                if sample_exit is not None:
                    child_exit = sample_exit
                    break
                if error is not None:
                    monitor_error = error
                    os.killpg(process.pid, signal.SIGTERM)
                    break
                if row is not None:
                    rows.append(row)
                time.sleep(args.interval_ms / 1000.0)
            if child_exit is None:
                child_exit = process.wait(timeout=30)
        except Exception as error:
            monitor_error = f"{type(error).__name__}: {error}"
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    child_exit = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    child_exit = process.wait()

    end_ns = time.monotonic_ns()
    after_vm = vmstat()
    after_events: dict[str, int] = {}
    try:
        after_events = read_values(parent_cgroup / "memory.events", " ")
    except OSError as error:
        monitor_error = monitor_error or f"cgroup post-read: {error}"
    competitors_after = kfd_processes({os.getpid(), os.getppid()})
    implementation_after = {
        relative: sha256(repo / relative) for relative in IMPLEMENTATION_PATHS
    }
    if implementation_after != implementation_before:
        monitor_error = monitor_error or "telemetry implementation changed"
    metrics_identity_after: dict[str, str] | None = None
    try:
        metrics_identity_after = gpu_metrics_identity(args.gpu_metrics)
        gpu_metrics(args.gpu_metrics)
        if metrics_identity_after != metrics_identity_before:
            monitor_error = monitor_error or "gpu_metrics identity changed"
    except (OSError, ValueError) as error:
        monitor_error = monitor_error or f"gpu_metrics post-read: {error}"
    atomic_csv(args.samples, rows)
    publish_partial(stdout_partial, args.stdout)
    publish_partial(stderr_partial, args.stderr)
    validity = classify(
        child_exit_code=child_exit, monitor_error=monitor_error, rows=rows,
        start_ns=start_ns, end_ns=end_ns, interval_ms=args.interval_ms,
        vm_before=before_vm, vm_after=after_vm,
        cgroup_before=before_events, cgroup_after=after_events,
        kfd_before=competitors_before, kfd_after=competitors_after,
        kfd_seen=competitors_seen)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "prospective_m1a_only",
        "historical_evidence_reclassified": False,
        "wrapper_complete": True,
        "command": args.command,
        "effective_control_environment": controls,
        "implementation_hashes": implementation_before,
        "implementation_hashes_after": implementation_after,
        "implementation_hashes_stable":
            implementation_before == implementation_after,
        "monitor_error": monitor_error,
        "child_exit_code": child_exit,
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": end_ns,
        "interval_ms": args.interval_ms,
        "gpu_metrics_path": str(args.gpu_metrics),
        "gpu_metrics_identity": metrics_identity_before,
        "gpu_metrics_identity_after": metrics_identity_after,
        "gpu_metrics_abi": {"structure_size": GPU_METRICS_SIZE_V3_0,
                            "format_revision": 3, "content_revision": 0},
        "sample_count": len(rows),
        "samples": str(args.samples.resolve()),
        "samples_sha256": sha256(args.samples),
        "stdout": str(args.stdout.resolve()),
        "stdout_sha256": sha256(args.stdout),
        "stderr": str(args.stderr.resolve()),
        "stderr_sha256": sha256(args.stderr),
        "vmstat_before": before_vm,
        "vmstat_after": after_vm,
        "cgroup_path": str(child_cgroup) if child_cgroup else None,
        "wrapper_cgroup_path": str(parent_cgroup),
        "cgroup_before": before_events,
        "cgroup_after": after_events,
        "kfd_competitors_before": competitors_before,
        "kfd_competitors_after": competitors_after,
        "kfd_competitor_pids_seen": sorted(competitors_seen),
        "validity": validity,
    }
    atomic_json(args.manifest, manifest)
    return 0 if validity["operational_soak_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
