#!/usr/bin/env python3
"""Run one command while recording process, graphics, disk, and energy data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            field = raw.strip().split()[0]
            if field.isdigit():
                values[key] = int(field)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return values


def process_sample(pid: int) -> dict[str, int | None]:
    status = read_key_values(Path(f"/proc/{pid}/status"))
    io = read_key_values(Path(f"/proc/{pid}/io"))
    return {
        "rss_kib": status.get("VmRSS"),
        "high_water_rss_kib": status.get("VmHWM"),
        "read_bytes": io.get("read_bytes"),
        "write_bytes": io.get("write_bytes"),
    }


def system_memory_sample() -> dict[str, int | None]:
    memory = read_key_values(Path("/proc/meminfo"))
    total = memory.get("MemTotal")
    available = memory.get("MemAvailable")
    return {
        "total_kib": total,
        "available_kib": available,
        "used_kib": total - available
        if total is not None and available is not None
        else None,
    }


def graphics_sample(gpu: int) -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
        if len(fields) != 3:
            raise ValueError("unexpected graphics query shape")
        return {
            "utilization_percent": float(fields[0]),
            "memory_mib": float(fields[1]),
            "power_watts": float(fields[2]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {
            "utilization_percent": None,
            "memory_mib": None,
            "power_watts": None,
        }


def resolve_block_device(mount_path: Path) -> str | None:
    try:
        source = subprocess.run(
            ["findmnt", "-no", "SOURCE", "--target", str(mount_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not source.startswith("/dev/"):
            return None
        parent = subprocess.run(
            ["lsblk", "-no", "PKNAME", source],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return parent or Path(source).name
    except (OSError, subprocess.SubprocessError):
        return None


def block_sample(device: str | None) -> list[int] | None:
    if not device:
        return None
    try:
        values = [int(value) for value in Path(
            f"/sys/class/block/{device}/stat"
        ).read_text().split()]
        return values if len(values) >= 11 else None
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def processor_energy_domains() -> dict[str, tuple[Path, int]]:
    domains: dict[str, tuple[Path, int]] = {}
    for energy_path in Path("/sys/class/powercap").glob("**/energy_uj"):
        try:
            name = energy_path.with_name("name").read_text().strip()
            maximum = int(energy_path.with_name("max_energy_range_uj").read_text())
            domains[name] = (energy_path, maximum)
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return domains


def energy_values(domains: dict[str, tuple[Path, int]]) -> dict[str, int]:
    values: dict[str, int] = {}
    for name, (path, _) in domains.items():
        try:
            values[name] = int(path.read_text())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return values


def energy_delta(
    before: dict[str, int],
    after: dict[str, int],
    domains: dict[str, tuple[Path, int]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in before.keys() & after.keys():
        maximum = domains[name][1]
        delta = after[name] - before[name]
        if delta < 0:
            delta += maximum
        result[name] = delta / 1_000_000.0
    return result


def maximum(samples: list[dict[str, Any]], key: str) -> float | int | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--mount-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.interval <= 0:
        parser.error("a command and a positive interval are required")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.samples_csv.parent.mkdir(parents=True, exist_ok=True)
    block_device = resolve_block_device(args.mount_path)
    block_before = block_sample(block_device)
    energy_domains = processor_energy_domains()
    processor_energy_before = energy_values(energy_domains)

    started_wall = time.time()
    started = time.monotonic()
    if args.stdout:
        args.stdout.parent.mkdir(parents=True, exist_ok=True)
    if args.stderr:
        args.stderr.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = args.stdout.open("w") if args.stdout else None
    stderr_handle = args.stderr.open("w") if args.stderr else None
    process = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
    )
    samples: list[dict[str, Any]] = []
    graphics_energy_joules = 0.0
    previous_graphics_time: float | None = None
    previous_graphics_power: float | None = None
    last_process_io: dict[str, int | None] = {}
    try:
        while True:
            now = time.monotonic()
            process_data = process_sample(process.pid)
            memory_data = system_memory_sample()
            graphics_data = graphics_sample(args.gpu)
            power = graphics_data["power_watts"]
            if (
                power is not None
                and previous_graphics_time is not None
                and previous_graphics_power is not None
            ):
                graphics_energy_joules += (
                    (power + previous_graphics_power)
                    * 0.5
                    * (now - previous_graphics_time)
                )
            if power is not None:
                previous_graphics_time = now
                previous_graphics_power = power
            last_process_io = process_data
            samples.append(
                {
                    "elapsed_seconds": now - started,
                    **process_data,
                    "system_used_kib": memory_data["used_kib"],
                    "graphics_utilization_percent": graphics_data[
                        "utilization_percent"
                    ],
                    "graphics_memory_mib": graphics_data["memory_mib"],
                    "graphics_power_watts": power,
                }
            )
            if process.poll() is not None:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    finally:
        if stdout_handle:
            stdout_handle.close()
        if stderr_handle:
            stderr_handle.close()

    ended = time.monotonic()
    block_after = block_sample(block_device)
    processor_energy_after = energy_values(energy_domains)
    disk: dict[str, Any] = {"device": block_device, "available": False}
    if block_before is not None and block_after is not None:
        disk = {
            "device": block_device,
            "available": True,
            "read_operations": block_after[0] - block_before[0],
            "read_bytes": (block_after[2] - block_before[2]) * 512,
            "write_operations": block_after[4] - block_before[4],
            "write_bytes": (block_after[6] - block_before[6]) * 512,
            "busy_milliseconds": block_after[9] - block_before[9],
        }

    result = {
        "schema": "lucebox-command-telemetry-v1",
        "command": command,
        "command_shell": shlex.join(command),
        "pid": process.pid,
        "exit_code": process.returncode,
        "started_unix_seconds": started_wall,
        "elapsed_seconds": ended - started,
        "sample_interval_seconds": args.interval,
        "sample_count": len(samples),
        "process": {
            "peak_rss_kib": maximum(samples, "rss_kib"),
            "reported_high_water_rss_kib": maximum(
                samples, "high_water_rss_kib"
            ),
            "last_read_bytes": last_process_io.get("read_bytes"),
            "last_write_bytes": last_process_io.get("write_bytes"),
        },
        "system_memory": {
            "peak_used_kib": maximum(samples, "system_used_kib"),
        },
        "graphics": {
            "index": args.gpu,
            "available": any(
                sample["graphics_power_watts"] is not None for sample in samples
            ),
            "peak_memory_mib": maximum(samples, "graphics_memory_mib"),
            "peak_utilization_percent": maximum(
                samples, "graphics_utilization_percent"
            ),
            "peak_power_watts": maximum(samples, "graphics_power_watts"),
            "integrated_energy_joules": graphics_energy_joules
            if previous_graphics_power is not None
            else None,
            "energy_method": "trapezoidal integration of sampled board power"
            if previous_graphics_power is not None
            else "unavailable",
        },
        "processor_energy": {
            "available": bool(processor_energy_before and processor_energy_after),
            "joules_by_domain": energy_delta(
                processor_energy_before, processor_energy_after, energy_domains
            ),
            "reason": None
            if processor_energy_before and processor_energy_after
            else "no readable processor energy counter was exposed",
        },
        "disk": disk,
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    with args.samples_csv.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
