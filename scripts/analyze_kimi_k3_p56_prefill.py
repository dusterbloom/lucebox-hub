#!/usr/bin/env python3
"""Separate K3 prefill/decode counters and model layer-major I/O geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


GIB = 1024**3
P56 = re.compile(r"\[kimi-k3-p56\] (?P<body>.*)")
STAGE = re.compile(r"\[kimi-k3-stage\] (?P<body>.*)")
WIDTHS = (1, 2, 4, 8, 16, 32, 64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key_values(body: str) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {}
    for item in body.split():
        key, value = item.split("=", 1)
        if key == "phase":
            result[key] = value
        elif any(character in value for character in ".eE"):
            result[key] = float(value)
        else:
            result[key] = int(value)
    return result


def parse_lines(path: Path, pattern: re.Pattern[str]) -> list[dict[str, int | float | str]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append(key_values(match.group("body")))
    return rows


def sum_census(rows: list[dict[str, int | float | str]], phase: str) -> dict[str, int | float]:
    selected = [row for row in rows if row["phase"] == phase]
    if not selected:
        raise ValueError(f"no P56 {phase} counters found")
    result: dict[str, int | float] = {}
    for key in selected[0]:
        if key == "phase" or key == "positions-per-second":
            continue
        result[key] = sum(row[key] for row in selected)  # type: ignore[arg-type]
    seconds = float(result["seconds"])
    positions = int(result["positions"])
    result["positions-per-second"] = positions / seconds if seconds else 0.0
    result["sequence-count"] = len(selected)
    return result


def stage_summary(
    rows: list[dict[str, int | float | str]], manifest: dict
) -> dict[str, dict[str, float | int]]:
    cursor = 0
    phases: dict[str, list[dict[str, int | float | str]]] = {
        "prefill": [], "decode": []}
    for sequence in manifest["sequences"]:
        prompt = int(sequence["prompt_token_count"])
        generated = len(sequence["output_tokens"])
        count = prompt + max(0, generated - 1)
        chunk = []
        consumed = 0
        while consumed < count:
            if cursor >= len(rows):
                raise ValueError("stage rows end before the suite manifest")
            row = rows[cursor]
            cursor += 1
            if int(row["position"]) != consumed:
                raise ValueError("stage positions do not reset at each sequence")
            tokens = int(row["tokens"])
            if tokens <= 0 or consumed + tokens > count:
                raise ValueError("stage row crosses a sequence boundary")
            chunk.append(row)
            consumed += tokens
        consumed = 0
        for row in chunk:
            tokens = int(row["tokens"])
            if consumed < prompt < consumed + tokens:
                raise ValueError("stage row crosses the prefill/decode boundary")
            target = "prefill" if consumed < prompt else "decode"
            phases[target].append(row)
            consumed += tokens
        if consumed != count:
            raise ValueError("stage rows cross a sequence boundary")
    if cursor != len(rows):
        raise ValueError("unassigned stage rows remain")
    fields = (
        "total_ms", "embedding_ms", "dense_ms", "routed_prep_ms",
        "offload_prep_ms", "experts_ms", "join_ms", "output_ms", "other_ms")
    result = {}
    for phase, phase_rows in phases.items():
        tokens = sum(int(row["tokens"]) for row in phase_rows)
        values: dict[str, float | int] = {"positions": tokens, "forwards": len(phase_rows)}
        for field in fields:
            values[f"{field}_per_position"] = (
                sum(float(row[field]) for row in phase_rows) / tokens if tokens else 0.0)
            normalized = [float(row[field]) / int(row["tokens"]) for row in phase_rows]
            values[f"{field}_median_forward_normalized"] = (
                statistics.median(normalized) if normalized else 0.0)
        result[phase] = values
    return result


def prefill_io_geometry(trace: Path, prompt_lengths: list[int]) -> dict:
    requests: list[tuple[int, int, int, str, int, int]] = []
    jobs: set[tuple[int, int, int, int, int]] = set()
    sequence = 0
    last_base: int | None = None
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            base = int(row["base_pos"])
            if last_base is not None and base < last_base:
                sequence += 1
            last_base = base
            layer = int(row["model_layer"])
            expert = int(row["expert_id"])
            token = int(row["token_index"])
            position = base + token
            if (sequence >= len(prompt_lengths) or
                    position >= prompt_lengths[sequence]):
                continue
            if row["region"] in {"gate", "up", "down", "native-exact-expert"}:
                jobs.add((sequence, position, token, layer, expert))
            physical = int(row["explicit_read_bytes"])
            offset = int(row["aligned_offset"])
            length = int(row["aligned_length"])
            if physical > 0 and offset >= 0 and length > 0:
                requests.append((sequence, position, layer, row["file_path"], offset, length))

    widths = {}
    for width in WIDTHS:
        groups: dict[tuple[int, int, int], set[tuple[str, int, int]]] = defaultdict(set)
        for sequence, base, layer, path, offset, length in requests:
            groups[(sequence, layer, base // width)].add((path, offset, length))
        unique_bytes = 0
        coalesced_requests = 0
        for ranges in groups.values():
            by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for path, offset, length in ranges:
                by_path[path].append((offset, offset + length))
            for intervals in by_path.values():
                current_end = -1
                for start, end in sorted(intervals):
                    if start > current_end:
                        coalesced_requests += 1
                        unique_bytes += end - start
                        current_end = end
                    elif end > current_end:
                        unique_bytes += end - current_end
                        current_end = end
        widths[str(width)] = {
            "layer_macro_groups": len(groups),
            "deduplicated_coalesced_requests": coalesced_requests,
            "deduplicated_physical_bytes": unique_bytes,
        }
    return {
        "status": "TRACE_GEOMETRY_NOT_TIMED_REPLAY",
        "physical_read_events": len(requests),
        "compact_jobs": len(jobs),
        "macro_widths": widths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--io-trace", type=Path)
    args = parser.parse_args()
    manifest_path = args.root / "suite" / "suite-manifest.json"
    stderr_path = args.root / "stderr.log"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("draft_path"):
        raise ValueError("P56 phase attribution requires speculative decoding off")
    census_rows = parse_lines(stderr_path, P56)
    stages = stage_summary(parse_lines(stderr_path, STAGE), manifest)
    prefill = sum_census(census_rows, "prefill")
    decode = sum_census(census_rows, "decode")
    physical_per_position = (
        int(prefill["physical-direct-read-bytes"]) / int(prefill["positions"]))
    baseline_rate = float(prefill["positions-per-second"])
    target_rate = baseline_rate * 10.0
    result = {
        "schema": "k3-p56-prefill-census-v1",
        "status": "MEASURED_PHASE_SPLIT",
        "scope": "NON_SPECULATIVE_PREFILL_AND_AR_DECODE",
        "provenance": {
            "root": str(args.root),
            "manifest_sha256": sha256(manifest_path),
            "stderr_sha256": sha256(stderr_path),
            "repository_commit": manifest.get("environment", {}).get(
                "KIMI_H16_REPOSITORY_COMMIT"),
        },
        "prefill": prefill,
        "decode": decode,
        "stages": stages,
        "ten_x_cold_target": {
            "baseline_positions_per_second": baseline_rate,
            "target_positions_per_second": target_rate,
            "target_suite_seconds": float(prefill["seconds"]) / 10.0,
            "physical_bytes_per_position": physical_per_position,
            "unoverlapped_storage_gib_per_second_required": (
                physical_per_position * target_rate / GIB),
        },
    }
    trace = args.io_trace or args.root / "io_trace.tsv"
    if trace.is_file():
        result["io_geometry"] = prefill_io_geometry(
            trace, [int(row["prompt_token_count"]) for row in manifest["sequences"]])
        result["provenance"]["io_trace_sha256"] = sha256(trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
