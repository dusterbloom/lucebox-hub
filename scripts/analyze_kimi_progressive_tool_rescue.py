#!/usr/bin/env python3
"""Validate the preregistered K3 singleton tool-rescue gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from pathlib import Path


P20 = re.compile(
    r"explicit-provider-reads=(\d+).*direct-physical-bytes=(\d+).*"
    r"direct-io-ns=(\d+)")
TRACE = re.compile(r"\[server-token-trace\].* (prompt_ids|generated_ids)=([0-9,]+)")
RESCUE = re.compile(r"\[kimi-k3-progressive-rescue\] base-pos=(\d+) slab-budget=(\d+)")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_manifest(root: Path) -> None:
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        if digest(path) != expected:
            raise ValueError(f"checksum mismatch: {path}")


def read_environment(path: Path) -> dict[str, str]:
    result = {}
    for item in path.read_bytes().split(b"\0"):
        if item:
            key, value = item.decode().split("=", 1)
            result[key] = value
    return result


def token_traces(stderr: str) -> dict[str, list[int]]:
    found = {name: [int(value) for value in values.split(",")]
             for name, values in TRACE.findall(stderr)}
    if found.keys() != {"prompt_ids", "generated_ids"}:
        raise ValueError("expected one prompt and generated token trace")
    return found


def i32le_digest(values: list[int]) -> str:
    return hashlib.sha256(
        struct.pack(f"<{len(values)}i", *values)).hexdigest()


def traffic(path: Path) -> dict[str, int | float]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if len(rows) != 92 or [int(row["model_layer"]) for row in rows] != list(range(1, 93)):
        raise ValueError("traffic must contain routed layers 1 through 92")
    positions = {int(row["tokens"]) for row in rows}
    if len(positions) != 1:
        raise ValueError("traffic rows disagree on provider positions")
    positions_value = positions.pop()
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "requested_nominal_slabs", "selected_slab_records",
            "selected_sidecar_bytes", "exact_fallback_bytes",
            "total_provider_bytes")
    }
    totals["provider_positions"] = positions_value
    totals["logical_bytes_per_position"] = (
        totals["total_provider_bytes"] / positions_value)
    totals["logical_gib_per_position"] = (
        totals["logical_bytes_per_position"] / (1024 ** 3))
    totals["fallback_gib_per_position"] = (
        totals["exact_fallback_bytes"] / positions_value / (1024 ** 3))
    return totals


def valid_weather_call(response: dict) -> tuple[bool, dict]:
    choice = response["choices"][0]
    calls = choice["message"].get("tool_calls", [])
    if len(calls) != 1:
        return False, {"finish_reason": choice["finish_reason"], "tool_calls": len(calls)}
    call = calls[0]
    try:
        arguments = json.loads(call["function"]["arguments"])
    except (KeyError, json.JSONDecodeError):
        arguments = None
    details = {
        "finish_reason": choice["finish_reason"],
        "tool_calls": len(calls),
        "tool_name": call.get("function", {}).get("name"),
        "arguments": arguments,
    }
    valid = (
        details["finish_reason"] == "tool_calls"
        and details["tool_name"] == "get_weather"
        and isinstance(arguments, dict)
        and arguments.get("location") == "San Francisco"
    )
    return valid, details


def analyze_arm(root: Path, expected_positions: str,
                expected_prompt_sha: str) -> dict:
    verify_manifest(root)
    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    if len(traces["prompt_ids"]) != 147:
        raise ValueError(f"{root}: prompt length changed")
    if i32le_digest(traces["prompt_ids"]) != expected_prompt_sha:
        raise ValueError(f"{root}: prompt token hash changed")
    environment = read_environment(root / "environment.nul")
    if environment.get("DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS", "") != expected_positions:
        raise ValueError(f"{root}: position override changed")
    if environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_REQUEST_B24") != "1":
        raise ValueError(f"{root}: request-wide B96 was not disabled")
    markers = [(int(pos), int(budget)) for pos, budget in RESCUE.findall(stderr)]
    if "request minimum slab budget=96" in stderr:
        raise ValueError(f"{root}: request-wide B96 marker found")
    p20 = P20.findall(stderr)
    if len(p20) != 1:
        raise ValueError(f"{root}: expected one P20 summary")
    explicit_reads, physical_bytes, direct_io_ns = map(int, p20[0])
    response = json.loads((root / "response.json").read_text())
    valid, tool = valid_weather_call(response)
    byte_metrics = traffic(root / "traffic.tsv")
    byte_metrics.update({
        "explicit_provider_reads": explicit_reads,
        "direct_physical_bytes": physical_bytes,
        "direct_physical_gib_per_position": (
            physical_bytes / byte_metrics["provider_positions"] / (1024 ** 3)),
        "direct_io_ns": direct_io_ns,
    })
    return {
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "command_sha256": digest(root / "command.nul"),
        "environment_sha256": digest(root / "environment.nul"),
        "response_sha256": digest(root / "response.json"),
        "logits_sha256": digest(root / "final.f32"),
        "traffic_sha256": digest(root / "traffic.tsv"),
        "prompt_token_ids_i32le_sha256": i32le_digest(traces["prompt_ids"]),
        "generated_ids": traces["generated_ids"],
        "rescue_markers": markers,
        "tool_call_valid": valid,
        "tool_call": tool,
        "usage": response["usage"],
        "traffic": byte_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--rescue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    expected_prompt_sha = prereg["fixture"]["prompt_token_ids_i32le_sha256"]
    base = analyze_arm(args.base_root, "", expected_prompt_sha)
    rescue = analyze_arm(args.rescue_root, "158:96", expected_prompt_sha)
    if base["source_commit"] != rescue["source_commit"]:
        raise ValueError("arms used different source commits")
    if base["executable_sha256"] != rescue["executable_sha256"]:
        raise ValueError("arms used different executables")
    if base["rescue_markers"]:
        raise ValueError("base arm contains a rescue marker")
    if rescue["rescue_markers"] != [(158, 96)]:
        raise ValueError("singleton arm marker changed")
    if base["tool_call_valid"] or not rescue["tool_call_valid"]:
        raise ValueError("tool validity did not cross the preregistered gate")

    differing_tokens = [
        index for index, pair in enumerate(zip(
            base["generated_ids"], rescue["generated_ids"]))
        if pair[0] != pair[1]
    ]
    byte_delta = {
        key: rescue["traffic"][key] - base["traffic"][key]
        for key in (
            "requested_nominal_slabs", "selected_slab_records",
            "selected_sidecar_bytes", "exact_fallback_bytes",
            "total_provider_bytes", "direct_physical_bytes", "direct_io_ns")
    }
    positions = base["traffic"]["provider_positions"]
    byte_delta.update({
        "logical_gib_total": byte_delta["total_provider_bytes"] / (1024 ** 3),
        "logical_gib_per_position": byte_delta["total_provider_bytes"] / positions / (1024 ** 3),
        "physical_gib_total": byte_delta["direct_physical_bytes"] / (1024 ** 3),
        "physical_gib_per_position": byte_delta["direct_physical_bytes"] / positions / (1024 ** 3),
    })
    result = {
        "schema": "kimi-k3-progressive-tool-rescue-result-v1",
        "status": "MEASURED_PROGRESSIVE_RESCUE_GO",
        "preregistration_sha256": digest(args.prereg),
        "source": prereg["source"],
        "base": base,
        "singleton_rescue": rescue,
        "comparison": {
            "generated_token_difference_indices_zero_based": differing_tokens,
            "base_boundary_ids": base["generated_ids"][10:13],
            "rescue_boundary_ids": rescue["generated_ids"][10:13],
            "byte_delta": byte_delta,
        },
        "gate": {
            "passed": True,
            "decision": "Skip the four-position window. Test a runtime grammar/tool-schema trigger; do not promote a label-derived position oracle.",
        },
        "limitations": [
            "The successful position is label-derived and is not a deployable risk signal.",
            "One native tool fixture does not establish broad reliability.",
            "The measured cold run is token-sequential and is not a serving-throughput claim.",
            "No terminal full-vocabulary KL teacher capture was made for this generated sequence.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": result["status"],
        "logical_gib_delta": byte_delta["logical_gib_total"],
        "physical_gib_delta": byte_delta["physical_gib_total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
