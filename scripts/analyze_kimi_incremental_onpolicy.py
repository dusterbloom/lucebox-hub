#!/usr/bin/env python3
"""Score the preregistered all-layer B16-plus-four on-policy gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_native_tool_first_token import read_logits, terminal_metrics
from analyze_kimi_progressive_tool_rescue import (
    P20,
    digest,
    read_environment,
    token_traces,
    traffic,
    verify_manifest,
)


MARKER = re.compile(
    r"\[kimi-k3-incremental-slab\] layer=(\d+) base-pos=(\d+) "
    r"token=(\d+) base=(\d+) full=(\d+) delta=(\d+)")
LOGICAL_KEYS = (
    "requested_nominal_slabs",
    "selected_slab_records",
    "selected_sidecar_bytes",
    "exact_fallback_bytes",
    "total_provider_bytes",
)


def analyze_arm(root: Path, incremental: bool) -> tuple[dict, list[float]]:
    verify_manifest(root)
    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    environment = read_environment(root / "environment.nul")
    expected_incremental = "16" if incremental else None
    if environment.get("DFLASH_KIMI_EXPERIMENT_INCREMENTAL_BASE_BUDGET") != \
            expected_incremental:
        raise ValueError(f"{root}: incremental environment changed")
    if environment.get("DFLASH_KIMI_PRODUCTION_DEFAULTS") != "0" or \
            environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12":
        raise ValueError(f"{root}: execution profile changed")
    markers = [tuple(map(int, values)) for values in MARKER.findall(stderr)]
    byte_metrics = traffic(root / "traffic.tsv")
    marker_ok = (
        len(markers) == 92 * byte_metrics["provider_positions"] and
        all(marker[3:] == (16, 20, 4) for marker in markers))
    if incremental != bool(markers):
        raise ValueError(f"{root}: incremental markers disagree with arm")
    p20 = P20.findall(stderr)
    if len(p20) != 1:
        raise ValueError(f"{root}: expected one P20 summary")
    explicit_reads, physical_bytes, direct_io_ns = map(int, p20[0])
    response = json.loads((root / "response.json").read_text())
    logits_path = root / "final.f32"
    logits = read_logits(logits_path)
    return ({
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "source_status_sha256": digest(root / "source-status.txt"),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "command_sha256": digest(root / "command.nul"),
        "environment_sha256": digest(root / "environment.nul"),
        "request_sha256": digest(root / "request.json"),
        "response_sha256": digest(root / "response.json"),
        "logits_sha256": digest(logits_path),
        "traffic_sha256": digest(root / "traffic.tsv"),
        "prompt_ids": traces["prompt_ids"],
        "generated_ids": traces["generated_ids"],
        "finish_reason": response["choices"][0]["finish_reason"],
        "content": response["choices"][0]["message"].get("content", ""),
        "usage": response["usage"],
        "incremental_marker_count": len(markers),
        "incremental_marker_contract_pass": marker_ok if incremental else not markers,
        "traffic": {
            **byte_metrics,
            "explicit_provider_reads": explicit_reads,
            "direct_physical_bytes": physical_bytes,
            "direct_io_ns": direct_io_ns,
        },
    }, logits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--fixture", action="append", required=True,
                        help="ID=CONTROL_ROOT,CANDIDATE_ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.prereg.read_text())
    expected = {fixture["id"]: fixture for fixture in prereg["fixtures"]}
    supplied = {}
    for raw in args.fixture:
        fixture_id, separator, roots = raw.partition("=")
        control, comma, candidate = roots.partition(",")
        if not separator or not comma or fixture_id in supplied:
            raise ValueError(f"invalid or duplicate --fixture: {raw}")
        supplied[fixture_id] = (Path(control), Path(candidate))
    if supplied.keys() != expected.keys():
        raise ValueError("supplied fixtures differ from preregistration")

    rows = []
    for fixture_id, fixture in expected.items():
        control, control_logits = analyze_arm(supplied[fixture_id][0], False)
        candidate, candidate_logits = analyze_arm(supplied[fixture_id][1], True)
        if control["source_commit"] != candidate["source_commit"] or \
                control["executable_sha256"] != candidate["executable_sha256"]:
            raise ValueError(f"{fixture_id}: source or executable differs")
        if not (control["request_sha256"] == candidate["request_sha256"] ==
                fixture["sha256"]):
            raise ValueError(f"{fixture_id}: request fixture changed")
        expected_ids = fixture["expected_ids"]
        aligned = control["generated_ids"] == candidate["generated_ids"] == expected_ids
        composition = terminal_metrics(control_logits, candidate_logits)
        teacher_path = Path(fixture["native_teacher_root"]) / "final.f32"
        if digest(teacher_path) != fixture["native_teacher_logits_sha256"]:
            raise ValueError(f"{fixture_id}: native teacher changed")
        teacher = read_logits(teacher_path)
        control_teacher = terminal_metrics(teacher, control_logits) if aligned else None
        candidate_teacher = terminal_metrics(teacher, candidate_logits) if aligned else None
        logical_equal = all(
            control["traffic"][key] == candidate["traffic"][key]
            for key in LOGICAL_KEYS)
        logit_equal = control["logits_sha256"] == candidate["logits_sha256"]
        structural = (
            control["incremental_marker_contract_pass"] and
            candidate["incremental_marker_contract_pass"] and logical_equal)
        behavioral = aligned and (
            fixture_id != "get-weather-first-token" or
            candidate["generated_ids"] == [163588])
        distributional = (
            logit_equal and float(composition["terminal_kl"]) == 0.0 and
            control_teacher is not None and candidate_teacher is not None and
            float(control_teacher["terminal_kl"]) ==
                float(candidate_teacher["terminal_kl"]))
        rows.append({
            "id": fixture_id,
            "control": control,
            "candidate": candidate,
            "generated_ids_aligned_and_expected": aligned,
            "logical_traffic_equal": logical_equal,
            "logical_traffic_delta": {
                key: candidate["traffic"][key] - control["traffic"][key]
                for key in LOGICAL_KEYS
            },
            "logits_byte_identical": logit_equal,
            "max_abs_logit_delta": max(
                abs(left - right)
                for left, right in zip(control_logits, candidate_logits)),
            "control_vs_candidate": composition,
            "control_vs_native_teacher": control_teacher,
            "candidate_vs_native_teacher": candidate_teacher,
            "structural_gate_pass": structural,
            "behavioral_gate_pass": behavioral,
            "distributional_gate_pass": distributional,
            "gate_pass": structural and behavioral and distributional,
        })

    passed = all(row["gate_pass"] for row in rows)
    tool = next(row for row in rows if row["id"] == "get-weather-first-token")
    result = {
        "schema": "kimi-k3-incremental-b16-b20-onpolicy-result-v1",
        "status": "MEASURED_GO" if passed else "MEASURED_NO_GO",
        "preregistration_sha256": digest(args.prereg),
        "fixtures": rows,
        "gate": {
            "passed": passed,
            "tool_boundary_recovered": tool["behavioral_gate_pass"],
            "decision": (
                "Earn a delta-only production-profile prototype; do not promote production."
                if passed else
                "Do not promote the current all-layer composition order. Preserve behavioral evidence separately from exact/distributional failure."
            ),
        },
        "limitations": [
            "Host-reference timing is not serving throughput.",
            "Equal generated tokens are not distributional equivalence.",
            "The candidate reads all 20 records and therefore measures no rescue-byte saving."
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "tool_boundary_recovered": result["gate"]["tool_boundary_recovered"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
