#!/usr/bin/env python3
"""Validate route12 schema rescue on tool-declared no-call controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import digest
from analyze_kimi_schema_rescue import analyze_arm


ROUTE_MARKER = re.compile(
    r"\[kimi-k3-route-limit\] top-routes=(\d+) weights=unchanged")


def plain_answer(identifier: str, response: dict) -> tuple[bool, dict]:
    choice = response["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls", [])
    content = " ".join(message.get("content", "").strip().split())
    if identifier == "ok":
        answer = content == "OK"
    elif identifier == "math":
        answer = re.search(r"(?<!\d)42(?!\d)", content) is not None
    else:
        raise ValueError(f"unregistered control {identifier}")
    return answer and not calls, {
        "finish_reason": choice["finish_reason"],
        "content": content,
        "tool_calls": len(calls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--off", action="append", required=True,
        help="control-id=artifact-root")
    parser.add_argument(
        "--on", action="append", required=True,
        help="control-id=artifact-root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    repo = Path(__file__).resolve().parent.parent
    for path, expected in (
            (Path(__file__), prereg["source"]["analyzer_sha256"]),
            (repo / "scripts/run_kimi_progressive_tool_rescue.sh",
             prereg["source"]["runner_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"registered harness hash changed: {path}")
    if digest(args.policy) != prereg["representation"]["policy_sha256"]:
        raise ValueError("Budget20 policy hash changed")
    off_roots = dict(item.split("=", 1) for item in args.off)
    on_roots = dict(item.split("=", 1) for item in args.on)
    controls = {row["id"]: row for row in prereg["controls"]}
    if set(off_roots) != set(controls) or set(on_roots) != set(controls):
        raise ValueError("control/root set does not match preregistration")

    rows = []
    all_passed = True
    source_commits = set()
    executable_hashes = set()
    for identifier in (row["id"] for row in prereg["controls"]):
        spec = controls[identifier]
        fixture = Path(spec["fixture"])
        if digest(fixture) != spec["fixture_sha256"]:
            raise ValueError(f"fixture hash changed: {identifier}")
        off = analyze_arm(Path(off_roots[identifier]))
        on = analyze_arm(Path(on_roots[identifier]))
        source_commits.update((off["source_commit"], on["source_commit"]))
        executable_hashes.update((off["executable_sha256"], on["executable_sha256"]))
        if (digest(Path(off_roots[identifier]) / "request.json") !=
                spec["fixture_sha256"] or
                digest(Path(on_roots[identifier]) / "request.json") !=
                spec["fixture_sha256"]):
            raise ValueError(f"measured fixture changed: {identifier}")
        for name, arm, enabled in (("off", off, False), ("on", on, True)):
            environment = arm["environment"]
            present = environment.get(
                "DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") == "1"
            if (present != enabled or
                    environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12" or
                    environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(args.policy) or
                    "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment):
                raise ValueError(f"environment changed: {identifier}/{name}")
            stderr = Path(arm["artifact_root"], "server.stderr").read_text()
            route_markers = [int(value) for value in ROUTE_MARKER.findall(stderr)]
            expected_configs = 1 if enabled else 0
            if (route_markers != [12] or
                    arm["schema_prefix_configurations"] != expected_configs or
                    arm["static_position_markers"] != 0 or
                    arm["request_wide_b96_markers"] != 0):
                raise ValueError(f"marker contract changed: {identifier}/{name}")
        off_response = json.loads(Path(off["artifact_root"], "response.json").read_text())
        on_response = json.loads(Path(on["artifact_root"], "response.json").read_text())
        off_valid, off_answer = plain_answer(identifier, off_response)
        on_valid, on_answer = plain_answer(identifier, on_response)
        equality = {
            "prompt_ids": off["prompt_token_ids_i32le_sha256"] ==
                          on["prompt_token_ids_i32le_sha256"],
            "generated_ids": off["generated_ids"] == on["generated_ids"],
            "final_logits": off["logits_sha256"] == on["logits_sha256"],
            "traffic": off["traffic_sha256"] == on["traffic_sha256"],
            "logical_bytes": off["traffic"]["total_provider_bytes"] ==
                             on["traffic"]["total_provider_bytes"],
            "fallback_bytes": off["traffic"]["exact_fallback_bytes"] ==
                              on["traffic"]["exact_fallback_bytes"],
            "physical_bytes": off["traffic"]["direct_physical_bytes"] ==
                              on["traffic"]["direct_physical_bytes"],
        }
        passed = (off_valid and on_valid and not on["schema_rescue_markers"] and
                  all(equality.values()))
        all_passed = all_passed and passed
        rows.append({
            "id": identifier,
            "passed": passed,
            "off": {**off, "plain_answer_valid": off_valid, "answer": off_answer},
            "on": {**on, "plain_answer_valid": on_valid, "answer": on_answer},
            "on_rescue_markers": on["schema_rescue_markers"],
            "equality": equality,
        })

    if len(source_commits) != 1:
        raise ValueError("arms used different source commits")
    if executable_hashes != {prereg["source"]["executable_sha256"]}:
        raise ValueError("arms used a different executable")
    result = {
        "schema": "kimi-k3-route12-tool-false-positive-result-v1",
        "status": ("MEASURED_ROUTE12_TOOL_FALSE_POSITIVE_GO" if all_passed else
                   "MEASURED_ROUTE12_TOOL_FALSE_POSITIVE_NO_GO"),
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": next(iter(source_commits)),
        },
        "controls": rows,
        "gate": {
            "passed": all_passed,
            "decision": (
                "Schema rescue is inert on both tool-declared no-call controls; proceed to broader structured/agentic quality."
                if all_passed else
                "Stop schema-rescue production consideration."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "controls": {row["id"]: row["passed"] for row in rows},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
