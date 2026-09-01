#!/usr/bin/env python3
"""Validate exact-native closure for tokenizer-misaligned route12 prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    digest,
    read_environment,
    token_traces,
    verify_manifest,
)
from analyze_kimi_route12_native_success import task_success


def exact_arm(root: Path) -> dict:
    verify_manifest(root)
    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    environment = read_environment(root / "environment.nul")
    response = json.loads((root / "response.json").read_text())
    choice = response["choices"][0]
    return {
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "environment": environment,
        "prompt_ids": traces["prompt_ids"],
        "generated_ids": traces["generated_ids"],
        "content": choice["message"].get("content", ""),
        "finish_reason": choice["finish_reason"],
        "usage": response["usage"],
        "response_sha256": digest(root / "response.json"),
        "logits_sha256": digest(root / "final.f32"),
        "client_time_seconds": float(
            (root / "client.time.tsv").read_text().split("\t", 1)[0]),
    }


def candidate_prompt(root: Path) -> dict:
    verify_manifest(root)
    traces = token_traces((root / "server.stderr").read_text())
    return {
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "prompt_ids": traces["prompt_ids"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="task-id=route12 artifact root")
    parser.add_argument(
        "--exact", action="append", required=True,
        help="task-id=exact-native artifact root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    candidates = dict(item.split("=", 1) for item in args.candidate)
    exacts = dict(item.split("=", 1) for item in args.exact)
    arms = {row["id"]: row for row in prereg["arms"]}
    if set(candidates) != set(arms) or set(exacts) != set(arms):
        raise ValueError("candidate/exact task set does not match preregistration")
    if digest(Path(prereg["source"]["exact_harness"])) != prereg["source"][
            "exact_harness_sha256"]:
        raise ValueError("exact harness hash changed")

    rows = []
    closure_passed = True
    for identifier in (row["id"] for row in prereg["arms"]):
        spec = arms[identifier]
        fixture = Path(spec["fixture"])
        exact_root = Path(exacts[identifier])
        if digest(fixture) != spec["fixture_sha256"]:
            raise ValueError(f"fixture hash changed: {identifier}")
        if digest(exact_root / "request.json") != spec["fixture_sha256"]:
            raise ValueError(f"measured fixture changed: {identifier}")
        exact = exact_arm(exact_root)
        candidate = candidate_prompt(Path(candidates[identifier]))
        environment = exact.pop("environment")
        forbidden = [
            key for key in environment
            if key.startswith("DFLASH_KIMI_EXPERIMENT_")
        ]
        environment_valid = (
            environment.get("DFLASH_KIMI_PRODUCTION_DEFAULTS") == "0" and
            environment.get("DFLASH_KIMI_LAYER1_PROVIDER") == "exact" and
            not forbidden)
        prompt_equal = exact["prompt_ids"] == candidate["prompt_ids"]
        native_success = task_success(identifier, exact["content"])
        binary_equal = (
            exact["executable_sha256"] == candidate["executable_sha256"] ==
            prereg["source"]["executable_sha256"])
        row_passed = environment_valid and prompt_equal and native_success and binary_equal
        closure_passed = closure_passed and row_passed
        rows.append({
            "id": identifier,
            "passed": row_passed,
            "exact": exact,
            "candidate_prompt": {
                **{key: value for key, value in candidate.items()
                   if key != "prompt_ids"},
                "prompt_token_count": len(candidate["prompt_ids"]),
            },
            "prompt_ids_equal": prompt_equal,
            "environment_valid": environment_valid,
            "forbidden_experiment_environment": forbidden,
            "binary_equal": binary_equal,
            "native_task_success": native_success,
        })

    result = {
        "schema": "kimi-k3-route12-native-closure-result-v1",
        "status": ("MEASURED_ROUTE12_NATIVE_CLOSURE_GO" if closure_passed else
                   "MEASURED_ROUTE12_NATIVE_CLOSURE_NO_GO"),
        "preregistration_sha256": digest(args.prereg),
        "source": prereg["source"],
        "arms": rows,
        "gate": {
            "passed": closure_passed,
            "decision": (
                "Build the preregistered mixed-provenance baseline and score the frozen route12 captures."
                if closure_passed else
                "Do not score the affected route12 tasks."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "arms": {row["id"]: row["passed"] for row in rows},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
