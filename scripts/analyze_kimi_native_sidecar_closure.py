#!/usr/bin/env python3
"""Validate source-matched Full192 native closure on Lucebox4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import (
    P20,
    digest,
    read_environment,
    token_traces,
    traffic,
    verify_manifest,
)
from analyze_kimi_route12_native_success import first_divergence, task_success


EXPECTED_MODEL_FIRST_SHA256 = (
    "5022014e7c49d8844e9f1bc7d9fb824c0d640214540aa845690518d800286083"
)


def analyze_arm(identifier: str, root: Path, fixture: Path,
                fixture_sha256: str, native: dict, prereg: dict) -> dict:
    verify_manifest(root)
    if digest(fixture) != fixture_sha256:
        raise ValueError(f"fixture changed: {identifier}")
    if digest(root / "request.json") != fixture_sha256:
        raise ValueError(f"measured fixture changed: {identifier}")

    environment = read_environment(root / "environment.nul")
    required = {
        "DFLASH_KIMI_PRODUCTION_DEFAULTS": "0",
        "DFLASH_KIMI_LAYER1_PROVIDER": "all-layers-calibrated96",
        "DFLASH_KIMI_SIDECAR_AUTHORITATIVE": "1",
        "DFLASH_KIMI_P20_SLAB_BUDGET": "192",
    }
    if any(environment.get(key) != value for key, value in required.items()):
        raise ValueError(f"Full192 environment changed: {identifier}")
    forbidden = [
        key for key in environment
        if key.startswith("DFLASH_KIMI_EXPERIMENT_")
        or key == "DFLASH_KIMI_H22_LAYER_BUDGETS"
    ]
    if forbidden:
        raise ValueError(f"experimental override present: {identifier}: {forbidden}")

    input_hashes = (root / "inputs.sha256").read_text().splitlines()
    if not input_hashes or input_hashes[0].split()[0] != EXPECTED_MODEL_FIRST_SHA256:
        raise ValueError(f"wrong native core first shard: {identifier}")
    executable = (root / "executable.sha256").read_text().split()[0]
    if executable != prereg["closure"]["executable_sha256"]:
        raise ValueError(f"executable changed: {identifier}")

    stderr = (root / "server.stderr").read_text()
    traces = token_traces(stderr)
    response = json.loads((root / "response.json").read_text())
    content = response["choices"][0]["message"].get("content", "")
    byte_metrics = traffic(root / "traffic.tsv")
    expected_positions = len(traces["prompt_ids"]) + max(
        0, len(traces["generated_ids"]) - 1
    )
    if byte_metrics["provider_positions"] != expected_positions:
        raise ValueError(f"provider-position mismatch: {identifier}")
    p20 = P20.findall(stderr)
    if len(p20) != 1:
        raise ValueError(f"expected one provider summary: {identifier}")
    explicit_reads, physical_bytes, direct_io_ns = map(int, p20[0])
    prompt_equal = traces["prompt_ids"] == native["prompt_tokens"]
    divergence = (
        first_divergence(native["output_tokens"], traces["generated_ids"])
        if prompt_equal else None
    )
    success = task_success(identifier, content)
    return {
        "id": identifier,
        "artifact_root": str(root),
        "source_commit": (root / "source-commit.txt").read_text().strip(),
        "manifest_sha256": digest(root / "SHA256SUMS"),
        "response_sha256": digest(root / "response.json"),
        "logits_sha256": digest(root / "final.f32"),
        "traffic_sha256": digest(root / "traffic.tsv"),
        "prompt_ids": traces["prompt_ids"],
        "generated_ids": traces["generated_ids"],
        "native_prompt_ids_equal": prompt_equal,
        "native_generated_ids_equal": divergence is None if prompt_equal else None,
        "first_generated_divergence": divergence,
        "content": content,
        "task_success": success,
        "finish_reason": response["choices"][0]["finish_reason"],
        "usage": response["usage"],
        "traffic": {
            **byte_metrics,
            "explicit_provider_reads": explicit_reads,
            "direct_physical_bytes": physical_bytes,
            "direct_physical_gib_per_position": (
                physical_bytes / expected_positions / (1024 ** 3)
            ),
            "direct_io_ns": direct_io_ns,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--fetch-result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="task-id=artifact-root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    baseline = json.loads(args.baseline.read_text())
    if digest(args.baseline) != prereg["closure"]["native_baseline_sha256"]:
        raise ValueError("native baseline changed")
    if digest(Path(prereg["closure"]["harness"])) != prereg["closure"][
            "harness_sha256"]:
        raise ValueError("closure harness changed")
    fetch_result = json.loads(args.fetch_result.read_text())
    if (fetch_result.get("status") != "VERIFIED_COMPLETE" or
            fetch_result.get("plan_sha256") != prereg["range_fetch"]["plan_sha256"]):
        raise ValueError("native sparse fetch did not close")

    fixture_rows = {row["id"]: row for row in prereg["closure"]["fixtures"]}
    native_rows = {row["id"]: row for row in baseline["sequences"]}
    roots = dict(item.split("=", 1) for item in args.arm)
    if set(roots) != set(fixture_rows):
        raise ValueError("closure arm set changed")

    arms = []
    for identifier in (row["id"] for row in prereg["closure"]["fixtures"]):
        fixture = fixture_rows[identifier]
        arms.append(analyze_arm(
            identifier,
            Path(roots[identifier]),
            Path(fixture["path"]),
            fixture["sha256"],
            native_rows[identifier],
            prereg,
        ))
    commits = {arm["source_commit"] for arm in arms}
    if len(commits) != 1:
        raise ValueError("closure arms used different commits")

    fact = next(arm for arm in arms if arm["id"] == "fact-capital")
    tasks_passed = sum(arm["task_success"] for arm in arms)
    gate_passed = (
        tasks_passed == len(arms)
        and fact["native_prompt_ids_equal"]
        and fact["native_generated_ids_equal"]
    )
    result = {
        "schema": "kimi-k3-native-sidecar-closure-result-v1",
        "status": ("MEASURED_NATIVE_SIDECAR_CLOSURE_GO" if gate_passed else
                   "MEASURED_NATIVE_SIDECAR_CLOSURE_NO_GO"),
        "preregistration_sha256": digest(args.prereg),
        "fetch": {
            "result": str(args.fetch_result),
            "result_sha256": digest(args.fetch_result),
            **fetch_result,
        },
        "source": {
            **prereg["source"],
            "measured_commit": next(iter(commits)),
            "executable_sha256": prereg["closure"]["executable_sha256"],
            "native_core_first_shard_sha256": EXPECTED_MODEL_FIRST_SHA256,
        },
        "arms": arms,
        "gate": {
            "passed": gate_passed,
            "tasks_passed": f"{tasks_passed}/{len(arms)}",
            "fact_prompt_and_generation_exact": (
                fact["native_prompt_ids_equal"] and
                fact["native_generated_ids_equal"]
            ),
            "decision": (
                "Rerun frozen H23 and route12 candidates against the source-matched native core."
                if gate_passed else
                "Do not score candidates; preserve failure without fixture tuning."
            ),
        },
        "limitations": [
            "Cross-HIP/CUDA byte-identical logits are not required; source bytes and native behavior are the closure criteria.",
            "Grammar and extraction prompt IDs may differ from the old capture because the current official template tokenizer is frozen separately.",
            "This closes teacher provenance; it is not a BWS quality or throughput result."
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "tasks_passed": result["gate"]["tasks_passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
