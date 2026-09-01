#!/usr/bin/env python3
"""Score the preregistered route12 policy on frozen native-success tasks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import digest, token_traces
from analyze_kimi_schema_rescue import analyze_arm


ROUTE_MARKER = re.compile(
    r"\[kimi-k3-route-limit\] top-routes=(\d+) weights=unchanged")


def normalized(text: str) -> str:
    subscript_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return " ".join(
        text.lower().replace("’", "'").translate(subscript_digits).split())


def task_success(identifier: str, text: str) -> bool:
    value = normalized(text)
    if identifier == "fact-capital":
        return "tokyo" in value
    if identifier == "fact-science":
        return ("carbon dioxide" in value or
                re.search(r"(?<![a-z0-9])co2(?![a-z0-9])", value) is not None)
    if identifier == "code-sum":
        return re.search(r"(?<!\d)10(?!\d)", value) is not None
    if identifier == "code-function":
        compact = value.replace(" ", "")
        return (
            "defsquare_even_numbers(" in compact and "%2" in compact and
            ("**2" in compact or
             re.search(r"([a-z])\*\1for", compact) is not None))
    if identifier == "reasoning-marble":
        return re.search(r"(?<!\d)42(?!\d)", value) is not None
    if identifier == "reasoning-rate":
        return re.search(r"(?<!\d)150(?!\d)", value) is not None
    if identifier == "grammar-apples":
        return ("she doesn't like apples" in value or
                "she does not like apples" in value)
    if identifier == "grammar-agreement":
        return "the list of items is on the table" in value
    if identifier == "translation-italian":
        return "buongiorno" in value or "buon giorno" in value
    if identifier == "translation-spanish":
        return "muchas gracias" in value
    if identifier == "extract-code":
        return "lime-742" in re.sub(r"\s+", "", value)
    if identifier == "extract-decoys":
        return "quartz-918" in re.sub(r"\s+", "", value)
    raise ValueError(f"unregistered task {identifier}")


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--root", action="append", required=True,
        help="task-id=artifact-root; repeat once per preregistered task")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    baseline = json.loads(args.baseline.read_text())
    if digest(args.baseline) != prereg["native_baseline"]["sha256"]:
        raise ValueError("native baseline hash changed")
    if digest(args.policy) != prereg["representation"]["policy_sha256"]:
        raise ValueError("Budget20 policy hash changed")
    baseline_rows = {row["id"]: row for row in baseline["sequences"]}
    root_args = dict(item.split("=", 1) for item in args.root)
    fixture_rows = {row["id"]: row for row in prereg["fixtures"]}
    if set(root_args) != set(fixture_rows) or set(root_args) != set(baseline_rows):
        raise ValueError("task/root set does not match preregistration")

    sequences = []
    source_commits = set()
    executable_hashes = set()
    total_positions = 0
    total_logical = 0
    total_fallback = 0
    total_physical = 0
    total_direct_io_ns = 0
    for identifier in (row["id"] for row in prereg["fixtures"]):
        root = Path(root_args[identifier])
        fixture = Path(fixture_rows[identifier]["path"])
        if digest(fixture) != fixture_rows[identifier]["sha256"]:
            raise ValueError(f"fixture hash changed: {identifier}")
        if digest(root / "request.json") != fixture_rows[identifier]["sha256"]:
            raise ValueError(f"measured fixture changed: {identifier}")
        arm = analyze_arm(root)
        source_commits.add(arm["source_commit"])
        executable_hashes.add(arm["executable_sha256"])
        environment = arm["environment"]
        if (environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") != "12" or
                environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1" or
                environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(args.policy)):
            raise ValueError(f"environment changed: {identifier}")
        if "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment:
            raise ValueError(f"position intervention present: {identifier}")
        stderr = (root / "server.stderr").read_text()
        route_markers = [int(value) for value in ROUTE_MARKER.findall(stderr)]
        if (route_markers != [12] or arm["schema_prefix_configurations"] != 0 or
                arm["schema_rescue_markers"] or arm["static_position_markers"] != 0 or
                arm["request_wide_b96_markers"] != 0):
            raise ValueError(f"route/schema marker contract changed: {identifier}")
        native = baseline_rows[identifier]
        traces = token_traces(stderr)
        prompt_equal = traces["prompt_ids"] == native["prompt_tokens"]
        response = json.loads((root / "response.json").read_text())
        content = response["choices"][0]["message"].get("content", "")
        generated = arm["generated_ids"]
        success = task_success(identifier, content) if prompt_equal else None
        divergence = (first_divergence(native["output_tokens"], generated)
                      if prompt_equal else None)
        positions = arm["traffic"]["provider_positions"]
        expected_positions = len(traces["prompt_ids"]) + max(0, len(generated) - 1)
        if positions != expected_positions:
            raise ValueError(f"provider-position alignment changed: {identifier}")
        total_positions += positions
        total_logical += arm["traffic"]["total_provider_bytes"]
        total_fallback += arm["traffic"]["exact_fallback_bytes"]
        total_physical += arm["traffic"]["direct_physical_bytes"]
        total_direct_io_ns += arm["traffic"]["direct_io_ns"]
        sequences.append({
            "id": identifier,
            "artifact_root": str(root),
            "manifest_sha256": arm["manifest_sha256"],
            "response_sha256": digest(root / "response.json"),
            "logits_sha256": arm["logits_sha256"],
            "traffic_sha256": arm["traffic_sha256"],
            "native_prompt_token_count": len(native["prompt_tokens"]),
            "candidate_prompt_token_count": len(traces["prompt_ids"]),
            "prompt_tokens_equal_native": prompt_equal,
            "native_tokens": native["output_tokens"],
            "candidate_tokens": generated,
            "first_generated_token_divergence": divergence,
            "exact_sequence": divergence is None if prompt_equal else None,
            "candidate_text": content if prompt_equal else None,
            "task_success": success,
            "route_limit_markers": route_markers,
            "provider_positions": positions,
            "logical_gib_per_position": arm["traffic"]["logical_gib_per_position"],
            "physical_gib_per_position": (
                arm["traffic"]["direct_physical_bytes"] / positions / (1024 ** 3)),
            "client_time_seconds": float(
                (root / "client.time.tsv").read_text().split("\t", 1)[0]),
        })

    if len(source_commits) != 1:
        raise ValueError("arms used different source commits")
    if executable_hashes != {prereg["source"]["executable_sha256"]}:
        raise ValueError("arms used a different executable")
    aggregate_logical = total_logical / total_positions / (1024 ** 3)
    aggregate_physical = total_physical / total_positions / (1024 ** 3)
    alignment_passed = all(
        row["prompt_tokens_equal_native"] for row in sequences)
    tasks_passed = sum(row["task_success"] is True for row in sequences)
    exact_sequences = sum(row["exact_sequence"] is True for row in sequences)
    gate_passed = (alignment_passed and tasks_passed == len(sequences) and
                   aggregate_logical < 1.2)
    status = (
        "MEASURED_ROUTE12_NATIVE_SUCCESS_INVALID_PROMPT_ALIGNMENT"
        if not alignment_passed else
        ("MEASURED_ROUTE12_NATIVE_SUCCESS_GO" if gate_passed else
         "MEASURED_ROUTE12_NATIVE_SUCCESS_NO_GO"))
    result = {
        "schema": "kimi-k3-route12-native-success-result-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": next(iter(source_commits)),
        },
        "native_baseline": prereg["native_baseline"],
        "candidate": {
            "tasks_passed": tasks_passed,
            "tasks": len(sequences),
            "exact_sequences": exact_sequences,
            "sequences": sequences,
        },
        "traffic": {
            "provider_positions": total_positions,
            "logical_authoritative_bytes": total_logical,
            "logical_gib_per_position": aggregate_logical,
            "exact_fallback_bytes": total_fallback,
            "exact_fallback_gib_per_position": (
                total_fallback / total_positions / (1024 ** 3)),
            "physical_direct_read_bytes": total_physical,
            "physical_gib_per_position": aggregate_physical,
            "direct_io_ns": total_direct_io_ns,
        },
        "controls": prereg["controls"],
        "terminal_kl": {
            "available": False,
            "reason": "Frozen native output IDs are retained, but native full-vocabulary logit tensors are not retained on Lucebox4.",
        },
        "gate": {
            "passed": gate_passed,
            "tasks_passed": tasks_passed == len(sequences),
            "prompt_alignment_passed": alignment_passed,
            "schema_false_activation_absent": True,
            "logical_below_1_2_gib": aggregate_logical < 1.2,
            "decision": (
                "Run exact binary closure only for the misaligned prompt IDs; do not score this invalid gate."
                if not alignment_passed else
                "Preregister tool-declared false-positive controls; keep research-only."
                if gate_passed else
                "Stop route12 production consideration and return to representation/progressive rescue work."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "tasks_passed": f"{tasks_passed}/{len(sequences)}",
        "exact_sequences": f"{exact_sequences}/{len(sequences)}",
        "logical_gib_per_position": aggregate_logical,
        "physical_gib_per_position": aggregate_physical,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
