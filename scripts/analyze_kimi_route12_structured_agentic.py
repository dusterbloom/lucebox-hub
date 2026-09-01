#!/usr/bin/env python3
"""Compare route12 with the preregistered H23 structured/agentic control."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

from analyze_kimi_progressive_tool_rescue import digest
from analyze_kimi_schema_rescue import analyze_arm


SAFE_CALLS = {
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "sorted": sorted,
}
SAFE_METHODS = {"append", "copy", "sort"}


def structured_json(content: str) -> tuple[bool, dict]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        return False, {"error": str(error), "content": content}
    expected = {"language": "Python", "passed_tests": 3, "safe": True}
    valid = (value == expected and list(value) == list(expected) and
             content.lstrip().startswith("{") and content.rstrip().endswith("}"))
    return valid, {"value": value, "key_order": list(value) if isinstance(value, dict) else None}


def python_source(content: str) -> tuple[str | None, str | None]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            return None, "unterminated code fence"
        value = "\n".join(lines[1:-1])
    return value, None


def safe_tree(source: str) -> tuple[ast.Module | None, str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return None, str(error)
    forbidden = (
        ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete, ast.Global,
        ast.Import, ast.ImportFrom, ast.Lambda, ast.Nonlocal, ast.Raise,
        ast.Try, ast.While, ast.With, ast.Yield, ast.YieldFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return None, f"forbidden syntax: {type(node).__name__}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return None, "dunder attribute forbidden"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_CALLS:
                    return None, f"call forbidden: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in SAFE_METHODS:
                    return None, f"method forbidden: {node.func.attr}"
            else:
                return None, "indirect call forbidden"
    return tree, None


def agentic_code(content: str) -> tuple[bool, dict]:
    source, error = python_source(content)
    if source is None:
        return False, {"error": error}
    tree, error = safe_tree(source)
    if tree is None:
        return False, {"error": error, "source": source}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if [node.name for node in functions] != ["merge_intervals"]:
        return False, {"error": "expected exactly merge_intervals", "source": source}
    namespace = {"__builtins__": SAFE_CALLS}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)
        function = namespace["merge_intervals"]
        cases = [
            ([], []),
            ([[1, 3], [2, 4], [8, 10], [10, 12]], [[1, 4], [8, 12]]),
            ([[5, 7], [1, 2]], [[1, 2], [5, 7]]),
            ([[1, 1]], [[1, 1]]),
            ([[-3, -1], [-2, 0], [4, 5]], [[-3, 0], [4, 5]]),
        ]
        outcomes = []
        for original, expected in cases:
            argument = copy.deepcopy(original)
            result = function(argument)
            outcomes.append({
                "result": result,
                "expected": expected,
                "input_unmodified": argument == original,
                "passed": result == expected and argument == original,
            })
        valid = all(row["passed"] for row in outcomes)
    except Exception as exception:
        return False, {"error": f"{type(exception).__name__}: {exception}", "source": source}
    return valid, {"source": source, "tests": outcomes}


def evaluate(identifier: str, root: Path) -> tuple[bool, dict, dict]:
    arm = analyze_arm(root)
    response = json.loads((root / "response.json").read_text())
    choice = response["choices"][0]
    message = choice["message"]
    if message.get("tool_calls", []):
        return False, {"error": "unexpected tool call"}, arm
    content = message.get("content", "")
    if identifier == "json":
        valid, details = structured_json(content)
    elif identifier == "agentic":
        valid, details = agentic_code(content)
    else:
        raise ValueError(f"unregistered fixture {identifier}")
    details.update({"content": content, "finish_reason": choice["finish_reason"]})
    return valid, details, arm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--control-policy", type=Path, required=True)
    parser.add_argument("--route12-policy", type=Path, required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--route12", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.prereg.read_text())
    repo = Path(__file__).resolve().parent.parent
    for path, expected in (
            (Path(__file__), prereg["source"]["analyzer_sha256"]),
            (repo / "scripts/run_kimi_progressive_tool_rescue.sh",
             prereg["source"]["runner_sha256"]),
            (args.control_policy, prereg["policies"]["control_sha256"]),
            (args.route12_policy, prereg["policies"]["route12_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"registered input hash changed: {path}")
    control_roots = dict(item.split("=", 1) for item in args.control)
    route12_roots = dict(item.split("=", 1) for item in args.route12)
    fixtures = {row["id"]: row for row in prereg["fixtures"]}
    if set(control_roots) != set(fixtures) or set(route12_roots) != set(fixtures):
        raise ValueError("fixture/root set does not match preregistration")

    rows = []
    source_commits = set()
    executable_hashes = set()
    all_valid = True
    for identifier in (row["id"] for row in prereg["fixtures"]):
        spec = fixtures[identifier]
        fixture = Path(spec["path"])
        if digest(fixture) != spec["sha256"]:
            raise ValueError(f"fixture hash changed: {identifier}")
        control_valid, control_details, control = evaluate(
            identifier, Path(control_roots[identifier]))
        candidate_valid, candidate_details, candidate = evaluate(
            identifier, Path(route12_roots[identifier]))
        source_commits.update((control["source_commit"], candidate["source_commit"]))
        executable_hashes.update((control["executable_sha256"], candidate["executable_sha256"]))
        for name, root, arm, policy, route12 in (
                ("control", Path(control_roots[identifier]), control,
                 args.control_policy, False),
                ("route12", Path(route12_roots[identifier]), candidate,
                 args.route12_policy, True)):
            if digest(root / "request.json") != spec["sha256"]:
                raise ValueError(f"measured fixture changed: {identifier}/{name}")
            environment = arm["environment"]
            if (environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != str(policy) or
                    (environment.get("DFLASH_KIMI_EXPERIMENT_ROUTE_LIMIT") == "12") != route12 or
                    environment.get("DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE") != "1" or
                    "DFLASH_KIMI_EXPERIMENT_POSITION_BUDGETS" in environment or
                    arm["schema_prefix_configurations"] != 0 or
                    arm["schema_rescue_markers"] or
                    arm["static_position_markers"] != 0 or
                    arm["request_wide_b96_markers"] != 0):
                raise ValueError(f"environment/marker contract changed: {identifier}/{name}")
        prompt_equal = (control["prompt_token_ids_i32le_sha256"] ==
                        candidate["prompt_token_ids_i32le_sha256"])
        if not prompt_equal:
            raise ValueError(f"control/candidate prompt mismatch: {identifier}")
        candidate_logical = candidate["traffic"]["logical_gib_per_position"]
        passed = control_valid and candidate_valid and candidate_logical < 1.2
        all_valid = all_valid and passed
        rows.append({
            "id": identifier,
            "passed": passed,
            "control_valid": control_valid,
            "candidate_valid": candidate_valid,
            "control_details": control_details,
            "candidate_details": candidate_details,
            "prompt_ids_equal": prompt_equal,
            "generated_ids_equal": control["generated_ids"] == candidate["generated_ids"],
            "final_logits_equal": control["logits_sha256"] == candidate["logits_sha256"],
            "control": control,
            "route12": candidate,
            "traffic_reduction_fraction": 1.0 - (
                candidate["traffic"]["total_provider_bytes"] /
                control["traffic"]["total_provider_bytes"]),
        })

    if len(source_commits) != 1:
        raise ValueError("arms used different source commits")
    if executable_hashes != {prereg["source"]["executable_sha256"]}:
        raise ValueError("arms used a different executable")
    controls_valid = all(row["control_valid"] for row in rows)
    status = (
        "MEASURED_ROUTE12_STRUCTURED_AGENTIC_INVALID_CONTROL"
        if not controls_valid else
        ("MEASURED_ROUTE12_STRUCTURED_AGENTIC_GO" if all_valid else
         "MEASURED_ROUTE12_STRUCTURED_AGENTIC_NO_GO"))
    result = {
        "schema": "kimi-k3-route12-structured-agentic-result-v1",
        "status": status,
        "preregistration_sha256": digest(args.prereg),
        "source": {
            **prereg["source"],
            "measured_commit": next(iter(source_commits)),
        },
        "fixtures": rows,
        "gate": {
            "passed": all_valid if controls_valid else False,
            "controls_valid": controls_valid,
            "decision": (
                "One or more H23 1.8-GiB controls failed; invalidate the affected fixture rather than scoring route12."
                if not controls_valid else
                "Route12 retains both control-valid structured/agentic tasks below 1.2 GiB/position; keep research-only pending broader quality and performance work."
                if all_valid else
                "Stop route12 production consideration."
            ),
        },
        "limitations": prereg["limitations"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "fixtures": {row["id"]: row["passed"] for row in rows},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
