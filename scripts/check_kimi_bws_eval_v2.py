#!/usr/bin/env python3
"""Apply one preregistered K3 BWS v2 behavioral checker."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import signal
from pathlib import Path


SAFE_CALLS = {
    "abs": abs,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
}
SAFE_METHODS = {"append", "copy", "extend", "insert", "pop", "reverse", "sort"}


def normalized(value: str) -> str:
    digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return " ".join(value.lower().replace("’", "'").translate(digits).split())


def ordered_equal(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return (list(left) == list(right) and all(
            ordered_equal(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            ordered_equal(a, b) for a, b in zip(left, right))
    return left == right


def safe_function(source: str, name: str) -> tuple[object | None, str | None]:
    if source.lstrip().startswith("```"):
        return None, "markdown code fence is not executable source"
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return None, str(error)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or [node.name for node in functions] != [name]:
        return None, f"expected exactly one function named {name}"
    forbidden = (
        ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete, ast.Global,
        ast.Import, ast.ImportFrom, ast.Nonlocal, ast.Raise, ast.Try, ast.While,
        ast.With, ast.Yield, ast.YieldFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return None, f"forbidden syntax: {type(node).__name__}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return None, "dunder attribute forbidden"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_CALLS and node.func.id != name:
                    return None, f"call forbidden: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in SAFE_METHODS:
                    return None, f"method forbidden: {node.func.attr}"
            else:
                return None, "indirect call forbidden"
    namespace = {"__builtins__": SAFE_CALLS}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    return namespace[name], None


def run_python_cases(content: str, checker: dict) -> tuple[bool, dict]:
    name = ("square_even_numbers" if checker["type"] ==
            "python_square_even_numbers" else "merge_intervals")
    function, error = safe_function(content.strip(), name)
    if function is None:
        return False, {"error": error}
    rows = []
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("candidate timed out")))
    try:
        for original, expected in checker["cases"]:
            argument = copy.deepcopy(original)
            signal.alarm(1)
            result = function(argument)
            signal.alarm(0)
            unmodified = argument == original
            passed = result == expected and (
                unmodified or not checker.get("input_must_not_mutate", False))
            rows.append({"result": result, "expected": expected,
                         "input_unmodified": unmodified, "passed": passed})
    except Exception as caught:
        signal.alarm(0)
        return False, {"error": f"{type(caught).__name__}: {caught}",
                       "tests": rows}
    finally:
        signal.signal(signal.SIGALRM, previous)
    return all(row["passed"] for row in rows), {"tests": rows}


def message_from_response(response: dict) -> tuple[dict, str | None]:
    try:
        return response["choices"][0]["message"], None
    except (KeyError, IndexError, TypeError) as error:
        return {}, f"invalid chat response: {error}"


def check_response(response: dict, checker: dict) -> tuple[bool, dict]:
    message, error = message_from_response(response)
    if error:
        return False, {"error": error}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    kind = checker["type"]

    if kind == "tool_call":
        if len(tool_calls) != 1:
            return False, {"error": "expected exactly one tool call",
                           "tool_calls": tool_calls}
        function = tool_calls[0].get("function", {})
        try:
            arguments = json.loads(function.get("arguments", ""))
        except json.JSONDecodeError as caught:
            return False, {"error": f"invalid tool arguments: {caught}"}
        valid = function.get("name") == checker["name"] and all(
            normalized(str(arguments.get(key, ""))) == normalized(str(value))
            for key, value in checker["arguments"].items())
        return valid, {"name": function.get("name"), "arguments": arguments}

    if tool_calls:
        return False, {"error": "unexpected tool call", "tool_calls": tool_calls}
    value = normalized(content)
    if kind == "normalized_contains":
        valid = normalized(checker["value"]) in value
    elif kind == "normalized_regex":
        valid = re.search(checker["value"], value) is not None
    elif kind == "whitespace_insensitive_contains":
        valid = re.sub(r"\s+", "", checker["value"].lower()) in re.sub(
            r"\s+", "", content.lower())
    elif kind in {"integer_token", "no_tool_integer"}:
        valid = re.search(
            rf"(?<!\d){re.escape(str(checker['value']))}(?!\d)", value) is not None
    elif kind == "json_exact":
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as caught:
            return False, {"error": str(caught), "content": content}
        valid = ordered_equal(parsed, checker["value"])
        return valid, {"value": parsed}
    elif kind.startswith("python_"):
        valid, details = run_python_cases(content, checker)
        return valid, {"content": content, **details}
    elif kind == "three_numbered_lines":
        lines = [line.strip() for line in content.strip().splitlines()]
        prefixes = [f"{index}." for index in range(1, 4)]
        bodies = [line[len(prefix):].strip() for line, prefix in zip(lines, prefixes)]
        one_sentence = len(lines) == 3 and all(
            line.startswith(prefix) and body.endswith((".", "?", "!")) and
            len([part for part in re.split(r"[.!?]+", body) if part.strip()]) == 1
            for line, prefix, body in zip(lines, prefixes, bodies))
        required = all(normalized(term) in value
                       for term in checker["required_terms"])
        valid = one_sentence and required
        return valid, {"lines": lines, "one_sentence_each": one_sentence,
                       "required_terms_present": required}
    else:
        raise ValueError(f"unregistered checker type: {kind}")
    return valid, {"content": content}


def self_test() -> None:
    response = {"choices": [{"message": {"content": "Tokyo"}}]}
    assert check_response(response, {"type": "normalized_contains",
                                     "value": "tokyo"})[0]
    response["choices"][0]["message"]["content"] = (
        "def square_even_numbers(values):\n"
        "    return [value ** 2 for value in values if value % 2 == 0]\n")
    assert check_response(response, {
        "type": "python_square_even_numbers",
        "cases": [[[], []], [[1, 2, 4], [4, 16]]],
        "input_must_not_mutate": True,
    })[0]
    response = {"choices": [{"message": {"content": None, "tool_calls": [{
        "function": {"name": "get_weather",
                     "arguments": "{\"location\":\"Rome\"}"}}]}}]}
    assert check_response(response, {"type": "tool_call",
                                     "name": "get_weather",
                                     "arguments": {"location": "Rome"}})[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path)
    parser.add_argument("--fixture")
    parser.add_argument("--response", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return 0
    if not (args.prereg and args.fixture and args.response):
        parser.error("--prereg, --fixture and --response are required")
    prereg = json.loads(args.prereg.read_text())
    fixtures = {row["id"]: row for row in prereg["fixtures"]}
    if args.fixture not in fixtures:
        raise ValueError(f"unregistered fixture: {args.fixture}")
    response = json.loads(args.response.read_text())
    valid, details = check_response(response, fixtures[args.fixture]["checker"])
    print(json.dumps({"fixture": args.fixture, "passed": valid,
                      "details": details}, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
