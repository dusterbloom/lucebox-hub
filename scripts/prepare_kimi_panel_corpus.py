#!/usr/bin/env python3
"""Build a deterministic, sequence-split smoke corpus for the Kimi panel probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def normalized_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n\n".join(
            part.strip() for part in value if isinstance(part, str) and part.strip()
        )
    return ""


def collect_mt_bench(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for ordinal, row in enumerate(read_json_lines(path)):
        text = normalized_text(row.get("turns"))
        if text:
            identifier = str(row.get("question_id", ordinal))
            rows.append((f"conversation-{identifier}", text))
    return rows


def collect_code(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for ordinal, row in enumerate(read_json_lines(path)):
        text = normalized_text(row.get("prompt"))
        if text:
            identifier = str(row.get("task_id", ordinal)).replace("/", "-")
            rows.append((f"code-{identifier}", text))
    return rows


def assign_split(index: int) -> str:
    # Sequence-level split, interleaved so even the 2,048-token smoke capture
    # contains held-out rows before its total-token cap is reached.
    return "validation" if index % 5 == 4 else "calibration"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=200)
    args = parser.parse_args()

    if args.max_sequences <= 0:
        parser.error("--max-sequences must be positive")

    domains = [
        collect_mt_bench(args.conversation),
        collect_code(args.code),
    ]
    if any(not domain for domain in domains):
        raise RuntimeError("each source must contribute at least one sequence")

    interleaved: list[tuple[str, str]] = []
    for index in range(max(len(domain) for domain in domains)):
        for domain in domains:
            if index < len(domain):
                interleaved.append(domain[index])
                if len(interleaved) >= args.max_sequences:
                    break
        if len(interleaved) >= args.max_sequences:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    split_counts = {"calibration": 0, "validation": 0}
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for index, (identifier, text) in enumerate(interleaved):
            split = assign_split(index)
            split_counts[split] += 1
            encoded = json.dumps(
                {"id": identifier, "split": split, "text": text},
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            destination.write(encoded)
            digest.update(encoded.encode("utf-8"))

    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest.hexdigest(),
                "sequences": len(interleaved),
                "split_counts": split_counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
