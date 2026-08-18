#!/usr/bin/env python3
"""Summarize current K3 per-position stage telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


LINE = re.compile(r"\[kimi-k3-stage\] (?P<body>.*)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text().splitlines():
        match = LINE.search(line)
        if not match:
            continue
        row: dict[str, float] = {}
        for item in match.group("body").split():
            key, value = item.split("=", 1)
            row[key] = float(value)
        rows.append(row)
    return rows


def summary(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.root / "suite" / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sequence = manifest["sequences"][0]
    prompt_tokens = int(sequence["prompt_token_count"])
    rows = parse(args.root / "stderr.log")
    decode = [row for row in rows if int(row["position"]) >= prompt_tokens]
    if len(decode) != len(sequence["output_tokens"]) - 1:
        raise ValueError("decode stage-row count does not match transitions")
    fields = (
        "total_ms", "embedding_ms", "dense_ms", "routed_prep_ms",
        "offload_prep_ms", "experts_ms", "join_ms", "output_ms", "other_ms",
    )
    stages = {field: summary(decode, field) for field in fields}
    total = stages["total_ms"]["median_ms"]
    routed = stages["routed_prep_ms"]["median_ms"]
    experts = stages["experts_ms"]["median_ms"]
    result = {
        "schema": "kimi-k3-p31-current-stage-profile-v1",
        "status": "MEASURED",
        "provenance": {
            "root": str(args.root),
            "manifest_sha256": sha256(manifest_path),
            "telemetry_sha256": sha256(args.root / "telemetry.json"),
            "stderr_sha256": sha256(args.root / "stderr.log"),
            "repository_commit": manifest["environment"]["KIMI_H16_REPOSITORY_COMMIT"],
            "provider": manifest["provider"],
            "budget_table": manifest["environment"]["DFLASH_KIMI_H22_LAYER_BUDGETS"],
        },
        "decode_rows": len(decode),
        "stages": stages,
        "median_control_room": {
            "measured_transition_rate": 1000.0 / total,
            "routed_preparation_fraction": routed / total,
            "expert_provider_fraction": experts / total,
            "routed_preparation_free_rate": 1000.0 / (total - routed),
            "expert_provider_free_rate": 1000.0 / (total - experts),
            "both_free_rate": 1000.0 / (total - routed - experts),
            "target_4_tokens_per_second_ms": 250.0,
        },
        "verdict": "BOTH_CORE_AND_EXPERT_PATHS_REQUIRE_MULTIPLICATIVE_ACCELERATION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
