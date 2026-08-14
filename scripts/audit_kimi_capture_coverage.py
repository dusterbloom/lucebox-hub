#!/usr/bin/env python3
"""Audit sequence splits and routed-expert coverage in a Kimi panel capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from train_kimi_panel_directional import read_capture


EXPERT_COUNT = 896


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, probability))
        for name, probability in (
            ("minimum", 0.0),
            ("p01", 0.01),
            ("p05", 0.05),
            ("p25", 0.25),
            ("median", 0.50),
            ("p75", 0.75),
            ("p95", 0.95),
            ("p99", 0.99),
            ("maximum", 1.0),
        )
    }


def main() -> int:
    args = parse_args()
    header, records = read_capture(args.capture)
    if header["top_k"] <= 0:
        raise ValueError("capture has no routed experts")

    split_ids: dict[int, list[np.ndarray]] = {0: [], 1: []}
    split_tokens = {0: 0, 1: 0}
    split_sequences = {0: 0, 1: 0}
    maximum_router_sum_error = 0.0
    finite = True
    for record in records:
        split = int(record["split"])
        split_ids[split].append(record["expert_ids"])
        split_tokens[split] += int(record["expert_ids"].shape[0])
        split_sequences[split] += 1
        maximum_router_sum_error = max(
            maximum_router_sum_error,
            float(
                np.max(
                    np.abs(record["router_weights"].sum(axis=1) - 1.0)
                )
            ),
        )
        finite = finite and bool(
            np.isfinite(record["latent"]).all()
            and np.isfinite(record["router_weights"]).all()
        )

    route_counts: dict[int, np.ndarray] = {}
    for split in (0, 1):
        ids = np.concatenate(split_ids[split]).reshape(-1)
        route_counts[split] = np.bincount(
            ids, minlength=EXPERT_COUNT
        ).astype(np.int64, copy=False)
    calibration = route_counts[0]
    validation = route_counts[1]
    unseen_calibration = calibration == 0

    result = {
        "schema": "kimi-k3-capture-coverage-v1",
        "status": "MEASURED",
        "capture": str(args.capture),
        "capture_bytes": args.capture.stat().st_size,
        "capture_sha256": sha256(args.capture),
        "header": header,
        "sequence_disjoint_split": True,
        "split_tokens": {
            "calibration": split_tokens[0],
            "validation": split_tokens[1],
        },
        "split_sequences": {
            "calibration": split_sequences[0],
            "validation": split_sequences[1],
        },
        "router_sum_maximum_absolute_error": maximum_router_sum_error,
        "all_values_finite": finite,
        "calibration_routes_per_expert": quantiles(calibration),
        "validation_routes_per_expert": quantiles(validation),
        "experts_without_calibration_routes": int(
            np.count_nonzero(unseen_calibration)
        ),
        "experts_without_validation_routes": int(
            np.count_nonzero(validation == 0)
        ),
        "validation_experts_unseen_in_calibration": int(
            np.count_nonzero(unseen_calibration & (validation > 0))
        ),
        "validation_routes_through_uncalibrated_experts": int(
            validation[unseen_calibration].sum()
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
