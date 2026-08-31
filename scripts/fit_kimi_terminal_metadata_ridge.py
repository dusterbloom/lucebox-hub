#!/usr/bin/env python3
"""Fit the preregistered discovery-only ridge and score one held-out screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    result.update(path.read_bytes())
    return result.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        tied = np.flatnonzero(values == value)
        result[tied] = result[tied].mean()
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float] | None:
    if left.size < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1]),
    }


def load_rows(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text())
    if document.get("schema") != "kimi-k3-terminal-full-screen-v1":
        raise ValueError(f"unsupported screen schema: {path}")
    rows = [row for row in document["interventions"]
            if row["action"] == "force" and row["calibrated_expert"]]
    if len(rows) < 24:
        raise ValueError(f"too few calibrated omitted rows: {path}")
    return rows


def raw_features(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    continuous = np.asarray([
        [
            np.log(max(abs(float(row["router_weight"])), 1e-12)),
            np.log(max(float(row["residual_importance"]), 1e-12)),
            float(row["route"]) / 15.0,
            float(row["ordered_rank"]) / 11.0,
        ]
        for row in rows
    ], dtype=np.float64)
    one_hot = np.zeros((len(rows), 12), dtype=np.float64)
    one_hot[np.arange(len(rows)), [int(row["natural_slab"]) for row in rows]] = 1.0
    return continuous, one_hot


def transform(continuous: np.ndarray, one_hot: np.ndarray,
              mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(continuous.shape[0]),
                            (continuous - mean) / std, one_hot))


def fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    regularizer = np.eye(x.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def cross_validate(rows: list[dict[str, object]], continuous: np.ndarray,
                   one_hot: np.ndarray, y: np.ndarray) -> tuple[float, list[dict[str, object]]]:
    routes = np.asarray([int(row["route"]) for row in rows])
    summaries = []
    for penalty in LAMBDAS:
        folds = []
        for route in sorted(set(routes)):
            test = routes == route
            if test.sum() < 3:
                continue
            train = ~test
            mean = continuous[train].mean(axis=0)
            std = continuous[train].std(axis=0)
            std[std == 0] = 1.0
            coefficients = fit(transform(continuous[train], one_hot[train], mean, std),
                               y[train], penalty)
            prediction = transform(continuous[test], one_hot[test], mean, std) @ coefficients
            score = correlation(prediction, y[test])
            if score is not None:
                folds.append({"route": int(route), "n": int(test.sum()), **score})
        if not folds:
            raise ValueError("no leave-one-route-out fold has at least three labels")
        summaries.append({
            "lambda": penalty,
            "eligible_folds": len(folds),
            "mean_spearman": float(np.mean([fold["spearman"] for fold in folds])),
            "folds": folds,
        })
    best = max(summaries, key=lambda row: (row["mean_spearman"], row["lambda"]))
    return float(best["lambda"]), summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    train_rows = load_rows(args.train)
    test_rows = load_rows(args.test)
    train_continuous, train_one_hot = raw_features(train_rows)
    test_continuous, test_one_hot = raw_features(test_rows)
    train_y = np.asarray([float(row["equal_byte_terminal_value"]) for row in train_rows])
    test_y = np.asarray([float(row["equal_byte_terminal_value"]) for row in test_rows])
    penalty, cross_validation = cross_validate(
        train_rows, train_continuous, train_one_hot, train_y)

    mean = train_continuous.mean(axis=0)
    std = train_continuous.std(axis=0)
    std[std == 0] = 1.0
    coefficients = fit(transform(train_continuous, train_one_hot, mean, std),
                       train_y, penalty)
    train_prediction = transform(train_continuous, train_one_hot, mean, std) @ coefficients
    test_prediction = transform(test_continuous, test_one_hot, mean, std) @ coefficients
    train_local = np.asarray([float(row["local_score"]) for row in train_rows])
    test_local = np.asarray([float(row["local_score"]) for row in test_rows])

    predictions = []
    for row, prediction, truth in zip(test_rows, test_prediction, test_y):
        predictions.append({
            "route": int(row["route"]),
            "expert": int(row["expert"]),
            "ordered_rank": int(row["ordered_rank"]),
            "natural_slab": int(row["natural_slab"]),
            "predicted_terminal_value": float(prediction),
            "measured_terminal_value": float(truth),
            "local_score": float(row["local_score"]),
        })
    predictions.sort(key=lambda row: row["predicted_terminal_value"], reverse=True)

    test_model = correlation(test_prediction, test_y)
    test_control = correlation(test_local, test_y)
    model_spearman = None if test_model is None else test_model["spearman"]
    control_spearman = None if test_control is None else test_control["spearman"]
    correlation_gate = (model_spearman is not None and control_spearman is not None and
                        model_spearman >= 0.45 and model_spearman - control_spearman >= 0.15)
    result = {
        "schema": "kimi-k3-terminal-metadata-ridge-v1",
        "status": "MEASURED_CORRELATION_GATE_ONLY",
        "inputs": {
            "train": str(args.train), "train_sha256": digest(args.train),
            "test": str(args.test), "test_sha256": digest(args.test),
        },
        "fit": {
            "lambda": penalty,
            "continuous_feature_mean": mean.tolist(),
            "continuous_feature_std": std.tolist(),
            "coefficients": coefficients.tolist(),
            "cross_validation": cross_validation,
        },
        "correlation": {
            "train_model": correlation(train_prediction, train_y),
            "train_local_control": correlation(train_local, train_y),
            "heldout_model": test_model,
            "heldout_local_control": test_control,
            "heldout_spearman_gain": None if model_spearman is None or control_spearman is None
                                       else model_spearman - control_spearman,
            "preregistered_gate_pass": correlation_gate,
        },
        "heldout_predictions": predictions,
        "remaining_gate": "Actual equal-byte conditional swaps and teacher top1 recovery are still required.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "lambda": penalty,
                      "correlation_gate_pass": correlation_gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
