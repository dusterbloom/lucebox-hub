#!/usr/bin/env python3
"""Fit identifiable low-rank behavioral metrics to Kimi H16 interventions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr

from analyze_kimi_h16_intervention import load_intervention


RANKS = (4, 8, 16, 32, 64)
RIDGES = (0.0, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
SAFE_THRESHOLDS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)


@dataclass
class Samples:
    delta: np.ndarray
    kl: np.ndarray
    family: np.ndarray
    split: np.ndarray
    sequence: np.ndarray
    sequence_row: np.ndarray

    def take(self, mask: np.ndarray) -> "Samples":
        return Samples(
            self.delta[mask], self.kl[mask], self.family[mask],
            self.split[mask], self.sequence[mask], self.sequence_row[mask],
        )


def load_suite(path: Path) -> Samples:
    analysis_path = path / "analysis.json"
    rows_path = path / "rows.csv"
    trace_path = path / "interventions.f32"
    analysis = json.loads(analysis_path.read_text())
    if (
        analysis.get("schema") != "kimi-k3-h16-suite-analysis-v1"
        or not analysis.get("paired")
        or not analysis.get("exact_reference", {}).get("byte_identical")
    ):
        raise ValueError(f"{path}: suite did not pass its exact-reference gate")
    family = f"{analysis['provider']}{analysis['budget']}"
    with rows_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    header, intervention = load_intervention(trace_path)
    if len(rows) != header["records"]:
        raise ValueError(f"{path}: rows and interventions disagree")
    global_rows = np.asarray([int(row["global_row"]) for row in rows])
    if not np.array_equal(global_rows, np.arange(len(rows))):
        raise ValueError(f"{path}: rows are not in intervention order")
    return Samples(
        delta=intervention["delta"].astype(np.float64),
        kl=np.asarray([float(row["terminal_kl"]) for row in rows]),
        family=np.full(len(rows), family, dtype=object),
        split=np.asarray([row["split"] for row in rows], dtype=object),
        sequence=np.asarray([row["sequence_id"] for row in rows], dtype=object),
        sequence_row=np.asarray([
            int(row["sequence_row"]) for row in rows
        ], dtype=np.int64),
    )


def concatenate(parts: list[Samples]) -> Samples:
    if not parts:
        raise ValueError("no provider suites were supplied")
    return Samples(*(
        np.concatenate([getattr(part, field) for part in parts], axis=0)
        for field in Samples.__dataclass_fields__
    ))


def finite_correlation(result: object) -> float | None:
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    samples: Samples,
) -> dict[str, object]:
    positive = actual[actual > 0.0]
    epsilon = max(
        1.0e-12,
        float(np.median(positive)) * 1.0e-8 if positive.size else 1.0e-12,
    )
    actual_safe = np.maximum(actual, epsilon)
    predicted_safe = np.maximum(predicted, epsilon)
    ratio = actual_safe / predicted_safe
    log_actual = np.log(actual_safe)
    log_predicted = np.log(predicted_safe)
    order = np.argsort(-ratio)[: min(10, ratio.size)]
    false_safe = [{
        "actual_kl": float(actual[index]),
        "predicted_kl": float(predicted[index]),
        "actual_over_predicted": float(ratio[index]),
        "provider_family": str(samples.family[index]),
        "sequence_id": str(samples.sequence[index]),
        "sequence_row": int(samples.sequence_row[index]),
    } for index in order]
    safety: dict[str, object] = {}
    for threshold in SAFE_THRESHOLDS:
        predicted_safe_mask = predicted <= threshold
        false_safe_count = int(np.count_nonzero(
            predicted_safe_mask & (actual > threshold)
        ))
        safety[f"{threshold:.0e}"] = {
            "predicted_safe_count": int(np.count_nonzero(predicted_safe_mask)),
            "predicted_safe_fraction": float(predicted_safe_mask.mean()),
            "false_safe_count": false_safe_count,
            "false_safe_fraction_of_predicted_safe": (
                false_safe_count / int(np.count_nonzero(predicted_safe_mask))
                if np.any(predicted_safe_mask) else None
            ),
        }
    return {
        "samples": int(actual.size),
        "spearman": finite_correlation(spearmanr(actual, predicted)),
        "pearson_log": finite_correlation(pearsonr(log_actual, log_predicted)),
        "median_actual_over_predicted": float(np.median(ratio)),
        "fraction_within_2x": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        "p95_underprediction_ratio": float(np.quantile(ratio, 0.95)),
        "p99_underprediction_ratio": float(np.quantile(ratio, 0.99)),
        "maximum_underprediction_ratio": float(ratio.max()),
        "mean_actual_kl": float(actual.mean()),
        "mean_predicted_kl": float(predicted.mean()),
        "safety_thresholds": safety,
        "worst_false_safe": false_safe,
    }


def calibration_factor(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = (actual > 0.0) & (predicted > 1.0e-15)
    if not np.any(valid):
        return 1.0
    return float(np.clip(
        np.median(actual[valid] / predicted[valid]), 1.0e-4, 1.0e4
    ))


def fit_nonnegative_metric(
    train_coordinates: np.ndarray,
    train_kl: np.ndarray,
    validation_coordinates: np.ndarray,
    validation_kl: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    train_features = 0.5 * np.square(train_coordinates)
    validation_features = 0.5 * np.square(validation_coordinates)
    scales = np.sqrt(np.mean(np.square(train_features), axis=0))
    scales = np.maximum(scales, 1.0e-30)
    normalized_train = train_features / scales
    normalized_validation = validation_features / scales
    best: tuple[float, np.ndarray, float, float] | None = None
    positive = validation_kl[validation_kl > 0.0]
    epsilon = max(
        1.0e-12,
        float(np.median(positive)) * 1.0e-8 if positive.size else 1.0e-12,
    )
    identity = np.eye(normalized_train.shape[1])
    for ridge in RIDGES:
        if ridge > 0.0:
            design = np.vstack([normalized_train, np.sqrt(ridge) * identity])
            target = np.concatenate([
                train_kl,
                np.zeros(normalized_train.shape[1]),
            ])
        else:
            design = normalized_train
            target = train_kl
        beta, _ = nnls(design, target, maxiter=100 * design.shape[1])
        weights = beta / scales
        validation_raw = validation_features @ weights
        factor = calibration_factor(validation_kl, validation_raw)
        validation_prediction = np.maximum(validation_raw * factor, epsilon)
        score = float(np.mean(np.square(
            np.log(np.maximum(validation_kl, epsilon))
            - np.log(validation_prediction)
        )))
        candidate = (score, weights, factor, ridge)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3], best[0]


def fit_identity(
    train: Samples, validation: Samples, test: Samples
) -> dict[str, object]:
    train_feature = 0.5 * np.sum(np.square(train.delta), axis=1)
    denominator = float(np.dot(train_feature, train_feature))
    weight = max(
        0.0,
        float(np.dot(train_feature, train.kl)) / denominator
        if denominator > 0.0 else 0.0,
    )
    validation_raw = 0.5 * np.sum(
        np.square(validation.delta), axis=1
    ) * weight
    factor = calibration_factor(validation.kl, validation_raw)
    predicted = 0.5 * np.sum(np.square(test.delta), axis=1) * weight * factor
    return {
        "metric": "scaled_identity",
        "weight": weight,
        "validation_calibration_factor": factor,
        "test": evaluate(test.kl, predicted, test),
    }


def fit_rank_ladder(
    train: Samples,
    validation: Samples,
    test: Samples,
    artifact_prefix: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    maximum_rank = min(max(RANKS), train.delta.shape[0], train.delta.shape[1])
    _, singular, right = np.linalg.svd(train.delta, full_matrices=False)
    right = right[:maximum_rank]
    total_energy = float(np.sum(np.square(singular)))
    ladder: list[dict[str, object]] = []
    artifacts: dict[str, np.ndarray] = {}
    for rank in RANKS:
        if rank > maximum_rank:
            ladder.append({
                "rank": rank,
                "status": "INSUFFICIENT_TRAINING_SAMPLES",
            })
            continue
        basis = right[:rank]
        train_coordinates = train.delta @ basis.T
        validation_coordinates = validation.delta @ basis.T
        test_coordinates = test.delta @ basis.T
        weights, factor, ridge, validation_score = fit_nonnegative_metric(
            train_coordinates, train.kl,
            validation_coordinates, validation.kl,
        )
        predicted = 0.5 * np.square(test_coordinates) @ weights * factor
        positive_weights = int(np.count_nonzero(weights > 0.0))
        ladder.append({
            "rank": rank,
            "status": "MEASURED",
            "parameterization": (
                "nonnegative diagonal PSD metric in uncentered training-delta "
                "principal coordinates"
            ),
            "positive_metric_directions": positive_weights,
            "training_delta_energy_fraction": (
                float(np.sum(np.square(singular[:rank]))) / total_energy
                if total_energy > 0.0 else 0.0
            ),
            "selected_ridge": ridge,
            "validation_log_mse": validation_score,
            "validation_calibration_factor": factor,
            "test": evaluate(test.kl, predicted, test),
        })
        if artifact_prefix is not None:
            artifacts[f"{artifact_prefix}_B_rank_{rank}"] = (
                np.sqrt(np.maximum(weights * factor, 0.0))[:, None] * basis
            ).astype(np.float32)
    return ladder, artifacts


def require_nonempty(samples: Samples, name: str) -> None:
    if samples.kl.size == 0:
        raise ValueError(f"empty {name} split")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()

    data = concatenate([load_suite(path) for path in args.suite])
    families = sorted(set(str(value) for value in data.family))
    if len(families) < 2:
        raise ValueError("at least two provider families are required")

    calibration = data.take(data.split == "calibration")
    validation = data.take(data.split == "validation")
    test = data.take(data.split == "test")
    for part, name in (
        (calibration, "calibration"),
        (validation, "validation"),
        (test, "test"),
    ):
        require_nonempty(part, name)

    sequence_ladder, artifacts = fit_rank_ladder(
        calibration, validation, test, "sequence_generalization"
    )
    provider_folds: list[dict[str, object]] = []
    for heldout in families:
        training = data.take(
            (data.split == "calibration") & (data.family != heldout)
        )
        tuning = data.take(
            (data.split == "validation") & (data.family != heldout)
        )
        heldout_test = data.take(
            (data.split == "test") & (data.family == heldout)
        )
        for part, name in (
            (training, f"{heldout} provider-fold training"),
            (tuning, f"{heldout} provider-fold validation"),
            (heldout_test, f"{heldout} provider-fold test"),
        ):
            require_nonempty(part, name)
        ladder, _ = fit_rank_ladder(training, tuning, heldout_test)
        provider_folds.append({
            "heldout_provider_family": heldout,
            "training_samples": int(training.kl.size),
            "validation_samples": int(tuning.kl.size),
            "test_samples": int(heldout_test.kl.size),
            "identity_baseline": fit_identity(training, tuning, heldout_test),
            "rank_ladder": ladder,
        })

    result = {
        "schema": "kimi-k3-h16-fisher-fit-v1",
        "status": "MEASURED",
        "provider_families": families,
        "sample_counts": {
            "total": int(data.kl.size),
            "calibration": int(calibration.kl.size),
            "validation": int(validation.kl.size),
            "test": int(test.kl.size),
        },
        "method": {
            "target": "KL(teacher || one-layer intervention)",
            "model": "predicted_KL = 0.5 * ||B delta||^2",
            "identifiability_constraint": (
                "B.T B is diagonal in the uncentered principal-coordinate "
                "basis learned only from training perturbations"
            ),
            "weight_constraint": "nonnegative",
            "ridge_candidates": list(RIDGES),
            "rank_candidates": list(RANKS),
            "calibration": (
                "one multiplicative factor selected from whole validation "
                "sequences only"
            ),
        },
        "sequence_generalization": {
            "training_split": "calibration",
            "validation_split": "validation",
            "test_split": "test",
            "identity_baseline": fit_identity(calibration, validation, test),
            "rank_ladder": sequence_ladder,
        },
        "provider_and_sequence_generalization": provider_folds,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **artifacts)
    print(json.dumps({
        "provider_families": families,
        "sample_counts": result["sample_counts"],
        "sequence_rank_ladder": [{
            "rank": row["rank"],
            "spearman": row.get("test", {}).get("spearman"),
            "p99_underprediction_ratio": row.get("test", {}).get(
                "p99_underprediction_ratio"
            ),
        } for row in sequence_ladder],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
