#!/usr/bin/env python3
"""Measure exact Kimi route locality, cacheability, and prefetchability."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from probe_kimi_response_atlas import summarize
from train_kimi_panel_directional import load_data


EXPERT_COUNT = 896
CACHE_CAPACITIES = (16, 32, 64, 128, 256, 512)
PREFETCH_BUDGETS = (16, 32, 64)
HISTORY_DEPTHS = (1, 2, 4)
EXPERT_PAYLOAD_MIB = 5.383 * 1024 / 896
MODEL_MOE_LAYERS = 92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def route_recall(predicted: set[int], actual: np.ndarray) -> float:
    return sum(int(expert) in predicted for expert in actual) / actual.size


def lru_miss_fraction(sequences: list[np.ndarray], capacity: int) -> float:
    misses = 0
    routes = 0
    for sequence in sequences:
        cache: OrderedDict[int, None] = OrderedDict()
        for route in sequence:
            for raw_expert in route:
                expert = int(raw_expert)
                routes += 1
                if expert in cache:
                    cache.move_to_end(expert)
                else:
                    misses += 1
                    cache[expert] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
    return misses / routes


def build_transition_scores(sequences: list[np.ndarray]) -> np.ndarray:
    transitions = np.zeros((EXPERT_COUNT, EXPERT_COUNT), dtype=np.float64)
    for sequence in sequences:
        for previous, current in zip(sequence[:-1], sequence[1:]):
            # Each previous expert votes equally for every next expert.
            transitions[np.ix_(previous, current)] += 1.0
    row_sum = transitions.sum(axis=1, keepdims=True)
    transitions /= np.maximum(row_sum, 1.0)
    return transitions


def main() -> int:
    args = parse_args()
    data = load_data(args.capture, args.teacher)
    calibration_sequences: list[np.ndarray] = []
    validation_sequences: list[np.ndarray] = []
    for split, (begin, end) in zip(data.sequence_splits, data.sequence_ranges):
        sequence = data.expert_ids[begin:end]
        (validation_sequences if split else calibration_sequences).append(sequence)

    history_results: dict[str, dict[str, float]] = {}
    exact_set_repeats: list[float] = []
    for depth in HISTORY_DEPTHS:
        recalls: list[float] = []
        for sequence in validation_sequences:
            for token in range(depth, sequence.shape[0]):
                predicted: set[int] = set()
                for offset in range(1, depth + 1):
                    predicted.update(int(x) for x in sequence[token - offset])
                recalls.append(route_recall(predicted, sequence[token]))
                if depth == 1:
                    exact_set_repeats.append(
                        float(set(map(int, sequence[token - 1])) == set(map(int, sequence[token])))
                    )
        history_results[f"previous_{depth}_tokens"] = summarize(np.asarray(recalls))

    calibration_frequency = np.bincount(
        np.concatenate(calibration_sequences).reshape(-1), minlength=EXPERT_COUNT
    )
    static_order = np.argsort(-calibration_frequency, kind="stable")
    static_results: dict[str, dict[str, float]] = {}
    for budget in PREFETCH_BUDGETS:
        predicted = set(map(int, static_order[:budget]))
        recalls = [
            route_recall(predicted, route)
            for sequence in validation_sequences
            for route in sequence
        ]
        static_results[f"top_{budget}"] = summarize(np.asarray(recalls))

    transitions = build_transition_scores(calibration_sequences)
    transition_results: dict[str, dict[str, float]] = {}
    for budget in PREFETCH_BUDGETS:
        recalls: list[float] = []
        for sequence in validation_sequences:
            for previous, current in zip(sequence[:-1], sequence[1:]):
                scores = transitions[previous].sum(axis=0)
                predicted = set(
                    map(int, np.argpartition(scores, -budget)[-budget:])
                )
                recalls.append(route_recall(predicted, current))
        transition_results[f"predict_{budget}"] = summarize(np.asarray(recalls))

    lru_results: dict[str, dict[str, float]] = {}
    for capacity in CACHE_CAPACITIES:
        miss_fraction = lru_miss_fraction(validation_sequences, capacity)
        lru_results[f"capacity_{capacity}"] = {
            "miss_fraction": miss_fraction,
            "hit_fraction": 1.0 - miss_fraction,
            "one_layer_cache_gib": capacity * EXPERT_PAYLOAD_MIB / 1024,
            "same_capacity_each_of_92_layers_gib": (
                capacity * EXPERT_PAYLOAD_MIB * MODEL_MOE_LAYERS / 1024
            ),
            "projected_expert_payload_gib_per_token": 8.844 * miss_fraction,
        }

    result = {
        "schema": "kimi-k3-layer01-route-locality-v1",
        "status": "EXPLORATORY",
        "capture": str(args.capture),
        "model_layer": data.model_layer,
        "top_k": data.top_k,
        "calibration_sequences": len(calibration_sequences),
        "validation_sequences": len(validation_sequences),
        "validation_tokens": int(sum(x.shape[0] for x in validation_sequences)),
        "history_set_recall": history_results,
        "consecutive_exact_route_set_repeat_fraction": float(
            np.mean(exact_set_repeats)
        ),
        "static_calibration_frequency_prefetch": static_results,
        "calibration_transition_prefetch": transition_results,
        "validation_lru_cache": lru_results,
        "warnings": [
            "Only layer one is measured.",
            "Prefetch recall can hide latency but does not remove bandwidth.",
            "The all-layer cache projection assumes the same expert capacity per layer.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"previous-token-recall={history_results['previous_1_tokens']['mean']:.6f} "
        f"transition-32={transition_results['predict_32']['mean']:.6f} "
        f"lru64-hit={lru_results['capacity_64']['hit_fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
