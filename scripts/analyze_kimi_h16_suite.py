#!/usr/bin/env python3
"""Analyze a sequence-disjoint Kimi H16 exact or paired suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_kimi_h16_intervention import load_intervention
from compare_kimi_logits import load_trace, log_softmax


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def load_manifest(directory: Path) -> tuple[Path, dict[str, object]]:
    path = directory / "suite-manifest.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "kimi-k3-h16-suite-v1"
        or not isinstance(manifest.get("sequences"), list)
        or not manifest["sequences"]
    ):
        raise ValueError(f"{path}: unsupported suite manifest")
    return path, manifest


def trace_path(directory: Path, sequence: dict[str, object], kind: str) -> Path:
    registered = sequence.get(f"{kind}_logits")
    if not isinstance(registered, str) or not registered:
        raise ValueError(f"sequence {sequence.get('id')}: missing {kind} logits")
    path = Path(registered)
    local = directory / path.name
    if local.is_file():
        return local
    if path.is_file():
        return path
    raise ValueError(f"sequence {sequence.get('id')}: cannot find {kind} logits")


def exact_reference_summary(
    directory: Path,
    manifest: dict[str, object],
    reference_directory: Path,
    reference_manifest: dict[str, object],
) -> dict[str, object]:
    reference_by_id = {
        sequence["id"]: sequence for sequence in reference_manifest["sequences"]
    }
    byte_identical = True
    maximum_difference = 0.0
    all_kl: list[np.ndarray] = []
    sequences: list[dict[str, object]] = []
    for sequence in manifest["sequences"]:
        identifier = sequence["id"]
        if identifier not in reference_by_id:
            raise ValueError(f"reference suite lacks sequence {identifier}")
        reference_sequence = reference_by_id[identifier]
        if (
            sequence["prompt_tokens"] != reference_sequence["prompt_tokens"]
            or sequence["output_tokens"] != reference_sequence["output_tokens"]
            or sequence["split"] != reference_sequence["split"]
        ):
            raise ValueError(f"reference behavior disagrees for {identifier}")
        candidate_path = trace_path(directory, sequence, "teacher")
        reference_path = trace_path(
            reference_directory, reference_sequence, "teacher"
        )
        candidate_header, candidate_logits, candidate_raw = load_trace(candidate_path)
        reference_header, reference_logits, reference_raw = load_trace(reference_path)
        if candidate_header != reference_header:
            raise ValueError(f"reference logit header disagrees for {identifier}")
        identical = candidate_raw == reference_raw
        byte_identical = byte_identical and identical
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(
                candidate_logits.astype(np.float64)
                - reference_logits.astype(np.float64)
            ))),
        )
        reference_logp = log_softmax(reference_logits.astype(np.float64))
        candidate_logp = log_softmax(candidate_logits.astype(np.float64))
        probability = np.exp(reference_logp)
        kl = np.maximum(
            np.sum(probability * (reference_logp - candidate_logp), axis=1),
            0.0,
        )
        all_kl.append(kl)
        sequences.append({
            "id": identifier,
            "rows": int(kl.size),
            "byte_identical": identical,
            "teacher_sha256": sha256(candidate_path),
            "reference_sha256": sha256(reference_path),
            "maximum_kl": float(kl.max()),
        })
    concatenated = np.concatenate(all_kl)
    return {
        "byte_identical": byte_identical,
        "maximum_absolute_logit_difference": maximum_difference,
        "teacher_to_reference_kl": summarize(concatenated),
        "sequences": sequences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_directory", type=Path)
    parser.add_argument("--reference-suite", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    manifest_path, manifest = load_manifest(args.suite_directory)
    reference_path, reference_manifest = load_manifest(args.reference_suite)
    reference = exact_reference_summary(
        args.suite_directory,
        manifest,
        args.reference_suite,
        reference_manifest,
    )
    result: dict[str, object] = {
        "schema": "kimi-k3-h16-suite-analysis-v1",
        "status": "MEASURED",
        "suite": str(args.suite_directory),
        "suite_manifest_sha256": sha256(manifest_path),
        "reference_suite": str(args.reference_suite),
        "reference_manifest_sha256": sha256(reference_path),
        "paired": bool(manifest["paired"]),
        "provider": manifest["provider"],
        "sequence_count": len(manifest["sequences"]),
        "exact_reference": reference,
    }

    if not manifest["paired"]:
        if args.output_csv:
            raise ValueError("an exact-repeat analysis has no intervention rows")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["exact_reference"], indent=2))
        return 0

    intervention_path = args.suite_directory / "interventions.f32"
    intervention_header, intervention = load_intervention(intervention_path)
    total_registered = sum(
        int(sequence["intervention_record_count"])
        for sequence in manifest["sequences"]
    )
    if intervention_header["records"] != total_registered:
        raise ValueError("intervention record count disagrees with manifest")

    rows: list[dict[str, object]] = []
    split_indices: dict[str, list[int]] = {}
    sequence_summaries: list[dict[str, object]] = []
    exact_all = intervention["exact"].astype(np.float64)
    approximate_all = intervention["approximate"].astype(np.float64)
    calculated_delta = approximate_all - exact_all
    stored_delta = intervention["delta"].astype(np.float64)
    delta_consistency = float(np.max(np.abs(stored_delta - calculated_delta)))

    for sequence in manifest["sequences"]:
        identifier = str(sequence["id"])
        split = str(sequence["split"])
        start = int(sequence["intervention_record_start"])
        count = int(sequence["intervention_record_count"])
        if count != int(sequence["prompt_token_count"]):
            raise ValueError(f"{identifier}: prompt and intervention counts disagree")
        end = start + count
        if start < 0 or end > intervention_header["records"]:
            raise ValueError(f"{identifier}: intervention range is invalid")
        positions = intervention["position"][start:end]
        token_offsets = intervention["token_offset"][start:end]
        if not np.array_equal(positions, np.arange(count, dtype=np.int32)):
            raise ValueError(f"{identifier}: positions are not a complete prefix")
        if np.any(token_offsets != 0):
            raise ValueError(f"{identifier}: unexpected batched token offset")

        teacher_path = trace_path(args.suite_directory, sequence, "teacher")
        candidate_path = trace_path(args.suite_directory, sequence, "candidate")
        teacher_header, teacher_logits, _ = load_trace(teacher_path)
        candidate_header, candidate_logits, _ = load_trace(candidate_path)
        if teacher_header != candidate_header or teacher_header["rows"] != count:
            raise ValueError(f"{identifier}: terminal trace shape disagrees")

        prompt_tokens = [int(value) for value in sequence["prompt_tokens"]]
        output_tokens = [int(value) for value in sequence["output_tokens"]]
        targets = np.asarray(prompt_tokens[1:] + output_tokens[:1], dtype=np.int64)
        if targets.size != count:
            raise ValueError(f"{identifier}: target chain is incomplete")

        exact = exact_all[start:end]
        approximate = approximate_all[start:end]
        delta = calculated_delta[start:end]
        cosine = np.sum(exact * approximate, axis=1) / np.maximum(
            np.linalg.norm(exact, axis=1) * np.linalg.norm(approximate, axis=1),
            1.0e-30,
        )
        relative_l2 = np.linalg.norm(delta, axis=1) / np.maximum(
            np.linalg.norm(exact, axis=1), 1.0e-30
        )
        teacher_logp = log_softmax(teacher_logits.astype(np.float64))
        candidate_logp = log_softmax(candidate_logits.astype(np.float64))
        probability = np.exp(teacher_logp)
        kl = np.maximum(
            np.sum(probability * (teacher_logp - candidate_logp), axis=1),
            0.0,
        )
        indices = np.arange(count)
        delta_nll = (
            -candidate_logp[indices, targets] + teacher_logp[indices, targets]
        )
        teacher_top = teacher_logits.argmax(axis=1)
        candidate_top = candidate_logits.argmax(axis=1)
        agreement = teacher_top == candidate_top

        sequence_row_indices: list[int] = []
        for sequence_row in range(count):
            global_row = start + sequence_row
            sequence_row_indices.append(global_row)
            rows.append({
                "global_row": global_row,
                "sequence_id": identifier,
                "split": split,
                "sequence_row": sequence_row,
                "position": int(positions[sequence_row]),
                "target_token": int(targets[sequence_row]),
                "layer_cosine": float(cosine[sequence_row]),
                "layer_relative_l2": float(relative_l2[sequence_row]),
                "terminal_kl": float(kl[sequence_row]),
                "target_token_delta_nll": float(delta_nll[sequence_row]),
                "teacher_top1": int(teacher_top[sequence_row]),
                "candidate_top1": int(candidate_top[sequence_row]),
                "top1_agreement": bool(agreement[sequence_row]),
            })
        split_indices.setdefault(split, []).extend(sequence_row_indices)
        sequence_summaries.append({
            "id": identifier,
            "split": split,
            "rows": count,
            "terminal_kl": summarize(kl),
            "mean_layer_cosine": float(cosine.mean()),
            "top1_agreement_rate": float(agreement.mean()),
            "teacher_logits_sha256": sha256(teacher_path),
            "candidate_logits_sha256": sha256(candidate_path),
        })

    ordered_rows = sorted(rows, key=lambda row: int(row["global_row"]))
    if [int(row["global_row"]) for row in ordered_rows] != list(
        range(intervention_header["records"])
    ):
        raise ValueError("sequence ranges do not cover the intervention trace once")
    cosine_all = np.asarray([row["layer_cosine"] for row in ordered_rows])
    relative_l2_all = np.asarray([
        row["layer_relative_l2"] for row in ordered_rows
    ])
    kl_all = np.asarray([row["terminal_kl"] for row in ordered_rows])
    delta_nll_all = np.asarray([
        row["target_token_delta_nll"] for row in ordered_rows
    ])
    agreement_all = np.asarray([
        row["top1_agreement"] for row in ordered_rows
    ], dtype=bool)
    by_split: dict[str, object] = {}
    for split, indices_list in split_indices.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        by_split[split] = {
            "rows": int(indices.size),
            "terminal_kl": summarize(kl_all[indices]),
            "mean_layer_cosine": float(cosine_all[indices].mean()),
            "top1_agreement_rate": float(agreement_all[indices].mean()),
        }

    provider_name = "slabs" if intervention_header["provider"] == 1 else "whole"
    result.update({
        "provider": provider_name,
        "budget": intervention_header["budget"],
        "exact_byte_fraction": (
            intervention_header["budget"] / 192
            if provider_name == "slabs"
            else intervention_header["budget"] / 16
        ),
        "intervention_trace": str(intervention_path),
        "intervention_trace_sha256": sha256(intervention_path),
        "intervention_header": intervention_header,
        "delta_storage_maximum_absolute_error": delta_consistency,
        "layer_routed_output": {
            "cosine": summarize(cosine_all),
            "relative_l2": summarize(relative_l2_all),
        },
        "terminal_teacher_to_intervention_kl": summarize(kl_all),
        "target_token_delta_nll": summarize(delta_nll_all),
        "top1_agreement": {
            "count": int(agreement_all.sum()),
            "denominator": int(agreement_all.size),
            "rate": float(agreement_all.mean()),
        },
        "by_split": by_split,
        "sequences": sequence_summaries,
    })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(ordered_rows[0]))
            writer.writeheader()
            writer.writerows(ordered_rows)
    print(json.dumps({
        "exact_reference": result["exact_reference"],
        "provider": result["provider"],
        "budget": result["budget"],
        "terminal_kl": result["terminal_teacher_to_intervention_kl"],
        "top1_agreement": result["top1_agreement"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
