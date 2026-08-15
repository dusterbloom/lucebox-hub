#!/usr/bin/env python3
"""Join one H16 routed intervention trace to its final-logit consequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

from compare_kimi_logits import load_trace, log_softmax


HEADER = struct.Struct("<8s6I4Q")
MAGIC = b"K3INT001"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p01": float(np.quantile(values, 0.01)),
        "maximum": float(values.max()),
    }


def parse_ids(log: Path, label: str) -> list[int]:
    pattern = re.compile(rf"^\[kimi-k3-smoke\] {label}:\s*(.*)$")
    for line in log.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return [int(value) for value in match.group(1).split()]
    raise ValueError(f"{log}: missing {label}")


def load_intervention(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated intervention trace")
    fields = HEADER.unpack_from(raw)
    (
        magic,
        version,
        provider,
        budget,
        dimension,
        top_k,
        model_layer,
        records,
        record_bytes,
        reserved0,
        reserved1,
    ) = fields
    expected_record_bytes = 8 + top_k * 8 + 3 * dimension * 4
    if (
        magic != MAGIC
        or version != 1
        or provider not in (1, 2, 3)
        or dimension != 3584
        or top_k != 16
        or model_layer < 1
        or record_bytes != expected_record_bytes
        or reserved0 != 0
        or reserved1 != 0
        or len(raw) != HEADER.size + records * record_bytes
    ):
        raise ValueError("unsupported intervention trace")
    positions = np.empty(records, dtype=np.int32)
    token_offsets = np.empty(records, dtype=np.int32)
    ids = np.empty((records, top_k), dtype=np.int32)
    weights = np.empty((records, top_k), dtype=np.float32)
    exact = np.empty((records, dimension), dtype=np.float32)
    approximate = np.empty_like(exact)
    delta = np.empty_like(exact)
    cursor = HEADER.size
    for row in range(records):
        positions[row], token_offsets[row] = struct.unpack_from("<ii", raw, cursor)
        cursor += 8
        ids[row] = np.frombuffer(raw, "<i4", top_k, cursor)
        cursor += top_k * 4
        weights[row] = np.frombuffer(raw, "<f4", top_k, cursor)
        cursor += top_k * 4
        for target in (exact, approximate, delta):
            target[row] = np.frombuffer(raw, "<f4", dimension, cursor)
            cursor += dimension * 4
    if not all(np.isfinite(value).all() for value in (weights, exact, approximate, delta)):
        raise ValueError("non-finite intervention data")
    return {
        "provider": provider,
        "budget": budget,
        "dimension": dimension,
        "top_k": top_k,
        "model_layer": model_layer,
        "records": records,
    }, {
        "position": positions,
        "token_offset": token_offsets,
        "expert_ids": ids,
        "router_weights": weights,
        "exact": exact,
        "approximate": approximate,
        "delta": delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_logits", type=Path)
    parser.add_argument("candidate_logits", type=Path)
    parser.add_argument("intervention_trace", type=Path)
    parser.add_argument("teacher_stdout", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    intervention_header, intervention = load_intervention(args.intervention_trace)
    teacher_header, teacher_logits, _ = load_trace(args.teacher_logits)
    candidate_header, candidate_logits, _ = load_trace(args.candidate_logits)
    if teacher_header != candidate_header:
        raise ValueError("teacher/candidate logit headers disagree")
    if teacher_header["rows"] != intervention_header["records"]:
        raise ValueError("routed and terminal trace row counts disagree")

    prompt_ids = parse_ids(args.teacher_stdout, "prompt_ids")
    output_ids = parse_ids(args.teacher_stdout, "output_ids")
    targets = np.asarray(prompt_ids[1:] + output_ids, dtype=np.int64)
    if targets.size != teacher_logits.shape[0]:
        raise ValueError("teacher token chain does not match logit rows")

    exact = intervention["exact"].astype(np.float64)
    approximate = intervention["approximate"].astype(np.float64)
    stored_delta = intervention["delta"].astype(np.float64)
    calculated_delta = approximate - exact
    delta_consistency = float(np.max(np.abs(stored_delta - calculated_delta)))
    cosine = np.sum(exact * approximate, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1) * np.linalg.norm(approximate, axis=1),
        1.0e-30,
    )
    relative_l2 = np.linalg.norm(calculated_delta, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1), 1.0e-30
    )
    teacher_logp = log_softmax(teacher_logits.astype(np.float64))
    candidate_logp = log_softmax(candidate_logits.astype(np.float64))
    teacher_probability = np.exp(teacher_logp)
    kl = np.maximum(
        np.sum(teacher_probability * (teacher_logp - candidate_logp), axis=1),
        0.0,
    )
    row_index = np.arange(targets.size)
    delta_nll = -candidate_logp[row_index, targets] + teacher_logp[row_index, targets]
    teacher_top = teacher_logits.argmax(axis=1)
    candidate_top = candidate_logits.argmax(axis=1)
    agreement = teacher_top == candidate_top

    rows: list[dict[str, object]] = []
    for row in range(targets.size):
        rows.append({
            "row": row,
            "position": int(intervention["position"][row]),
            "token_offset": int(intervention["token_offset"][row]),
            "target_token": int(targets[row]),
            "layer_cosine": float(cosine[row]),
            "layer_relative_l2": float(relative_l2[row]),
            "terminal_kl": float(kl[row]),
            "target_token_delta_nll": float(delta_nll[row]),
            "teacher_top1": int(teacher_top[row]),
            "candidate_top1": int(candidate_top[row]),
            "top1_agreement": bool(agreement[row]),
        })
    provider_name = {
        1: "slabs",
        2: "whole",
        3: "slabs-recomposed",
    }[intervention_header["provider"]]
    result = {
        "schema": "kimi-h16-frozen-intervention-v1",
        "status": "MEASURED",
        "provider": provider_name,
        "budget": intervention_header["budget"],
        "exact_byte_fraction": (
            intervention_header["budget"] / 192
            if provider_name in ("slabs", "slabs-recomposed")
            else intervention_header["budget"] / 16
        ),
        "artifacts": {
            "teacher_logits": str(args.teacher_logits),
            "teacher_logits_sha256": sha256(args.teacher_logits),
            "candidate_logits": str(args.candidate_logits),
            "candidate_logits_sha256": sha256(args.candidate_logits),
            "intervention_trace": str(args.intervention_trace),
            "intervention_trace_sha256": sha256(args.intervention_trace),
        },
        "trace": intervention_header,
        "delta_storage_max_abs_error": delta_consistency,
        "layer_one_routed_output": {
            "cosine": summarize(cosine),
            "relative_l2": summarize(relative_l2),
        },
        "terminal_teacher_to_intervention_kl": summarize(kl),
        "target_token_delta_nll": summarize(delta_nll),
        "top1_agreement": {
            "count": int(agreement.sum()),
            "denominator": int(agreement.size),
            "rate": float(agreement.mean()),
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: result[key] for key in (
        "provider", "budget", "layer_one_routed_output",
        "terminal_teacher_to_intervention_kl", "top1_agreement")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
