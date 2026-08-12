#!/usr/bin/env python3
"""Compare two Kimi full-vocabulary logit traces."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8sIIQQQII")
MAGIC = b"K3LOG001"


def load_trace(path: Path) -> tuple[dict[str, int], np.ndarray, bytes]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: truncated header")
    magic, version, vocabulary, rows, prompt, generated, storage, reserved = (
        HEADER.unpack_from(raw)
    )
    if (
        magic != MAGIC
        or version != 1
        or vocabulary == 0
        or rows == 0
        or storage != 0
        or reserved != 0
    ):
        raise ValueError(f"{path}: unsupported header")
    expected = HEADER.size + rows * vocabulary * 4
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(raw)}")
    logits = np.frombuffer(raw, dtype="<f4", offset=HEADER.size).reshape(
        rows, vocabulary
    )
    if not np.isfinite(logits).all():
        raise ValueError(f"{path}: non-finite logits")
    return {
        "version": version,
        "vocabulary": vocabulary,
        "rows": rows,
        "prompt_tokens": prompt,
        "generated_tokens": generated,
    }, logits, raw


def log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = values.max(axis=1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_header, reference, reference_raw = load_trace(args.reference)
    candidate_header, candidate, candidate_raw = load_trace(args.candidate)
    if reference_header != candidate_header:
        raise ValueError("trace shapes or request metadata differ")

    reference_log_probability = log_softmax(reference.astype(np.float64))
    candidate_log_probability = log_softmax(candidate.astype(np.float64))
    reference_probability = np.exp(reference_log_probability)
    divergence = np.sum(
        reference_probability
        * (reference_log_probability - candidate_log_probability),
        axis=1,
    )
    # Roundoff can produce a few negative values around 1e-16.
    divergence = np.maximum(divergence, 0.0)
    top_reference = reference.argmax(axis=1)
    top_candidate = candidate.argmax(axis=1)
    absolute = np.abs(candidate.astype(np.float64) - reference)
    result = {
        "schema": "kimi-logit-comparison-v1",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "header": reference_header,
        "byte_identical": reference_raw == candidate_raw,
        "logits_byte_identical": reference.tobytes() == candidate.tobytes(),
        "maximum_absolute_logit_difference": float(absolute.max()),
        "teacher_to_candidate_divergence": summarize(divergence),
        "top_choice_agreement": {
            "count": int(np.count_nonzero(top_reference == top_candidate)),
            "denominator": int(reference.shape[0]),
            "rate": float(np.mean(top_reference == top_candidate)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
