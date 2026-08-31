#!/usr/bin/env python3
"""Score preregistered slab interventions against frozen K3 teacher logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8sIIQQQII")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def logits(path: Path, raw_vocab: int | None = None) -> tuple[dict[str, int], np.ndarray]:
    raw = path.read_bytes()
    # The production capture hook writes a single raw F32 terminal row.  Keep
    # that form usable for the first causal screens while retaining the richer
    # trace container for multi-position experiments.
    if raw_vocab is not None and len(raw) == 4 * raw_vocab:
        values = np.frombuffer(raw, dtype="<f4").reshape(1, raw_vocab)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite logits")
        return {"vocabulary": raw_vocab, "rows": 1,
                "prompt_tokens": -1, "generated_tokens": -1}, values
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: truncated K3 logit header")
    magic, version, vocab, rows, prompt, generated, storage, reserved = HEADER.unpack_from(raw)
    if magic != b"K3LOG001" or version != 1 or not vocab or not rows or storage or reserved:
        raise ValueError(f"{path}: unsupported K3 logit trace")
    if len(raw) != HEADER.size + 4 * rows * vocab:
        raise ValueError(f"{path}: logit extent mismatch")
    values = np.frombuffer(raw, dtype="<f4", offset=HEADER.size).reshape(rows, vocab)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite logits")
    return {"vocabulary": vocab, "rows": rows, "prompt_tokens": prompt, "generated_tokens": generated}, values


def terminal(teacher_path: Path, candidate_path: Path,
             raw_vocab: int | None = None) -> dict[str, object]:
    teacher_header, teacher = logits(teacher_path, raw_vocab)
    candidate_header, candidate = logits(candidate_path, raw_vocab)
    if teacher_header != candidate_header:
        raise ValueError(f"unaligned teacher/candidate traces: {teacher_path} {candidate_path}")
    teacher64 = teacher.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    teacher_logp = teacher64 - np.logaddexp.reduce(teacher64, axis=1)[:, None]
    candidate_logp = candidate64 - np.logaddexp.reduce(candidate64, axis=1)[:, None]
    divergence = np.maximum(0.0, np.sum(np.exp(teacher_logp) * (teacher_logp - candidate_logp), axis=1))
    top_teacher = teacher.argmax(axis=1)
    top_candidate = candidate.argmax(axis=1)
    top_margin = np.partition(teacher, -2, axis=1)[:, -1] - np.partition(teacher, -2, axis=1)[:, -2]
    candidate_margin = candidate[np.arange(candidate.shape[0]), top_teacher] - np.max(np.where(np.arange(candidate.shape[1])[None, :] == top_teacher[:, None], -np.inf, candidate), axis=1)
    return {
        "teacher_sha256": digest(teacher_path), "candidate_sha256": digest(candidate_path),
        "rows": int(teacher.shape[0]), "mean_kl": float(divergence.mean()),
        "median_kl": float(np.median(divergence)), "p95_kl": float(np.quantile(divergence, .95)),
        "maximum_kl": float(divergence.max()), "top1_agreement": int((top_teacher == top_candidate).sum()),
        "top1_denominator": int(teacher.shape[0]), "top1_changed": int((top_teacher != top_candidate).sum()),
        "mean_logit_margin_change": float((candidate_margin - top_margin).mean()),
    }


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        same = np.flatnonzero(values == value)
        result[same] = result[same].mean()
    return result


def correlation(rows: list[dict[str, object]], key: str) -> dict[str, float] | None:
    usable = [row for row in rows if key in row and row[key] is not None]
    if len(usable) < 3:
        return None
    score = np.asarray([float(row[key]) for row in usable])
    utility = np.asarray([float(row["utility_kl_per_byte"]) for row in usable])
    if not np.isfinite(score).all() or not np.isfinite(utility).all() or np.ptp(score) == 0 or np.ptp(utility) == 0:
        return None
    return {"n": len(usable), "pearson": float(np.corrcoef(score, utility)[0, 1]), "spearman": float(np.corrcoef(rank(score), rank(utility))[0, 1])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="immutable JSON intervention plan")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != "kimi-k3-terminal-slab-plan-v1":
        raise ValueError("unsupported intervention plan")
    rows: list[dict[str, object]] = []
    for intervention in plan["interventions"]:
        teacher = Path(intervention["teacher_logits"])
        candidate = Path(intervention["candidate_logits"])
        bytes_ = int(intervention["authoritative_bytes"])
        if bytes_ <= 0:
            raise ValueError(f"{intervention['id']}: authoritative_bytes must be positive")
        raw_vocab = intervention.get("raw_f32_vocabulary")
        metrics = terminal(teacher, candidate,
                           None if raw_vocab is None else int(raw_vocab))
        baseline = float(intervention.get("baseline_mean_kl", 0.0))
        recovered = max(0.0, baseline - float(metrics["mean_kl"]))
        rows.append({**intervention, "terminal": metrics, "terminal_kl_recovered": recovered,
                     "utility_kl_per_byte": recovered / bytes_})
    result = {"schema": "kimi-k3-terminal-slab-screen-v1", "status": "MEASURED",
              "plan_sha256": digest(args.plan), "selector": plan["selector"], "interventions": rows,
              "rank_correlation": {key: correlation(rows, key) for key in ("local_residual", "local_cosine", "gradient_score", "fisher_score")},
              "limitations": ["Frozen teacher histories only; this is not an on-policy quality result.", "Utility is conditional on the plan's declared baseline and is not assumed additive."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "interventions": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
