#!/usr/bin/env python3
"""Exploratory aggregate-direction training for the Kimi-K3 diagonal panel.

This deliberately writes a separate result from the registered closed-form
fit.  The original validation sequences remain untouched until the final
report; early stopping uses a sequence-disjoint subset of calibration data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file

from export_kimi_panel_safetensors import load_panel


CAPTURE_HEADER = struct.Struct("<8sIiIIQQII4Q")
RECORD_HEADER = struct.Struct("<IB3sI")
TEACHER_HEADER = struct.Struct("<8sIiIIQQ2Q")
CAPTURE_MAGIC = b"K3PNL001"
TEACHER_MAGIC = b"K3TGT001"


@dataclass
class PanelData:
    latent: np.ndarray
    expert_ids: np.ndarray
    router_weights: np.ndarray
    teacher: np.ndarray
    train_indices: np.ndarray
    development_indices: np.ndarray
    validation_indices: np.ndarray
    sequence_ids: list[str]
    sequence_splits: list[int]
    sequence_ranges: list[tuple[int, int]]
    model_layer: int
    dimension: int
    top_k: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bf16_to_float(values: np.ndarray) -> np.ndarray:
    words = values.astype(np.uint32, copy=False) << 16
    return words.view(np.float32)


def read_capture(path: Path) -> tuple[dict[str, int], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    with path.open("rb") as source:
        raw = source.read(CAPTURE_HEADER.size)
        if len(raw) != CAPTURE_HEADER.size:
            raise ValueError("capture header is truncated")
        (
            magic,
            version,
            model_layer,
            dimension,
            top_k,
            sequence_count,
            token_count,
            latent_storage,
            weight_storage,
            *reserved,
        ) = CAPTURE_HEADER.unpack(raw)
        if (
            magic != CAPTURE_MAGIC
            or version != 1
            or model_layer < 0
            or dimension <= 0
            or top_k <= 0
            or sequence_count <= 0
            or token_count <= 0
            or latent_storage != 1
            or weight_storage != 0
            or any(reserved)
        ):
            raise ValueError("capture header is invalid or unsupported")
        observed_tokens = 0
        for _ in range(sequence_count):
            raw = source.read(RECORD_HEADER.size)
            if len(raw) != RECORD_HEADER.size:
                raise ValueError("capture record header is truncated")
            identifier_bytes, split, record_reserved, count = RECORD_HEADER.unpack(raw)
            if (
                identifier_bytes <= 0
                or split not in (0, 1)
                or record_reserved != b"\0\0\0"
                or count <= 0
            ):
                raise ValueError("capture record header is invalid")
            identifier = source.read(identifier_bytes).decode("utf-8")
            tokens = np.fromfile(source, dtype="<i4", count=count)
            latent_bf16 = np.fromfile(
                source, dtype="<u2", count=count * dimension
            )
            expert_ids = np.fromfile(
                source, dtype="<i4", count=count * top_k
            )
            router_weights = np.fromfile(
                source, dtype="<f4", count=count * top_k
            )
            if (
                tokens.size != count
                or latent_bf16.size != count * dimension
                or expert_ids.size != count * top_k
                or router_weights.size != count * top_k
            ):
                raise ValueError("capture record payload is truncated")
            records.append(
                {
                    "id": identifier,
                    "split": split,
                    "latent": bf16_to_float(latent_bf16).reshape(count, dimension),
                    "expert_ids": expert_ids.reshape(count, top_k),
                    "router_weights": router_weights.reshape(count, top_k),
                }
            )
            observed_tokens += count
        if observed_tokens != token_count or source.read(1):
            raise ValueError("capture length does not match its header")
    return {
        "model_layer": model_layer,
        "dimension": dimension,
        "top_k": top_k,
        "sequence_count": sequence_count,
        "token_count": token_count,
    }, records


def read_teacher(path: Path, expected: dict[str, int]) -> np.ndarray:
    with path.open("rb") as source:
        raw = source.read(TEACHER_HEADER.size)
        if len(raw) != TEACHER_HEADER.size:
            raise ValueError("teacher header is truncated")
        (
            magic,
            version,
            model_layer,
            dimension,
            storage,
            sequence_count,
            token_count,
            *reserved,
        ) = TEACHER_HEADER.unpack(raw)
        if (
            magic != TEACHER_MAGIC
            or version != 1
            or storage != 0
            or model_layer != expected["model_layer"]
            or dimension != expected["dimension"]
            or sequence_count != expected["sequence_count"]
            or token_count != expected["token_count"]
            or any(reserved)
        ):
            raise ValueError("teacher header does not match the capture")
        values = np.fromfile(source, dtype="<f4", count=token_count * dimension)
        if values.size != token_count * dimension or source.read(1):
            raise ValueError("teacher payload is truncated or extended")
    if not np.isfinite(values).all():
        raise ValueError("teacher contains non-finite values")
    return values.reshape(token_count, dimension)


def load_data(capture_path: Path, teacher_path: Path) -> PanelData:
    header, records = read_capture(capture_path)
    teacher = read_teacher(teacher_path, header)
    latent = np.concatenate([record["latent"] for record in records])
    expert_ids = np.concatenate([record["expert_ids"] for record in records])
    router_weights = np.concatenate(
        [record["router_weights"] for record in records]
    )
    sequence_ids: list[str] = []
    sequence_splits: list[int] = []
    sequence_ranges: list[tuple[int, int]] = []
    train: list[int] = []
    development: list[int] = []
    validation: list[int] = []
    cursor = 0
    calibration_ordinal = 0
    for record in records:
        count = record["latent"].shape[0]
        indices = range(cursor, cursor + count)
        split = int(record["split"])
        if split == 1:
            validation.extend(indices)
        elif calibration_ordinal % 8 == 0:
            development.extend(indices)
            calibration_ordinal += 1
        else:
            train.extend(indices)
            calibration_ordinal += 1
        sequence_ids.append(str(record["id"]))
        sequence_splits.append(split)
        sequence_ranges.append((cursor, cursor + count))
        cursor += count
    if not train or not development or not validation:
        raise ValueError("capture does not provide three non-empty splits")
    if not np.allclose(router_weights.sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("native router weights do not sum to one")
    return PanelData(
        latent=latent,
        expert_ids=expert_ids,
        router_weights=router_weights,
        teacher=teacher,
        train_indices=np.asarray(train, dtype=np.int64),
        development_indices=np.asarray(development, dtype=np.int64),
        validation_indices=np.asarray(validation, dtype=np.int64),
        sequence_ids=sequence_ids,
        sequence_splits=sequence_splits,
        sequence_ranges=sequence_ranges,
        model_layer=header["model_layer"],
        dimension=header["dimension"],
        top_k=header["top_k"],
    )


def predict(
    latent: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    offset: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor:
    selected_offset = offset[expert_ids]
    selected_gain = gain[expert_ids]
    return (
        (selected_offset + selected_gain * latent[:, None, :])
        * router_weights[:, :, None]
    ).sum(dim=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.5)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


@torch.no_grad()
def evaluate(
    indices: torch.Tensor,
    latent: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    teacher: torch.Tensor,
    offset: torch.Tensor,
    gain: torch.Tensor,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    cosine_parts: list[torch.Tensor] = []
    relative_parts: list[torch.Tensor] = []
    for begin in range(0, indices.numel(), batch_size):
        selected = indices[begin : begin + batch_size]
        estimate = predict(
            latent[selected], expert_ids[selected], router_weights[selected],
            offset, gain
        )
        exact = teacher[selected]
        cosine_parts.append(functional.cosine_similarity(estimate, exact, dim=1))
        relative_parts.append(
            torch.linalg.vector_norm(estimate - exact, dim=1)
            / torch.linalg.vector_norm(exact, dim=1).clamp_min(1e-12)
        )
    cosine = torch.cat(cosine_parts).float().cpu().numpy()
    relative = torch.cat(relative_parts).float().cpu().numpy()
    return {"cosine": summarize(cosine), "relative_l2": summarize(relative)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260713)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 0 or args.batch_size <= 0 or args.evaluate_every <= 0:
        parser.error("training bounds must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.monotonic()
    data = load_data(args.capture, args.teacher)
    metadata, arrays = load_panel(args.panel)
    if (
        metadata["model_layer"] != data.model_layer
        or metadata["latent_dimension"] != data.dimension
    ):
        raise ValueError("panel does not match capture")

    latent = torch.from_numpy(data.latent.copy()).to(device)
    expert_ids = torch.from_numpy(data.expert_ids.copy()).to(device)
    router_weights = torch.from_numpy(data.router_weights.copy()).to(device)
    teacher = torch.from_numpy(data.teacher.copy()).to(device)
    train_indices = torch.from_numpy(data.train_indices).to(device)
    development_indices = torch.from_numpy(data.development_indices).to(device)
    validation_indices = torch.from_numpy(data.validation_indices).to(device)
    initial_offset = torch.from_numpy(arrays["unweighted_offset"].copy()).to(device)
    initial_gain = torch.from_numpy(arrays["unweighted_gain"].copy()).to(device)
    offset = torch.nn.Parameter(initial_offset.clone())
    gain = torch.nn.Parameter(initial_gain.clone())

    initial_development = evaluate(
        development_indices, latent, expert_ids, router_weights, teacher,
        offset, gain, args.evaluation_batch_size
    )
    initial_validation = evaluate(
        validation_indices, latent, expert_ids, router_weights, teacher,
        offset, gain, args.evaluation_batch_size
    )
    optimizer = torch.optim.Adam([offset, gain], lr=args.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    history: list[dict[str, float | int]] = []
    best_development = initial_development["cosine"]["mean"]
    best_step = 0
    best_offset = offset.detach().cpu().clone()
    best_gain = gain.detach().cpu().clone()
    evaluations_without_improvement = 0

    for step in range(1, args.steps + 1):
        sampled = torch.randint(
            train_indices.numel(), (args.batch_size,),
            generator=generator, device=device
        )
        indices = train_indices[sampled]
        estimate = predict(
            latent[indices], expert_ids[indices], router_weights[indices],
            offset, gain
        )
        loss = (1.0 - functional.cosine_similarity(
            estimate, teacher[indices], dim=1
        )).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([offset, gain], 1.0)
        optimizer.step()

        if step % args.evaluate_every == 0 or step == args.steps:
            development = evaluate(
                development_indices, latent, expert_ids, router_weights,
                teacher, offset, gain, args.evaluation_batch_size
            )
            development_mean = development["cosine"]["mean"]
            history.append(
                {
                    "step": step,
                    "training_batch_loss": float(loss.detach().cpu()),
                    "development_mean_cosine": development_mean,
                }
            )
            print(
                f"step={step} loss={float(loss.detach().cpu()):.7f} "
                f"development_cosine={development_mean:.9f}",
                flush=True,
            )
            if development_mean > best_development + 1e-7:
                best_development = development_mean
                best_step = step
                best_offset = offset.detach().cpu().clone()
                best_gain = gain.detach().cpu().clone()
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                if evaluations_without_improvement >= args.patience:
                    break

    offset.data.copy_(best_offset.to(device))
    gain.data.copy_(best_gain.to(device))
    final_development = evaluate(
        development_indices, latent, expert_ids, router_weights, teacher,
        offset, gain, args.evaluation_batch_size
    )
    final_validation = evaluate(
        validation_indices, latent, expert_ids, router_weights, teacher,
        offset, gain, args.evaluation_batch_size
    )
    final_training = evaluate(
        train_indices, latent, expert_ids, router_weights, teacher,
        offset, gain, args.evaluation_batch_size
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_prefix.with_suffix(".safetensors")
    save_file(
        {
            "offset": best_offset.contiguous(),
            "gain": best_gain.contiguous(),
        },
        str(artifact_path),
        metadata={
            "schema": "kimi-k3-directional-diagonal-v1",
            "model_layer": str(data.model_layer),
            "objective": "aggregate_cosine",
            "seed": str(args.seed),
        },
    )
    validation_mean = final_validation["cosine"]["mean"]
    if validation_mean >= 0.9998:
        verdict = "GREEN"
    elif validation_mean >= 0.99:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    result = {
        "schema": "kimi-k3-directional-diagonal-v1",
        "verdict": verdict,
        "status": "exploratory_followup_not_registered_primary_gate",
        "model_layer": data.model_layer,
        "objective": "mean aggregate cosine",
        "capture_sha256": sha256(args.capture),
        "teacher_sha256": sha256(args.teacher),
        "initial_panel_sha256": sha256(args.panel),
        "artifact": str(artifact_path),
        "seed": args.seed,
        "steps_requested": args.steps,
        "best_step": best_step,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "tokens": {
            "training": int(train_indices.numel()),
            "development": int(development_indices.numel()),
            "held_out_validation": int(validation_indices.numel()),
        },
        "initial": {
            "development": initial_development,
            "held_out_validation": initial_validation,
        },
        "trained": {
            "training": final_training,
            "development": final_development,
            "held_out_validation": final_validation,
        },
        "history": history,
        "final_logit_kl": None,
        "final_logit_kl_unavailable_reason": (
            "This bounded layer-one experiment stops before the remaining "
            "ninety-one model layers; full-model execution is the next memory gate."
        ),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0
        ),
    }
    result_path = args.output_prefix.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"VERDICT: {verdict}")
    print(
        "held-out mean cosine: "
        f"{initial_validation['cosine']['mean']:.9f} -> "
        f"{validation_mean:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
