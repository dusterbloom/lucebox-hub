#!/usr/bin/env python3
"""Fail-closed auto planner for the future Lucebox --lsa-auto flag."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENCODER_SCHEMA = "luce.lsa.qwen35.encoder.v1"
AUTO_PLAN_SCHEMA = "luce.lsa.qwen35.auto_plan.v1"
AUTO_CALIBRATION_SCHEMA = "luce.lsa.qwen35.auto_calibration.v1"
EVALUATION_SCHEMA = "luce.lsa.qwen35.evaluation.v1"

DEFAULT_AVAILABLE_VRAM_GIB = 24.0
DEFAULT_WEIGHTS_GIB = 18.0
DEFAULT_RUNTIME_OVERHEAD_GIB = 2.0
DEFAULT_DRAFT_GIB = 0.7
DEFAULT_MAX_ARENA_CHUNKS = 256
DEFAULT_MIN_ARENA_CHUNKS = 32
DEFAULT_LOCAL_WINDOW_TOKENS = 8192
DEFAULT_INTERVAL = 64
DEFAULT_BLOCK_SIZE = 64
DEFAULT_KV_HEADS = 4
DEFAULT_HEAD_DIM = 256
DEFAULT_FULL_ATTN_LAYERS = 16

BITS_PER_VALUE = {
    "tq3_0": 3.5,
    "q4_0": 4.5,
    "q4_1": 5.0,
    "q5_0": 5.5,
    "q5_1": 6.0,
    "q8_0": 8.0,
    "f16": 16.0,
    "bf16": 16.0,
}


@dataclass(frozen=True)
class LsaAutoInputs:
    max_context_tokens: int
    local_window_tokens: int = DEFAULT_LOCAL_WINDOW_TOKENS
    block_size: int = DEFAULT_BLOCK_SIZE
    cache_type: str = "tq3_0"
    available_vram_gib: float = DEFAULT_AVAILABLE_VRAM_GIB
    weights_gib: float = DEFAULT_WEIGHTS_GIB
    runtime_overhead_gib: float = DEFAULT_RUNTIME_OVERHEAD_GIB
    draft_gib: float = DEFAULT_DRAFT_GIB
    draft_residency: str = "auto"
    max_arena_chunks: int = DEFAULT_MAX_ARENA_CHUNKS
    min_arena_chunks: int = DEFAULT_MIN_ARENA_CHUNKS
    kv_heads: int = DEFAULT_KV_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    full_attention_layers: int = DEFAULT_FULL_ATTN_LAYERS
    agentic: bool = False
    parity_validated: bool = False
    host_cache_validated: bool = False


@dataclass(frozen=True)
class EncoderSummary:
    path: str
    schema: str
    model_fingerprint: str
    hidden_size: int
    kv_heads: int
    head_dim: int
    rank: int
    parameters: int


@dataclass(frozen=True)
class CalibrationSummary:
    path: str
    schema: str
    max_trained_context_tokens: int | None
    learned_recall_10: float | None
    learned_recall_20: float | None
    random_recall_20: float | None
    recent_recall_20: float | None
    local_only_keep: float | None
    recommended_k: int | None
    all_chunks_parity: bool
    host_cache_validated: bool


@dataclass(frozen=True)
class RuntimeObservation:
    memory_pressure: float | None = None
    score_entropy: float | None = None
    top_score_margin: float | None = None
    selected_churn: float | None = None
    dflash_acceptance: float | None = None
    no_context_probability: float | None = None
    long_memory_hint: bool = False


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"{path}: cannot read JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_encoder_summary(path: Path) -> EncoderSummary:
    path = Path(path)
    manifest = _load_json(path / "encoder.json")
    if manifest.get("schema") != ENCODER_SCHEMA:
        raise ValueError(f"{path}: unsupported encoder schema {manifest.get('schema')!r}")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"{path}: encoder manifest is missing dataset metadata")
    weight = manifest.get("weight_file")
    if not isinstance(weight, dict):
        raise ValueError(f"{path}: encoder manifest is missing weight_file")
    weight_name = Path(str(weight.get("name", "")))
    if weight_name.is_absolute() or weight_name.name != str(weight_name):
        raise ValueError(f"{path}: encoder weight path must be local")
    weight_path = path / weight_name
    if not weight_path.exists():
        raise ValueError(f"{path}: encoder weight file is missing")
    if int(weight.get("size_bytes", -1)) != weight_path.stat().st_size:
        raise ValueError(f"{path}: encoder weight size does not match manifest")
    return EncoderSummary(
        path=str(path),
        schema=str(manifest["schema"]),
        model_fingerprint=str(dataset.get("model_fingerprint", "unknown")),
        hidden_size=int(dataset["hidden_size"]),
        kv_heads=int(dataset["kv_heads"]),
        head_dim=int(dataset["head_dim"]),
        rank=int(manifest["rank"]),
        parameters=int(manifest.get("parameters", 0)),
    )


def load_calibration_summary(path: Path) -> CalibrationSummary:
    path = Path(path)
    payload = _load_json(path)
    schema = payload.get("schema")
    if schema == AUTO_CALIBRATION_SCHEMA:
        metrics = payload.get("metrics", {})
        recommended = payload.get("recommended", {})
        gates = payload.get("gates", {})
        return CalibrationSummary(
            path=str(path),
            schema=str(schema),
            max_trained_context_tokens=_optional_int(
                payload.get("max_trained_context_tokens")
            ),
            learned_recall_10=_optional_float(metrics.get("learned_recall@0.100")),
            learned_recall_20=_optional_float(metrics.get("learned_recall@0.200")),
            random_recall_20=_optional_float(metrics.get("random_recall@0.200")),
            recent_recall_20=_optional_float(metrics.get("recent_recall@0.200")),
            local_only_keep=_optional_float(metrics.get("local_only_keep")),
            recommended_k=_optional_int(recommended.get("k")),
            all_chunks_parity=bool(gates.get("all_chunks_parity", False)),
            host_cache_validated=bool(gates.get("host_cache_validated", False)),
        )
    if schema == EVALUATION_SCHEMA:
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: evaluation report is missing metrics")
        return CalibrationSummary(
            path=str(path),
            schema=str(schema),
            max_trained_context_tokens=None,
            learned_recall_10=_optional_float(metrics.get("learned_recall@0.100")),
            learned_recall_20=_optional_float(metrics.get("learned_recall@0.200")),
            random_recall_20=_optional_float(metrics.get("random_recall@0.200")),
            recent_recall_20=_optional_float(metrics.get("recent_recall@0.200")),
            local_only_keep=_optional_float(metrics.get("local_only_keep")),
            recommended_k=None,
            all_chunks_parity=False,
            host_cache_validated=False,
        )
    raise ValueError(f"{path}: unsupported calibration schema {schema!r}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("calibration contains a non-finite metric")
    return out


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out < 0:
        raise ValueError("calibration contains a negative integer")
    return out


def load_runtime_observation(path: Path) -> RuntimeObservation:
    payload = _load_json(path)
    return RuntimeObservation(
        memory_pressure=_optional_float(payload.get("memory_pressure")),
        score_entropy=_optional_float(payload.get("score_entropy")),
        top_score_margin=_optional_float(payload.get("top_score_margin")),
        selected_churn=_optional_float(payload.get("selected_churn")),
        dflash_acceptance=_optional_float(payload.get("dflash_acceptance")),
        no_context_probability=_optional_float(payload.get("no_context_probability")),
        long_memory_hint=bool(payload.get("long_memory_hint", False)),
    )


def kv_bytes_per_token(inputs: LsaAutoInputs) -> float:
    bits = BITS_PER_VALUE.get(inputs.cache_type)
    if bits is None:
        raise ValueError(f"unsupported cache type: {inputs.cache_type!r}")
    return (
        2.0
        * inputs.kv_heads
        * inputs.head_dim
        * inputs.full_attention_layers
        * bits
        / 8.0
    )


def estimate_fit(inputs: LsaAutoInputs) -> dict[str, float | int]:
    if min(
        inputs.max_context_tokens,
        inputs.local_window_tokens,
        inputs.block_size,
        inputs.max_arena_chunks,
        inputs.min_arena_chunks,
        inputs.kv_heads,
        inputs.head_dim,
        inputs.full_attention_layers,
    ) <= 0:
        raise ValueError("auto-plan inputs must be positive")
    if inputs.min_arena_chunks > inputs.max_arena_chunks:
        raise ValueError("min arena chunks cannot exceed max arena chunks")
    draft_gib = (
        0.0
        if inputs.draft_residency == "request-scoped"
        else inputs.draft_gib
    )
    reserved_gib = inputs.weights_gib + inputs.runtime_overhead_gib + draft_gib
    headroom_gib = max(0.0, inputs.available_vram_gib - reserved_gib)
    bytes_per_token = kv_bytes_per_token(inputs)
    bytes_per_chunk = bytes_per_token * inputs.block_size
    fit_by_memory = int((headroom_gib * (1024**3)) // bytes_per_chunk)
    cold_tokens = max(0, inputs.max_context_tokens - inputs.local_window_tokens)
    cold_chunks = math.ceil(cold_tokens / inputs.block_size)
    return {
        "bytes_per_token": bytes_per_token,
        "bytes_per_chunk": bytes_per_chunk,
        "reserved_gib": reserved_gib,
        "headroom_gib": headroom_gib,
        "fit_by_memory_chunks": fit_by_memory,
        "cold_chunks": cold_chunks,
    }


def quality_gates_pass(calibration: CalibrationSummary | None) -> bool:
    if calibration is None:
        return False
    if calibration.learned_recall_20 is None or calibration.learned_recall_10 is None:
        return False
    if calibration.learned_recall_20 < 0.90 or calibration.learned_recall_10 < 0.80:
        return False
    if calibration.local_only_keep is not None and calibration.local_only_keep > 0.05:
        return False
    if calibration.random_recall_20 is not None:
        if calibration.learned_recall_20 - calibration.random_recall_20 < 0.30:
            return False
    return True


def adapt_lsa_plan(
    plan: dict[str, Any],
    observation: RuntimeObservation,
) -> dict[str, Any]:
    adapted = dict(plan)
    inputs = plan.get("inputs", {})
    if not plan.get("enabled"):
        adapted["adaptive"] = {
            "applied": False,
            "reason": "base plan is not enabled",
            "observation": asdict(observation),
        }
        return adapted

    max_k = int(plan.get("max_arena_chunks", DEFAULT_MAX_ARENA_CHUNKS))
    min_k = int(inputs.get("min_arena_chunks", DEFAULT_MIN_ARENA_CHUNKS))
    k = _clamp_int(int(plan.get("k", min_k)), min_k, max_k)
    interval = int(plan.get("interval", DEFAULT_INTERVAL))
    policy = str(plan.get("policy", "topk-capped"))
    actions: list[str] = []

    if (
        observation.no_context_probability is not None
        and observation.no_context_probability >= 0.80
        and not observation.long_memory_hint
    ):
        k = min_k
        interval = max(interval, 128)
        policy = "local-first"
        actions.append("high no-context probability: shrink to minimum arena")

    uncertain = False
    if observation.score_entropy is not None and observation.score_entropy >= 0.85:
        uncertain = True
    if observation.top_score_margin is not None and observation.top_score_margin <= 0.05:
        uncertain = True
    if observation.selected_churn is not None and observation.selected_churn >= 0.75:
        uncertain = True
    if uncertain:
        k = _clamp_int(max(k + max(1, k // 4), min_k), min_k, max_k)
        interval = min(interval, 64)
        if bool(inputs.get("agentic", False)):
            policy = "topk-stratified"
        actions.append("uncertain selector telemetry: widen active arena")

    if observation.long_memory_hint:
        k = _clamp_int(max(k, int(math.ceil(max_k * 0.75))), min_k, max_k)
        interval = min(interval, 64)
        policy = "topk-stratified"
        actions.append("long-memory hint: reserve stratified history coverage")

    if observation.memory_pressure is not None and observation.memory_pressure >= 0.90:
        k = _clamp_int(int(math.floor(k * 0.75)), min_k, max_k)
        actions.append("high memory pressure: reduce active arena")

    if observation.dflash_acceptance is not None and observation.dflash_acceptance < 0.35:
        interval = max(interval, 128)
        actions.append("low DFlash acceptance: slow LSA churn")

    adapted["k"] = k
    adapted["interval"] = interval
    adapted["policy"] = policy
    adapted["adaptive"] = {
        "applied": bool(actions),
        "actions": actions,
        "observation": asdict(observation),
    }
    return adapted


def plan_lsa_auto(
    inputs: LsaAutoInputs,
    *,
    encoder: EncoderSummary | None,
    calibration: CalibrationSummary | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    reasons: list[str] = []
    fit = estimate_fit(inputs)
    runtime_ready = False

    if encoder is None:
        return _disabled_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "missing trained encoder artifact",
            warnings,
        )

    if encoder.kv_heads != inputs.kv_heads or encoder.head_dim != inputs.head_dim:
        return _disabled_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "encoder geometry does not match runtime KV geometry",
            warnings,
        )

    if calibration is None:
        return _diagnostic_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "missing calibration report; use oracle diagnostics only",
            warnings,
        )

    if (
        calibration.max_trained_context_tokens is not None
        and inputs.max_context_tokens > calibration.max_trained_context_tokens
    ):
        return _diagnostic_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "requested context exceeds trained calibration length",
            warnings,
        )

    if not quality_gates_pass(calibration):
        return _diagnostic_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "calibration quality gates did not pass",
            warnings,
        )

    parity_validated = inputs.parity_validated or calibration.all_chunks_parity
    if not parity_validated:
        return _diagnostic_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "all-chunks packed-KV parity is not validated",
            warnings,
        )

    max_fit = min(
        inputs.max_arena_chunks,
        int(fit["fit_by_memory_chunks"]),
        int(fit["cold_chunks"]),
    )
    if max_fit < inputs.min_arena_chunks:
        return _disabled_plan(
            inputs,
            fit,
            encoder,
            calibration,
            "insufficient VRAM headroom for minimum active arena",
            warnings,
        )

    k = calibration.recommended_k or min(DEFAULT_MAX_ARENA_CHUNKS, max_fit)
    k = max(inputs.min_arena_chunks, min(int(k), max_fit))
    policy = "topk-stratified" if inputs.agentic else "topk-capped"
    host_cache_ok = inputs.host_cache_validated or calibration.host_cache_validated
    mode = "host-cache" if host_cache_ok else "oracle"
    if mode == "oracle":
        warnings.append("host-cache residency is not validated; memory savings are diagnostic only")
    reasons.append("trained encoder, calibration gates, and packed-KV parity are valid")
    return _plan(
        inputs,
        fit,
        encoder,
        calibration,
        enabled=True,
        runtime_ready=runtime_ready,
        mode=mode,
        policy=policy,
        k=k,
        interval=DEFAULT_INTERVAL,
        reasons=reasons,
        warnings=warnings,
    )


def _disabled_plan(
    inputs: LsaAutoInputs,
    fit: dict[str, float | int],
    encoder: EncoderSummary | None,
    calibration: CalibrationSummary | None,
    reason: str,
    warnings: list[str],
) -> dict[str, Any]:
    return _plan(
        inputs,
        fit,
        encoder,
        calibration,
        enabled=False,
        runtime_ready=False,
        mode="disabled",
        policy="none",
        k=0,
        interval=DEFAULT_INTERVAL,
        reasons=[reason],
        warnings=warnings,
    )


def _diagnostic_plan(
    inputs: LsaAutoInputs,
    fit: dict[str, float | int],
    encoder: EncoderSummary | None,
    calibration: CalibrationSummary | None,
    reason: str,
    warnings: list[str],
) -> dict[str, Any]:
    warnings = [*warnings, "live sparse retrieval remains disabled"]
    return _plan(
        inputs,
        fit,
        encoder,
        calibration,
        enabled=False,
        runtime_ready=False,
        mode="oracle",
        policy="diagnostic",
        k=0,
        interval=DEFAULT_INTERVAL,
        reasons=[reason],
        warnings=warnings,
    )


def _plan(
    inputs: LsaAutoInputs,
    fit: dict[str, float | int],
    encoder: EncoderSummary | None,
    calibration: CalibrationSummary | None,
    *,
    enabled: bool,
    runtime_ready: bool,
    mode: str,
    policy: str,
    k: int,
    interval: int,
    reasons: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema": AUTO_PLAN_SCHEMA,
        "implementation_stage": "offline-plan-only",
        "runtime_ready": runtime_ready,
        "enabled": enabled,
        "mode": mode,
        "policy": policy,
        "k": k,
        "interval": interval,
        "local_window_tokens": inputs.local_window_tokens,
        "max_arena_chunks": inputs.max_arena_chunks,
        "inputs": asdict(inputs),
        "fit": fit,
        "encoder": asdict(encoder) if encoder is not None else None,
        "calibration": asdict(calibration) if calibration is not None else None,
        "reasons": reasons,
        "warnings": warnings,
    }


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-ctx", type=int, required=True)
    parser.add_argument("--local-window", type=int, default=DEFAULT_LOCAL_WINDOW_TOKENS)
    parser.add_argument("--cache-type", default="tq3_0", choices=sorted(BITS_PER_VALUE))
    parser.add_argument("--available-vram-gib", type=float, default=DEFAULT_AVAILABLE_VRAM_GIB)
    parser.add_argument("--weights-gib", type=float, default=DEFAULT_WEIGHTS_GIB)
    parser.add_argument("--runtime-overhead-gib", type=float, default=DEFAULT_RUNTIME_OVERHEAD_GIB)
    parser.add_argument("--draft-gib", type=float, default=DEFAULT_DRAFT_GIB)
    parser.add_argument(
        "--draft-residency",
        choices=("auto", "persistent", "request-scoped"),
        default="auto",
    )
    parser.add_argument("--max-arena-chunks", type=int, default=DEFAULT_MAX_ARENA_CHUNKS)
    parser.add_argument("--min-arena-chunks", type=int, default=DEFAULT_MIN_ARENA_CHUNKS)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--full-attention-layers", type=int, default=DEFAULT_FULL_ATTN_LAYERS)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--parity-ok", action="store_true")
    parser.add_argument("--host-cache-ok", action="store_true")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inputs = LsaAutoInputs(
        max_context_tokens=args.max_ctx,
        local_window_tokens=args.local_window,
        cache_type=args.cache_type,
        available_vram_gib=args.available_vram_gib,
        weights_gib=args.weights_gib,
        runtime_overhead_gib=args.runtime_overhead_gib,
        draft_gib=args.draft_gib,
        draft_residency=args.draft_residency,
        max_arena_chunks=args.max_arena_chunks,
        min_arena_chunks=args.min_arena_chunks,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        full_attention_layers=args.full_attention_layers,
        agentic=args.agentic,
        parity_validated=args.parity_ok,
        host_cache_validated=args.host_cache_ok,
    )
    encoder = load_encoder_summary(args.encoder) if args.encoder else None
    calibration = (
        load_calibration_summary(args.calibration) if args.calibration else None
    )
    plan = plan_lsa_auto(inputs, encoder=encoder, calibration=calibration)
    if args.observation:
        plan = adapt_lsa_plan(plan, load_runtime_observation(args.observation))
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
