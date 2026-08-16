#!/usr/bin/env python3
"""Export the honest, per-layer Kimi K3 calibrated-slab runtime substrate.

The recovered 2,048-token capture is a PILOT.  This exporter preserves that
qualification, the per-expert calibration hit counts, and every provenance
edge needed to reject a stale layer.  It never promotes an expert with fewer
than ``--minimum-expert-hits`` calibration routes: those experts remain exact
at runtime.

The natural-order slab sidecars are reused in place.  No 545 GiB repack is
performed.  Their registered SHA-256 values are embedded into each runtime
artifact and may optionally be reverified with ``--verify-sidecar-sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


FIRST_ROUTED_LAYER = 1
LAST_ROUTED_LAYER = 92
EXPERT_COUNT = 896
DIMENSION = 3584
SLAB_SIZE = 256
SLAB_COUNT = 12
ALIGNMENT = 4096
MAGIC = b"K3AUX001"
# v1 prefix, calibrated mask/count spans, then four binary SHA-256 bindings.
HEADER_V2 = struct.Struct("<8s8I14Q128s")
CAPTURE_HEADER = struct.Struct("<8sIiIIQQII4Q")
CAPTURE_RECORD = struct.Struct("<IB3sI")
CAPTURE_MAGIC = b"K3PNL001"
SIDECAR_HEADER_V1 = struct.Struct("<8s8I5Q")
SIDECAR_HEADER_V2 = struct.Struct("<8s8I8Q")
SIDECAR_MAGIC = b"K3SLB001"


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_model_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"invalid model checksum registry line: {line!r}")
        int(fields[0], 16)
        name = Path(fields[1].lstrip("*")).name
        if name in result:
            raise ValueError(f"duplicate model checksum for {name}")
        result[name] = fields[0].lower()
    if len(result) != 14:
        raise ValueError("the Kimi model checksum registry must name 14 shards")
    return result


def runtime_calibrated_mask(
    source_calibrated: np.ndarray,
    hit_counts: np.ndarray,
    minimum_expert_hits: int,
) -> np.ndarray:
    if source_calibrated.shape != (EXPERT_COUNT,) or hit_counts.shape != (EXPERT_COUNT,):
        raise ValueError("calibration masks must have one value per expert")
    if minimum_expert_hits <= 0:
        raise ValueError("minimum expert hits must be positive")
    if not np.array_equal(source_calibrated != 0, hit_counts != 0):
        raise ValueError("fit-state calibrated mask disagrees with capture hits")
    return ((source_calibrated != 0) & (hit_counts >= minimum_expert_hits)).astype(np.uint8)


def capture_calibration_hits(path: Path, expected_layer: int) -> tuple[np.ndarray, dict[str, int]]:
    """Read only route ids; seek over the large latent payload."""
    hits = np.zeros(EXPERT_COUNT, dtype="<u4")
    with path.open("rb") as source:
        raw = source.read(CAPTURE_HEADER.size)
        if len(raw) != CAPTURE_HEADER.size:
            raise ValueError(f"truncated capture header: {path}")
        (
            magic, version, model_layer, dimension, top_k,
            sequence_count, token_count, latent_storage, weight_storage,
            *reserved,
        ) = CAPTURE_HEADER.unpack(raw)
        if (
            magic != CAPTURE_MAGIC or version != 1 or model_layer != expected_layer
            or dimension != DIMENSION or top_k != 16 or sequence_count <= 0
            or token_count <= 0 or latent_storage != 1 or weight_storage != 0
            or any(reserved)
        ):
            raise ValueError(f"incompatible capture header: {path}")
        observed_tokens = 0
        calibration_tokens = 0
        for _ in range(sequence_count):
            raw = source.read(CAPTURE_RECORD.size)
            if len(raw) != CAPTURE_RECORD.size:
                raise ValueError(f"truncated capture record: {path}")
            identifier_bytes, split, record_reserved, count = CAPTURE_RECORD.unpack(raw)
            if identifier_bytes <= 0 or split not in (0, 1) or record_reserved != b"\0\0\0" or count <= 0:
                raise ValueError(f"invalid capture record: {path}")
            source.seek(identifier_bytes + count * 4 + count * DIMENSION * 2, os.SEEK_CUR)
            ids = np.fromfile(source, dtype="<i4", count=count * top_k)
            if ids.size != count * top_k or np.any(ids < 0) or np.any(ids >= EXPERT_COUNT):
                raise ValueError(f"invalid routed ids in capture: {path}")
            source.seek(count * top_k * 4, os.SEEK_CUR)  # router weights
            if split == 0:
                hits += np.bincount(ids, minlength=EXPERT_COUNT).astype("<u4")
                calibration_tokens += count
            observed_tokens += count
        if observed_tokens != token_count or source.tell() != path.stat().st_size:
            raise ValueError(f"capture length disagrees with header: {path}")
    return hits, {
        "sequence_count": int(sequence_count),
        "token_count": int(token_count),
        "calibration_token_count": calibration_tokens,
    }


def sidecar_layout(path: Path, expected_layer: int) -> dict[str, int]:
    with path.open("rb") as source:
        prefix = source.read(SIDECAR_HEADER_V1.size)
        if len(prefix) != SIDECAR_HEADER_V1.size:
            raise ValueError(f"truncated sidecar header: {path}")
        values = SIDECAR_HEADER_V1.unpack(prefix)
        if values[0] != SIDECAR_MAGIC or values[1] not in (1, 2):
            raise ValueError(f"unsupported sidecar header: {path}")
        if values[1] == 2:
            source.seek(0)
            raw = source.read(SIDECAR_HEADER_V2.size)
            if len(raw) != SIDECAR_HEADER_V2.size:
                raise ValueError(f"truncated v2 sidecar header: {path}")
            values = SIDECAR_HEADER_V2.unpack(raw)
    (
        _, version, layer, experts, dimension, expert_width, slab_size,
        slab_count, alignment, order_offset, order_bytes, payload_offset,
        slab_bytes, record_bytes, *component_bytes,
    ) = values
    if version == 1:
        component_bytes = [179200, 179200, 179200]
    gate_bytes, up_bytes, down_bytes = component_bytes
    payload_bytes = EXPERT_COUNT * record_bytes
    if (
        layer != expected_layer or experts != EXPERT_COUNT or dimension != DIMENSION
        or expert_width != SLAB_SIZE * SLAB_COUNT or slab_size != SLAB_SIZE
        or slab_count != SLAB_COUNT or alignment != ALIGNMENT
        or order_bytes != EXPERT_COUNT * SLAB_COUNT * 2
        or slab_bytes != gate_bytes + up_bytes + down_bytes
        or record_bytes != SLAB_COUNT * slab_bytes
        or payload_offset + payload_bytes != path.stat().st_size
    ):
        raise ValueError(f"sidecar geometry/length mismatch: {path}")
    return {
        "header_version": version,
        "order_offset": order_offset,
        "order_bytes": order_bytes,
        "payload_offset": payload_offset,
        "slab_bytes": slab_bytes,
        "gate_slab_bytes": gate_bytes,
        "up_slab_bytes": up_bytes,
        "down_slab_bytes": down_bytes,
        "record_bytes": record_bytes,
        "file_bytes": path.stat().st_size,
    }


def export_layer(
    *, layer: int, fit_state: Path, capture: Path, sidecar: Path,
    sidecar_manifest: Path, output: Path, manifest: Path,
    model_hashes: dict[str, str], model_registry_sha: str,
    minimum_expert_hits: int, verify_sidecar_sha256: bool,
) -> dict[str, object]:
    sidecar_record = json.loads(sidecar_manifest.read_text())
    layout = sidecar_layout(sidecar, layer)
    if (
        sidecar_record.get("model_layer") != layer
        or sidecar_record.get("output_bytes") != layout["file_bytes"]
        or Path(sidecar_record.get("output", "")) != sidecar
        or len(sidecar_record.get("output_sha256", "")) != 64
    ):
        raise ValueError(f"sidecar manifest mismatch for layer {layer}")
    registered_sidecar_sha = sidecar_record["output_sha256"].lower()
    if verify_sidecar_sha256 and sha256(sidecar) != registered_sidecar_sha:
        raise ValueError(f"sidecar checksum mismatch for layer {layer}")

    tensors: dict[str, dict[str, object]] = {}
    for component in ("gate", "up", "down"):
        source = sidecar_record["source_shards"][component]
        shard = Path(source["path"])
        if shard.name not in model_hashes or source["bytes"] != shard.stat().st_size:
            raise ValueError(f"unregistered source shard for layer {layer} {component}")
        tensors[component] = {
            "tensor": f"blk.{layer}.ffn_{component}_exps.weight",
            "shard": str(shard),
            "shard_bytes": source["bytes"],
            "shard_sha256": model_hashes[shard.name],
        }

    hits, capture_info = capture_calibration_hits(capture, layer)
    with np.load(fit_state, allow_pickle=False) as state:
        expected = {
            "slab_means", "slab_expected_norm", "slab_expected_residual_norm",
            "native_means", "native_expected_norm", "calibrated_experts",
        }
        if set(state.files) != expected:
            raise ValueError(f"unexpected fit-state fields for layer {layer}")
        means = np.asarray(state["slab_means"], dtype="<f4")
        importance = np.asarray(state["slab_expected_residual_norm"], dtype="<f4")
        source_calibrated = np.asarray(state["calibrated_experts"], dtype=np.uint8)
    if means.shape != (EXPERT_COUNT, SLAB_COUNT, DIMENSION):
        raise ValueError(f"bad slab means shape for layer {layer}")
    if importance.shape != (EXPERT_COUNT, SLAB_COUNT) or not np.isfinite(importance).all():
        raise ValueError(f"bad slab importance for layer {layer}")
    if source_calibrated.shape != (EXPERT_COUNT,):
        raise ValueError(f"bad calibrated mask for layer {layer}")
    order = np.argsort(-importance, axis=1, kind="stable").astype("<u2")
    ordered_means = np.take_along_axis(means, order[:, :, None].astype(np.int64), axis=1).astype("<f4", copy=False)
    ordered_importance = np.take_along_axis(importance, order.astype(np.int64), axis=1).astype("<f4", copy=False)
    try:
        runtime_calibrated = runtime_calibrated_mask(
            source_calibrated, hits, minimum_expert_hits
        )
    except ValueError as exc:
        raise ValueError(f"layer {layer}: {exc}") from exc

    arrays = [
        ("order", order),
        ("ordered_slab_means", ordered_means),
        ("ordered_residual_importance", ordered_importance),
        # v1 compatibility fields are retained but not used by calibrated96.
        ("native_means", np.zeros((0,), dtype="<f4")),
        ("native_expected_norm", np.zeros((0,), dtype="<f4")),
        ("calibrated_experts", runtime_calibrated),
        ("calibration_hit_counts", hits.astype("<u4", copy=False)),
    ]
    offsets: list[int] = []
    sizes: list[int] = []
    cursor = ALIGNMENT
    for _, array in arrays:
        cursor = align(cursor)
        offsets.append(cursor)
        sizes.append(array.nbytes)
        cursor += array.nbytes

    fit_sha = sha256(fit_state)
    capture_sha = sha256(capture)
    provenance = b"".join(bytes.fromhex(value) for value in (
        fit_sha, capture_sha, registered_sidecar_sha, model_registry_sha,
    ))
    span_values = [value for pair in zip(offsets[:5], sizes[:5]) for value in pair]
    span_values.extend([offsets[5], sizes[5], offsets[6], sizes[6]])
    header = HEADER_V2.pack(
        MAGIC, 2, layer, EXPERT_COUNT, DIMENSION, SLAB_SIZE, SLAB_COUNT,
        0, ALIGNMENT, *span_values, provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb", buffering=0) as sink:
        block = header + bytes(ALIGNMENT - len(header))
        sink.write(block)
        digest.update(block)
        position = ALIGNMENT
        for (_, array), offset in zip(arrays, offsets):
            padding = bytes(offset - position)
            sink.write(padding)
            digest.update(padding)
            raw = array.tobytes(order="C")
            sink.write(raw)
            digest.update(raw)
            position = offset + len(raw)
        sink.flush()
        os.fsync(sink.fileno())
    temporary.replace(output)

    record: dict[str, object] = {
        "schema": "kimi-k3-calibrated-slab-runtime-v2",
        "status": "PILOT_INSUFFICIENT_FOR_QUALITY_CERTIFICATION",
        "model_layer": layer,
        "policy": {
            "requested_nominal_slab_budget": 96,
            "minimum_calibration_hits_per_expert": minimum_expert_hits,
            "insufficient_or_unseen_expert_action": "exact routed-expert fallback",
            "missing_or_invalid_layer_action": "exact routed-layer fallback",
            "recomposition": "natural positions and one full-width down reduction",
        },
        "coverage": {
            **capture_info,
            "experts_with_any_calibration_hit": int(np.count_nonzero(hits)),
            "calibrated_experts": int(np.count_nonzero(runtime_calibrated)),
            "exact_fallback_experts": int(np.count_nonzero(runtime_calibrated == 0)),
            "minimum_hits": int(hits.min()),
            "median_hits": float(np.median(hits)),
            "maximum_hits": int(hits.max()),
        },
        "provenance": {
            "fit_state": str(fit_state), "fit_state_sha256": fit_sha,
            "capture": str(capture), "capture_sha256": capture_sha,
            "sidecar": str(sidecar), "sidecar_bytes": layout["file_bytes"],
            "sidecar_sha256": registered_sidecar_sha,
            "sidecar_sha256_reverified": verify_sidecar_sha256,
            "model_checksum_registry_sha256": model_registry_sha,
            "source_tensors": tensors,
        },
        "layout": layout,
        "output": str(output), "output_bytes": output.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "arrays": {
            name: {"offset": offset, "bytes": size, "dtype": str(array.dtype), "shape": list(array.shape)}
            for (name, array), offset, size in zip(arrays, offsets, sizes)
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit_state_root", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("sidecar_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model-checksums", type=Path, default=Path(__file__).with_name("kimi_k3_ud_iq1s.sha256"))
    parser.add_argument("--minimum-expert-hits", type=int, default=8)
    parser.add_argument("--capture-tokens", type=int, default=2048)
    parser.add_argument("--first-layer", type=int, default=FIRST_ROUTED_LAYER)
    parser.add_argument("--last-layer", type=int, default=LAST_ROUTED_LAYER)
    parser.add_argument("--verify-sidecar-sha256", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (1 <= args.first_layer <= args.last_layer <= 92):
        raise ValueError("layer range must be within 1..92")
    if args.minimum_expert_hits <= 0:
        raise ValueError("minimum expert hits must be positive")
    if args.capture_tokens <= 0:
        raise ValueError("capture token count must be positive")
    model_hashes = registered_model_hashes(args.model_checksums)
    model_registry_sha = sha256(args.model_checksums)
    records = []
    for layer in range(args.first_layer, args.last_layer + 1):
        stem = f"kimi_layer{layer:02d}"
        paths = {
            "fit_state": args.fit_state_root / f"{stem}_neuron_slabs_calibration.npz",
            "capture": args.capture_root / f"{stem}_{args.capture_tokens}.bin",
            "sidecar": args.sidecar_root / f"{stem}_natural_slabs.k3slab",
            "sidecar_manifest": args.sidecar_root / f"{stem}_natural_slabs.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing layer artifacts: " + ", ".join(missing))
        layout = sidecar_layout(paths["sidecar"], layer)
        sidecar_record = json.loads(paths["sidecar_manifest"].read_text())
        if sidecar_record.get("output_sha256") is None:
            raise ValueError(f"layer {layer} sidecar has no registered checksum")
        if args.preflight_only:
            if (
                sidecar_record.get("model_layer") != layer
                or sidecar_record.get("output_bytes") != layout["file_bytes"]
                or Path(sidecar_record.get("output", "")) != paths["sidecar"]
            ):
                raise ValueError(f"sidecar manifest mismatch at layer {layer}")
            for component in ("gate", "up", "down"):
                source = sidecar_record["source_shards"][component]
                shard = Path(source["path"])
                if (
                    shard.name not in model_hashes
                    or not shard.is_file()
                    or source["bytes"] != shard.stat().st_size
                ):
                    raise ValueError(
                        f"unregistered {component} source at layer {layer}"
                    )
            hits, capture_info = capture_calibration_hits(paths["capture"], layer)
            with np.load(paths["fit_state"], allow_pickle=False) as state:
                mask = np.asarray(state["calibrated_experts"])
                if mask.shape != (EXPERT_COUNT,):
                    raise ValueError(f"bad calibrated mask at layer {layer}")
                importance = np.asarray(state["slab_expected_residual_norm"])
                if importance.shape != (EXPERT_COUNT, SLAB_COUNT) or not np.isfinite(importance).all():
                    raise ValueError(f"bad slab importance at layer {layer}")
            calibrated = runtime_calibrated_mask(
                mask, hits, args.minimum_expert_hits
            )
            records.append({
                "layer": layer,
                "sidecar_header_version": layout["header_version"],
                "sidecar_bytes": layout["file_bytes"],
                "fit_state_bytes": paths["fit_state"].stat().st_size,
                **capture_info,
                "calibrated_experts": int(np.count_nonzero(calibrated)),
                "exact_fallback_experts": int(np.count_nonzero(calibrated == 0)),
            })
            continue
        record = export_layer(
            layer=layer, output=args.output_root / f"{stem}_calibrated96.k3aux",
            manifest=args.output_root / f"{stem}_calibrated96.json",
            model_hashes=model_hashes, model_registry_sha=model_registry_sha,
            minimum_expert_hits=args.minimum_expert_hits,
            verify_sidecar_sha256=args.verify_sidecar_sha256, **paths,
        )
        records.append(record)
        print(f"[calibrated96-export] layer={layer} calibrated={record['coverage']['calibrated_experts']} fallback={record['coverage']['exact_fallback_experts']}", flush=True)
    aggregate = {
        "schema": "kimi-k3-all-layer-calibrated-slab-runtime-v2",
        "status": "PILOT_INSUFFICIENT_FOR_QUALITY_CERTIFICATION",
        "quality_claim": "NONE",
        "speed_claim": "NONE",
        "first_layer": args.first_layer, "last_layer": args.last_layer,
        "layer_count": len(records), "requested_nominal_slab_budget": 96,
        "minimum_calibration_hits_per_expert": args.minimum_expert_hits,
        "exact_fallback_bytes": "reported from observed routes by the runtime; not hidden in the nominal budget",
        "preflight_only": args.preflight_only,
        "layers": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / ("preflight.json" if args.preflight_only else "all_layers_calibrated96_manifest.json")
    target.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: aggregate[key] for key in aggregate if key != "layers"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
