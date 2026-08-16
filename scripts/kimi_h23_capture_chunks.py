#!/usr/bin/env python3
"""Plan, validate, resume, and merge crash-bounded K3 all-layer captures.

The native capture executable publishes all 92 layer files only at the end of
one process.  This wrapper keeps that trusted writer, but limits each process to
a deterministic corpus chunk.  Completed chunks are immutable; a crash loses
only the active chunk.  Merging is byte-preserving and independently resumable
per layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


CAPTURE_HEADER = struct.Struct("<8sIiIIQQII4Q")
CAPTURE_RECORD = struct.Struct("<IB3sI")
CAPTURE_MAGIC = b"K3PNL001"
FIRST_LAYER = 1
LAST_LAYER = 92
DIMENSION = 3584
TOP_K = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


def corpus_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not all(isinstance(row.get(key), str) for key in ("id", "split", "text"))
            or not row["id"]
            or not row["text"]
            or row["split"] not in ("calibration", "validation")
        ):
            raise ValueError(f"invalid corpus row {number}")
        if row["id"] in identifiers:
            raise ValueError(f"duplicate corpus id {row['id']!r}")
        identifiers.add(row["id"])
        rows.append({key: row[key] for key in ("id", "split", "text")})
    if not rows or not any(row["split"] == "calibration" for row in rows) or not any(
        row["split"] == "validation" for row in rows
    ):
        raise ValueError("corpus must contain calibration and validation rows")
    return rows


def expected_plan(corpus: Path, root: Path, total_tokens: int, rows_per_chunk: int) -> dict[str, Any]:
    if total_tokens <= 0 or rows_per_chunk <= 0:
        raise ValueError("token and row bounds must be positive")
    rows = corpus_rows(corpus)
    chunks = []
    for index, begin in enumerate(range(0, len(rows), rows_per_chunk)):
        selected = rows[begin : begin + rows_per_chunk]
        text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected)
        chunks.append(
            {
                "index": index,
                "row_begin": begin,
                "row_end": begin + len(selected),
                "ids": [row["id"] for row in selected],
                "splits": [row["split"] for row in selected],
                "corpus_relative_path": f"corpora/chunk-{index:04d}.jsonl",
                "corpus_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "corpus_text": text,
            }
        )
    return {
        "schema": "kimi-k3-h23-chunked-capture-plan-v1",
        "source_corpus": str(corpus.resolve()),
        "source_corpus_sha256": sha256(corpus),
        "total_tokens": total_tokens,
        "rows_per_chunk": rows_per_chunk,
        "layers": [FIRST_LAYER, LAST_LAYER],
        "chunks": chunks,
    }


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(plan))
    for chunk in value["chunks"]:
        chunk.pop("corpus_text", None)
    return value


def prepare(corpus: Path, root: Path, total_tokens: int, rows_per_chunk: int) -> dict[str, Any]:
    plan = expected_plan(corpus, root, total_tokens, rows_per_chunk)
    published = public_plan(plan)
    plan_path = root / "capture-plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text())
        if existing != published:
            raise ValueError(f"existing capture plan disagrees: {plan_path}")
    elif plan_path.exists():
        raise ValueError(f"capture plan path is not a file: {plan_path}")
    else:
        atomic_json(plan_path, published)
    for chunk in plan["chunks"]:
        path = root / chunk["corpus_relative_path"]
        if path.is_file():
            if sha256(path) != chunk["corpus_sha256"]:
                raise ValueError(f"chunk corpus disagrees: {path}")
        elif path.exists():
            raise ValueError(f"chunk corpus path is not a file: {path}")
        else:
            atomic_text(path, chunk["corpus_text"])
    return published


def load_plan(root: Path) -> dict[str, Any]:
    path = root / "capture-plan.json"
    if not path.is_file():
        raise ValueError(f"missing capture plan: {path}")
    plan = json.loads(path.read_text())
    if plan.get("schema") != "kimi-k3-h23-chunked-capture-plan-v1":
        raise ValueError("unsupported capture plan")
    if plan.get("layers") != [FIRST_LAYER, LAST_LAYER]:
        raise ValueError("capture plan layer range disagrees")
    for expected_index, chunk in enumerate(plan.get("chunks", [])):
        if chunk.get("index") != expected_index:
            raise ValueError("capture plan chunk indices are not contiguous")
        corpus = root / chunk["corpus_relative_path"]
        if not corpus.is_file() or sha256(corpus) != chunk["corpus_sha256"]:
            raise ValueError(f"capture chunk corpus is absent or changed: {corpus}")
    return plan


def inspect_capture(path: Path, expected_layer: int | None = None) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as source:
        raw = source.read(CAPTURE_HEADER.size)
        if len(raw) != CAPTURE_HEADER.size:
            raise ValueError(f"truncated capture header: {path}")
        values = CAPTURE_HEADER.unpack(raw)
        (
            magic,
            version,
            layer,
            dimension,
            top_k,
            sequence_count,
            token_count,
            latent_storage,
            weight_storage,
            *reserved,
        ) = values
        if (
            magic != CAPTURE_MAGIC
            or version != 1
            or (expected_layer is not None and layer != expected_layer)
            or dimension != DIMENSION
            or top_k != TOP_K
            or sequence_count <= 0
            or token_count <= 0
            or latent_storage != 1
            or weight_storage != 0
            or any(reserved)
        ):
            raise ValueError(f"incompatible capture header: {path}")
        records = []
        observed_tokens = 0
        for _ in range(sequence_count):
            raw = source.read(CAPTURE_RECORD.size)
            if len(raw) != CAPTURE_RECORD.size:
                raise ValueError(f"truncated capture record: {path}")
            identifier_bytes, split, record_reserved, count = CAPTURE_RECORD.unpack(raw)
            if identifier_bytes <= 0 or split not in (0, 1) or record_reserved != b"\0\0\0" or count <= 0:
                raise ValueError(f"invalid capture record: {path}")
            identifier_raw = source.read(identifier_bytes)
            if len(identifier_raw) != identifier_bytes:
                raise ValueError(f"truncated capture identifier: {path}")
            try:
                identifier = identifier_raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"invalid UTF-8 capture identifier: {path}") from error
            payload = count * (4 + dimension * 2 + top_k * 4 + top_k * 4)
            source.seek(payload, os.SEEK_CUR)
            if source.tell() > size:
                raise ValueError(f"truncated capture payload: {path}")
            records.append({"id": identifier, "split": split, "tokens": count})
            observed_tokens += count
        if observed_tokens != token_count or source.tell() != size:
            raise ValueError(f"capture length/token count disagrees: {path}")
    return {
        "path": str(path),
        "header": values,
        "layer": layer,
        "sequence_count": sequence_count,
        "token_count": token_count,
        "records": records,
        "bytes": size,
    }


def validate_chunk(root: Path, chunk: dict[str, Any]) -> dict[str, Any]:
    directory = root / "chunks" / f"chunk-{chunk['index']:04d}"
    manifest_path = directory / "all_layers_capture_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing completed chunk manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "kimi-k3-panel-multi-layer-capture-v1"
        or manifest.get("layer_count") != LAST_LAYER - FIRST_LAYER + 1
        or manifest.get("first_routed_layer") != FIRST_LAYER
        or manifest.get("last_routed_layer") != LAST_LAYER
    ):
        raise ValueError(f"invalid chunk manifest: {manifest_path}")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or len(captures) != 92:
        raise ValueError(f"invalid chunk capture table: {manifest_path}")
    reference_records = None
    paths = []
    for layer, record in enumerate(captures, FIRST_LAYER):
        if record.get("model_layer") != layer:
            raise ValueError(f"chunk layer order disagrees: {manifest_path}")
        path = Path(record.get("path", ""))
        if path.parent.resolve() != directory.resolve() or not path.is_file():
            raise ValueError(f"chunk capture path escapes or is absent: {path}")
        info = inspect_capture(path, layer)
        if info["token_count"] != manifest.get("token_count") or info["sequence_count"] != manifest.get("sequence_count"):
            raise ValueError(f"chunk capture count disagrees: {path}")
        if reference_records is None:
            reference_records = info["records"]
        elif info["records"] != reference_records:
            raise ValueError(f"chunk sequence table differs by layer: {path}")
        paths.append(path)
        index_path = Path(str(path) + ".json")
        if not index_path.is_file():
            raise ValueError(f"chunk capture index is absent: {index_path}")
        index = json.loads(index_path.read_text())
        if (
            index.get("model_layer") != layer
            or index.get("token_count") != info["token_count"]
            or index.get("sequence_count") != info["sequence_count"]
            or index.get("capture_path") != str(path)
        ):
            raise ValueError(f"chunk capture index disagrees: {index_path}")
    expected_ids = chunk["ids"]
    expected_splits = [0 if value == "calibration" else 1 for value in chunk["splits"]]
    if [row["id"] for row in reference_records] != expected_ids[: len(reference_records)]:
        raise ValueError(f"chunk captured non-prefix corpus rows: {directory}")
    if [row["split"] for row in reference_records] != expected_splits[: len(reference_records)]:
        raise ValueError(f"chunk capture splits disagree: {directory}")
    return {
        "index": chunk["index"],
        "directory": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "sequence_count": manifest["sequence_count"],
        "token_count": manifest["token_count"],
        "paths": [str(path) for path in paths],
    }


def status(root: Path) -> dict[str, Any]:
    plan = load_plan(root)
    completed = []
    total = 0
    invalid_next = None
    for chunk in plan["chunks"]:
        directory = root / "chunks" / f"chunk-{chunk['index']:04d}"
        if total >= plan["total_tokens"]:
            if directory.exists():
                raise ValueError(f"unexpected chunk exists after target was reached: {directory}")
            continue
        if not directory.exists():
            break
        try:
            record = validate_chunk(root, chunk)
        except (OSError, ValueError) as error:
            invalid_next = {"index": chunk["index"], "directory": str(directory), "error": str(error)}
            break
        if total + record["token_count"] > plan["total_tokens"]:
            raise ValueError("completed chunks exceed capture target")
        completed.append(record)
        total += record["token_count"]
    next_index = None if total == plan["total_tokens"] else len(completed)
    if next_index is not None and next_index >= len(plan["chunks"]):
        raise ValueError(f"source corpus exhausted at {total}/{plan['total_tokens']} tokens")
    next_chunk = None if next_index is None else plan["chunks"][next_index]
    chunk_root = root / "chunks"
    if chunk_root.is_dir():
        allowed = {f"chunk-{index:04d}" for index in range(len(completed) + (0 if next_index is None else 1))}
        unexpected = sorted(path.name for path in chunk_root.iterdir() if path.name not in allowed)
        if unexpected:
            raise ValueError(f"out-of-order or unregistered chunk directories: {unexpected}")
    return {
        "schema": "kimi-k3-h23-chunked-capture-status-v1",
        "root": str(root),
        "plan_sha256": sha256(root / "capture-plan.json"),
        "target_tokens": plan["total_tokens"],
        "completed_tokens": total,
        "remaining_tokens": plan["total_tokens"] - total,
        "completed_chunks": completed,
        "next_chunk": next_chunk,
        "invalid_next": invalid_next,
        "complete": total == plan["total_tokens"],
    }


def merge(root: Path, output: Path) -> dict[str, Any]:
    state = status(root)
    if not state["complete"]:
        raise ValueError("cannot merge an incomplete chunked capture")
    output.mkdir(parents=True, exist_ok=True)
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=True)
    source_manifest_hashes = [row["manifest_sha256"] for row in state["completed_chunks"]]
    merged_records = []
    for layer in range(FIRST_LAYER, LAST_LAYER + 1):
        destination = output / f"kimi_layer{layer:02d}_{state['target_tokens']}.bin"
        index_path = Path(str(destination) + ".json")
        receipt_path = receipts / f"layer{layer:02d}.json"
        source_paths = [Path(row["paths"][layer - FIRST_LAYER]) for row in state["completed_chunks"]]
        reusable = False
        if destination.is_file() and index_path.is_file() and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            reusable = (
                receipt.get("plan_sha256") == state["plan_sha256"]
                and receipt.get("source_manifest_sha256") == source_manifest_hashes
                and receipt.get("output_sha256") == sha256(destination)
            )
            if reusable:
                info = inspect_capture(destination, layer)
                reusable = info["token_count"] == state["target_tokens"]
        if not reusable:
            if destination.exists() or index_path.exists() or receipt_path.exists():
                raise ValueError(f"invalid existing merged layer; quarantine output root: layer {layer}")
            infos = [inspect_capture(path, layer) for path in source_paths]
            sequence_count = sum(info["sequence_count"] for info in infos)
            token_count = sum(info["token_count"] for info in infos)
            header = list(infos[0]["header"])
            header[5] = sequence_count
            header[6] = token_count
            temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
            digest = hashlib.sha256()
            with temporary.open("wb") as target:
                raw_header = CAPTURE_HEADER.pack(*header)
                target.write(raw_header)
                digest.update(raw_header)
                for path in source_paths:
                    with path.open("rb") as source:
                        source.seek(CAPTURE_HEADER.size)
                        for block in iter(lambda: source.read(8 << 20), b""):
                            target.write(block)
                            digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(destination)
            combined_records = [record for info in infos for record in info["records"]]
            atomic_json(
                index_path,
                {
                    "schema": "kimi-k3-panel-capture-v1",
                    "model_layer": layer,
                    "latent_dimension": DIMENSION,
                    "top_k": TOP_K,
                    "latent_storage": "bfloat16",
                    "router_weight_storage": "float32",
                    "sequence_count": sequence_count,
                    "token_count": token_count,
                    "capture_path": str(destination),
                    "capture_mode": "merged-crash-bounded-all-routed-layers",
                    "sequences": [
                        {"id": row["id"], "split": "calibration" if row["split"] == 0 else "validation", "tokens": row["tokens"]}
                        for row in combined_records
                    ],
                },
            )
            atomic_json(
                receipt_path,
                {
                    "schema": "kimi-k3-h23-merged-layer-receipt-v1",
                    "layer": layer,
                    "plan_sha256": state["plan_sha256"],
                    "source_manifest_sha256": source_manifest_hashes,
                    "source_paths": [str(path) for path in source_paths],
                    "output": str(destination),
                    "output_sha256": digest.hexdigest(),
                    "output_bytes": destination.stat().st_size,
                },
            )
        receipt = json.loads(receipt_path.read_text())
        merged_records.append(
            {
                "model_layer": layer,
                "path": str(destination),
                "sha256": receipt["output_sha256"],
                "bytes": receipt["output_bytes"],
            }
        )
    manifest = {
        "schema": "kimi-k3-panel-multi-layer-capture-v1",
        "capture_mode": "merged-crash-bounded-all-routed-layers",
        "sequence_count": sum(row["sequence_count"] for row in state["completed_chunks"]),
        "token_count": state["target_tokens"],
        "first_routed_layer": FIRST_LAYER,
        "last_routed_layer": LAST_LAYER,
        "layer_count": 92,
        "plan": str(root / "capture-plan.json"),
        "plan_sha256": state["plan_sha256"],
        "source_chunk_manifest_sha256": source_manifest_hashes,
        "captures": merged_records,
    }
    atomic_json(output / "all_layers_capture_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--corpus", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--total-tokens", type=int, default=10_000)
    prepare_parser.add_argument("--rows-per-chunk", type=int, default=8)
    for name in ("status", "validate-chunk"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
        if name == "validate-chunk":
            child.add_argument("--index", type=int, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--root", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.corpus, args.root, args.total_tokens, args.rows_per_chunk)
    elif args.command == "status":
        value = status(args.root)
    elif args.command == "validate-chunk":
        plan = load_plan(args.root)
        if args.index < 0 or args.index >= len(plan["chunks"]):
            raise ValueError("chunk index is outside the plan")
        value = validate_chunk(args.root, plan["chunks"][args.index])
    else:
        value = merge(args.root, args.output)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
