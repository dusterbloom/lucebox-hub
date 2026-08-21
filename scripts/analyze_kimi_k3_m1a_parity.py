#!/usr/bin/env python3
"""Qualify one prospective M1a 12-prompt HTTP/SSE parity capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import run_kimi_k3_m1a_parity as capture


SCHEMA = "kimi_k3_m1a_parity_result_v1"
TRACE_PREFIX = "[server-token-trace] "
TRACE_SCHEMA = "dflash-committed-token-trace-v1"


def regular(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular non-symlink: {path}")


def bound_file(path_text: Any, expected: Any) -> bytes:
    if not isinstance(path_text, str) or not isinstance(expected, str):
        raise ValueError("artifact binding absent")
    path = Path(path_text)
    regular(path)
    raw = path.read_bytes()
    if capture.sha256_bytes(raw) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")
    return raw


def validate_registered_files(launch: dict[str, Any]) -> None:
    for section, path_key in (("model_card", "path"),
                              ("executable", "path"), ("cmake", "path"),
                              ("chat_template", "path")):
        bound_file(launch[section][path_key], launch[section]["sha256"])
    model = launch["model"]
    bound_file(model["frozen_p55_manifest"],
               model["frozen_p55_manifest_sha256"])
    bound_file(model["path"], model["shard1_sha256"])
    for section in ("aux_manifest", "sidecar_manifest", "h23_policy"):
        bound_file(launch[section]["path"], launch[section]["sha256"])
    repo = Path(launch["repo"])
    for relative, expected in launch["source_hashes"].items():
        bound_file(str(repo / relative), expected)
    model_dir = Path(model["path"]).parent
    observed = []
    for row in model["shards"]:
        path = model_dir / row["name"]
        regular(path)
        observed.append({"name": path.name, "size": path.stat().st_size})
    if observed != model["shards"] or sum(x["size"] for x in observed) != model["total_bytes"]:
        raise ValueError("checkpoint shard inventory mismatch")


def parse_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed {name} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"malformed {name} object")
    return value


def validate_endpoints(rows: Any, launch: dict[str, Any]) -> None:
    if not isinstance(rows, list) or [x.get("name") for x in rows] != [
            "health", "models", "props"]:
        raise ValueError("endpoint cardinality/order mismatch")
    values = {}
    for row in rows:
        if row.get("status") != 200:
            raise ValueError("endpoint HTTP failure")
        if "json" not in str(row.get("content_type", "")).lower():
            raise ValueError("endpoint content type mismatch")
        values[row["name"]] = parse_json(
            bound_file(row.get("path"), row.get("sha256")), row["name"])
    if values["health"].get("status") not in ("ok", "healthy"):
        raise ValueError("server health response mismatch")
    models = values["models"].get("data")
    if not isinstance(models, list) or not any(
            isinstance(x, dict) and x.get("id") == "kimi-k3" for x in models):
        raise ValueError("kimi-k3 absent from models")
    props = values["props"]
    card = props.get("model_card")
    source = props.get("budget_envelope", {}).get("model_card_source")
    runtime = props.get("runtime", {})
    prefix = props.get("prefix_cache", {})
    full = props.get("full_cache", {})
    try:
        source_matches = (Path(str(source)).resolve(strict=True) ==
                          Path(launch["model_card"]["source"]).resolve(strict=True))
    except (OSError, RuntimeError):
        source_matches = False
    if props.get("model_alias") != "kimi-k3" or not isinstance(card, dict) or \
            card.get("name") != "Kimi K3" or not source or \
            card.get("source") != launch["model_card"]["upstream_source"] or \
            not source_matches or \
            runtime.get("target_device") != "hip:1" or \
            runtime.get("draft_device") is not None or \
            props.get("speculative", {}).get("enabled") is not False or \
            prefix.get("capacity") != 0 or full.get("enabled") is not False or \
            full.get("capacity") != 0 or full.get("disk_policy") != "off":
        raise ValueError("props production/card/cache contract mismatch")


def parse_traces(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    regular(path)
    traces: dict[str, dict[str, Any]] = {}
    for line in path.read_text(errors="strict").splitlines():
        if not line.startswith(TRACE_PREFIX):
            continue
        try:
            row = json.loads(line[len(TRACE_PREFIX):])
        except json.JSONDecodeError as error:
            raise ValueError("malformed token-trace row") from error
        if not isinstance(row, dict) or row.get("schema") != TRACE_SCHEMA:
            raise ValueError("malformed token-trace schema")
        response_id = row.get("response_id")
        if response_id not in expected_ids:
            raise ValueError("foreign token-trace response ID")
        if response_id in traces:
            raise ValueError("duplicate token-trace response ID")
        traces[response_id] = row
    if set(traces) != expected_ids:
        raise ValueError("missing token-trace response ID")
    return traces


def validate_trace_parity(rows: list[dict[str, Any]],
                          references: list[dict[str, Any]],
                          traces: dict[str, dict[str, Any]]) -> None:
    by_task = {row["id"]: row for row in references}
    expected_order = [(row["id"], stream)
                      for row in references for stream in (False, True)]
    for row, (task_id, stream) in zip(rows, expected_order):
        trace = traces[row["response_id"]]
        reference = by_task[task_id]
        if trace.get("stream") is not stream or trace.get("ok") is not True or \
                trace.get("prompt_tokens") != reference["prompt_tokens"] or \
                trace.get("output_tokens") != reference["output_tokens"]:
            raise ValueError("token-trace parity mismatch")
    for task_id in capture.TASK_IDS:
        pair = [traces[row["response_id"]] for row in rows
                if row["task_id"] == task_id]
        if len(pair) != 2 or pair[0]["prompt_tokens"] != pair[1]["prompt_tokens"] or \
                pair[0]["output_tokens"] != pair[1]["output_tokens"]:
            raise ValueError("stream/nonstream token parity mismatch")


def analyze(manifest_path: Path) -> dict[str, Any]:
    regular(manifest_path)
    manifest = parse_json(manifest_path.read_bytes(), "capture manifest")
    if manifest.get("schema") != capture.SCHEMA or \
            manifest.get("scope") != "prospective_m1a_g2_only" or \
            manifest.get("reference_sha256") != capture.REFERENCE_SHA256 or \
            manifest.get("generation_request_count") != len(capture.TASK_IDS) * 2 or \
            manifest.get("analysis_performed") is not False:
        raise ValueError("capture manifest contract mismatch")
    references = capture.load_references(Path(manifest["reference"]))
    reference_manifest = parse_json(
        Path(manifest["reference"]).read_bytes(), "P55 reference manifest")
    launch_raw = bound_file(manifest.get("launch_manifest"),
                            manifest.get("launch_manifest_sha256"))
    launch = capture.validate_launch(Path(manifest["launch_manifest"]))
    if parse_json(launch_raw, "launch manifest").get("schema") != capture.LAUNCH_SCHEMA:
        raise ValueError("launch manifest changed during analysis")
    validate_registered_files(launch)
    if Path(reference_manifest.get("model_path", "")).resolve() != \
            Path(launch["model"]["path"]).resolve():
        raise ValueError("P55 reference checkpoint mismatch")
    validate_endpoints(manifest.get("endpoints"), launch)
    expected_order = [(row["id"], stream)
                      for row in references for stream in (False, True)]
    rows = manifest.get("requests")
    ids = manifest.get("expected_response_ids")
    expected_count = len(expected_order)
    if not isinstance(rows, list) or len(rows) != expected_count or \
            not isinstance(ids, list) or len(ids) != expected_count or \
            len(set(ids)) != expected_count or \
            any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("request/response ID cardinality mismatch")
    by_task = {row["id"]: row for row in references}
    observed = []
    for ordinal, (row, expected) in enumerate(zip(rows, expected_order)):
        task_id, stream = expected
        if row.get("ordinal") != ordinal or row.get("task_id") != task_id or \
                row.get("stream") is not stream or row.get("status") != 200:
            raise ValueError("request order/status mismatch")
        content_type = str(row.get("content_type", "")).lower()
        if (stream and "text/event-stream" not in content_type) or \
                (not stream and "json" not in content_type):
            raise ValueError("generation content type mismatch")
        body = parse_json(bound_file(row.get("request"),
                                     row.get("request_sha256")), "request")
        expected_body = {"model": "kimi-k3", "messages": [{"role": "user",
                         "content": by_task[task_id]["prompt"]}],
                         "max_tokens": 24, "temperature": 0, "seed": 0,
                         "chat_template_kwargs": {"enable_thinking": False},
                         "stream": stream}
        if body != expected_body:
            raise ValueError("generation body mismatch")
        raw = bound_file(row.get("response"), row.get("response_sha256"))
        response_id = capture.observed_response_id(raw, stream)
        if row.get("response_id") != response_id or ids[ordinal] != response_id:
            raise ValueError("response ID binding mismatch")
        observed.append(response_id)
    if observed != ids:
        raise ValueError("response ID order mismatch")
    bound_file(manifest.get("server_stderr"), manifest.get("server_stderr_sha256"))
    traces = parse_traces(Path(manifest["server_stderr"]), set(ids))
    validate_trace_parity(rows, references, traces)
    return {"schema": SCHEMA, "status": "PASS", "decision": "M1A_G2_PARITY_PASS",
            "scope": "prospective_m1a_g2_only", "quality_claim": None,
            "performance_claim": None, "capture_manifest": str(manifest_path.resolve()),
            "capture_manifest_sha256": capture.sha256(manifest_path),
            "reference_sha256": capture.REFERENCE_SHA256,
            "request_count": expected_count, "trace_count": expected_count,
            "response_ids": ids, "task_ids": list(capture.TASK_IDS)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    if args.result.exists() or args.result.with_name(args.result.name + ".partial").exists():
        parser.error("result must be fresh")
    try:
        result = analyze(args.capture_manifest)
        capture.atomic_json(args.result, result)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
