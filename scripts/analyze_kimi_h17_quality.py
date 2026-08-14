#!/usr/bin/env python3
"""Exploratory free-generation quality comparison for H17 providers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from compare_kimi_logits import load_trace, log_softmax


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def first_integer(text: str) -> int | None:
    match = re.search(r"(?<!\d)-?\d+(?!\d)", text.replace(",", ""))
    return int(match.group()) if match else None


def task_success(identifier: str, text: str) -> bool | None:
    value = normalize(text)
    if identifier == "math-multiply":
        return first_integer(value) == 703
    if identifier == "code-sum":
        return first_integer(value) == 10
    if identifier == "fact-capital":
        return value.startswith("tokyo")
    if identifier == "logic-raven":
        return "bird" in value
    if identifier == "translation-italian":
        return "buongiorno" in value or "buon giorno" in value
    if identifier == "word-synonym":
        return any(word in value for word in (
            "robust", "strong", "durable", "tenacious", "adaptable",
            "tough", "hardy", "resourceful",
        ))
    if identifier == "writing-moonlight":
        return len(re.findall(r"\b[\w'-]+\b", value)) == 5
    if identifier == "science-ice":
        return "less dense" in value or (
            "density" in value and "water" in value
        )
    if identifier == "math-power":
        return first_integer(value) == 1024
    if identifier == "computer-queue":
        return "queue" in value
    if identifier == "grammar-apples":
        return (
            "she doesn't like apples" in value
            or "she does not like apples" in value
        )
    if identifier == "science-photosynthesis":
        return (
            any(word in value for word in ("light", "sunlight"))
            and any(word in value for word in ("energy", "glucose", "sugar"))
        )
    return None


def degeneration(tokens: list[int], text: str) -> dict[str, object]:
    longest_run = 0
    run = 0
    previous = None
    for token in tokens:
        if token == previous:
            run += 1
        else:
            run = 1
            previous = token
        longest_run = max(longest_run, run)
    trigrams = [tuple(tokens[index:index + 3]) for index in range(len(tokens) - 2)]
    repeated_trigram = max(
        (trigrams.count(value) for value in set(trigrams)), default=0
    )
    normalized = normalize(text)
    flagged = longest_run >= 4 or repeated_trigram >= 3 or not normalized
    return {
        "flagged": flagged,
        "longest_identical_token_run": longest_run,
        "maximum_trigram_occurrences": repeated_trigram,
    }


def first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def trace_path(directory: Path, sequence: dict[str, object]) -> Path:
    registered = sequence.get("output_logits") or sequence.get("teacher_logits")
    if not isinstance(registered, str) or not registered:
        raise ValueError(f"{sequence.get('id')}: missing output trace")
    local = directory / Path(registered).name
    return local if local.is_file() else Path(registered)


def logit_diagnostic(
    native_path: Path,
    candidate_path: Path,
) -> dict[str, object]:
    native_header, native, _ = load_trace(native_path)
    candidate_header, candidate, _ = load_trace(candidate_path)
    if native_header["vocabulary"] != candidate_header["vocabulary"]:
        raise ValueError("quality-screen vocabularies differ")
    rows = min(native.shape[0], candidate.shape[0])
    native_logp = log_softmax(native[:rows].astype(np.float64))
    candidate_logp = log_softmax(candidate[:rows].astype(np.float64))
    probability = np.exp(native_logp)
    kl = np.maximum(
        np.sum(probability * (native_logp - candidate_logp), axis=1), 0.0
    )
    return {
        "aligned_rows": rows,
        "native_rows": int(native.shape[0]),
        "candidate_rows": int(candidate.shape[0]),
        "mean_kl": float(kl.mean()),
        "maximum_kl": float(kl.max()),
        "first_row_kl": float(kl[0]),
        "warning": "Diagnostic only; rows after token divergence condition on different histories.",
    }


def load_manifest(directory: Path) -> dict[str, object]:
    manifest = json.loads((directory / "suite-manifest.json").read_text())
    if manifest.get("schema") != "kimi-k3-h16-suite-v1":
        raise ValueError(f"{directory}: unsupported suite manifest")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument(
        "--mode", action="append", default=[], metavar="NAME=DIRECTORY"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    native_manifest = load_manifest(args.native)
    native_by_id = {
        str(sequence["id"]): sequence
        for sequence in native_manifest["sequences"]
    }
    modes: list[tuple[str, Path, dict[str, object]]] = [
        ("native", args.native, native_manifest)
    ]
    for raw in args.mode:
        if "=" not in raw:
            raise ValueError("--mode must be NAME=DIRECTORY")
        name, directory = raw.split("=", 1)
        modes.append((name, Path(directory), load_manifest(Path(directory))))

    mode_results: dict[str, object] = {}
    markdown = [
        "# H17 exploratory free-generation quality screen",
        "",
        "> Identity status is unchanged. Results for static96/oracle96/oracle144 "
        "are EXPLORATORY — confounded by all-192 arithmetic divergence.",
        "",
    ]
    for name, directory, manifest in modes:
        sequences = []
        successes = []
        native_successes = []
        preserved_native_successes = []
        math_code_rows = []
        exact_matches = 0
        for sequence in manifest["sequences"]:
            identifier = str(sequence["id"])
            native = native_by_id.get(identifier)
            if native is None or native["prompt_tokens"] != sequence["prompt_tokens"]:
                raise ValueError(f"{name}: frozen prompt mismatch for {identifier}")
            native_tokens = [int(value) for value in native["output_tokens"]]
            candidate_tokens = [int(value) for value in sequence["output_tokens"]]
            native_text = str(native.get("output_text", ""))
            candidate_text = str(sequence.get("output_text", ""))
            native_success = task_success(identifier, native_text)
            success = task_success(identifier, candidate_text)
            if success is not None:
                successes.append(success)
            if native_success is not None:
                native_successes.append(native_success)
            if native_success:
                preserved_native_successes.append(bool(success))
            exact = candidate_tokens == native_tokens
            exact_matches += int(exact)
            row = {
                "id": identifier,
                "prompt": sequence["text"],
                "native_text": native_text,
                "candidate_text": candidate_text,
                "native_tokens": native_tokens,
                "candidate_tokens": candidate_tokens,
                "first_generated_token_divergence": first_divergence(
                    native_tokens, candidate_tokens
                ),
                "output_token_count": len(candidate_tokens),
                "token_exact_vs_native": exact,
                "native_task_success": native_success,
                "task_success": success,
                "task_success_matches_native": success == native_success,
                "degeneration": degeneration(candidate_tokens, candidate_text),
                "terminal_metrics": logit_diagnostic(
                    trace_path(args.native, native),
                    trace_path(directory, sequence),
                ),
            }
            if identifier in {"math-multiply", "math-power", "code-sum"}:
                math_code_rows.append({
                    "id": identifier,
                    "native_success": native_success,
                    "candidate_success": success,
                    "preserved_native_success": (
                        bool(success) if native_success else None
                    ),
                })
            sequences.append(row)
            markdown.extend([
                f"## {identifier} — {name}",
                "",
                f"**Prompt:** {sequence['text']}",
                "",
                f"**Native:** `{native_text}`",
                "",
                f"**{name}:** `{candidate_text}`",
                "",
                f"Task success: `{success}` · first token divergence: "
                f"`{row['first_generated_token_divergence']}` · degeneration: "
                f"`{row['degeneration']['flagged']}`",
                "",
            ])
        mode_results[name] = {
            "label": (
                "EXPLORATORY — confounded by all-192 arithmetic divergence"
                if name in {"static96", "oracle96", "oracle144"}
                else "EXPLORATORY PRACTICAL QUALITY"
            ),
            "directory": str(directory),
            "provider": manifest.get("provider"),
            "provider_scope_note": (
                "Natural-order contiguous six-slab prefix per active expert, "
                "zero omitted tail; this is not the unavailable calibrated "
                "all-layer residual-norm selector."
                if name == "static96"
                else "Natural-order greedy prefix oracle with zero omitted tail; "
                "all 192 slabs are evaluated before selection, so no speed is claimed."
                if name in {"oracle96", "oracle144"}
                else None
            ),
            "sequence_count": len(sequences),
            "token_exact_vs_native": {
                "count": exact_matches,
                "rate": exact_matches / max(len(sequences), 1),
            },
            "task_success": {
                "count": int(sum(successes)),
                "denominator": len(successes),
                "rate": (
                    sum(successes) / len(successes) if successes else None
                ),
            },
            "task_success_vs_native": {
                "native_count": int(sum(native_successes)),
                "native_denominator": len(native_successes),
                "preserved_native_success_count": int(sum(
                    preserved_native_successes
                )),
                "preserved_native_success_denominator": len(
                    preserved_native_successes
                ),
                "preserved_native_success_rate": (
                    sum(preserved_native_successes) /
                    len(preserved_native_successes)
                    if preserved_native_successes else None
                ),
            },
            "coding_and_math_exact_success": math_code_rows,
            "degeneration_count": sum(
                int(row["degeneration"]["flagged"]) for row in sequences
            ),
            "sequences": sequences,
        }

    result = {
        "schema": "kimi-k3-h17-exploratory-free-generation-v1",
        "claim_status": "EXPLORATORY_CONFOUNDED",
        "identity_gate": "FAILED at all-192; unchanged by this screen",
        "modes": mode_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(markdown) + "\n")
    print(json.dumps({
        name: {
            "task_success": summary["task_success"],
            "task_success_vs_native": summary["task_success_vs_native"],
            "token_exact_vs_native": summary["token_exact_vs_native"],
            "degeneration_count": summary["degeneration_count"],
        }
        for name, summary in mode_results.items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
