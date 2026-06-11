"""Paper-aligned corpus planning for the Qwen3.5 LSA pilot.

The DeepSeek LSA paper uses long documents as source material and derives the
retriever labels offline from a frozen teacher. This module keeps that boundary
explicit: it creates a deterministic manifest and extraction boundary plan, but
does not download data or invoke the target model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "luce.lsa.qwen35.corpus_plan.v1"

BLOCK_SIZE = 64
LOOKAHEAD_HORIZON = 64
BOUNDARIES_PER_DOCUMENT = 8
SINK_TOKENS = 64
RECENT_TOKENS = 8192

PILOT_LENGTH_STRATA = (
    (16_384, 24),
    (32_768, 20),
    (65_536, 16),
    (131_072, 4),
)

SCALED_LENGTH_STRATA = (
    (16_384, 128),
    (32_768, 128),
    (65_536, 96),
    (131_072, 32),
)

SOURCE_MIX = (
    (
        "agent_coding_trace",
        0.40,
        (
            "local agent transcripts",
            "long tool-output traces",
            "repository issue and PR context",
        ),
    ),
    (
        "synthetic_retrieval",
        0.30,
        (
            "NeedleBench",
            "RULER-style needles",
            "locally generated scattered-evidence tasks",
        ),
    ),
    (
        "technical_document",
        0.20,
        (
            "LongBench-v2 technical subsets",
            "LEval code/legal/meeting subsets",
            "project docs and source bundles",
        ),
    ),
    (
        "local_or_no_context_control",
        0.10,
        (
            "short local-only questions",
            "prompt-independent controls",
            "negative retrieval controls",
        ),
    ),
)

HF_SOURCE_CANDIDATES = (
    {
        "name": "THUDM/LongBench-v2",
        "role": "realistic long-context source and eval gate",
        "notes": "Use contexts as raw material; do not use answer labels for LSA training.",
    },
    {
        "name": "THUDM/LongBench",
        "role": "shorter long-context source and baseline eval",
        "notes": "Useful for retrieval/code diversity below the 128K pilot ceiling.",
    },
    {
        "name": "opencompass/NeedleBench",
        "role": "synthetic retrieval source and eval gate",
        "notes": "Good for single-needle, multi-needle, and reasoning needles.",
    },
    {
        "name": "L4NLP/LEval",
        "role": "technical, legal, meeting, and code source material",
        "notes": "Small enough for fast pilot sampling.",
    },
    {
        "name": "tau/scrolls",
        "role": "long document diversity",
        "notes": "Use as source material only; labels still come from Qwen teacher extraction.",
    },
    {
        "name": "deepmind/pg19",
        "role": "long raw text controls",
        "notes": "Sample sparingly; broad text distribution but weak coding-agent match.",
    },
)


@dataclass(frozen=True)
class CorpusDocumentPlan:
    document_id: str
    split: str
    source_class: str
    target_tokens: int
    boundary_positions: tuple[int, ...]
    prompt_path: str
    boundary_path: str
    raw_capture_path: str
    npz_path: str


@dataclass(frozen=True)
class CorpusPlan:
    schema: str
    plan_name: str
    description: str
    label_source: str
    tokenization_contract: str
    extraction_contract: dict[str, object]
    length_strata: tuple[dict[str, int], ...]
    source_mix: tuple[dict[str, object], ...]
    hf_source_candidates: tuple[dict[str, str], ...]
    documents: tuple[CorpusDocumentPlan, ...]

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def total_source_tokens(self) -> int:
        return sum(document.target_tokens for document in self.documents)

    @property
    def total_boundaries(self) -> int:
        return sum(len(document.boundary_positions) for document in self.documents)


def aligned_boundary_positions(
    target_tokens: int,
    *,
    count: int = BOUNDARIES_PER_DOCUMENT,
    block_size: int = BLOCK_SIZE,
    lookahead_horizon: int = LOOKAHEAD_HORIZON,
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("boundary count must be positive")
    if block_size <= 0 or lookahead_horizon <= 0:
        raise ValueError("block size and lookahead horizon must be positive")
    first_block = 1
    last_block = (target_tokens - lookahead_horizon) // block_size
    if last_block - first_block + 1 < count:
        raise ValueError(
            f"{target_tokens} tokens cannot fit {count} aligned boundaries"
        )
    span = last_block - first_block
    positions = tuple(
        (first_block + (span * index) // (count + 1)) * block_size
        for index in range(1, count + 1)
    )
    if len(set(positions)) != len(positions):
        raise ValueError("boundary generation produced duplicate positions")
    return positions


def _category_counts(total_documents: int) -> dict[str, int]:
    if total_documents <= 0:
        raise ValueError("total document count must be positive")
    raw = [
        (name, total_documents * fraction)
        for name, fraction, _sources in SOURCE_MIX
    ]
    counts = {name: int(value) for name, value in raw}
    remainder = total_documents - sum(counts.values())
    ranked = sorted(
        raw,
        key=lambda item: (item[1] - int(item[1]), item[0]),
        reverse=True,
    )
    for name, _value in ranked[:remainder]:
        counts[name] += 1
    return counts


def _round_robin_labels(counts: dict[str, int]) -> list[str]:
    remaining = dict(counts)
    total = sum(remaining.values())
    labels: list[str] = []
    for _index in range(total):
        label = max(
            (name for name, count in remaining.items() if count > 0),
            key=lambda name: (remaining[name] / counts[name], remaining[name], name),
        )
        remaining[label] -= 1
        labels.append(label)
    return labels


def _split_for_index(index: int) -> str:
    remainder = index % 8
    if remainder == 6:
        return "validation"
    if remainder == 7:
        return "test"
    return "train"


def _iter_target_tokens(length_strata: Iterable[tuple[int, int]]) -> list[int]:
    targets: list[int] = []
    for target_tokens, count in length_strata:
        if target_tokens <= 0 or count <= 0:
            raise ValueError("length strata must contain positive values")
        targets.extend([target_tokens] * count)
    return targets


def build_corpus_plan(plan_name: str = "pilot") -> CorpusPlan:
    if plan_name == "pilot":
        length_strata = PILOT_LENGTH_STRATA
        description = "64-document Qwen3.5 LSA pilot: 16K-128K source tokens."
    elif plan_name == "scaled":
        length_strata = SCALED_LENGTH_STRATA
        description = "384-document Qwen3.5 LSA scale-up: 16K-128K source tokens."
    else:
        raise ValueError(f"unknown corpus plan: {plan_name!r}")

    targets = _iter_target_tokens(length_strata)
    category_counts = _category_counts(len(targets))
    categories = _round_robin_labels(category_counts)
    documents = []
    for index, (target_tokens, category) in enumerate(
        zip(targets, categories, strict=True)
    ):
        document_id = f"{plan_name}-{index:04d}"
        documents.append(
            CorpusDocumentPlan(
                document_id=document_id,
                split=_split_for_index(index),
                source_class=category,
                target_tokens=target_tokens,
                boundary_positions=aligned_boundary_positions(target_tokens),
                prompt_path=f"prompts/{document_id}.tokens.i32",
                boundary_path=f"boundaries/{document_id}.txt",
                raw_capture_path=f"raw/{document_id}",
                npz_path=f"npz/{document_id}.npz",
            )
        )

    extraction_contract = {
        "block_size": BLOCK_SIZE,
        "retrieval_interval": BLOCK_SIZE,
        "lookahead_horizon": LOOKAHEAD_HORIZON,
        "boundaries_per_document": BOUNDARIES_PER_DOCUMENT,
        "sink_tokens": SINK_TOKENS,
        "recent_tokens": RECENT_TOKENS,
        "hidden_tap": "layer46.post_ffn",
        "key_tap": "layer47.k_norm.pre_rope",
        "oracle_layers": list(range(3, 64, 4)),
        "teacher": "frozen Qwen3.5 target graph; labels from causal future attention mass",
    }

    strata_payload = tuple(
        {"target_tokens": target_tokens, "documents": count}
        for target_tokens, count in length_strata
    )
    source_payload = tuple(
        {
            "name": name,
            "target_fraction": fraction,
            "document_count": category_counts[name],
            "candidate_sources": tuple(sources),
        }
        for name, fraction, sources in SOURCE_MIX
    )
    return CorpusPlan(
        schema=SCHEMA,
        plan_name=plan_name,
        description=description,
        label_source=(
            "Do not train from HF task answers. Use long source documents only; "
            "derive labels from the Qwen teacher/oracle extraction pipeline."
        ),
        tokenization_contract=(
            "Each prompt_path must contain little-endian int32 token IDs from the "
            "same tokenizer/model fingerprint used by lsa_extract_qwen35."
        ),
        extraction_contract=extraction_contract,
        length_strata=strata_payload,
        source_mix=source_payload,
        hf_source_candidates=HF_SOURCE_CANDIDATES,
        documents=tuple(documents),
    )


def write_corpus_plan(path: Path, plan: CorpusPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    payload["summary"] = {
        "total_documents": plan.total_documents,
        "total_source_tokens": plan.total_source_tokens,
        "total_boundaries": plan.total_boundaries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_boundary_files(root: Path, plan: CorpusPlan) -> None:
    for document in plan.documents:
        path = root / document.boundary_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(f"{position}\n" for position in document.boundary_positions)
        )


def load_corpus_plan(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported corpus plan schema: {payload.get('schema')!r}")
    summary = payload.get("summary")
    documents = payload.get("documents")
    if not isinstance(summary, dict) or not isinstance(documents, list):
        raise ValueError("corpus plan is missing summary or documents")
    if summary.get("total_documents") != len(documents):
        raise ValueError("corpus plan document count is inconsistent")
    total_tokens = 0
    total_boundaries = 0
    for document in documents:
        boundaries = document.get("boundary_positions")
        target_tokens = int(document.get("target_tokens", 0))
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("corpus plan document has no boundaries")
        if any(position % BLOCK_SIZE != 0 for position in boundaries):
            raise ValueError("corpus plan contains non-aligned boundaries")
        if any(position + LOOKAHEAD_HORIZON > target_tokens for position in boundaries):
            raise ValueError("corpus plan boundary cannot fit lookahead horizon")
        total_tokens += target_tokens
        total_boundaries += len(boundaries)
    if summary.get("total_source_tokens") != total_tokens:
        raise ValueError("corpus plan token total is inconsistent")
    if summary.get("total_boundaries") != total_boundaries:
        raise ValueError("corpus plan boundary total is inconsistent")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", choices=("pilot", "scaled"), default="pilot")
    parser.add_argument(
        "--write-boundaries",
        action="store_true",
        help="also write extraction boundary files under the output directory",
    )
    args = parser.parse_args()

    plan = build_corpus_plan(args.plan)
    write_corpus_plan(args.output, plan)
    if args.write_boundaries:
        write_boundary_files(args.output.parent, plan)


if __name__ == "__main__":
    main()
