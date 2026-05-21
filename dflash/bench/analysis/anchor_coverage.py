"""
Anchor coverage analysis for PFlash.

Measures 2-gram, 4-gram, and 6-gram anchor hit distributions across bench
case files. Identifies the "anchor-zero corpus" — prompts where 4-gram
returns zero hits — which is the test bed for any future backstop mechanism.

Usage:
    python anchor_coverage.py --results-dir dflash/bench/results/2026-05-21_envelope

Outputs (all relative to the script directory):
    anchor_distribution.csv   — per-case row with all anchor metrics
    anchor_summary.md         — aggregate tables and cross-tab analysis
    anchor_zero_corpus.jsonl  — cases where 4-gram hits == 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Pure anchor functions — matched to C++ compute_anchor_hits() behavior.
#
# C++ behavior (qwen3_drafter.cpp):
#   - query window: last `query_tokens` token positions
#   - q0 = max(0, S - query_tokens)
#   - search_end = max(0, q0 - NGRAM)   (excludes overlap with query tail)
#   - For each q in [q0, S-NGRAM]: scan p in [0, search_end]
#   - Per-q: collect positions if matching; only appended if hits <= max_hits_per_q
#   - Returns total positions written (capped at max_hits_buf)
# ---------------------------------------------------------------------------

def compute_anchor_hits_pure(
    ids: list[int],
    n: int,
    query_tokens: int = 96,
    max_hits_per_q: int = 8,
    max_hits_buf: int = 16,
) -> list[tuple[int, int]]:
    """Return list of (q_pos, body_pos) n-gram matches.

    Pure Python equivalent of C++ compute_anchor_hits().
    Args:
        ids:           full token id sequence (body + query tail)
        n:             n-gram size (2, 4, or 6)
        query_tokens:  how many trailing tokens form the query window
        max_hits_per_q: max body matches per query n-gram (mirrors C++ local[16])
        max_hits_buf:  total max positions returned
    Returns:
        List of (q_pos, body_pos) tuples.
    """
    S = len(ids)
    q0 = max(0, S - query_tokens)
    search_end = max(0, q0 - n)
    result: list[tuple[int, int]] = []

    for q in range(q0, S - n + 1):  # P1a: no `total < max_hits_buf` outer guard
        hits = 0
        local: list[int] = []
        # Inner scan over body (p=0..search_end inclusive, matches C++)
        for p in range(0, search_end + 1):
            if p + n > S:
                break
            match = True
            for k in range(n):
                if ids[p + k] != ids[q + k]:
                    match = False
                    break
            if match:
                if hits < 16:  # local scratch buffer size from C++
                    local.append(p)
                hits += 1
        # Only commit if hits > 0 and within per-q cap
        if 0 < hits <= max_hits_per_q:
            for p in local:
                if len(result) >= max_hits_buf:
                    break
                result.append((q, p))

    return result


def count_anchor_hits(
    ids: list[int],
    n: int,
    query_tokens: int = 96,
) -> int:
    """Return the count of anchor hits (mirrors C++ return value)."""
    return len(compute_anchor_hits_pure(ids, n, query_tokens))


# ---------------------------------------------------------------------------
# Unit tests verifying Python behavior matches C++ on hand-crafted input.
# Run with --test flag.
# ---------------------------------------------------------------------------

def _run_unit_tests() -> None:
    print("Running unit tests...")

    # Test 1: exact repeat at start — 4-gram should find one hit.
    # ids = [1,2,3,4, filler..., 1,2,3,4]
    # body has [1,2,3,4] at pos 0; query tail has [1,2,3,4] at tail.
    # S=200, query_tokens=96, q0=104, search_end=max(0,104-4)=100
    # q=196 (last valid 4-gram start in [104..196]): p=0 matches ids[0..3]
    filler = list(range(10, 200))  # 190 distinct tokens; 4 + 186 + 4 = 194
    ids = [1, 2, 3, 4] + filler[:-4] + [1, 2, 3, 4]
    S = len(ids)
    assert S == 194, f"Expected 194, got {S}"
    hits = count_anchor_hits(ids, n=4, query_tokens=96)
    assert hits >= 1, f"Expected >=1 4-gram hit, got {hits}"
    print(f"  Test 1 PASS: repeat 4-gram, hits={hits}")

    # Test 2: no repeat — 4-gram should find 0 hits.
    ids_unique = list(range(200))
    hits_zero = count_anchor_hits(ids_unique, n=4, query_tokens=96)
    assert hits_zero == 0, f"Expected 0, got {hits_zero}"
    print(f"  Test 2 PASS: unique tokens, hits={hits_zero}")

    # Test 3: 2-gram finds more hits than 4-gram on a filler-heavy sequence.
    # Insert a 2-gram repeat but NOT a 4-gram repeat.
    ids3 = list(range(50, 250))  # 200 unique
    # Insert bigram [7, 8] twice: at pos 0 and near tail (in query window).
    ids3[0] = 7
    ids3[1] = 8
    ids3[150] = 7   # q pos within query window when S=200, q0=104
    ids3[151] = 8
    # Also break 4-gram: make pos 2,3 differ from 152,153
    ids3[2] = 99
    ids3[152] = 100
    hits2 = count_anchor_hits(ids3, n=2, query_tokens=96)
    hits4 = count_anchor_hits(ids3, n=4, query_tokens=96)
    assert hits2 >= 1, f"Expected >=1 2-gram hit, got {hits2}"
    print(f"  Test 3 PASS: 2-gram hits={hits2}, 4-gram hits={hits4}")

    # Test 4: search_end boundary — query n-gram should not match itself.
    # If q0=104 and NGRAM=4, search_end=100. A match at p=103 (overlaps query)
    # should NOT be found since p <= search_end=100.
    ids4 = list(range(200))
    ids4[103] = ids4[104]   # overlap region, should not be reached
    hits_boundary = count_anchor_hits(ids4, n=4, query_tokens=96)
    # Even if we set p=103 equal to q=104, search_end=100 so p=103 is excluded.
    # The value at p=103 should not matter.
    print(f"  Test 4 PASS: boundary exclusion works (hits={hits_boundary})")

    # Test 5: max_hits_buf cap — should never return more than max_hits_buf positions.
    # [1,2,3,4]*50 triggers hits>max_hits_per_q on every query n-gram so returns 0.
    # This validates the per-q over-count filter (same behavior as C++ local[]).
    ids5 = [1, 2, 3, 4] * 50
    hits5 = compute_anchor_hits_pure(ids5, n=4, query_tokens=96, max_hits_buf=16)
    assert len(hits5) <= 16, f"Expected <=16, got {len(hits5)}"
    # Sparse version: unique body + distinct single-hit repeats -> fills buffer.
    ids5b = list(range(200, 300))  # 100 unique
    # Place 20 distinct 4-grams in the query window that each match once in body.
    # query_tokens=20 -> q0=80; search_end=max(0,80-4)=76
    # We put matching grams at body[0..79] (one each) and in query at 80..99.
    for i in range(20):
        ids5b[i * 4] = 1000 + i
        ids5b[i * 4 + 1] = 2000 + i
        ids5b[i * 4 + 2] = 3000 + i
        ids5b[i * 4 + 3] = 4000 + i
        ids5b[80 + i] = 1000 + i  # only partial — not enough for 4-gram match here
    hits5b = compute_anchor_hits_pure(ids5b, n=4, query_tokens=96, max_hits_buf=16)
    assert len(hits5b) <= 16, f"Buffer exceeded: {len(hits5b)}"
    print(f"  Test 5 PASS: max_hits_buf cap respected (positions={len(hits5)}, sparse={len(hits5b)})")

    print("All unit tests PASSED.\n")


# ---------------------------------------------------------------------------
# TF-IDF weighting: downweight frequent n-grams in the body.
# ---------------------------------------------------------------------------

def compute_tfidf_weighted_hits(
    ids: list[int],
    n: int = 4,
    query_tokens: int = 96,
) -> float:
    """Sum of (1 / body_freq) for each matched body n-gram (TF-IDF proxy)."""
    S = len(ids)
    q0 = max(0, S - query_tokens)
    search_end = max(0, q0 - n)

    # Count frequency of each n-gram in the body (positions 0..search_end).
    body_ngram_freq: Counter[tuple[int, ...]] = Counter()
    for p in range(0, search_end + 1):
        if p + n > S:
            break
        gram = tuple(ids[p:p + n])
        body_ngram_freq[gram] += 1

    hits = compute_anchor_hits_pure(ids, n=n, query_tokens=query_tokens)
    weight = 0.0
    for q_pos, body_pos in hits:
        gram = tuple(ids[body_pos:body_pos + n])
        freq = body_ngram_freq.get(gram, 1)
        weight += 1.0 / freq

    return weight


# ---------------------------------------------------------------------------
# Case loading: from cases.jsonl (niah_single) or regenerated (vt/fwe/mqa).
# ---------------------------------------------------------------------------

def _load_tokenizer():
    """Load Qwen3 tokenizer (CPU only, no model)."""
    from transformers import AutoTokenizer
    # Qwen3-0.6B and Qwen3-27B share the same tokenizer vocabulary.
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    except Exception:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


# Inline minimal copies of gen_vt / gen_fwe / gen_mqa from ruler_diag3.py
# so this script is self-contained. Kept as close to the originals as possible.

_VT_CHAINS = 6
_VT_MAX_HOPS = 4
_FWE_VOCAB_SIZE = 200
_FWE_BODY_WORDS = 2000
_FWE_TOP_K = 3


def _rand_varname(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def _rand_literal(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=length))


def _count_tokens(text: str, tok) -> int:
    return len(tok.encode(text))


def _fit_filler(scaffold: str, target_tokens: int, tok) -> str:
    """Pad scaffold with filler sentences to hit target_tokens."""
    filler_unit = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    current = _count_tokens(scaffold, tok)
    if current >= target_tokens:
        return scaffold
    filler_needed = target_tokens - current
    filler_block = (filler_unit * ((filler_needed // _count_tokens(filler_unit, tok)) + 2))
    # Binary search to find the right filler length.
    lo, hi = 0, len(filler_block)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        candidate = filler_block[:mid] + scaffold
        if _count_tokens(candidate, tok) < target_tokens:
            lo = mid
        else:
            hi = mid
    prompt = filler_block[:hi] + scaffold
    return prompt


def _gen_vt(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    chains: list[list[str]] = []
    for _ in range(_VT_CHAINS):
        n_hops = rng.randint(2, _VT_MAX_HOPS)
        vars_ = [_rand_varname(rng) for _ in range(n_hops)]
        chains.append(vars_)

    literals = [_rand_literal(rng) for _ in range(_VT_CHAINS)]
    all_assigns: list[str] = []
    for ci, chain in enumerate(chains):
        prev = literals[ci]
        for vi, var in enumerate(chain):
            if vi == 0:
                all_assigns.append(f"{var} = {prev}")
            else:
                all_assigns.append(f"{var} = {chain[vi-1]}")

    rng.shuffle(all_assigns)
    assigns_text = "\n".join(all_assigns)
    query_var = chains[0][-1]
    answer = literals[0]
    question = f"What is the final value of {query_var}?"
    scaffold = f"{assigns_text}\n\nQuestion: {question}\nAnswer:"
    prompt = _fit_filler(scaffold, target_tokens, tok)
    return {"prompt": prompt, "answer": answer, "n_tokens": _count_tokens(prompt, tok)}


def _gen_fwe(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
             for _ in range(_FWE_VOCAB_SIZE)]
    seen: set[str] = set()
    uniq: list[str] = []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    extra_seed = seed + 100000
    while len(uniq) < _FWE_VOCAB_SIZE:
        candidate = "zz" + "".join(random.Random(extra_seed).choices(string.ascii_lowercase, k=6))
        extra_seed += 1
        if candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
    vocab = uniq[:_FWE_VOCAB_SIZE]

    from collections import Counter as _Counter
    weights_list = [1.0 / (i + 1) ** 1.2 for i in range(_FWE_VOCAB_SIZE)]
    total_w = sum(weights_list)
    weights_list = [w / total_w for w in weights_list]

    rng.shuffle(vocab)
    body_words = rng.choices(vocab, weights=weights_list, k=_FWE_BODY_WORDS)
    body = " ".join(body_words)

    counts = _Counter(body_words)
    top3 = {w for w, _ in counts.most_common(_FWE_TOP_K)}
    question = (
        "Based on the word list above, what are the 3 most frequently occurring words? "
        "List them separated by commas, nothing else."
    )
    scaffold = f"{body}\n\nQuestion: {question}\nAnswer:"
    prompt = _fit_filler(scaffold, target_tokens, tok)
    return {"prompt": prompt, "answer": sorted(top3), "n_tokens": _count_tokens(prompt, tok)}


def _gen_mqa(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    n_needles = 5
    keys = []
    while len(keys) < n_needles:
        k = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        if k not in keys:
            keys.append(k)
    values = ["".join(rng.choices(string.digits, k=7)) for _ in range(n_needles)]
    needles = [f"The value of {k} is {v}." for k, v in zip(keys, values)]
    question = f"What is the value of {keys[0]}?"
    answer = values[0]
    scaffold = " ".join(needles) + f"\n\nQuestion: {question}\nAnswer:"
    prompt = _fit_filler(scaffold, target_tokens, tok)
    return {"prompt": prompt, "answer": answer, "n_tokens": _count_tokens(prompt, tok)}


_GEN_FNS = {
    "vt": _gen_vt,
    "fwe": _gen_fwe,
    "mqa": _gen_mqa,
}


class CaseRecord(NamedTuple):
    case_idx: int
    task: str
    ctx_bucket: int
    prompt_len_tokens: int
    anchor_4gram_hits: int
    anchor_2gram_hits: int
    anchor_6gram_hits: int
    tf_idf_weighted_4gram: float
    answer: object
    prompt_preview: str


def _parse_dir_name(name: str) -> tuple[str, int] | None:
    """Parse 'task_ctx_keep_mode' directory name into (task, ctx)."""
    parts = name.split("_")
    # Find the integer ctx part (could be at various positions)
    # Format: task_ctx_keep_mode  e.g. niah_single_4096_0.1_off, vt_4096_0.1_off
    ctx = None
    task_parts = []
    for p in parts:
        try:
            ctx = int(p)
            break
        except ValueError:
            task_parts.append(p)
    if ctx is None:
        return None
    task = "_".join(task_parts)
    return task, ctx


def load_cases_from_dir(
    result_dir: Path,
    tok,
) -> list[tuple[int, str, int, list[int], object]]:
    """Load (case_idx, task, ctx, token_ids, answer) from a result directory.

    For niah_single: reads cases.jsonl (has prompt text).
    For vt/fwe/mqa: regenerates prompts from seed stored in raw.json.
    """
    parsed = _parse_dir_name(result_dir.name)
    if parsed is None:
        return []
    task, ctx = parsed

    cases_jsonl = result_dir / "cases.jsonl"
    raw_files = sorted(result_dir.glob("case_*.raw.json"))
    if not raw_files:
        return []

    out = []

    if cases_jsonl.exists():
        # niah_single path: prompt text is in cases.jsonl
        with open(cases_jsonl) as f:
            jsonl_cases = [json.loads(line) for line in f if line.strip()]

        for i, case in enumerate(jsonl_cases):
            prompt_text = case.get("prompt", "")
            answer = case.get("answer", "")
            if not prompt_text:
                continue
            ids = tok.encode(prompt_text)
            out.append((i, task, ctx, ids, answer))
    else:
        # vt/fwe/mqa path: regenerate from seed stored in raw.json
        gen_fn = _GEN_FNS.get(task)
        if gen_fn is None:
            print(f"  [warn] No gen function for task={task}, skipping {result_dir.name}")
            return []

        for rf in raw_files:
            with open(rf) as f:
                raw = json.load(f)
            case_idx = raw.get("case_idx", 0)
            seed = raw.get("seed")
            if seed is None:
                print(f"  [warn] No seed in {rf}, skipping")
                continue
            answer = raw.get("answer", "")
            try:
                case = gen_fn(seed, ctx, tok)
            except Exception as e:
                print(f"  [warn] gen failed for {rf}: {e}")
                continue
            ids = tok.encode(case["prompt"])
            out.append((case_idx, task, ctx, ids, answer))

    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tok = _load_tokenizer()

    # Discover all result subdirectories
    all_subdirs = [d for d in sorted(results_dir.iterdir())
                   if d.is_dir() and _parse_dir_name(d.name) is not None]

    print(f"Found {len(all_subdirs)} candidate directories", flush=True)

    records: list[CaseRecord] = []
    zero_corpus: list[dict] = []

    for subdir in all_subdirs:
        print(f"  Processing {subdir.name}...", flush=True)
        cases = load_cases_from_dir(subdir, tok)
        if not cases:
            continue

        for case_idx, task, ctx, ids, answer in cases:
            n4 = count_anchor_hits(ids, n=4)
            n2 = count_anchor_hits(ids, n=2)
            n6 = count_anchor_hits(ids, n=6)
            tfidf = compute_tfidf_weighted_hits(ids, n=4)
            prompt_text = tok.decode(ids)
            preview = prompt_text[:200].replace("\n", " ")

            rec = CaseRecord(
                case_idx=case_idx,
                task=task,
                ctx_bucket=ctx,
                prompt_len_tokens=len(ids),
                anchor_4gram_hits=n4,
                anchor_2gram_hits=n2,
                anchor_6gram_hits=n6,
                tf_idf_weighted_4gram=round(tfidf, 4),
                answer=answer,
                prompt_preview=preview,
            )
            records.append(rec)

            if n4 == 0:
                zero_corpus.append({
                    "case_idx": case_idx,
                    "task": task,
                    "ctx": ctx,
                    "prompt_preview": preview,
                    "ground_truth_answer": answer,
                    "anchor_2gram_hits": n2,
                    "anchor_6gram_hits": n6,
                })

    print(f"\nAnalyzed {len(records)} cases total.", flush=True)

    # Write CSV
    csv_path = out_dir / "anchor_distribution.csv"
    fieldnames = list(CaseRecord._fields)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r._asdict())
    print(f"Wrote {csv_path}", flush=True)

    # Write anchor-zero corpus
    zero_path = out_dir / "anchor_zero_corpus.jsonl"
    with open(zero_path, "w") as f:
        for item in zero_corpus:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {zero_path} ({len(zero_corpus)} cases)", flush=True)

    # Generate summary markdown
    summary = _build_summary(records, zero_corpus, tok)
    summary_path = out_dir / "anchor_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Wrote {summary_path}", flush=True)


def _build_summary(
    records: list[CaseRecord],
    zero_corpus: list[dict],
    tok,
) -> str:
    total = len(records)
    if total == 0:
        return "# Anchor Coverage Summary\n\nNo cases analyzed.\n"

    lines = ["# Anchor Coverage Summary\n"]
    lines.append(f"Total cases analyzed: **{total}**\n")

    # Overall zero rates
    n_zero_4g = sum(1 for r in records if r.anchor_4gram_hits == 0)
    n_zero_2g = sum(1 for r in records if r.anchor_2gram_hits == 0)
    n_zero_6g = sum(1 for r in records if r.anchor_6gram_hits == 0)
    pct4 = 100.0 * n_zero_4g / total
    pct2 = 100.0 * n_zero_2g / total
    pct6 = 100.0 * n_zero_6g / total

    lines.append("## Overall Zero-Hit Rates\n")
    lines.append(f"| N-gram | Zero-hit cases | % of total |")
    lines.append(f"|--------|---------------|------------|")
    lines.append(f"| 2-gram | {n_zero_2g} | {pct2:.1f}% |")
    lines.append(f"| 4-gram | {n_zero_4g} | {pct4:.1f}% |")
    lines.append(f"| 6-gram | {n_zero_6g} | {pct6:.1f}% |")
    lines.append("")

    # Per-task x ctx breakdown
    lines.append("## 4-Gram Zero-Hit Rate by Task x Context\n")
    lines.append("| Task | Ctx | Total | Zero-hit | Zero% |")
    lines.append("|------|-----|-------|----------|-------|")

    cell_data: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in records:
        cell_data[(r.task, r.ctx_bucket)].append(r.anchor_4gram_hits)

    hottest_rate = 0.0
    hot_row = ""
    for (task, ctx), hits_list in sorted(cell_data.items()):
        n = len(hits_list)
        z = sum(1 for h in hits_list if h == 0)
        rate = 100.0 * z / n
        row = f"| {task} | {ctx} | {n} | {z} | {rate:.1f}% |"
        lines.append(row)
        if rate > hottest_rate:
            hottest_rate = rate
            hot_row = row
    lines.append("")

    lines.append(f"**Hottest row (highest anchor-zero rate):** `{hot_row}`\n")

    # Hit count histograms
    lines.append("## Hit Count Distribution\n")
    for ngram in (2, 4, 6):
        attr = f"anchor_{ngram}gram_hits"
        counts = Counter(getattr(r, attr) for r in records)
        lines.append(f"### {ngram}-gram hit distribution\n")
        lines.append("| Hits | Cases |")
        lines.append("|------|-------|")
        for k in sorted(counts.keys()):
            lines.append(f"| {k} | {counts[k]} |")
        lines.append("")

    # Multi-resolution rescue cross-tab
    rescued_2g = sum(
        1 for r in records
        if r.anchor_4gram_hits == 0 and r.anchor_2gram_hits > 0
    )
    rescued_6g = sum(
        1 for r in records
        if r.anchor_4gram_hits == 0 and r.anchor_6gram_hits > 0
    )
    zero4 = max(1, n_zero_4g)  # avoid div-by-zero

    lines.append("## Multi-Resolution Rescue Analysis\n")
    lines.append("Cases where 4-gram=0 but another n-gram finds hits:\n")
    lines.append("| Rescue source | Cases rescued | % of 4-gram-zero |")
    lines.append("|---------------|--------------|-----------------|")
    lines.append(f"| 2-gram | {rescued_2g} | {100.0*rescued_2g/zero4:.1f}% |")
    lines.append(f"| 6-gram | {rescued_6g} | {100.0*rescued_6g/zero4:.1f}% |")
    lines.append("")

    # TF-IDF top-3 n-grams for 10 random cases
    lines.append("## TF-IDF Weighted Top-3 Anchor 4-Grams (10 Random Cases)\n")
    sample_records = records[:] if len(records) <= 10 else random.sample(records, 10)
    for r in sample_records:
        if r.anchor_4gram_hits == 0:
            continue
        # Recompute to get actual n-gram text for display
        try:
            ids = tok.encode(r.prompt_preview)
            hits = compute_anchor_hits_pure(ids, n=4)
            # TF-IDF: compute body freq
            S = len(ids)
            q0 = max(0, S - 96)
            search_end = max(0, q0 - 4)
            body_freq: Counter[tuple[int, ...]] = Counter()
            for p in range(search_end + 1):
                gram = tuple(ids[p:p + 4])
                body_freq[gram] += 1
            scored = []
            for q_pos, body_pos in hits:
                gram = tuple(ids[body_pos:body_pos + 4])
                scored.append((1.0 / body_freq.get(gram, 1), gram))
            scored.sort(reverse=True)
            top3 = scored[:3]
            decoded = [tok.decode(list(g)) for _, g in top3]
            lines.append(f"- case {r.case_idx} ({r.task}/{r.ctx_bucket}): `{decoded}`")
        except Exception:
            pass
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor coverage analysis for PFlash bench cases.")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "2026-05-21_envelope",
        help="Directory containing result subdirectories.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory (default: same dir as this script).",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests only, then exit.",
    )
    args = ap.parse_args()

    if args.test:
        _run_unit_tests()
        sys.exit(0)

    if not args.results_dir.exists():
        print(f"Error: results-dir does not exist: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    analyze(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
