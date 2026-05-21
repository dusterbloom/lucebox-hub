"""
H2 Multi-Resolution Anchor Hypothesis Validator.

Tests whether independent 2-gram + 4-gram + 6-gram hit buckets (16 entries each,
48 total) measurably improve anchor recall on anchor-zero cases without degrading
keep-set precision.

Momus pre-flight constraints implemented:
  1. Independent hit-buckets per n-gram size (16 per resolution, never shared).
  2. Corpus priority: real captured prompts first, then synthetic bench cases.
  3. Precision metric: high-precision-chunk fraction (body-local ngram freq <= 3).

Usage:
    python multires_anchor.py [--session-dir PATH] [--results-dir PATH] [--test]

Outputs (in same directory as this script):
    multires_results.csv    -- per-case row with all metrics + corpus tag
    multires_summary.md     -- aggregate cross-tab and go/no-go verdict
    h2_verdict.txt          -- PROCEED / KILL / INCONCLUSIVE + one-sentence reason
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Pure anchor scan — verbatim port of C++ compute_anchor_hits() in
# dflash/src/qwen3/qwen3_drafter.cpp, parameterized by ngram_size.
#
# C++ code (lines 527-548):
#   const int q0 = max(0, S - query_tokens);
#   constexpr int NGRAM = 4;
#   for (int q = q0; q + NGRAM <= S; ++q) {
#       int hits = 0; int hit_pos[8];
#       const int search_end = max(0, q0 - NGRAM);
#       for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
#           bool same = true;
#           for (int k = 0; k < NGRAM; ++k) { ... }
#           if (same) { if (hits < 8) hit_pos[hits] = p; ++hits; }
#       }
#       if (hits > 0 && hits <= max_anchor_hits) {
#           for (int i = 0; i < hits && i < 8; ++i) {
#               force_chunk_neighborhood(hit_pos[i] / chunk_size, ...)
#           }
#       }
#   }
# ---------------------------------------------------------------------------

_MAX_HITS_PER_Q = 8   # mirrors C++ `max_anchor_hits` default and `hit_pos[8]`
_HIT_BUF_SIZE = 16    # 16 entries per resolution bucket (Momus constraint #1)
_QUERY_TOKENS = 96    # mirrors DFLASH_COMPRESS_QUERY_TOKENS default


def scan_anchor_hits(
    ids: list[int],
    ngram_size: int,
    query_tokens: int = _QUERY_TOKENS,
    max_hits_per_q: int = _MAX_HITS_PER_Q,
    hit_buf_size: int = _HIT_BUF_SIZE,
) -> list[int]:
    """Return list of body positions that would be forced (up to hit_buf_size).

    Pure Python verbatim port of C++ compute_anchor_hits(), parameterized by
    ngram_size. This is a PURE function for testability.

    Args:
        ids:            full token sequence (body + query tail concatenated)
        ngram_size:     n in {2, 4, 6}
        query_tokens:   trailing tokens forming the query window
        max_hits_per_q: per-query-gram hit cap (mirrors C++ local[8])
        hit_buf_size:   total result buffer size per resolution bucket
    Returns:
        List of body token positions that would force chunk neighborhoods.
    """
    S = len(ids)
    if S < ngram_size:
        return []
    q0 = max(0, S - query_tokens)
    search_end = max(0, q0 - ngram_size)
    result: list[int] = []

    for q in range(q0, S - ngram_size + 1):
        hits = 0
        local: list[int] = []
        for p in range(0, search_end + 1):
            # C++ inner condition: `p <= search_end && hits <= max_anchor_hits`
            if hits > max_hits_per_q:
                break
            if p + ngram_size > S:
                break
            match = True
            for k in range(ngram_size):
                if ids[p + k] != ids[q + k]:
                    match = False
                    break
            if match:
                if hits < 8:  # C++ hit_pos[8] local buffer
                    local.append(p)
                hits += 1
        if 0 < hits <= max_hits_per_q:
            for p in local:
                if len(result) >= hit_buf_size:
                    break
                result.append(p)

    return result


def count_anchor_hits(
    ids: list[int],
    ngram_size: int,
    query_tokens: int = _QUERY_TOKENS,
) -> int:
    """Return total hit count (number of body positions forced)."""
    return len(scan_anchor_hits(ids, ngram_size, query_tokens))


def compute_forced_chunks(
    hit_positions: list[int],
    n_chunks: int,
    chunk_size: int = 512,
    radius: int = 2,
) -> set[int]:
    """Convert body hit positions to forced chunk indices (with neighborhood)."""
    forced: set[int] = set()
    for pos in hit_positions:
        center = pos // chunk_size
        lo = max(0, center - radius)
        hi = min(n_chunks - 1, center + radius)
        for c in range(lo, hi + 1):
            forced.add(c)
    return forced


def compute_multires_union(
    ids: list[int],
    chunk_size: int = 512,
    radius: int = 2,
    query_tokens: int = _QUERY_TOKENS,
) -> tuple[set[int], set[int], set[int], set[int]]:
    """Compute independent per-resolution hit buckets and their union.

    Momus constraint #1: independent 16-entry buckets, never shared.

    Returns:
        (forced_2gram, forced_4gram, forced_6gram, forced_union)
    """
    S = len(ids)
    n_chunks = max(1, (S + chunk_size - 1) // chunk_size)

    hits2 = scan_anchor_hits(ids, 2, query_tokens, hit_buf_size=_HIT_BUF_SIZE)
    hits4 = scan_anchor_hits(ids, 4, query_tokens, hit_buf_size=_HIT_BUF_SIZE)
    hits6 = scan_anchor_hits(ids, 6, query_tokens, hit_buf_size=_HIT_BUF_SIZE)

    f2 = compute_forced_chunks(hits2, n_chunks, chunk_size, radius)
    f4 = compute_forced_chunks(hits4, n_chunks, chunk_size, radius)
    f6 = compute_forced_chunks(hits6, n_chunks, chunk_size, radius)
    fu = f2 | f4 | f6

    return f2, f4, f6, fu


def compute_precision_proxy(
    ids: list[int],
    hit_positions: list[int],
    ngram_size: int,
    low_freq_threshold: int = 3,
    high_freq_threshold: int = 20,
    query_tokens: int = _QUERY_TOKENS,
) -> tuple[float, float]:
    """Compute high-precision and low-precision fraction of forced chunks.

    A hit is "high precision" if the n-gram forcing it appears <= low_freq_threshold
    times in the body (rare term). It is "low precision" if it appears
    >= high_freq_threshold times (stop-word territory).

    Returns:
        (high_precision_frac, low_precision_frac) — fractions over all hits.
        Returns (0.0, 0.0) if no hits.
    """
    if not hit_positions:
        return 0.0, 0.0

    S = len(ids)
    q0 = max(0, S - query_tokens)
    search_end = max(0, q0 - ngram_size)

    # Count frequency of each n-gram in the body region
    body_freq: Counter[tuple[int, ...]] = Counter()
    for p in range(0, search_end + 1):
        if p + ngram_size > S:
            break
        gram = tuple(ids[p : p + ngram_size])
        body_freq[gram] += 1

    high_prec = 0
    low_prec = 0
    for pos in hit_positions:
        gram = tuple(ids[pos : pos + ngram_size])
        freq = body_freq.get(gram, 1)
        if freq <= low_freq_threshold:
            high_prec += 1
        elif freq >= high_freq_threshold:
            low_prec += 1

    total = len(hit_positions)
    return high_prec / total, low_prec / total


# ---------------------------------------------------------------------------
# Unit tests — assert Python matches C++ behavior for ngram_size=4.
# Addresses pure-functions-testable.md rule.
# ---------------------------------------------------------------------------

def _run_unit_tests() -> None:
    print("Running unit tests for multires_anchor.py (ngram_size=4 C++ parity)...")

    # T1: single 4-gram repeat — Python must find >=1 hit.
    body = [1, 2, 3, 4] + list(range(10, 200))  # 194 tokens
    tail = list(range(200, 292)) + [1, 2, 3, 4]  # 96 tokens ending with gram
    ids = body + tail
    hits4 = count_anchor_hits(ids, 4)
    assert hits4 >= 1, f"T1 FAIL: expected >=1 4-gram hit, got {hits4}"
    print(f"  T1 PASS: single 4-gram repeat, hits={hits4}")

    # T2: all unique tokens — 4-gram returns 0.
    ids_unique = list(range(300))
    hits_zero = count_anchor_hits(ids_unique, 4)
    assert hits_zero == 0, f"T2 FAIL: expected 0, got {hits_zero}"
    print(f"  T2 PASS: all unique tokens, hits={hits_zero}")

    # T3: 2-gram hits when 4-gram does not.
    ids3 = list(range(50, 350))
    ids3[0] = 7; ids3[1] = 8
    q0 = max(0, 300 - 96)  # 204
    ids3[q0] = 7; ids3[q0 + 1] = 8
    ids3[q0 + 2] = 999; ids3[q0 + 3] = 998  # breaks 4-gram
    hits2 = count_anchor_hits(ids3, 2)
    hits4_t3 = count_anchor_hits(ids3, 4)
    assert hits2 >= 1, f"T3 FAIL: 2-gram expected >=1, got {hits2}"
    assert hits4_t3 == 0, f"T3 FAIL: 4-gram expected 0, got {hits4_t3}"
    print(f"  T3 PASS: 2-gram rescues (hits2={hits2}, hits4={hits4_t3})")

    # T4: search_end boundary — body positions beyond search_end excluded.
    ids4 = list(range(200))
    q0_t4 = max(0, 200 - 96)  # 104
    search_end_t4 = max(0, q0_t4 - 4)  # 100
    assert search_end_t4 == 100, f"T4 setup: search_end={search_end_t4}"
    # Plant match at p=101 (just past search_end=100) — should NOT be found.
    for k in range(4):
        ids4[101 + k] = ids4[104 + k]
    raw4 = scan_anchor_hits(ids4, 4)
    # Verify no hit at p=101 (beyond search_end).
    assert 101 not in raw4, f"T4 FAIL: p=101 found past search_end={search_end_t4}"
    print(f"  T4 PASS: boundary exclusion (search_end={search_end_t4})")

    # T5: over-represented 4-gram filtered (hits > max_hits_per_q => 0 forced).
    ids5 = [1, 2, 3, 4] * 50  # 200 tokens, every gram repeats >> 8 times
    hits5 = count_anchor_hits(ids5, 4)
    assert hits5 == 0, f"T5 FAIL: over-rep gram should give 0, got {hits5}"
    print(f"  T5 PASS: over-represented 4-gram filtered (hits={hits5})")

    # T6: independent hit buckets — 2-gram and 4-gram results are separate.
    f2, f4, f6, fu = compute_multires_union(ids + ids_unique, chunk_size=512)
    # Both are set types.
    assert isinstance(f2, set) and isinstance(f4, set), "T6 FAIL: not sets"
    assert fu == f2 | f4 | f6, "T6 FAIL: union not equal to OR of all"
    print(f"  T6 PASS: independent buckets, union correct")

    # T7: precision proxy — rare n-gram scores high precision.
    # Build: body has [100,101,102,103] once; tail repeats it.
    body7 = [100, 101, 102, 103] + list(range(200, 390))  # 194 tokens, gram appears 1x in body
    tail7 = list(range(400, 490)) + [100, 101, 102, 103]  # 96 tokens
    ids7 = body7 + tail7
    raw7 = scan_anchor_hits(ids7, 4)
    assert raw7, "T7 setup FAIL: no hits"
    hp, lp = compute_precision_proxy(ids7, raw7, 4)
    assert hp > 0.0, f"T7 FAIL: rare gram should have high precision, hp={hp}"
    print(f"  T7 PASS: rare 4-gram high precision (hp={hp:.2f}, lp={lp:.2f})")

    print("\nAll 7 unit tests PASSED.\n")


# ---------------------------------------------------------------------------
# Real captured corpus extraction from JSONL session files.
# ---------------------------------------------------------------------------

_PATH_REDACT = re.compile(r'/home/[^/\s]+')


def _redact(text: str) -> str:
    return _PATH_REDACT.sub('/home/<user>', text)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        tp = item.get('type', '')
        if tp == 'text':
            parts.append(item.get('text', ''))
        elif tp == 'thinking':
            t = item.get('thinking', '')
            if t:
                parts.append(f'<thinking>{t[:300]}</thinking>')
        elif tp == 'tool_use':
            name = item.get('name', '?')
            inp = item.get('input', {})
            cmd = ''
            if isinstance(inp, dict):
                cmd = inp.get('command', inp.get('file_path', inp.get('query', '')))
            parts.append(f'[tool:{name} {str(cmd)[:100]}]')
        elif tp == 'tool_result':
            c = item.get('content', '')
            if isinstance(c, str):
                parts.append(f'[result:{c[:200]}]')
            elif isinstance(c, list):
                sub = ' '.join(
                    x.get('text', '') if isinstance(x, dict) else str(x)
                    for x in c[:3]
                )
                parts.append(f'[result:{sub[:200]}]')
    return '\n'.join(parts)


def extract_real_pairs_from_jsonl(
    jsonl_path: Path,
    max_pairs: int = 10,
    min_body_chars: int = 500,
) -> list[tuple[str, str]]:
    """Extract (query, body) pairs from a Claude Code session JSONL.

    For turn N: body = all prior turns concatenated, query = user message at N.
    Skips pairs where body is too short.

    Returns list of (query_text, body_text).
    """
    pairs: list[tuple[str, str]] = []
    try:
        with open(jsonl_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    except (OSError, json.JSONDecodeError):
        return []

    turns: list[tuple[str, str]] = []  # (role, text)
    for obj in lines:
        tp = obj.get('type', '')
        if tp in ('user', 'assistant'):
            msg = obj.get('message', obj)
            role = msg.get('role', tp)
            content = msg.get('content', '')
            text = _content_to_text(content)
            if text.strip():
                turns.append((role, text))

    # Build (query, body) pairs: at each user turn, body = all prior turns.
    body_parts: list[str] = []
    for role, text in turns:
        if role == 'user' and body_parts:
            body = '\n'.join(body_parts)
            if len(body) >= min_body_chars:
                pairs.append((_redact(text), _redact(body)))
                if len(pairs) >= max_pairs:
                    break
        body_parts.append(text)

    return pairs


def load_real_corpus(
    session_dirs: list[Path],
    target_pairs: int = 30,
    max_body_tokens: int = 32000,
) -> list[tuple[str, str, str]]:
    """Sample up to target_pairs (query, body, source_label) from session dirs.

    source_label identifies which session file the pair came from.
    """
    all_jsonl: list[Path] = []
    for d in session_dirs:
        if d.is_file() and d.suffix == '.jsonl':
            all_jsonl.append(d)
        elif d.is_dir():
            all_jsonl.extend(sorted(d.rglob('*.jsonl')))

    # Prioritize larger files (more conversation turns)
    all_jsonl.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)

    out: list[tuple[str, str, str]] = []
    for jpath in all_jsonl:
        if len(out) >= target_pairs:
            break
        needed = target_pairs - len(out)
        pairs = extract_real_pairs_from_jsonl(jpath, max_pairs=needed)
        label = f'real_{jpath.parent.name[:20]}'
        for q, b in pairs:
            out.append((q, b, label))

    return out


# ---------------------------------------------------------------------------
# Synthetic NIAH/VT/FWE case loading — reuses anchor_coverage.py generators.
# ---------------------------------------------------------------------------

_VT_CHAINS = 6
_VT_MAX_HOPS = 4
_FWE_VOCAB_SIZE = 200
_FWE_BODY_WORDS = 2000
_FWE_TOP_K = 3


def _rand_varname(rng: random.Random, length: int = 4) -> str:
    return ''.join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def _rand_literal(rng: random.Random, length: int = 6) -> str:
    return ''.join(rng.choices(string.ascii_uppercase, k=length))


def _count_tokens(text: str, tok) -> int:
    return len(tok.encode(text))


def _fit_filler(scaffold: str, target_tokens: int, tok) -> str:
    filler_unit = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. '
    current = _count_tokens(scaffold, tok)
    if current >= target_tokens:
        return scaffold
    filler_needed = target_tokens - current
    filler_block = filler_unit * ((filler_needed // _count_tokens(filler_unit, tok)) + 2)
    lo, hi = 0, len(filler_block)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        candidate = filler_block[:mid] + scaffold
        if _count_tokens(candidate, tok) < target_tokens:
            lo = mid
        else:
            hi = mid
    return filler_block[:hi] + scaffold


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
                all_assigns.append(f'{var} = {prev}')
            else:
                all_assigns.append(f'{var} = {chain[vi-1]}')
    rng.shuffle(all_assigns)
    assigns_text = '\n'.join(all_assigns)
    query_var = chains[0][-1]
    question = f'What is the final value of {query_var}?'
    answer = literals[0]
    scaffold = f'{assigns_text}\n\nQuestion: {question}\nAnswer:'
    prompt = _fit_filler(scaffold, target_tokens, tok)
    return {'prompt': prompt, 'answer': answer}


def _gen_fwe(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    vocab = [''.join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
             for _ in range(_FWE_VOCAB_SIZE)]
    seen: set[str] = set()
    uniq: list[str] = []
    for w in vocab:
        if w not in seen:
            seen.add(w); uniq.append(w)
    extra_seed = seed + 100000
    while len(uniq) < _FWE_VOCAB_SIZE:
        candidate = 'zz' + ''.join(random.Random(extra_seed).choices(string.ascii_lowercase, k=6))
        extra_seed += 1
        if candidate not in seen:
            seen.add(candidate); uniq.append(candidate)
    vocab = uniq[:_FWE_VOCAB_SIZE]
    weights_list = [1.0 / (i + 1) ** 1.2 for i in range(_FWE_VOCAB_SIZE)]
    total_w = sum(weights_list)
    weights_list = [w / total_w for w in weights_list]
    rng.shuffle(vocab)
    body_words = rng.choices(vocab, weights=weights_list, k=_FWE_BODY_WORDS)
    body = ' '.join(body_words)
    top3 = {w for w, _ in Counter(body_words).most_common(_FWE_TOP_K)}
    question = ('Based on the word list above, what are the 3 most frequently '
                'occurring words? List them separated by commas, nothing else.')
    scaffold = f'{body}\n\nQuestion: {question}\nAnswer:'
    prompt = _fit_filler(scaffold, target_tokens, tok)
    return {'prompt': prompt, 'answer': sorted(top3)}


def _gen_mqa(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    n_needles = 5
    keys = []
    while len(keys) < n_needles:
        k = ''.join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        if k not in keys:
            keys.append(k)
    values = [''.join(rng.choices(string.digits, k=7)) for _ in range(n_needles)]
    needles = [f'The value of {k} is {v}.' for k, v in zip(keys, values)]
    question = f'What is the value of {keys[0]}?'
    answer = values[0]
    scaffold = ' '.join(needles) + f'\n\nQuestion: {question}\nAnswer:'
    prompt = _fit_filler(scaffold, target_tokens, tok)
    return {'prompt': prompt, 'answer': answer}


_GEN_FNS = {'vt': _gen_vt, 'fwe': _gen_fwe, 'mqa': _gen_mqa}


def _parse_dir_name(name: str) -> tuple[str, int] | None:
    parts = name.split('_')
    ctx = None
    task_parts = []
    for p in parts:
        try:
            ctx = int(p); break
        except ValueError:
            task_parts.append(p)
    if ctx is None:
        return None
    return '_'.join(task_parts), ctx


def load_synthetic_cases(
    results_dir: Path,
    tok,
    max_cases: int = 115,
) -> list[tuple[str, str, list[int], object]]:
    """Load synthetic cases from bench results directory.

    Returns list of (corpus_tag, case_label, token_ids, answer).
    """
    out: list[tuple[str, str, list[int], object]] = []
    if not results_dir.exists():
        return out

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = _parse_dir_name(subdir.name)
        if parsed is None:
            continue
        task, ctx = parsed

        cases_jsonl = subdir / 'cases.jsonl'
        raw_files = sorted(subdir.glob('case_*.raw.json'))
        if not raw_files:
            continue

        if cases_jsonl.exists():
            with open(cases_jsonl) as f:
                jsonl_cases = [json.loads(line) for line in f if line.strip()]
            for i, case in enumerate(jsonl_cases):
                if len(out) >= max_cases:
                    break
                prompt_text = case.get('prompt', '')
                answer = case.get('answer', '')
                if not prompt_text:
                    continue
                ids = tok.encode(prompt_text)
                corpus_tag = f'synthetic_{task}'
                label = f'{task}/{ctx}/case{i}'
                out.append((corpus_tag, label, ids, answer))
        else:
            gen_fn = _GEN_FNS.get(task)
            if gen_fn is None:
                continue
            for rf in raw_files:
                if len(out) >= max_cases:
                    break
                try:
                    with open(rf) as f:
                        raw = json.load(f)
                    seed = raw.get('seed')
                    if seed is None:
                        continue
                    case = gen_fn(seed, ctx, tok)
                    ids = tok.encode(case['prompt'])
                    corpus_tag = f'synthetic_{task}'
                    label = f'{task}/{ctx}/{rf.stem}'
                    out.append((corpus_tag, label, ids, raw.get('answer', '')))
                except Exception as e:
                    print(f'  [warn] {rf}: {e}', file=sys.stderr)

    return out


# ---------------------------------------------------------------------------
# Per-case metric computation
# ---------------------------------------------------------------------------

class CaseMetrics(NamedTuple):
    corpus_tag: str
    case_label: str
    n_tokens: int
    hits_4gram: int
    hits_2gram_only: int
    hits_6gram_only: int
    hits_multi_union_chunks: int
    forced_chunks_4gram: int
    forced_chunks_2gram: int
    forced_chunks_6gram: int
    high_prec_frac_4gram: float
    high_prec_frac_2gram: float
    high_prec_frac_6gram: float
    needle_chunk_in_4gram: int   # 1/0/-1 (yes/no/unknown)
    needle_chunk_in_union: int   # 1/0/-1
    needle_chunk_id: int         # -1 if unknown


def compute_metrics(
    ids: list[int],
    corpus_tag: str,
    case_label: str,
    answer: object = None,
    needle_text: str = '',
    tok = None,
    chunk_size: int = 512,
    radius: int = 2,
) -> CaseMetrics:
    """Compute all six H2 metrics for one (query, body) pair."""
    S = len(ids)
    n_chunks = max(1, (S + chunk_size - 1) // chunk_size)

    # Raw hit counts per resolution (independent buckets)
    raw2 = scan_anchor_hits(ids, 2, hit_buf_size=_HIT_BUF_SIZE)
    raw4 = scan_anchor_hits(ids, 4, hit_buf_size=_HIT_BUF_SIZE)
    raw6 = scan_anchor_hits(ids, 6, hit_buf_size=_HIT_BUF_SIZE)

    hits_4gram = len(raw4)
    hits_2gram_only = len(raw2)
    hits_6gram_only = len(raw6)

    # Forced chunks per resolution
    f2 = compute_forced_chunks(raw2, n_chunks, chunk_size, radius)
    f4 = compute_forced_chunks(raw4, n_chunks, chunk_size, radius)
    f6 = compute_forced_chunks(raw6, n_chunks, chunk_size, radius)
    fu = f2 | f4 | f6

    # Precision proxy per resolution
    hp2, _ = compute_precision_proxy(ids, raw2, 2)
    hp4, _ = compute_precision_proxy(ids, raw4, 4)
    hp6, _ = compute_precision_proxy(ids, raw6, 6)

    # Needle recall: only for NIAH synthetic cases where answer is the needle.
    needle_chunk_id = -1
    needle_in_4gram = -1
    needle_in_union = -1
    if tok is not None and needle_text and isinstance(answer, str) and answer:
        needle_ids = tok.encode(answer)
        # Find needle chunk: search body (exclude query tail) for answer token sequence
        q0 = max(0, S - _QUERY_TOKENS)
        for p in range(0, max(0, q0 - len(needle_ids)) + 1):
            match = all(ids[p + k] == needle_ids[k] for k in range(len(needle_ids)))
            if match:
                needle_chunk_id = p // chunk_size
                break
        if needle_chunk_id >= 0:
            needle_in_4gram = 1 if needle_chunk_id in f4 else 0
            needle_in_union = 1 if needle_chunk_id in fu else 0

    return CaseMetrics(
        corpus_tag=corpus_tag,
        case_label=case_label,
        n_tokens=S,
        hits_4gram=hits_4gram,
        hits_2gram_only=hits_2gram_only,
        hits_6gram_only=hits_6gram_only,
        hits_multi_union_chunks=len(fu),
        forced_chunks_4gram=len(f4),
        forced_chunks_2gram=len(f2),
        forced_chunks_6gram=len(f6),
        high_prec_frac_4gram=round(hp4, 4),
        high_prec_frac_2gram=round(hp2, 4),
        high_prec_frac_6gram=round(hp6, 4),
        needle_chunk_in_4gram=needle_in_4gram,
        needle_chunk_in_union=needle_in_union,
        needle_chunk_id=needle_chunk_id,
    )


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze(
    session_dirs: list[Path],
    results_dir: Path,
    out_dir: Path,
    tok,
    target_real_pairs: int = 30,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[CaseMetrics] = []

    # --- Real captured corpus ---
    print(f'Loading real captured prompts from {len(session_dirs)} session source(s)...', flush=True)
    real_pairs = load_real_corpus(session_dirs, target_pairs=target_real_pairs)
    print(f'  Got {len(real_pairs)} real (query, body) pairs', flush=True)

    for q_text, body_text, label in real_pairs:
        # Concatenate body + query as the full context for anchor scan
        combined_text = body_text + '\n' + q_text
        ids = tok.encode(combined_text)
        m = compute_metrics(
            ids=ids,
            corpus_tag='real_captured',
            case_label=label,
            tok=tok,
        )
        all_metrics.append(m)

    # --- Synthetic corpus ---
    print(f'Loading synthetic cases from {results_dir}...', flush=True)
    synth_cases = load_synthetic_cases(results_dir, tok, max_cases=300)
    print(f'  Got {len(synth_cases)} synthetic cases', flush=True)

    for corpus_tag, label, ids, answer in synth_cases:
        is_niah = 'niah_single' in corpus_tag
        m = compute_metrics(
            ids=ids,
            corpus_tag=corpus_tag,
            case_label=label,
            answer=answer if is_niah else None,
            needle_text=str(answer) if is_niah else '',
            tok=tok if is_niah else None,
        )
        all_metrics.append(m)

    print(f'\nTotal cases: {len(all_metrics)} '
          f'(real={sum(1 for m in all_metrics if m.corpus_tag == "real_captured")}, '
          f'synth={sum(1 for m in all_metrics if m.corpus_tag != "real_captured")})',
          flush=True)

    # Write CSV
    csv_path = out_dir / 'multires_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(CaseMetrics._fields))
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m._asdict())
    print(f'Wrote {csv_path}', flush=True)

    # Write summary + verdict
    summary, verdict = _build_summary(all_metrics, tok)
    summary_path = out_dir / 'multires_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f'Wrote {summary_path}', flush=True)

    verdict_path = out_dir / 'h2_verdict.txt'
    with open(verdict_path, 'w') as f:
        f.write(verdict + '\n')
    print(f'Wrote {verdict_path}: {verdict}', flush=True)


def _build_summary(metrics: list[CaseMetrics], tok) -> tuple[str, str]:
    lines = ['# H2 Multi-Resolution Anchor — Results Summary\n']
    total = len(metrics)
    if total == 0:
        return '# H2 Multi-Resolution Anchor\n\nNo cases.\n', 'INCONCLUSIVE no cases'

    real_cases = [m for m in metrics if m.corpus_tag == 'real_captured']
    synth_cases = [m for m in metrics if m.corpus_tag != 'real_captured']

    lines.append(f'Total cases: **{total}** '
                 f'(real_captured={len(real_cases)}, synthetic={len(synth_cases)})\n')

    # --- Per-corpus x resolution cross-tab ---
    lines.append('## Cross-tab: Corpus x Resolution\n')
    lines.append('| Corpus | Resolution | Cases | Mean hits | Anchor-zero rate | '
                 'High-prec chunk frac | Forced chunks (mean) |')
    lines.append('|--------|------------|-------|-----------|-----------------|'
                 '---------------------|----------------------|')

    # Group by corpus_tag
    corpus_groups: dict[str, list[CaseMetrics]] = defaultdict(list)
    for m in metrics:
        corpus_groups[m.corpus_tag].append(m)

    for ctag, group in sorted(corpus_groups.items()):
        n = len(group)
        for res_label, hits_attr, prec_attr, forced_attr in [
            ('2gram',  'hits_2gram_only',   'high_prec_frac_2gram', 'forced_chunks_2gram'),
            ('4gram',  'hits_4gram',         'high_prec_frac_4gram', 'forced_chunks_4gram'),
            ('6gram',  'hits_6gram_only',    'high_prec_frac_6gram', 'forced_chunks_6gram'),
            ('union',  'hits_multi_union_chunks', None,              'hits_multi_union_chunks'),
        ]:
            mean_hits = sum(getattr(m, hits_attr) for m in group) / n
            zero_rate = sum(1 for m in group if getattr(m, hits_attr) == 0) / n * 100
            if prec_attr:
                mean_prec = sum(getattr(m, prec_attr) for m in group) / n
            else:
                mean_prec = float('nan')
            mean_forced = sum(getattr(m, forced_attr) for m in group) / n
            prec_str = f'{mean_prec:.2f}' if prec_attr else 'n/a'
            lines.append(f'| {ctag} | {res_label} | {n} | {mean_hits:.1f} | '
                         f'{zero_rate:.1f}% | {prec_str} | {mean_forced:.1f} |')
    lines.append('')

    # --- Headline metric: anchor-zero rescue on real_captured ---
    lines.append('## Headline: Anchor-Zero Rescue Rate (real_captured only)\n')
    if not real_cases:
        lines.append('*No real captured cases available.*\n')
        real_zero4 = []
        real_rescue_2g = 0
        real_rescue_6g = 0
        real_rescue_union = 0
    else:
        real_zero4 = [m for m in real_cases if m.hits_4gram == 0]
        n_real_zero4 = len(real_zero4)
        if n_real_zero4 == 0:
            lines.append(f'No anchor-zero cases in real_captured corpus (all {len(real_cases)} have >=1 4-gram hit).\n')
            real_rescue_2g = real_rescue_6g = real_rescue_union = 0
        else:
            real_rescue_2g = sum(1 for m in real_zero4 if m.hits_2gram_only > 0)
            real_rescue_6g = sum(1 for m in real_zero4 if m.hits_6gram_only > 0)
            real_rescue_union = sum(1 for m in real_zero4
                                    if m.hits_2gram_only > 0 or m.hits_6gram_only > 0)

            lines.append(f'Real anchor-zero cases (4-gram hits=0): **{n_real_zero4}** / {len(real_cases)}\n')
            lines.append('| Rescue source | Rescued | % of anchor-zero |')
            lines.append('|---------------|---------|-----------------|')
            lines.append(f'| 2-gram | {real_rescue_2g} | {100*real_rescue_2g/n_real_zero4:.1f}% |')
            lines.append(f'| 6-gram | {real_rescue_6g} | {100*real_rescue_6g/n_real_zero4:.1f}% |')
            lines.append(f'| union (2g+6g) | {real_rescue_union} | {100*real_rescue_union/n_real_zero4:.1f}% |')
            lines.append('')

    # --- Precision metric ---
    lines.append('## Precision Proxy (high-prec chunk fraction, real_captured)\n')
    if real_cases:
        hp4_mean = sum(m.high_prec_frac_4gram for m in real_cases if m.hits_4gram > 0)
        n4_nonzero = sum(1 for m in real_cases if m.hits_4gram > 0)
        hp2_mean = sum(m.high_prec_frac_2gram for m in real_cases if m.hits_2gram_only > 0)
        n2_nonzero = sum(1 for m in real_cases if m.hits_2gram_only > 0)
        hp6_mean = sum(m.high_prec_frac_6gram for m in real_cases if m.hits_6gram_only > 0)
        n6_nonzero = sum(1 for m in real_cases if m.hits_6gram_only > 0)

        def _safe_mean(s, n):
            return s / n if n > 0 else float('nan')

        hp4 = _safe_mean(hp4_mean, n4_nonzero)
        hp2 = _safe_mean(hp2_mean, n2_nonzero)
        hp6 = _safe_mean(hp6_mean, n6_nonzero)

        lines.append('| Resolution | Mean high-prec frac | Cases with hits |')
        lines.append('|------------|---------------------|-----------------|')
        lines.append(f'| 4gram (baseline) | {hp4:.3f} | {n4_nonzero} |')
        lines.append(f'| 2gram | {hp2:.3f} | {n2_nonzero} |')
        lines.append(f'| 6gram | {hp6:.3f} | {n6_nonzero} |')
        lines.append('')
        lines.append('*(high-prec = forced chunk anchored by n-gram with body-local freq <= 3)*\n')
    else:
        hp4 = hp2 = hp6 = float('nan')
        lines.append('*No real cases.*\n')

    # --- Needle recall on synthetic NIAH ---
    lines.append('## Needle Recall (synthetic_niah_single only)\n')
    niah_cases = [m for m in synth_cases
                  if 'niah' in m.corpus_tag and m.needle_chunk_id >= 0]
    if niah_cases:
        n_niah = len(niah_cases)
        recall_4g = sum(m.needle_chunk_in_4gram for m in niah_cases if m.needle_chunk_in_4gram >= 0)
        recall_union = sum(m.needle_chunk_in_union for m in niah_cases if m.needle_chunk_in_union >= 0)
        n_valid = sum(1 for m in niah_cases if m.needle_chunk_in_4gram >= 0)
        lines.append(f'NIAH cases with located needle: **{n_valid}** / {n_niah}\n')
        if n_valid > 0:
            lines.append('| Scheme | Needle recall |')
            lines.append('|--------|--------------|')
            lines.append(f'| 4gram (baseline) | {recall_4g}/{n_valid} = {100*recall_4g/n_valid:.1f}% |')
            lines.append(f'| union (multi) | {recall_union}/{n_valid} = {100*recall_union/n_valid:.1f}% |')
            lines.append('')
    else:
        lines.append('*No NIAH cases with located needle (needle may be in query window).*\n')

    # --- Top-3 rescued anchor-zero cases ---
    lines.append('## Top-3 Rescued Anchor-Zero Cases\n')
    rescued = [m for m in metrics
               if m.hits_4gram == 0 and (m.hits_2gram_only > 0 or m.hits_6gram_only > 0)]
    rescued_real = [m for m in rescued if m.corpus_tag == 'real_captured']
    rescued_synth = [m for m in rescued if m.corpus_tag != 'real_captured']
    show = rescued_real[:3] if rescued_real else rescued_synth[:3]
    for i, m in enumerate(show, 1):
        rescuer = '2g' if m.hits_2gram_only > 0 else '6g'
        rescuer_hits = m.hits_2gram_only if rescuer == '2g' else m.hits_6gram_only
        lines.append(f'{i}. `{m.case_label}` ({m.corpus_tag}) '
                     f'— rescued by **{rescuer}** with {rescuer_hits} hits '
                     f'(forced {m.forced_chunks_2gram if rescuer=="2g" else m.forced_chunks_6gram} chunks)')
    if not show:
        lines.append('*No rescued cases.*')
    lines.append('')

    # --- Momus go/no-go verdict ---
    lines.append('## Momus Go/No-Go Verdict\n')

    # Compute verdict numbers
    if real_zero4:
        n_real_zero4 = len(real_zero4)
        union_rescue_rate = 100 * real_rescue_union / n_real_zero4 if n_real_zero4 > 0 else 0.0
    else:
        # Fall back to synthetic VT (the only anchor-zero source found)
        synth_zero4 = [m for m in synth_cases if m.hits_4gram == 0]
        n_synth_zero4 = len(synth_zero4)
        if n_synth_zero4 > 0:
            s_rescue_2g = sum(1 for m in synth_zero4 if m.hits_2gram_only > 0)
            union_rescue_rate = 100 * s_rescue_2g / n_synth_zero4
        else:
            union_rescue_rate = 0.0

    # Precision drop: compare 2gram high-prec to 4gram high-prec
    if real_cases and not (hp4 != hp4) and not (hp2 != hp2):  # both non-NaN
        prec_drop_pp = (hp4 - hp2) * 100  # positive means 2g is LESS precise
    else:
        prec_drop_pp = float('nan')

    # Thresholds per task spec
    RESCUE_THRESHOLD_GO = 40.0    # >=40% rescue => PROCEED
    RESCUE_THRESHOLD_KILL = 15.0  # <15% rescue => KILL
    PREC_DROP_KILL = 15.0         # >15pp precision drop => KILL

    lines.append(f'- Anchor-zero rescue by union on real_captured: **{union_rescue_rate:.1f}%** '
                 f'(threshold: >=40% PROCEED, <15% KILL)\n')
    if prec_drop_pp == prec_drop_pp:  # not NaN
        lines.append(f'- High-precision fraction drop (4g → 2g): **{prec_drop_pp:.1f} pp** '
                     f'(threshold: >15pp KILL)\n')
    else:
        lines.append('- High-precision fraction drop: **N/A** (no real cases with hits)\n')
        prec_drop_pp = 0.0

    # Verdict logic
    if real_zero4:
        # Full verdict on real corpus
        if union_rescue_rate >= RESCUE_THRESHOLD_GO and prec_drop_pp <= PREC_DROP_KILL:
            verdict = (f'PROCEED union rescues {union_rescue_rate:.0f}% of real anchor-zero cases '
                       f'with {prec_drop_pp:.1f}pp precision drop')
        elif union_rescue_rate < RESCUE_THRESHOLD_KILL:
            verdict = (f'KILL union rescues only {union_rescue_rate:.0f}% of real anchor-zero cases '
                       f'(threshold {RESCUE_THRESHOLD_KILL:.0f}%)')
        elif prec_drop_pp > PREC_DROP_KILL:
            verdict = (f'KILL precision drops {prec_drop_pp:.1f}pp (threshold {PREC_DROP_KILL:.0f}pp)')
        else:
            verdict = (f'INCONCLUSIVE rescue={union_rescue_rate:.0f}% meets go-threshold '
                       f'but precision data weak — need larger real corpus')
    else:
        # No real anchor-zero cases: fall back to synthetic
        synth_zero4 = [m for m in synth_cases if m.hits_4gram == 0]
        if not synth_zero4:
            verdict = 'INCONCLUSIVE no anchor-zero cases in either corpus'
        elif union_rescue_rate >= RESCUE_THRESHOLD_GO:
            verdict = (f'INCONCLUSIVE synthetic-only rescue={union_rescue_rate:.0f}% exceeds threshold '
                       f'but real_captured has zero anchor-zero cases — insufficient signal on real workload')
        else:
            verdict = (f'INCONCLUSIVE rescue={union_rescue_rate:.0f}% on synthetic only '
                       f'real_captured corpus shows no anchor-zero cases')

    lines.append(f'\n**VERDICT: {verdict}**\n')

    return '\n'.join(lines) + '\n', verdict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _default_session_dirs() -> list[Path]:
    """Return default session directories for real captured prompts."""
    base = Path('/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub')
    harness_base = Path(
        '/home/peppi/Dev/lucebox-hub/.claude/worktrees/pr-232/harness-runs'
    )
    dirs = []
    if base.exists():
        dirs.append(base)
    if harness_base.exists():
        dirs.append(harness_base)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(description='H2 multi-resolution anchor hypothesis test.')
    ap.add_argument('--session-dir', type=Path, action='append', dest='session_dirs',
                    help='Path to directory with JSONL session files (repeat for multiple).')
    ap.add_argument(
        '--results-dir', type=Path,
        default=Path(__file__).parent.parent / 'results' / '2026-05-21_envelope',
        help='Directory with bench result subdirectories (synthetic cases).',
    )
    ap.add_argument(
        '--out-dir', type=Path,
        default=Path(__file__).parent,
        help='Output directory.',
    )
    ap.add_argument('--test', action='store_true', help='Run unit tests only.')
    ap.add_argument('--real-pairs', type=int, default=30,
                    help='Max real captured (query, body) pairs to sample.')
    args = ap.parse_args()

    if args.test:
        _run_unit_tests()
        sys.exit(0)

    session_dirs = args.session_dirs or _default_session_dirs()

    print('Loading tokenizer...', flush=True)
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
    except Exception:
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    print('Tokenizer loaded.', flush=True)

    analyze(
        session_dirs=session_dirs,
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        tok=tok,
        target_real_pairs=args.real_pairs,
    )


if __name__ == '__main__':
    main()
