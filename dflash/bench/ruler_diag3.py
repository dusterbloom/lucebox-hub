#!/usr/bin/env python3
"""RULER diagnostic-3 harness: VT, FWE, MQA tasks for local C++ HTTP server.

Three tasks from the RULER benchmark suite, re-implemented minimally for an
empirical operating-envelope study against the dflash_server Anthropic Messages
endpoint.

Usage examples:
  python ruler_diag3.py --task vt  --ctx-tokens 16384 --n 10 --out /tmp/vt_16k  --mode auto
  python ruler_diag3.py --task fwe --ctx-tokens 16384 --n 10 --out /tmp/fwe_16k --mode off
  python ruler_diag3.py --task mqa --ctx-tokens 16384 --n 10 --out /tmp/mqa_16k --mode always
"""
from __future__ import annotations
import argparse, json, math, os, random, re, string, time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen3.6-27B"
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
SYSTEM_PROMPT = (
    "You are a careful long-context assistant. "
    "Answer in one short line, no extra prose."
)

# ──────────────────────────────────────────────────────────────────────────────
# Tokeniser-length binary search (same pattern as niah_gen.py)
# ──────────────────────────────────────────────────────────────────────────────

def _count_tokens(text: str, tok) -> int:
    return len(tok.encode(text))


def _fit_filler(scaffold: str, target_tokens: int, tok, tolerance: float = 0.005) -> str:
    """Return filler-padded text whose token count is ≤ target_tokens."""
    scaffold_toks = _count_tokens(scaffold, tok)
    if scaffold_toks >= target_tokens:
        raise ValueError(
            f"scaffold alone is {scaffold_toks} tokens, exceeds target {target_tokens}"
        )
    budget = target_tokens - scaffold_toks
    lo, hi = 2.0, 6.0
    target_chars = int(budget * (lo + hi) / 2)

    def build(chars: int) -> str:
        pad = (FILLER * (chars // len(FILLER) + 2))[:max(0, chars)]
        return pad + "\n\n" + scaffold

    prompt = build(target_chars)
    actual = _count_tokens(prompt, tok)
    for _ in range(20):
        if abs(actual - target_tokens) / target_tokens < tolerance:
            break
        if actual > target_tokens:
            hi = (lo + hi) / 2
        else:
            lo = (lo + hi) / 2
        target_chars = int(budget * (lo + hi) / 2)
        prompt = build(target_chars)
        actual = _count_tokens(prompt, tok)

    # Hard-trim: guarantee we never exceed target_tokens.
    for step in (256, 64, 16, 1):
        while actual > target_tokens and target_chars >= step:
            target_chars -= step
            prompt = build(target_chars)
            actual = _count_tokens(prompt, tok)

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Task: Variable Tracking (VT)
# ──────────────────────────────────────────────────────────────────────────────
# 4 independent variable chains, each up to 5 hops.
# Format: "VAR1 = LITERAL\n... VAR2 = VAR1\n... VAR3 = VAR2\n..."
# Query:  "What is the final value of <last_var>?"
# Score:  fraction of cases where the final literal value appears in the answer.

_VT_CHAINS = 4
_VT_MAX_HOPS = 5


def _rand_varname(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def _rand_literal(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=length))


def gen_vt(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    chains: list[list[str]] = []
    for _ in range(_VT_CHAINS):
        n_hops = rng.randint(2, _VT_MAX_HOPS)
        literal = _rand_literal(rng)
        vars_ = [_rand_varname(rng) for _ in range(n_hops)]
        # Ensure no collisions across chains.
        chains.append(vars_)
        chains[-1]  # reference

    # Collect all assignments: shuffle them across the filler body.
    # Each chain: var[0] = LITERAL; var[1] = var[0]; ...; var[-1] = var[-2]
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
    # Query asks for the last var in chain 0.
    query_var = chains[0][-1]
    answer = literals[0]
    question = f"What is the final value of {query_var}?"
    scaffold = f"{assigns_text}\n\nQuestion: {question}\nAnswer:"
    try:
        prompt = _fit_filler(scaffold, target_tokens, tok)
    except ValueError as e:
        raise ValueError(f"VT gen_vt seed={seed}: {e}") from e

    return {
        "prompt": prompt,
        "answer": answer,
        "query_var": query_var,
        "n_tokens": _count_tokens(prompt, tok),
    }


def score_vt(response: str, expected: str) -> float:
    return 1.0 if expected.upper() in response.upper() else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Task: Frequent Words Extraction (FWE)
# ──────────────────────────────────────────────────────────────────────────────
# Build a body of ~2000 words drawn from a 200-word vocabulary with Zipf weights.
# Query: "List the 3 most frequent words."
# Score: mean Jaccard of model's top-3 set vs ground-truth top-3 set.

_FWE_VOCAB_SIZE = 200
_FWE_BODY_WORDS = 2000
_FWE_TOP_K = 3


def _zipf_weights(n: int, exponent: float = 1.2) -> list[float]:
    raw = [1.0 / (i + 1) ** exponent for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def gen_fwe(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
             for _ in range(_FWE_VOCAB_SIZE)]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    # Pad back to _FWE_VOCAB_SIZE if dedup removed words.
    extra_seed = seed + 100000
    while len(uniq) < _FWE_VOCAB_SIZE:
        candidate = "zz" + "".join(random.Random(extra_seed).choices(string.ascii_lowercase, k=6))
        extra_seed += 1
        if candidate not in seen:
            seen.add(candidate)
            uniq.append(candidate)
    vocab = uniq[:_FWE_VOCAB_SIZE]

    weights = _zipf_weights(_FWE_VOCAB_SIZE)
    # Shuffle vocab before weighting so each run gets a different high-freq word.
    rng.shuffle(vocab)
    body_words = rng.choices(vocab, weights=weights, k=_FWE_BODY_WORDS)
    body = " ".join(body_words)

    # Ground truth: top-3 by actual frequency.
    from collections import Counter
    counts = Counter(body_words)
    top3 = {w for w, _ in counts.most_common(_FWE_TOP_K)}

    question = (
        "Based on the word list above, what are the 3 most frequently occurring words? "
        "List them separated by commas, nothing else."
    )
    scaffold = f"{body}\n\nQuestion: {question}\nAnswer:"
    try:
        prompt = _fit_filler(scaffold, target_tokens, tok)
    except ValueError as e:
        raise ValueError(f"FWE gen_fwe seed={seed}: {e}") from e

    return {
        "prompt": prompt,
        "answer": sorted(top3),
        "n_tokens": _count_tokens(prompt, tok),
    }


def score_fwe(response: str, expected: list[str]) -> float:
    """Jaccard of model's top-3 words vs ground-truth top-3 words."""
    # Extract all lowercase alpha tokens from the response.
    tokens = re.findall(r"[a-z]+", response.lower())
    # Take the first 3 unique words as the model's answer.
    seen: set[str] = set()
    pred: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            pred.append(t)
        if len(pred) >= _FWE_TOP_K:
            break
    pred_set = set(pred)
    exp_set = set(w.lower() for w in expected)
    if not pred_set and not exp_set:
        return 1.0
    return len(pred_set & exp_set) / len(pred_set | exp_set)


# ──────────────────────────────────────────────────────────────────────────────
# Task: Multi-Query Answering (MQA)  k=4 distractors
# ──────────────────────────────────────────────────────────────────────────────
# Insert 5 needles (1 query + 4 distractors) with distinct keys.
# Ask for the query needle's value.
# Score: fraction correct (exact match of value in response).

_MQA_DISTRACTORS = 4
NEEDLE_TMPL = "The special {key} number is: {value}."
MQA_QUESTION_TMPL = "What is the special {key} number? Answer with just the number."


def _rand_key(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))


def _rand_value(rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=7))


def gen_mqa(seed: int, target_tokens: int, tok) -> dict:
    rng = random.Random(seed)
    n_needles = _MQA_DISTRACTORS + 1
    keys = []
    while len(keys) < n_needles:
        k = _rand_key(rng)
        if k not in keys:
            keys.append(k)
    values = [_rand_value(rng) for _ in range(n_needles)]
    query_idx = 0
    needles = [NEEDLE_TMPL.format(key=keys[i], value=values[i]) for i in range(n_needles)]
    question = MQA_QUESTION_TMPL.format(key=keys[query_idx])
    answer = values[query_idx]
    scaffold = "\n".join(needles) + f"\n\nQuestion: {question}\nAnswer:"
    try:
        prompt = _fit_filler(scaffold, target_tokens, tok)
    except ValueError as e:
        raise ValueError(f"MQA gen_mqa seed={seed}: {e}") from e

    return {
        "prompt": prompt,
        "answer": answer,
        "query_key": keys[query_idx],
        "n_tokens": _count_tokens(prompt, tok),
    }


def score_mqa(response: str, expected: str) -> float:
    return 1.0 if expected in response else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# HTTP client
# ──────────────────────────────────────────────────────────────────────────────

def post_messages(
    server_url: str,
    prompt: str,
    mode: str,
    keep_ratio: float,
    max_tokens: int,
    timeout: int = 300,
) -> tuple[str, float, float, dict]:
    """POST to /v1/messages; return (response_text, wall_s, ttft_s, raw_json)."""
    url = server_url.rstrip("/") + "/v1/messages"
    payload: dict[str, Any] = {
        "model": "local",
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "extra_body": {
            "pflash_mode": mode,
            "keep_ratio": keep_ratio,
        },
    }
    t0 = time.time()
    ttft_s = float("nan")
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        wall_s = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        wall_s = time.time() - t0
        return f"[error] {exc}", wall_s, ttft_s, {}

    # Extract text from Anthropic Messages response shape.
    content = data.get("content", [])
    text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
    response_text = "".join(text_parts).strip()

    # TTFT: some servers emit usage.time_to_first_token_ms in the response.
    usage = data.get("usage", {})
    if "time_to_first_token_ms" in usage:
        ttft_s = usage["time_to_first_token_ms"] / 1000.0

    return response_text, wall_s, ttft_s, data


# ──────────────────────────────────────────────────────────────────────────────
# Per-task dispatch
# ──────────────────────────────────────────────────────────────────────────────

_TASK_CONFIG = {
    "vt":  {"gen": gen_vt,  "score": score_vt,  "max_tokens": 256},
    "fwe": {"gen": gen_fwe, "score": score_fwe, "max_tokens": 128},
    "mqa": {"gen": gen_mqa, "score": score_mqa, "max_tokens": 256},
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="RULER diagnostic-3 harness (VT / FWE / MQA) against local C++ HTTP server."
    )
    ap.add_argument("--task", required=True, choices=["vt", "fwe", "mqa"])
    ap.add_argument("--ctx-tokens", type=int, default=8192, metavar="N",
                    help="Target context length in tokens (default 8192).")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of test cases (default 5).")
    ap.add_argument("--out", required=True,
                    help="Output directory for summary.json and raw case files.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed; case i uses seed+i (default 42).")
    ap.add_argument("--mode", default="off", choices=["off", "auto", "always"],
                    help="pflash_mode to pass in extra_body (default off).")
    ap.add_argument("--keep-ratio", type=float, default=0.10,
                    help="pflash keep ratio (default 0.10).")
    ap.add_argument("--server-url", default="http://127.0.0.1:8080",
                    help="Base URL of the local C++ HTTP server (default http://127.0.0.1:8080).")
    ap.add_argument("--tokenizer", default=TOKENIZER_NAME,
                    help=f"HF tokenizer for length targeting (default {TOKENIZER_NAME}).")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-request HTTP timeout in seconds (default 300).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ruler_diag3] loading tokenizer {args.tokenizer} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    cfg = _TASK_CONFIG[args.task]
    gen_fn = cfg["gen"]
    score_fn = cfg["score"]
    max_tokens = cfg["max_tokens"]

    scores: list[float] = []
    wall_times: list[float] = []

    for i in range(args.n):
        seed_i = args.seed + i
        print(f"[case {i}] generating task={args.task} ctx={args.ctx_tokens} seed={seed_i}", flush=True)
        try:
            case = gen_fn(seed_i, args.ctx_tokens, tok)
        except ValueError as e:
            print(f"[case {i}] gen error: {e}", flush=True)
            continue

        print(f"[case {i}] prompt_len={case['n_tokens']} posting to {args.server_url} ...", flush=True)
        response_text, wall_s, ttft_s, raw = post_messages(
            args.server_url,
            case["prompt"],
            args.mode,
            args.keep_ratio,
            max_tokens,
            args.timeout,
        )
        sc = score_fn(response_text, case["answer"])
        scores.append(sc)
        wall_times.append(wall_s)

        print(f"[case {i}] score={sc:.3f} wall={wall_s:.1f}s ans={case['answer']!r} "
              f"resp={response_text[:80]!r}", flush=True)

        raw_record = {
            "case_idx": i,
            "seed": seed_i,
            "task": args.task,
            "prompt_len": case["n_tokens"],
            "answer": case["answer"],
            "response_text": response_text,
            "score": sc,
            "wall_s": wall_s,
            "ttft_s": ttft_s,
            "mode_used": args.mode,
            "keep_ratio": args.keep_ratio,
            "ctx_tokens": args.ctx_tokens,
            "server_raw": raw,
        }
        raw_path = out_dir / f"case_{i:04d}.raw.json"
        raw_path.write_text(json.dumps(raw_record, indent=2))

    if not scores:
        print("[ruler_diag3] no cases completed", flush=True)
        return

    accuracy = sum(scores) / len(scores)
    wall_sorted = sorted(wall_times)
    p50 = wall_sorted[len(wall_sorted) // 2]
    p95_idx = max(0, int(math.ceil(0.95 * len(wall_sorted))) - 1)
    p95 = wall_sorted[p95_idx]

    summary = {
        "task": args.task,
        "ctx_tokens": args.ctx_tokens,
        "n_cases": len(scores),
        "n_completed": len(scores),
        "accuracy": accuracy,
        "scores": scores,
        "wall_p50": p50,
        "wall_p95": p95,
        "mode": args.mode,
        "keep_ratio": args.keep_ratio,
        "server_url": args.server_url,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[ruler_diag3] done. accuracy={accuracy:.3f} ({len(scores)}/{args.n}) "
          f"wall_p50={p50:.1f}s wall_p95={p95:.1f}s", flush=True)
    print(f"[ruler_diag3] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
