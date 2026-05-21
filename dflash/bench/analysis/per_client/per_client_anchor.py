#!/usr/bin/env python3
"""
Per-client anchor coverage analysis from real harness session data.

Usage:
    python per_client_anchor.py --client claude_code --runs-dir <path> --out <dir>
    python per_client_anchor.py --client opencode --runs-dir <path> --out <dir>
    python per_client_anchor.py --runs-dir <path> --out <dir>  # all clients

Replicates C++ compute_anchor_hits() exactly (NGRAM=4, query_window=96,
max_hits_per_q=8, max_hits_buf=16).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken  # cl100k_base used as Qwen3 tokenizer proxy (BPE family)

HARNESS_RUNS_SUBDIR = ""  # runs-dir IS the harness-runs dir

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_ENC = tiktoken.get_encoding("cl100k_base")


def tokenize(text: str) -> list[int]:
    """Tokenize via tiktoken cl100k_base (Qwen3 BPE proxy)."""
    return list(_ENC.encode(text))


# ── Pure anchor computation — mirrors C++ compute_anchor_hits() ───────────────

NGRAM = 4
QUERY_TOKENS = 96
MAX_HITS_PER_Q = 8
MAX_HITS_BUF = 16


def compute_anchor_hits_pure(
    ids: list[int],
    query_tokens: int = QUERY_TOKENS,
    max_hits_per_q: int = MAX_HITS_PER_Q,
    max_hits_buf: int = MAX_HITS_BUF,
    ngram: int = NGRAM,
) -> tuple[int, list[int]]:
    """Pure Python replica of C++ compute_anchor_hits().

    Returns (hit_count, hit_positions).

    Algorithm (matching C++ exactly):
    - query window: ids[max(0, S-query_tokens) .. S)
    - search zone:  ids[0 .. max(0, q0-NGRAM))
    - for each q in query window: find all p in search zone where
      ids[p:p+NGRAM] == ids[q:q+NGRAM], collect up to max_hits_per_q per q
    - accumulate into hit_pos[], stop filling when total >= max_hits_buf
      but keep scanning all q positions (P1a fix — outer loop never exits early)
    """
    S = len(ids)
    if S < ngram:
        return 0, []
    q0 = max(0, S - query_tokens)
    search_end = max(0, q0 - ngram)

    total = 0
    hit_pos: list[int] = []

    for q in range(q0, S - ngram + 1):  # P1a: no `total < max_hits_buf` guard
        hits = 0
        local: list[int] = []
        qslice = ids[q : q + ngram]
        for p in range(search_end + 1):  # 0 <= p <= search_end
            if ids[p : p + ngram] == qslice:
                if hits < 16:
                    local.append(p)
                hits += 1
                if hits > max_hits_per_q:
                    break
        if 0 < hits <= max_hits_per_q:
            for pos in local:
                if total < max_hits_buf:
                    hit_pos.append(pos)
                    total += 1
    return total, hit_pos


def _ngram_hits_generic(ids: list[int], n: int) -> int:
    """Generic n-gram scan: does any body n-gram match any query n-gram? Returns hit count."""
    S = len(ids)
    if S < n:
        return 0
    q0 = max(0, S - QUERY_TOKENS)
    search_end = max(0, q0 - n)
    query_set: set[tuple[int, ...]] = set()
    for q in range(q0, S - n + 1):
        query_set.add(tuple(ids[q : q + n]))
    total = 0
    for p in range(search_end + 1):
        if tuple(ids[p : p + n]) in query_set:
            total += 1
    return total


def anchor_hits_2gram(ids: list[int]) -> int:
    return _ngram_hits_generic(ids, 2)


def anchor_hits_6gram(ids: list[int]) -> int:
    return _ngram_hits_generic(ids, 6)


# ── Unit tests ─────────────────────────────────────────────────────────────────

def _run_unit_tests() -> None:
    """Verify compute_anchor_hits_pure against hand-crafted inputs."""

    # Test 1: Exact repeat — query tail repeats a body prefix → 1 hit
    ids = list(range(20)) + list(range(4))  # body=[0..19], query tail=[0,1,2,3]
    hits, pos = compute_anchor_hits_pure(ids, query_tokens=10, ngram=4)
    assert hits >= 1, f"T1 expected >=1 hit, got {hits}"
    assert 0 in pos, f"T1 expected pos 0 in hits, got {pos}"

    # Test 2: No repeat — random body, unrelated query → 0 hits
    ids2 = list(range(50))
    hits2, _ = compute_anchor_hits_pure(ids2, query_tokens=5, ngram=4)
    assert hits2 == 0, f"T2 expected 0 hits (no repeat), got {hits2}"

    # Test 3: Query inside exclusion zone — no match allowed from search zone
    # body=[0..9], query_tokens=5 → q0=5, search_end=max(0,5-4)=1
    # So search zone is p in [0,1]. Query is ids[5..9]. They differ from ids[0:4].
    ids3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    hits3, _ = compute_anchor_hits_pure(ids3, query_tokens=5, ngram=4)
    assert hits3 == 0, f"T3 expected 0 hits, got {hits3}"

    # Test 4: Exact 4-gram at boundary of exclusion zone
    ids4 = [10, 20, 30, 40] + list(range(100)) + [10, 20, 30, 40]
    S = len(ids4)
    hits4, pos4 = compute_anchor_hits_pure(ids4, query_tokens=10, ngram=4)
    assert hits4 >= 1, f"T4 expected >=1 hit, got {hits4}"

    # Test 5: max_hits_buf capping — many repeats but buf capped at 16
    ids5 = [1, 2, 3, 4] * 50  # 200 tokens, lots of 4-gram repeats
    hits5, pos5 = compute_anchor_hits_pure(ids5, query_tokens=20, ngram=4)
    assert len(pos5) <= MAX_HITS_BUF, f"T5 hit_pos overflow: {len(pos5)}"

    # Test 6: Short prompt < NGRAM → 0 hits
    ids6 = [1, 2, 3]
    hits6, _ = compute_anchor_hits_pure(ids6)
    assert hits6 == 0, f"T6 expected 0 hits for short prompt, got {hits6}"

    print("All unit tests passed.")


# ── Session data extractors ────────────────────────────────────────────────────

@dataclass
class Turn:
    session_id: str
    turn_idx: int
    query_text: str        # user's message text (anonymized in output)
    body_text: str         # accumulated body (system + prior turns + current)
    body_tokens: int = 0
    anchor_hits_4g: int = 0
    anchor_hits_2g: int = 0
    anchor_hits_6g: int = 0


def _redact(text: str) -> str:
    """Redact local filesystem paths."""
    text = re.sub(r"/home/[^/\s]+", "<HOME>", text)
    text = re.sub(r"/tmp/[^\s]+", "<TMP>", text)
    return text


def extract_cc_turns(runs_dir: Path) -> list[Turn]:
    """Extract turns from Claude Code JSONL session files."""
    turns: list[Turn] = []
    pattern = runs_dir / "cc-*" / "**" / "*.jsonl"
    import glob
    files = glob.glob(str(pattern), recursive=True)
    files += glob.glob(str(runs_dir / "cc-*" / "claude-home*" / ".claude" / "projects" / "**" / "*.jsonl"), recursive=True)

    skip_cutoff = _ten_min_ago()
    seen: set[str] = set()

    for fpath in files:
        fpath = Path(fpath)
        if str(fpath) in seen:
            continue
        seen.add(str(fpath))
        try:
            mtime = fpath.stat().st_mtime
        except OSError:
            continue
        if mtime > skip_cutoff:
            continue  # skip live files
        try:
            _extract_cc_from_jsonl(fpath, turns)
        except Exception as e:
            print(f"  [warn] skipping {fpath}: {e}", file=sys.stderr)
    return turns


def _extract_cc_from_jsonl(fpath: Path, turns: list[Turn]) -> None:
    """Parse one Claude Code session JSONL into turns."""
    session_id = fpath.stem
    messages: list[dict] = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") in ("user", "assistant"):
                messages.append(d)

    # Build turns: each user message is a turn; body = everything seen so far
    accumulated: list[str] = []
    turn_idx = 0
    for msg in messages:
        role = msg.get("type")
        content = msg.get("message", {}).get("content", "")
        text = _extract_text(content)
        if not text:
            continue
        if role == "user":
            body = "\n".join(accumulated + [text])
            t = Turn(
                session_id=session_id,
                turn_idx=turn_idx,
                query_text=text[:500],
                body_text=body,
            )
            turns.append(t)
            turn_idx += 1
        accumulated.append(text)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return " ".join(parts).strip()
    return ""


def extract_opencode_turns(runs_dir: Path) -> list[Turn]:
    """Extract turns from opencode SQLite databases."""
    turns: list[Turn] = []
    # Use os.walk so hidden dirs (.local) are traversed
    dbs: list[Path] = []
    for root, _dirs, files in os.walk(str(runs_dir)):
        for fname in files:
            if fname == "opencode.db":
                dbs.append(Path(root) / fname)
    skip_cutoff = _ten_min_ago()

    for db_path in dbs:
        db_path = Path(db_path)
        # Use the co-located server.log mtime as the freshness proxy:
        # our sqlite3 queries touched the DB mtime; the server.log is untouched.
        run_dir = db_path
        for _ in range(6):
            run_dir = run_dir.parent
            if (run_dir / "server.log").exists():
                break
        ref_file = run_dir / "server.log" if (run_dir / "server.log").exists() else db_path
        try:
            mtime = ref_file.stat().st_mtime
        except OSError:
            continue
        if mtime > skip_cutoff:
            continue
        try:
            _extract_opencode_from_db(db_path, turns)
        except Exception as e:
            print(f"  [warn] skipping {db_path}: {e}", file=sys.stderr)
    return turns


def _extract_opencode_from_db(db_path: Path, turns: list[Turn]) -> None:
    """Parse opencode DB parts table for text content."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT p.session_id, p.message_id, p.data "
            "FROM part p ORDER BY p.time_created"
        ).fetchall()
    finally:
        conn.close()

    # Group by session, reconstruct conversation order
    sessions: dict[str, list[tuple[str, str]]] = {}
    for session_id, msg_id, data_str in rows:
        try:
            d = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "text":
            text = d.get("text", "").strip()
            if text:
                sessions.setdefault(session_id, []).append((msg_id, text))

    for session_id, msg_texts in sessions.items():
        accumulated: list[str] = []
        for turn_idx, (msg_id, text) in enumerate(msg_texts):
            body = "\n".join(accumulated + [text])
            t = Turn(
                session_id=f"opencode_{session_id}",
                turn_idx=turn_idx,
                query_text=text[:500],
                body_text=body,
            )
            turns.append(t)
            accumulated.append(text)


def _ten_min_ago() -> float:
    import time
    return time.time() - 600


# ── Drafter-skip server log parser ────────────────────────────────────────────

@dataclass
class DrafterSkipLine:
    run: str
    kept_tokens: int
    n_chunks: int
    forced_anchors: int
    S: int  # body size (S= field)


def parse_server_logs(runs_dir: Path) -> dict[str, list[DrafterSkipLine]]:
    """Parse [drafter-skip] lines from all server.log files, keyed by client."""
    # Map run-dir prefixes to client
    client_map = {
        "cc-": "claude_code",
        "probe-skip-anchor": "opencode",
        "pair-llamacpp": "opencode",
    }
    result: dict[str, list[DrafterSkipLine]] = {
        "claude_code": [],
        "opencode": [],
    }
    pattern = re.compile(
        r"\[drafter-skip\] kept (\d+) tokens from (\d+) chunks \((\d+) forced incl\. anchors\)"
        r".*?S=(\d+)"
    )

    for log_file in sorted(runs_dir.glob("*/server.log")):
        run = log_file.parent.name
        client = None
        for prefix, c in client_map.items():
            if run.startswith(prefix) or run == prefix.rstrip("-"):
                client = c
                break
        if client is None:
            continue

        with open(log_file) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    result[client].append(DrafterSkipLine(
                        run=run,
                        kept_tokens=int(m.group(1)),
                        n_chunks=int(m.group(2)),
                        forced_anchors=int(m.group(3)),
                        S=int(m.group(4)),
                    ))
    return result


# ── Per-turn anchor computation ────────────────────────────────────────────────

def analyze_turns(turns: list[Turn]) -> list[Turn]:
    for t in turns:
        ids = tokenize(t.body_text)
        t.body_tokens = len(ids)
        t.anchor_hits_4g, _ = compute_anchor_hits_pure(ids)
        t.anchor_hits_2g = anchor_hits_2gram(ids)
        t.anchor_hits_6g = anchor_hits_6gram(ids)
    return turns


# ── Aggregation ───────────────────────────────────────────────────────────────

@dataclass
class ClientStats:
    client: str
    n_turns: int
    mean_body_tokens: float
    p95_body_tokens: float
    pct_anchor_zero: float
    hist_4g: Counter = field(default_factory=Counter)
    rescue_2g_pct: float = 0.0  # % of anchor-zero turns rescued by 2-gram
    rescue_6g_pct: float = 0.0  # % of anchor-zero turns rescued by 6-gram
    anchor_zero_queries: list[str] = field(default_factory=list)


def aggregate(client: str, turns: list[Turn]) -> ClientStats:
    if not turns:
        return ClientStats(client, 0, 0, 0, 0)

    body_toks = sorted(t.body_tokens for t in turns)
    n = len(body_toks)
    mean_bt = sum(body_toks) / n
    p95_bt = body_toks[int(0.95 * n)]

    zero_turns = [t for t in turns if t.anchor_hits_4g == 0]
    pct_zero = 100.0 * len(zero_turns) / n

    hist = Counter(t.anchor_hits_4g for t in turns)

    rescue_2g = sum(1 for t in zero_turns if t.anchor_hits_2g > 0)
    rescue_6g = sum(1 for t in zero_turns if t.anchor_hits_6g > 0)
    r2 = 100.0 * rescue_2g / len(zero_turns) if zero_turns else 0.0
    r6 = 100.0 * rescue_6g / len(zero_turns) if zero_turns else 0.0

    # 3 representative anchor-zero queries (anonymized)
    sample: list[str] = []
    for t in zero_turns[:3]:
        sample.append(_redact(t.query_text[:200]))

    return ClientStats(
        client=client,
        n_turns=n,
        mean_body_tokens=mean_bt,
        p95_body_tokens=p95_bt,
        pct_anchor_zero=pct_zero,
        hist_4g=hist,
        rescue_2g_pct=r2,
        rescue_6g_pct=r6,
        anchor_zero_queries=sample,
    )


# ── Output writers ─────────────────────────────────────────────────────────────

def write_anchor_zero_jsonl(out_dir: Path, client: str, turns: list[Turn]) -> None:
    path = out_dir / "per_client_anchor_zero.jsonl"
    mode = "a"  # append across clients
    with open(path, mode) as f:
        for t in turns:
            if t.anchor_hits_4g == 0:
                rec = {
                    "client": client,
                    "session": t.session_id,
                    "turn": t.turn_idx,
                    "body_tokens": t.body_tokens,
                    "anchor_hits_4g": t.anchor_hits_4g,
                    "anchor_hits_2g": t.anchor_hits_2g,
                    "anchor_hits_6g": t.anchor_hits_6g,
                    "query_sample": _redact(t.query_text[:200]),
                }
                f.write(json.dumps(rec) + "\n")


def write_summary(
    out_dir: Path,
    stats_list: list[ClientStats],
    server_log_data: dict[str, list[DrafterSkipLine]],
) -> None:
    path = out_dir / "summary.md"
    lines = ["# Per-Client Anchor Coverage Analysis\n"]

    lines.append("## Client × Anchor Coverage\n")
    lines.append(
        "| client | turns | mean body tok | p95 body tok"
        " | % anchor-zero | 2g rescue % | 6g rescue % |\n"
    )
    lines.append("|" + "---|" * 7 + "\n")
    for s in stats_list:
        lines.append(
            f"| {s.client} | {s.n_turns} | {s.mean_body_tokens:.0f}"
            f" | {s.p95_body_tokens:.0f} | {s.pct_anchor_zero:.1f}%"
            f" | {s.rescue_2g_pct:.1f}% | {s.rescue_6g_pct:.1f}% |\n"
        )

    lines.append("\n## 4-gram Hit Histograms\n")
    for s in stats_list:
        if not s.hist_4g:
            continue
        lines.append(f"### {s.client}\n")
        for k in sorted(s.hist_4g.keys()):
            bar = "#" * s.hist_4g[k]
            lines.append(f"  hits={k:2d}: {s.hist_4g[k]:3d} turns  {bar}\n")

    lines.append("\n## Representative Anchor-Zero Queries\n")
    for s in stats_list:
        lines.append(f"### {s.client}\n")
        for i, q in enumerate(s.anchor_zero_queries):
            lines.append(f"  [{i+1}] {repr(q)}\n")

    lines.append("\n## Server Log [drafter-skip] Evidence\n")
    lines.append("Direct production anchor counts (from server.log):\n\n")
    for client, entries in server_log_data.items():
        if not entries:
            continue
        lines.append(f"### {client} ({len(entries)} requests)\n")
        anchors = [e.forced_anchors for e in entries]
        body_sizes = [e.S for e in entries]
        hist = Counter(anchors)
        lines.append("forced-anchor distribution:\n")
        for k in sorted(hist.keys()):
            lines.append(f"  anchors={k:3d}: {hist[k]} requests\n")
        if body_sizes:
            lines.append(
                f"body size S: min={min(body_sizes)} mean={sum(body_sizes)//len(body_sizes)}"
                f" max={max(body_sizes)}\n"
            )
        lines.append("\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Summary written: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-client anchor coverage analysis")
    parser.add_argument("--client", choices=["claude_code", "opencode", "hermes", "all"],
                        default="all")
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("/home/peppi/Dev/lucebox-hub/.claude/worktrees/pr-232/harness-runs"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/bench/analysis/per_client"))
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit")
    args = parser.parse_args()

    if args.test:
        _run_unit_tests()
        return

    _run_unit_tests()  # always verify pure functions before analysis

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear anchor_zero jsonl for fresh run
    azpath = out_dir / "per_client_anchor_zero.jsonl"
    if azpath.exists():
        azpath.unlink()

    runs_dir: Path = args.runs_dir
    clients = [args.client] if args.client != "all" else ["claude_code", "opencode"]

    all_stats: list[ClientStats] = []
    all_turns: dict[str, list[Turn]] = {}

    for client in clients:
        print(f"\n--- {client} ---")
        if client == "claude_code":
            turns = extract_cc_turns(runs_dir)
        elif client == "opencode":
            turns = extract_opencode_turns(runs_dir)
        else:
            print(f"  No session data found for {client}; skipping.")
            continue

        if not turns:
            print(f"  No turns found.")
            continue

        print(f"  Extracted {len(turns)} turns, computing anchors...")
        turns = analyze_turns(turns)
        stats = aggregate(client, turns)
        all_stats.append(stats)
        all_turns[client] = turns
        write_anchor_zero_jsonl(out_dir, client, turns)

        print(f"  turns={stats.n_turns} body_mean={stats.mean_body_tokens:.0f}"
              f" anchor_zero={stats.pct_anchor_zero:.1f}%"
              f" rescue_2g={stats.rescue_2g_pct:.1f}%"
              f" rescue_6g={stats.rescue_6g_pct:.1f}%")

    # Server log analysis
    server_data = parse_server_logs(runs_dir)

    write_summary(out_dir, all_stats, server_data)

    print("\nDone.")
    return all_stats


if __name__ == "__main__":
    main()
