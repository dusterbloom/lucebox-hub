#!/usr/bin/env python3
"""
quality_matrix.py — validate output quality for 3 speculators × 3 workloads.

Speculators: ar, mtp_d3, dflash_b22
Workloads:   humaneval, gsm8k, math500
Model:       Qwen3.6-27B Q4_K_M, n=8 per cell (seed=42)

Run:
    export MTP_GGUF=/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf
    export DFLASH_TARGET=/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf
    export DFLASH_DRAFT=/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf
    export DFLASH_BIN=/home/peppi/Dev/lucebox-hub/dflash/build/test_dflash
    python3 dflash/scripts/quality_matrix.py

Output: /tmp/quality_matrix_results.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
SCRIPTS = ROOT / "dflash" / "scripts"
sys.path.insert(0, str(SCRIPTS))

MTP_GGUF      = os.environ.get("MTP_GGUF",      "/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf")
DFLASH_TARGET = os.environ.get("DFLASH_TARGET",  "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DFLASH_DRAFT  = os.environ.get("DFLASH_DRAFT",   "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf")
DFLASH_BIN    = os.environ.get("DFLASH_BIN",     str(ROOT / "dflash" / "build" / "test_dflash"))
TOKENIZER_ID  = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B")

TMPDIR = Path(tempfile.gettempdir()) / "quality_matrix"
TMPDIR.mkdir(parents=True, exist_ok=True)

OUT_MD   = Path("/tmp/quality_matrix_results.md")
DEBUG_LOG = Path("/tmp/quality_matrix_debug.log")

SLEEP_BETWEEN_RUNS = 15.0   # WSL2 CUDA teardown mitigation
SUBPROCESS_TIMEOUT = 300    # 5 min per test_dflash invocation
EXEC_TIMEOUT_SECS  = 5      # signal.SIGALRM for HumanEval exec()

N_SAMPLE = 8
SEED     = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_debug_log = open(DEBUG_LOG, "w", buffering=1)


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _debug_log.write(line + "\n")


def log_debug(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _debug_log.write(f"[{ts}] DEBUG: {msg}\n")

# ---------------------------------------------------------------------------
# Tokenizer — HF preferred; GPT2 byte-decoder GGUF fallback
# ---------------------------------------------------------------------------

_tok_cache: Any = None


def _get_tokenizer():
    global _tok_cache
    if _tok_cache is not None:
        return _tok_cache
    try:
        from transformers import AutoTokenizer
        _tok_cache = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
        log(f"Loaded HF tokenizer: {TOKENIZER_ID}")
        return _tok_cache
    except Exception as e:
        log(f"HF tokenizer unavailable ({e}), falling back to GGUF byte-decoder")
        return None


# GPT2 byte-level decoder — reconstructs token strings from GGUF vocab without HF.
# Ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_DECODER: Dict[str, int] = {}   # filled lazily


def _ensure_byte_decoder() -> Dict[str, int]:
    global _BYTE_DECODER
    if _BYTE_DECODER:
        return _BYTE_DECODER
    enc = _gpt2_bytes_to_unicode()
    _BYTE_DECODER = {v: k for k, v in enc.items()}
    return _BYTE_DECODER


_gguf_vocab_cache: Optional[List[str]] = None


def _load_gguf_vocab(gguf_path: str) -> List[str]:
    global _gguf_vocab_cache
    if _gguf_vocab_cache is not None:
        return _gguf_vocab_cache
    import gguf
    reader = gguf.GGUFReader(gguf_path, mode="r")
    # Standard field name: tokenizer.ggml.tokens
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        raise RuntimeError("GGUF has no tokenizer.ggml.tokens field")
    vocab: List[str] = []
    for i in range(len(tokens_field.data)):
        raw = bytes(tokens_field.parts[tokens_field.data[i]])
        vocab.append(raw.decode("utf-8", errors="replace"))
    _gguf_vocab_cache = vocab
    log(f"Loaded GGUF vocab: {len(vocab)} tokens from {gguf_path}")
    return vocab


def detokenize_ids(token_ids: List[int], gguf_path: Optional[str] = None) -> str:
    """Convert a list of token IDs to text.

    Prefers HF AutoTokenizer.decode(); falls back to GGUF byte-level decode.
    """
    tok = _get_tokenizer()
    if tok is not None:
        return tok.decode(token_ids, skip_special_tokens=True)
    # GGUF fallback
    if gguf_path is None:
        gguf_path = MTP_GGUF
    vocab = _load_gguf_vocab(gguf_path)
    bd = _ensure_byte_decoder()
    text_pieces: List[str] = []
    for tid in token_ids:
        if 0 <= tid < len(vocab):
            piece = vocab[tid]
            # Decode GPT2-style byte tokens (e.g. "Ġ" → space, "Ċ" → newline)
            try:
                raw_bytes = bytes([bd[c] for c in piece])
                text_pieces.append(raw_bytes.decode("utf-8", errors="replace"))
            except KeyError:
                text_pieces.append(piece)
    return "".join(text_pieces)


def read_token_bin(path: str) -> List[int]:
    """Read little-endian int32 token IDs written by test_dflash --out."""
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    return list(struct.unpack(f"<{n}i", raw[:n * 4]))


# ---------------------------------------------------------------------------
# Dataset loading — mirrors workload modules (same seed, same selection)
# ---------------------------------------------------------------------------

def _load_humaneval(n_sample: int = N_SAMPLE, seed: int = SEED):
    """Returns list of dicts with keys: prompt, test, entry_point, task_id."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", None, split="test")
    ds_sel = ds.shuffle(seed=seed).select(range(min(n_sample, len(ds))))
    return list(ds_sel)


def _load_gsm8k(n_sample: int = N_SAMPLE, seed: int = SEED):
    """Returns list of dicts with keys: question, answer."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    ds_sel = ds.shuffle(seed=seed).select(range(min(n_sample, len(ds))))
    return list(ds_sel)


def _load_math500(n_sample: int = N_SAMPLE, seed: int = SEED):
    """Returns list of dicts with keys: problem, answer."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", None, split="test")
    ds_sel = ds.shuffle(seed=seed).select(range(min(n_sample, len(ds))))
    return list(ds_sel)


# ---------------------------------------------------------------------------
# Prompt construction — mirrors workload modules exactly
# ---------------------------------------------------------------------------

def _apply_chat_template(tok, raw_prompt: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        return raw_prompt


def build_prompts_humaneval(samples, tmpdir: Path) -> List[Dict[str, Any]]:
    tok = _get_tokenizer()
    if tok is None:
        raise RuntimeError("HF tokenizer required for prompt building")
    result = []
    for idx, s in enumerate(samples):
        raw = s["prompt"]
        text = _apply_chat_template(tok, raw)
        bin_path = tmpdir / f"humaneval_{idx:03d}.bin"
        _write_token_bin(tok, text, bin_path)
        result.append({
            "idx": idx,
            "task_id": s.get("task_id", f"humaneval_{idx}"),
            "bin_path": str(bin_path),
            "raw_prompt": raw,
            "test": s.get("test", ""),
            "entry_point": s.get("entry_point", ""),
            "gold_answer": None,
        })
    return result


def build_prompts_gsm8k(samples, tmpdir: Path) -> List[Dict[str, Any]]:
    tok = _get_tokenizer()
    if tok is None:
        raise RuntimeError("HF tokenizer required for prompt building")
    result = []
    for idx, s in enumerate(samples):
        raw = f"Question: {s['question']}\nAnswer: "
        text = _apply_chat_template(tok, raw)
        bin_path = tmpdir / f"gsm8k_{idx:03d}.bin"
        _write_token_bin(tok, text, bin_path)
        gold = _extract_gsm8k_gold(s["answer"])
        result.append({
            "idx": idx,
            "task_id": f"gsm8k_{idx}",
            "bin_path": str(bin_path),
            "raw_prompt": raw,
            "test": None,
            "entry_point": None,
            "gold_answer": gold,
            "gold_raw": s["answer"],
        })
    return result


def build_prompts_math500(samples, tmpdir: Path) -> List[Dict[str, Any]]:
    tok = _get_tokenizer()
    if tok is None:
        raise RuntimeError("HF tokenizer required for prompt building")
    result = []
    for idx, s in enumerate(samples):
        raw = (
            f"Problem: {s['problem']}\n"
            r"Solution: Put your final answer in \boxed{}."
            "\n"
        )
        text = _apply_chat_template(tok, raw)
        bin_path = tmpdir / f"math500_{idx:03d}.bin"
        _write_token_bin(tok, text, bin_path)
        result.append({
            "idx": idx,
            "task_id": f"math500_{idx}",
            "bin_path": str(bin_path),
            "raw_prompt": raw,
            "test": None,
            "entry_point": None,
            "gold_answer": s["answer"],
        })
    return result


def _write_token_bin(tok, text: str, path: Path) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))


# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------

def _extract_gsm8k_gold(answer_str: str) -> Optional[str]:
    """Extract numeric answer after '####' in GSM8K gold format."""
    m = re.search(r"####\s*([0-9,\-\.]+)", answer_str)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def _extract_gsm8k_pred(generated_text: str) -> Optional[str]:
    """Extract last number from generated text (last #### pattern or last digit sequence)."""
    # Try #### pattern first (model following the GSM8K convention)
    matches = re.findall(r"####\s*([0-9,\-\.]+)", generated_text)
    if matches:
        return matches[-1].replace(",", "").strip()
    # Fall back to last number in the text
    numbers = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", generated_text)
    if numbers:
        return numbers[-1].strip()
    return None


def _normalize_math(s: str) -> str:
    """Normalize a math answer for comparison."""
    s = s.strip().lower()
    s = s.replace("$", "").replace("\\,", "").replace(" ", "")
    # Normalize fractions: \frac{a}{b} → a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    # Remove \left and \right
    s = re.sub(r"\\(left|right)", "", s)
    # Remove trailing zeros from decimals
    try:
        f = float(s)
        return f"{f:g}"
    except (ValueError, TypeError):
        pass
    return s


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of last \\boxed{...} in text."""
    # Handle nested braces
    results = []
    i = 0
    while i < len(text):
        pos = text.find(r"\boxed{", i)
        if pos == -1:
            break
        start = pos + len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j - 1])
        i = pos + 1
    return results[-1] if results else None


# Subprocess-based grader for HumanEval (SIGALRM sandbox).
_GRADE_RUNNER = r'''
import json, sys, signal, traceback

def _alarm_handler(*_):
    raise TimeoutError("exec timeout")

signal.signal(signal.SIGALRM, _alarm_handler)
signal.alarm(int(sys.argv[1]))

src = sys.stdin.read()
try:
    g = {}
    exec(compile(src, "<grade>", "exec"), g)
    print(json.dumps({"ok": True}))
except TimeoutError as e:
    print(json.dumps({"ok": False, "err": "timeout"}))
except Exception as e:
    print(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc()[-500:]}))
'''


def grade_humaneval(prompt_info: Dict, generated_text: str) -> Dict[str, Any]:
    """Grade a HumanEval completion via subprocess exec."""
    raw_prompt = prompt_info["raw_prompt"]
    test_code  = prompt_info["test"]
    ep         = prompt_info["entry_point"]

    # Extract generated function body from output.
    # The model may emit a full function definition or just a body.
    generated_text = generated_text.strip()

    # Look for last ```python ... ``` block
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", generated_text, re.DOTALL)
    code = fences[-1].strip() if fences else generated_text

    # Check if the generated code defines the entry function
    func_re = re.compile(
        rf"^(def\s+{re.escape(ep)}\s*\(.*?\)[^:]*:)",
        re.MULTILINE | re.DOTALL,
    )
    m = func_re.search(code)
    if m:
        # Model produced full function — use it replacing the prompt's stub
        src = code + "\n\n" + test_code + f"\ncheck({ep})\n"
    else:
        # Treat as function body to append after the prompt signature
        src = raw_prompt + "\n" + code + "\n\n" + test_code + f"\ncheck({ep})\n"

    try:
        p = subprocess.run(
            [sys.executable, "-c", _GRADE_RUNNER, str(EXEC_TIMEOUT_SECS)],
            input=src.encode(),
            capture_output=True,
            timeout=EXEC_TIMEOUT_SECS + 5,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "subprocess-timeout", "predicted": generated_text[:200], "expected": "pass"}

    out_lines = (p.stdout or b"").decode().strip().splitlines()
    if not out_lines:
        stderr_tail = (p.stderr or b"")[-200:].decode(errors="replace")
        return {"ok": False, "err": f"no-output stderr={stderr_tail!r}", "predicted": generated_text[:200], "expected": "pass"}
    try:
        result = json.loads(out_lines[-1])
    except Exception:
        result = {"ok": False, "err": f"bad-json: {out_lines[-1]!r}"}
    result["predicted"] = generated_text[:300]
    result["expected"]  = "pass"
    return result


def grade_gsm8k(prompt_info: Dict, generated_text: str) -> Dict[str, Any]:
    gold = prompt_info["gold_answer"]
    pred = _extract_gsm8k_pred(generated_text)
    if gold is None:
        return {"ok": False, "err": "no-gold", "predicted": pred, "expected": gold}
    if pred is None:
        return {"ok": False, "err": "no-prediction", "predicted": None, "expected": gold}
    # Normalize: strip commas, compare as floats when possible
    ok = False
    try:
        ok = abs(float(gold) - float(pred)) < 1e-6
    except (ValueError, TypeError):
        ok = gold.strip() == pred.strip()
    return {"ok": ok, "predicted": pred, "expected": gold, "raw_output": generated_text[:200]}


def grade_math500(prompt_info: Dict, generated_text: str) -> Dict[str, Any]:
    gold = prompt_info["gold_answer"]
    pred_raw = _extract_boxed(generated_text)
    if pred_raw is None:
        # Fallback: last number in output
        nums = re.findall(r"-?[0-9]+(?:[.,][0-9]+)?", generated_text)
        pred_raw = nums[-1] if nums else None
    if pred_raw is None:
        return {"ok": False, "err": "no-boxed-answer", "predicted": None, "expected": gold}
    norm_gold = _normalize_math(gold)
    norm_pred = _normalize_math(pred_raw)
    ok = norm_gold == norm_pred
    return {"ok": ok, "predicted": pred_raw, "expected": gold, "norm_gold": norm_gold, "norm_pred": norm_pred}


GRADERS = {
    "humaneval": grade_humaneval,
    "gsm8k":     grade_gsm8k,
    "math500":   grade_math500,
}

# ---------------------------------------------------------------------------
# Speculator CLI builders — match exact flags from matrix speculators
# ---------------------------------------------------------------------------

def _make_env_ar_mtp() -> Dict[str, str]:
    """Clean env for AR / MTP runs."""
    env = {
        "HOME":             os.environ.get("HOME", ""),
        "PATH":             "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH":  "/usr/lib/wsl/lib:/usr/local/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "BENCH_THINKING":   "1",
    }
    return env


def _make_env_dflash() -> Dict[str, str]:
    """Clean env for DFlash runs (adds DFLASH27B_DRAFT_SWA)."""
    env = _make_env_ar_mtp()
    env["DFLASH27B_DRAFT_SWA"] = "2048"
    return env


def build_cmd_ar(prompt_bin: str, out_path: str, n_gen: int = 256, max_ctx: int = 4096) -> List[str]:
    return [
        DFLASH_BIN, MTP_GGUF,
        "--mtp-gguf", MTP_GGUF,
        "--prompt-bin", prompt_bin,
        "--n-gen", str(n_gen),
        "--gamma", "0",
        "--prompt-id", "0",
        "--max-ctx", str(max_ctx),
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--draft-source", "chain",
        "--out", out_path,
    ]


def build_cmd_mtp(prompt_bin: str, out_path: str, n_gen: int = 256, max_ctx: int = 4096) -> List[str]:
    return [
        DFLASH_BIN, MTP_GGUF,
        "--mtp-gguf", MTP_GGUF,
        "--prompt-bin", prompt_bin,
        "--n-gen", str(n_gen),
        "--gamma", "3",
        "--prompt-id", "0",
        "--max-ctx", str(max_ctx),
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--draft-source", "chain",
        "--chain-depth", "3",
        "--out", out_path,
    ]


def build_cmd_dflash(prompt_bin: str, out_path: str, n_gen: int = 256, max_ctx: int = 4096) -> List[str]:
    # DFlash positional: test_dflash <target> <draft> <prompt_bin> <n_gen> <out_path> --flags
    return [
        DFLASH_BIN, DFLASH_TARGET, DFLASH_DRAFT,
        prompt_bin, str(n_gen), out_path,
        "--max-ctx", str(max_ctx),
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--fast-rollback", "--ddtree",
        "--ddtree-budget=22",
    ]


SPECULATOR_CMDS = {
    "ar":         (build_cmd_ar,     _make_env_ar_mtp),
    "mtp_d3":     (build_cmd_mtp,    _make_env_ar_mtp),
    "dflash_b22": (build_cmd_dflash, _make_env_dflash),
}

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_one(speculator: str, prompt_info: Dict, out_path: str, n_gen: int = 256) -> Optional[List[int]]:
    """Run test_dflash for one (speculator, prompt) pair. Returns token IDs or None on error."""
    cmd_builder, env_builder = SPECULATOR_CMDS[speculator]
    cmd = cmd_builder(prompt_info["bin_path"], out_path, n_gen=n_gen)
    env = env_builder()

    log_debug(f"  cmd: {' '.join(cmd)}")
    log(f"  run {speculator} prompt={prompt_info['task_id']} ...")

    time.sleep(SLEEP_BETWEEN_RUNS)

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT ({SUBPROCESS_TIMEOUT}s) for {speculator} / {prompt_info['task_id']}")
        return None

    log_debug(f"  returncode={r.returncode}")
    log_debug(f"  stdout tail: {r.stdout[-500:]!r}")
    if r.returncode != 0:
        log(f"  FAIL rc={r.returncode} for {speculator} / {prompt_info['task_id']}")
        log_debug(f"  stderr: {r.stderr[-500:]!r}")
        return None

    if not Path(out_path).exists():
        log(f"  no output file written by {speculator} / {prompt_info['task_id']}")
        return None

    return read_token_bin(out_path)


CellResult = Dict[str, Any]


def run_workload(workload: str, prompts: List[Dict], speculator: str, n_gen: int = 256) -> List[CellResult]:
    """Run all prompts for one (workload, speculator) pair. Returns per-prompt results."""
    grader = GRADERS[workload]
    results: List[CellResult] = []

    for p in prompts:
        out_path = str(TMPDIR / f"{workload}_{speculator}_{p['idx']:03d}.out.bin")
        token_ids = run_one(speculator, p, out_path)

        if token_ids is None:
            results.append({
                "task_id": p["task_id"],
                "ok": False,
                "err": "run-failed",
                "predicted": None,
                "expected": p.get("gold_answer"),
            })
            continue

        try:
            generated_text = detokenize_ids(token_ids)
        except Exception as e:
            log(f"  detokenize error: {e}")
            results.append({
                "task_id": p["task_id"],
                "ok": False,
                "err": f"detokenize: {e}",
                "predicted": None,
                "expected": p.get("gold_answer"),
            })
            continue

        log_debug(f"  generated ({len(token_ids)} toks): {generated_text[:200]!r}")

        try:
            grade = grader(p, generated_text)
        except Exception as e:
            log(f"  grader error: {e}")
            grade = {"ok": False, "err": f"grader-exception: {e}"}

        grade["task_id"] = p["task_id"]
        grade.setdefault("predicted", generated_text[:300])
        grade.setdefault("expected", p.get("gold_answer"))
        results.append(grade)

        status = "PASS" if grade.get("ok") else "FAIL"
        log(f"    {status} | expected={grade.get('expected')!r} | predicted={grade.get('predicted', '')[:80]!r}")

    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _summary_table(workload_label: str, metric: str, all_results: Dict[str, List[CellResult]]) -> str:
    lines = [f"## {workload_label} {metric}", ""]
    lines.append("| speculator | passed | total | rate |")
    lines.append("| ---------- | ------ | ----- | ---- |")
    for spec, results in all_results.items():
        passed = sum(1 for r in results if r.get("ok"))
        total  = len(results)
        rate   = f"{100 * passed / total:.0f}%" if total else "N/A"
        lines.append(f"| {spec} | {passed} | {total} | {rate} |")
    lines.append("")
    return "\n".join(lines)


def _detail_table(workload: str, all_results: Dict[str, List[CellResult]]) -> str:
    lines = [f"### {workload} per-cell detail", ""]
    lines.append("| speculator | prompt_id | expected | predicted | pass |")
    lines.append("| ---------- | --------- | -------- | --------- | ---- |")
    for spec, results in all_results.items():
        for r in results:
            ok       = "YES" if r.get("ok") else "NO"
            expected = str(r.get("expected", ""))[:40].replace("|", "\\|")
            predicted = str(r.get("predicted", ""))[:60].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {spec} | {r.get('task_id', '?')} | {expected} | {predicted} | {ok} |")
    lines.append("")
    return "\n".join(lines)


def _quality_vs_throughput(full_results: Dict[str, Dict[str, List[CellResult]]]) -> str:
    lines = ["## Quality vs throughput trade-off", ""]
    regressions: List[str] = []
    # ar is the baseline; flag if mtp_d3 or dflash_b22 has lower pass rate
    for workload, spec_results in full_results.items():
        ar_results = spec_results.get("ar", [])
        ar_passed  = sum(1 for r in ar_results if r.get("ok"))
        ar_total   = len(ar_results)
        ar_rate    = ar_passed / ar_total if ar_total else 0.0

        for spec, results in spec_results.items():
            if spec == "ar":
                continue
            passed = sum(1 for r in results if r.get("ok"))
            total  = len(results)
            rate   = passed / total if total else 0.0
            delta  = rate - ar_rate
            lines.append(f"- **{workload} / {spec}**: {passed}/{total} ({100*rate:.0f}%) vs AR {ar_passed}/{ar_total} ({100*ar_rate:.0f}%) — delta={delta:+.0%}")
            if delta < -0.15:
                regressions.append(f"{workload}/{spec} dropped {-delta:.0%} below AR")
                # List specific failures not in AR
                ar_fail_ids = {r["task_id"] for r in ar_results if not r.get("ok")}
                spec_fail_ids = {r["task_id"] for r in results if not r.get("ok")}
                new_failures = spec_fail_ids - ar_fail_ids
                if new_failures:
                    lines.append(f"  - New failures (passed AR but failed {spec}): {sorted(new_failures)}")

    lines.append("")
    if regressions:
        lines.append("**QUALITY REGRESSIONS DETECTED:**")
        for reg in regressions:
            lines.append(f"- {reg}")
    else:
        lines.append("No significant quality regressions detected (all speculators within 15% of AR).")
    lines.append("")
    return "\n".join(lines)


def render_report(
    full_results: Dict[str, Dict[str, List[CellResult]]],
    n_sample: int,
    date_str: str,
) -> str:
    parts: List[str] = []

    parts.append(f"# Quality validation — {date_str}")
    parts.append("")
    parts.append(f"GPU: RTX 3090 (sm_86) @ 301W, Qwen3.6-27B Q4_K_M, n={n_sample} per cell")
    parts.append("")

    workload_labels = {
        "humaneval": ("HumanEval pass@1", "pass@1"),
        "gsm8k":     ("GSM8K exact_match", "exact_match"),
        "math500":   ("Math500 exact_match", "exact_match"),
    }

    for workload, (label, metric) in workload_labels.items():
        if workload in full_results:
            parts.append(_summary_table(label, "", full_results[workload]))

    parts.append("## Per-cell detail")
    parts.append("")
    for workload in workload_labels:
        if workload in full_results:
            parts.append(_detail_table(workload, full_results[workload]))

    parts.append(_quality_vs_throughput(full_results))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Smoke test — verify grader logic on synthetic correct outputs
# ---------------------------------------------------------------------------

def smoke_test_graders() -> bool:
    """Run 5 synthetic grader checks. Returns True if all pass."""
    log("Running grader smoke tests ...")
    all_ok = True

    # HumanEval: correct simple function
    he_sample = {
        "raw_prompt": "def add(a, b):\n    ",
        "test": "def check(add):\n    assert add(1, 2) == 3\n    assert add(0, 0) == 0",
        "entry_point": "add",
        "gold_answer": None,
    }
    he_output = "def add(a, b):\n    return a + b\n"
    r = grade_humaneval(he_sample, he_output)
    ok = r.get("ok")
    log(f"  [smoke HE-correct] ok={ok}  (expected True)")
    if not ok:
        all_ok = False

    # HumanEval: wrong function
    he_output_bad = "def add(a, b):\n    return a - b\n"
    r = grade_humaneval(he_sample, he_output_bad)
    ok = not r.get("ok")
    log(f"  [smoke HE-wrong]   fail-detected={ok}  (expected True)")
    if not ok:
        all_ok = False

    # GSM8K: correct
    gsm_sample = {"gold_answer": "42", "raw_prompt": ""}
    r = grade_gsm8k(gsm_sample, "The answer is #### 42")
    ok = r.get("ok")
    log(f"  [smoke GSM-correct] ok={ok}  (expected True)")
    if not ok:
        all_ok = False

    # GSM8K: wrong
    r = grade_gsm8k(gsm_sample, "The answer is #### 99")
    ok = not r.get("ok")
    log(f"  [smoke GSM-wrong]   fail-detected={ok}  (expected True)")
    if not ok:
        all_ok = False

    # Math500: correct boxed
    math_sample = {"gold_answer": "\\frac{1}{2}", "raw_prompt": ""}
    r = grade_math500(math_sample, r"The answer is $\boxed{\frac{1}{2}}$.")
    ok = r.get("ok")
    log(f"  [smoke MATH-correct] ok={ok}  (expected True) norm_gold={r.get('norm_gold')!r} norm_pred={r.get('norm_pred')!r}")
    if not ok:
        all_ok = False

    log(f"Smoke tests: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Quality validation matrix for dflash speculators")
    ap.add_argument("--workloads",    default="humaneval,gsm8k,math500",
                    help="Comma-separated workloads to run")
    ap.add_argument("--speculators",  default="ar,mtp_d3,dflash_b22",
                    help="Comma-separated speculators to run")
    ap.add_argument("--n-sample",     type=int, default=N_SAMPLE)
    ap.add_argument("--n-gen",        type=int, default=256)
    ap.add_argument("--seed",         type=int, default=SEED)
    ap.add_argument("--out",          default=str(OUT_MD), help="Output markdown path")
    ap.add_argument("--smoke-only",   action="store_true",
                    help="Only run grader smoke tests (no model invocations)")
    ap.add_argument("--skip-smoke",   action="store_true",
                    help="Skip grader smoke tests")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    log(f"quality_matrix.py — {datetime.date.today()}")
    log(f"MTP_GGUF:      {MTP_GGUF}")
    log(f"DFLASH_TARGET: {DFLASH_TARGET}")
    log(f"DFLASH_DRAFT:  {DFLASH_DRAFT}")
    log(f"DFLASH_BIN:    {DFLASH_BIN}")
    log(f"TOKENIZER:     {TOKENIZER_ID}")

    # Grader smoke tests
    if not args.skip_smoke:
        ok = smoke_test_graders()
        if not ok:
            log("ERROR: grader smoke tests failed — fix graders before running model")
            return 1
    if args.smoke_only:
        log("--smoke-only: done")
        return 0

    # Validate binary exists
    if not Path(DFLASH_BIN).exists():
        log(f"ERROR: DFLASH_BIN not found: {DFLASH_BIN}")
        return 1

    workloads   = [w.strip() for w in args.workloads.split(",") if w.strip()]
    speculators = [s.strip() for s in args.speculators.split(",") if s.strip()]
    n_sample    = args.n_sample
    seed        = args.seed
    n_gen       = args.n_gen

    log(f"Workloads:   {workloads}")
    log(f"Speculators: {speculators}")
    log(f"n_sample={n_sample}, seed={seed}, n_gen={n_gen}")

    # Load tokenizer once
    tok = _get_tokenizer()
    if tok is None:
        log("ERROR: HF tokenizer unavailable — cannot build prompts")
        return 1

    # Prompt builders keyed by workload name
    prompt_builders = {
        "humaneval": (_load_humaneval, build_prompts_humaneval),
        "gsm8k":     (_load_gsm8k,    build_prompts_gsm8k),
        "math500":   (_load_math500,  build_prompts_math500),
    }

    full_results: Dict[str, Dict[str, List[CellResult]]] = {}

    for workload in workloads:
        if workload not in prompt_builders:
            log(f"Unknown workload: {workload}, skipping")
            continue

        log(f"\n=== Workload: {workload} ===")
        load_fn, build_fn = prompt_builders[workload]

        log(f"Loading dataset ({workload}) ...")
        try:
            samples = load_fn(n_sample=n_sample, seed=seed)
        except Exception as e:
            log(f"ERROR loading {workload}: {e}")
            traceback.print_exc()
            continue

        log(f"Building prompts ({workload}) ...")
        workload_tmpdir = TMPDIR / workload
        workload_tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            prompts = build_fn(samples, workload_tmpdir)
        except Exception as e:
            log(f"ERROR building prompts for {workload}: {e}")
            traceback.print_exc()
            continue

        full_results[workload] = {}

        for spec in speculators:
            if spec not in SPECULATOR_CMDS:
                log(f"Unknown speculator: {spec}, skipping")
                continue
            log(f"\n--- {workload} / {spec} ---")
            results = run_workload(workload, prompts, spec, n_gen=n_gen)
            full_results[workload][spec] = results
            passed = sum(1 for r in results if r.get("ok"))
            log(f"  {spec}: {passed}/{len(results)} passed")

    # Render and write report
    date_str = str(datetime.date.today())
    report = render_report(full_results, n_sample=n_sample, date_str=date_str)

    out_path = Path(args.out)
    out_path.write_text(report)
    log(f"\nReport written to: {out_path}")
    log(f"Debug log:         {DEBUG_LOG}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
