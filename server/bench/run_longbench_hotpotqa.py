#!/usr/bin/env python3
"""LongBench hotpotqa bench: F1 scoring against real multi-hop QA cases."""
import argparse
import json
import os
import re
import string
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import requests

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET = Path(os.environ.get("TARGET", "/path/to/your/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"))
DRAFTER = Path(os.environ.get("PFLASH_DRAFT", "/path/to/your/models/Qwen3-0.6B-Q8_0.gguf"))
PORT = 18099
BASE_URL = f"http://127.0.0.1:{PORT}"

CONDITION_SPECS = {"ee7": (7, 7)}  # (EARLY_EXIT_N, SCORE_LAYERS)

DEFAULT_DATA = Path("/tmp/longbench_hotpotqa.jsonl")


# --- F1 scoring (LongBench reference implementation) ---

def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred, gt):
    pred_tokens = normalize_answer(pred).split()
    gt_tokens = normalize_answer(gt).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred, answers):
    """Best F1 over all gold answers (LongBench convention)."""
    if not answers:
        return 0.0
    return max(f1_score(pred, gt) for gt in answers)


# --- Server lifecycle ---

def start_server(condition, log_path, compression_mode="always", keep_ratio=0.10):
    srv_compression = "off" if compression_mode == "none" else compression_mode
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env.pop("PFLASH_DRAFTER_EARLY_EXIT_N", None)
    env.pop("PFLASH_DRAFTER_SCORE_LAYERS", None)
    if condition in CONDITION_SPECS:
        ee_n, sl = CONDITION_SPECS[condition]
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = str(ee_n)
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = str(sl)
    # Pass through anchor-transitive / bandit / gated-cascade env flags if set
    for k in (
        "PFLASH_COMPRESS_ANCHOR_TRANSITIVE",
        "PFLASH_COMPRESS_ANCHOR_MAX_ITERS",
        "PFLASH_COMPRESS_RARE_MAX_FREQ",
        "PFLASH_COMPRESS_ANCHOR_NGRAM",
        "PFLASH_COMPRESS_CASCADE_MIN_ANCHOR_FRAC",
        "PFLASH_COMPRESS_MAX_FORCED_RATIO",
        "DFLASH_BANDIT_KEEP",
        "DFLASH_BANDIT_ENABLED",
    ):
        if k in os.environ:
            env[k] = os.environ[k]

    DECODE_DRAFT = os.environ.get(
        "DECODE_DRAFT",
        os.environ.get("DECODE_DRAFT", "/path/to/your/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf"),
    )
    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--draft", DECODE_DRAFT,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "139264",
    ]
    # Server default is OFF; passing "off" explicitly triggers a parse error in
    # this binary version — omit compression flags entirely for baseline runs.
    if srv_compression != "off":
        cmd += [
            "--prefill-compression", srv_compression,
            "--prefill-keep-ratio", str(keep_ratio),
            "--prefill-drafter", str(DRAFTER),
            "--lazy-draft",
        ]
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc, timeout=180):
    for _ in range(timeout):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
        if proc.poll() is not None:
            return False
    return False


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(3)


def extract_metrics_from_log(log_path):
    metrics = {"drafter_score_s": None, "tail_score_s": None, "ggml_assert_count": 0}
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\[drafter\]\s+forward\+score in ([\d.]+)s", line)
                if m:
                    metrics["drafter_score_s"] = float(m.group(1))
                m2 = re.search(r"tail.?score\s+([\d.]+)s", line, re.IGNORECASE)
                if m2:
                    metrics["tail_score_s"] = float(m2.group(1))
                if "ggml_view_3d" in line and "assert" in line.lower():
                    metrics["ggml_assert_count"] += 1
    except Exception:
        pass
    return metrics


# --- Per-case execution ---

def run_one_case(condition, case, case_idx, results_dir, compression_mode, keep_ratio):
    log_path = results_dir / f"{condition}_case{case_idx}_server.log"
    proc = start_server(condition, log_path, compression_mode=compression_mode, keep_ratio=keep_ratio)
    result = {
        "id": case.get("id", f"row_{case_idx}"),
        "answers": case["answers"],
        "latency_s": None,
        "text": "",
        "f1": 0.0,
        "error": None,
        "drafter_score_s": None,
        "tail_score_s": None,
        "ggml_assert_count": 0,
    }
    try:
        if not wait_server(proc, timeout=180):
            tail = ""
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-30:])
            except Exception:
                pass
            result["error"] = f"server_start_failed: {tail[:500]}"
            return result

        payload = {
            "model": "dflash",
            "messages": [{"role": "user", "content": case["input"]}],
            "max_tokens": 128,
            "stream": False,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=600)
            result["latency_s"] = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            result["text"] = text[:400]
            result["f1"] = best_f1(text, case["answers"])
        except Exception as e:
            result["latency_s"] = time.perf_counter() - t0
            result["error"] = str(e)
    finally:
        stop_server(proc)

    metrics = extract_metrics_from_log(log_path)
    result.update(metrics)
    return result


def run_condition(condition, cases, results_dir, compression_mode, keep_ratio):
    print(f"\n[bench] condition={condition} compression={compression_mode} keep={keep_ratio} ({len(cases)} cases)", flush=True)
    case_results = []
    for i, case in enumerate(cases):
        print(f"  case {i}: id={case.get('id')} length={case.get('length')} answers={case['answers'][:2]}", flush=True)
        r = run_one_case(condition, case, i, results_dir, compression_mode, keep_ratio)
        case_results.append(r)
        latency_s = f"{r['latency_s']:.2f}s" if r["latency_s"] is not None else "N/A"
        print(f"  case {i}: latency={latency_s} f1={r['f1']:.3f} resp={r['text'][:80]!r}", flush=True)
        if r["error"]:
            print(f"  case {i}: error={r['error'][:200]}", flush=True)
        if r.get("ggml_assert_count", 0) > 0:
            print(f"  [STOP] ggml_view_3d assert in case {i} — Bug #42 regression!", flush=True)
            sys.exit(1)

    f1_scores = [r["f1"] for r in case_results]
    drafter_times = [r["drafter_score_s"] for r in case_results if r["drafter_score_s"] is not None]
    return {
        "condition": condition,
        "compression_mode": compression_mode,
        "keep_ratio": keep_ratio,
        "case_results": case_results,
        "mean_f1": mean(f1_scores) if f1_scores else 0.0,
        "drafter_times_s": drafter_times,
        "drafter_mean_s": mean(drafter_times) if drafter_times else None,
        "ggml_asserts": sum(r.get("ggml_assert_count", 0) for r in case_results),
    }


def write_summary(all_results, results_dir):
    lines = []
    lines.append("# LongBench hotpotqa — pflash compression F1 bench\n\n")
    lines.append("Scoring: best-of-answers F1 (LongBench reference).\n\n")
    lines.append("## Results\n\n")
    lines.append("| condition | compression | keep | mean_f1 | drafter_mean_s | ggml_asserts |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for r in all_results:
        dm = f"{r['drafter_mean_s']:.3f}s" if r.get("drafter_mean_s") else "N/A"
        lines.append(
            f"| {r['condition']} | {r['compression_mode']} | {r['keep_ratio']} "
            f"| {r['mean_f1']:.3f} | {dm} | {r['ggml_asserts']} |\n"
        )

    lines.append("\n## Failure Examples\n\n")
    for r in all_results:
        failed = [cr for cr in r["case_results"] if cr["f1"] < 0.5]
        if failed:
            lines.append(f"### {r['condition']} (keep={r['keep_ratio']})\n\n")
            for cr in failed[:5]:
                lines.append(f"- id={cr['id']} f1={cr['f1']:.3f} answers={cr['answers']} resp={cr['text'][:120]!r}\n")

    summary_path = results_dir / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.writelines(lines)
    print(f"\n[done] {summary_path}", flush=True)
    print("".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="path to longbench_hotpotqa.jsonl")
    ap.add_argument("--out-dir", default="dflash/bench/results/2026-05-25_longbench_hotpotqa")
    ap.add_argument("--max-cases", type=int, default=50, help="cap cases for shorter runs")
    ap.add_argument("--compression-mode", default="always")
    ap.add_argument("--keep-ratio", type=float, default=0.10)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"[error] data file not found: {data_path}  -- run download step first")

    with open(data_path) as f:
        cases = [json.loads(line) for line in f]
    cases = cases[: args.max_cases]
    if not cases:
        sys.exit(f"[error] no cases loaded from {data_path}")
    print(f"[init] {len(cases)} cases from {data_path}", flush=True)
    print(f"[init] length range: {min(c['length'] for c in cases)}-{max(c['length'] for c in cases)}", flush=True)

    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    # Arm 1: baseline (decode-only, no prefill compression)
    r_base = run_condition("baseline", cases, results_dir, "none", args.keep_ratio)
    all_results.append(r_base)
    with open(results_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    # Arm 2: ee7 (prefill compression + cascade)
    for condition in CONDITION_SPECS:
        r = run_condition(condition, cases, results_dir, args.compression_mode, args.keep_ratio)
        all_results.append(r)
        with open(results_dir / "raw_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    write_summary(all_results, results_dir)


if __name__ == "__main__":
    main()
