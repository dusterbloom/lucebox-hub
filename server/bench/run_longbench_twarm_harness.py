#!/usr/bin/env python3
"""
LongBench hotpotqa two-arm bench (baseline vs pflash+cascade) via direct HTTP.

Runs 50 cases per arm. Uses the same F1 scorer as run_longbench_hotpotqa.py.
Outputs aggregated metrics in the required format.

Usage:
  python3 run_longbench_twarm_harness.py --out-dir /path/to/out
"""
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
from statistics import mean, median

import requests

REPO = Path(__file__).resolve().parents[2]
BINARY = os.environ.get("DFLASH_SERVER_BIN", "/path/to/your/Dev/lucebox-hub/dflash/build/dflash_server")
TARGET = os.environ.get("TARGET", "/path/to/your/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DECODE_DRAFT = os.environ.get("DECODE_DRAFT", "/path/to/your/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf")
PFLASH_DRAFTER = os.environ.get("PFLASH_DRAFT", "/path/to/your/models/Qwen3-0.6B-Q8_0.gguf")
PORT = 19099
BASE_URL = f"http://127.0.0.1:{PORT}"
DEFAULT_DATA = Path("/tmp/longbench_hotpotqa.jsonl")


# --- F1 scoring (LongBench reference) ---

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
    if not answers:
        return 0.0
    return max(f1_score(pred, gt) for gt in answers)


# --- Server lifecycle ---

def kill_port(port: int) -> None:
    try:
        r = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
        pids = r.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), 15)
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(3)
            r2 = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
            for pid in r2.stdout.strip().split():
                try:
                    os.kill(int(pid), 9)
                except ProcessLookupError:
                    pass
            time.sleep(1)
    except FileNotFoundError:
        subprocess.run(
            f"pkill -f 'dflash_server.*{port}' || true", shell=True, capture_output=True
        )
        time.sleep(3)


def start_server(arm_label: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"

    cmd = [
        BINARY, TARGET,
        "--draft", DECODE_DRAFT,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "139264",
        "--max-tokens", "256",
        "--model-name", "luce-dflash",
        "--ddtree",
        "--ddtree-budget", "16",
    ]
    if arm_label == "pflash":
        cmd += [
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.05",
            "--prefill-drafter", PFLASH_DRAFTER,
            "--lazy-draft",
        ]
        env["PFLASH_DRAFTER_EARLY_EXIT_N"] = "7"
        env["PFLASH_DRAFTER_SCORE_LAYERS"] = "7"
        env["PFLASH_COMPRESS_ANCHOR_TRANSITIVE"] = "1"

    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    print(f"[lb] server PID={proc.pid} arm={arm_label}", flush=True)
    return proc


def wait_server(proc: subprocess.Popen, log_path: Path, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    health_ok = False
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                health_ok = True
                break
        except Exception:
            pass
        time.sleep(1)
        if proc.poll() is not None:
            return False
    if not health_ok:
        return False
    # Wait for model to finish loading
    dl2 = time.time() + 300
    while time.time() < dl2:
        if proc.poll() is not None:
            return False
        if log_path.exists() and "listening on" in log_path.read_text():
            return True
        time.sleep(2)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(2)


def extract_server_metrics_all(log_path: Path) -> dict:
    """Extract all drafter, prefill, keep_ratio occurrences from server log."""
    result = {"drafter_score_s": [], "prefill_s": [], "keep_ratio": []}
    if not log_path.exists():
        return result
    text = log_path.read_text()
    for m in re.finditer(r"\[drafter\]\s+forward\+score in ([\d.]+)s", text):
        result["drafter_score_s"].append(float(m.group(1)))
    for m in re.finditer(r"prefill=([\d.]+)s", text):
        result["prefill_s"].append(float(m.group(1)))
    for m in re.finditer(r"\[pflash\] \d+ -> \d+ -> \d+ tokens \(([\d.]+)% kept\)", text):
        result["keep_ratio"].append(float(m.group(1)))
    return result


# --- Per-case execution ---

def run_one_case(case: dict, case_idx: int, arm_label: str, results_dir: Path) -> dict:
    result = {
        "id": case.get("id", f"row_{case_idx}"),
        "answers": case["answers"],
        "latency_s": None,
        "text": "",
        "f1": 0.0,
        "error": None,
    }
    payload = {
        "model": "luce-dflash",
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
    return result


def run_arm(arm_label: str, cases: list, results_dir: Path) -> dict:
    log_path = results_dir / f"server_{arm_label}.log"
    kill_port(PORT)

    print(f"\n[lb] === arm={arm_label} ({len(cases)} cases) ===", flush=True)
    proc = start_server(arm_label, log_path)

    if not wait_server(proc, log_path, timeout=300):
        tail = log_path.read_text()[-2000:] if log_path.exists() else ""
        print(f"[lb] server failed to start for {arm_label}:\n{tail}", flush=True)
        stop_server(proc)
        return {
            "arm": arm_label, "error": "server_start_failed",
            "mean_f1": None, "wall_mean_s": None, "prefill_mean_s": None,
            "drafter_mean_s": None, "keep_ratio_mean": None,
            "n_cases": 0, "case_results": [],
        }

    print(f"[lb] server ready for {arm_label}", flush=True)
    case_results = []
    try:
        for i, case in enumerate(cases):
            print(f"  [{arm_label}] case {i}/{len(cases)-1} id={case.get('id')} ans={case['answers'][:1]}", flush=True)
            cr = run_one_case(case, i, arm_label, results_dir)
            case_results.append(cr)
            lat = f"{cr['latency_s']:.2f}s" if cr["latency_s"] else "N/A"
            print(f"    f1={cr['f1']:.3f} latency={lat} resp={cr['text'][:60]!r}", flush=True)
            if cr.get("error"):
                print(f"    error={cr['error'][:100]}", flush=True)
    finally:
        stop_server(proc)
        kill_port(PORT)

    # Parse server metrics
    srv = extract_server_metrics_all(log_path)
    drafters = srv["drafter_score_s"]
    prefills = srv["prefill_s"]
    keeps = srv["keep_ratio"]

    f1_scores = [r["f1"] for r in case_results]
    latencies = [r["latency_s"] for r in case_results if r["latency_s"]]

    print(f"  [{arm_label}] drafter samples: {len(drafters)} prefill samples: {len(prefills)}", flush=True)

    return {
        "arm": arm_label,
        "n_cases": len(case_results),
        "mean_f1": mean(f1_scores) if f1_scores else None,
        "wall_mean_s": mean(latencies) if latencies else None,
        "wall_median_s": median(latencies) if latencies else None,
        "drafter_mean_s": mean(drafters) if drafters else None,
        "prefill_mean_s": mean(prefills) if prefills else None,
        "keep_ratio_mean": mean(keeps) if keeps else None,
        "case_results": case_results,
    }


def write_metrics(baseline: dict, pflash: dict, out_dir: Path) -> None:
    metrics_path = out_dir / "metrics_hotpotqa.txt"

    n = baseline.get("n_cases", 0)
    b_f1 = baseline.get("mean_f1")
    p_f1 = pflash.get("mean_f1")
    f1_delta = (p_f1 - b_f1) if (b_f1 is not None and p_f1 is not None) else None

    b_wall = baseline.get("wall_mean_s")
    p_wall = pflash.get("wall_mean_s")
    e2e_speedup = (b_wall / p_wall) if (b_wall and p_wall and p_wall > 0) else None

    b_prefill = baseline.get("prefill_mean_s")
    p_prefill = pflash.get("prefill_mean_s")
    prefill_speedup = (b_prefill / p_prefill) if (b_prefill and p_prefill and p_prefill > 0) else None

    keep = pflash.get("keep_ratio_mean")
    keep_str = f"{keep:.1f}%" if keep is not None else "N/A"

    lines = [
        f"n_cases={n}",
        f"prompt_tokens_p50=N/A  (variable per case, range ~8K-12K tokens)",
        "",
        "[baseline]",
        (f"F1={b_f1:.3f}" if b_f1 is not None else "F1=N/A") +
        (f"   e2e_wall_mean={b_wall:.1f}s" if b_wall else "   e2e_wall_mean=N/A") +
        (f"   prefill_mean={b_prefill:.2f}s" if b_prefill else "   prefill_mean=N/A"),
        "",
        "[pflash ee7 + cascade]",
        (f"F1={p_f1:.3f}" if p_f1 is not None else "F1=N/A") +
        (f"   e2e_wall_mean={p_wall:.1f}s" if p_wall else "   e2e_wall_mean=N/A") +
        (f"   prefill_mean={p_prefill:.2f}s" if p_prefill else "   prefill_mean=N/A") +
        f"   tokens_kept_mean={keep_str}",
        "",
        "[headline]",
        (f"F1_delta={f1_delta:+.3f}pp" if f1_delta is not None else "F1_delta=N/A") +
        (f"   e2e_speedup={e2e_speedup:.2f}x" if e2e_speedup else "   e2e_speedup=N/A") +
        (f"   prefill_speedup={prefill_speedup:.2f}x" if prefill_speedup else "   prefill_speedup=N/A"),
    ]
    metrics_path.write_text("\n".join(lines) + "\n")
    print(f"\n[lb] metrics -> {metrics_path}", flush=True)
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument(
        "--out-dir",
        default="/tmp/lucebox-bench-pr/server/bench/results/2026-05-27_full_harness/_longbench_hotpotqa",
    )
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"[error] data file not found: {data_path}")

    with open(data_path) as f:
        cases = [json.loads(l) for l in f]
    cases = cases[: args.limit]
    print(f"[lb] {len(cases)} cases from {data_path}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_arm("baseline", cases, out_dir)
    pflash = run_arm("pflash", cases, out_dir)

    # Save raw results
    raw = {
        "baseline": {k: v for k, v in baseline.items() if k != "case_results"},
        "pflash": {k: v for k, v in pflash.items() if k != "case_results"},
        "n_cases": len(cases),
    }
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(raw, f, indent=2)

    write_metrics(baseline, pflash, out_dir)

    print("\n[lb] === FINAL ===", flush=True)
    print(f"  baseline F1={baseline.get('mean_f1'):.3f}  pflash F1={pflash.get('mean_f1'):.3f}", flush=True)


if __name__ == "__main__":
    main()
