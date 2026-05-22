#!/usr/bin/env python3
"""
Qwen3-Reranker-0.6B Q8_0 vs Qwen3-0.6B-BF16 as PFlash scoring drafter.
NIAH@32K, keep_ratio in {0.05, 0.10, 0.20}, ee7 fixed.
ONE SERVER PER CASE (server crashes on 2nd request due to ggml view bug).
"""
import json
import os
import subprocess
import sys
import time
import re
import requests
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DRAFTER_BF16 = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
DRAFTER_RERANKER = Path("/home/peppi/models/Qwen3-Reranker-0.6B-Q8_0.gguf")
PORT = 18102
BASE_URL = f"http://127.0.0.1:{PORT}"
CTX = 32768
MAX_CTX = 36864
NIAH_FILE = Path("/tmp/niah_reranker.jsonl")
OUT_DIR = REPO / "dflash/bench/results/2026-05-22_reranker_drafter"

CONDITIONS = [
    {"name": "baseline_qwen3",    "drafter": DRAFTER_BF16,     "keep": 0.05, "ee": 7},
    {"name": "reranker_keep05",   "drafter": DRAFTER_RERANKER,  "keep": 0.05, "ee": 7},
    {"name": "reranker_keep10",   "drafter": DRAFTER_RERANKER,  "keep": 0.10, "ee": 7},
    {"name": "reranker_keep20",   "drafter": DRAFTER_RERANKER,  "keep": 0.20, "ee": 7},
]


def start_server(cond, log_path):
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env["DFLASH_DRAFTER_EARLY_EXIT_N"] = str(cond["ee"])
    env["DFLASH_DRAFTER_SCORE_LAYERS"] = str(cond["ee"])
    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--prefill-compression", "always",
        "--prefill-keep-ratio", str(cond["keep"]),
        "--prefill-drafter", str(cond["drafter"]),
    ]
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc, timeout=120):
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
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(2)


def run_one_case(cond, case, case_idx, results_dir):
    log_path = results_dir / f"{cond['name']}_case{case_idx}_server.log"
    proc = start_server(cond, log_path)
    result = {
        "ttft_s": None, "text": "", "found": False, "error": None,
        "drafter_fwd_s": None, "tail_score_s": None, "embed_error": False,
    }
    try:
        if not wait_server(proc):
            tail = ""
            try:
                tail = open(log_path).read()[-500:]
            except Exception:
                pass
            result["error"] = f"server_start_failed: {tail}"
            return result

        payload = {
            "model": "dflash",
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 64,
            "stream": False,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=300)
            result["ttft_s"] = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            result["text"] = text[:300]
            result["found"] = case["answer"] in text
        except Exception as e:
            result["ttft_s"] = time.perf_counter() - t0
            result["error"] = str(e)
    finally:
        stop_server(proc)

    try:
        log_content = open(log_path).read()
        for line in log_content.splitlines():
            m = re.search(r"\[drafter\] forward\+score in ([\d.]+)s", line)
            if m:
                result["drafter_fwd_s"] = float(m.group(1))
            m2 = re.search(r"tail-score ([\d.]+)s", line)
            if m2:
                result["tail_score_s"] = float(m2.group(1))
        if "embed" in log_content and ("false" in log_content.lower() or "error" in log_content.lower()):
            result["embed_error"] = True
    except Exception:
        pass

    return result


def run_condition(cond, cases):
    print(f"\n[bench] {cond['name']} drafter={Path(cond['drafter']).name} keep={cond['keep']} ee={cond['ee']}", flush=True)
    case_results = []
    for i, case in enumerate(cases):
        print(f"  case {i}: ans={case.get('answer','?')}", flush=True)
        r = run_one_case(cond, case, i, OUT_DIR)
        case_results.append(r)
        status = "OK" if r["found"] else "FAIL"
        drafter_s = f"{r['drafter_fwd_s']:.3f}s" if r["drafter_fwd_s"] else "N/A"
        tail_s = f"{r['tail_score_s']:.3f}s" if r["tail_score_s"] else "N/A"
        embed_warn = " EMBED_ERR" if r["embed_error"] else ""
        print(f"  case {i}: ttft={r['ttft_s']:.2f}s drafter={drafter_s} tail={tail_s} [{status}]{embed_warn}", flush=True)
        if r["text"]:
            print(f"  case {i}: text={r['text'][:80]!r}", flush=True)
        if r["error"]:
            print(f"  case {i}: error={r['error'][:200]}", flush=True)

    drafter_times = [r["drafter_fwd_s"] for r in case_results if r["drafter_fwd_s"] is not None]
    tail_times = [r["tail_score_s"] for r in case_results if r["tail_score_s"] is not None]
    niah_pass = sum(1 for c in case_results if c["found"])

    return {
        "condition": cond["name"],
        "drafter": str(cond["drafter"].name),
        "keep": cond["keep"],
        "case_results": case_results,
        "drafter_p50_s": median(drafter_times) if drafter_times else None,
        "tail_score_p50_s": median(tail_times) if tail_times else None,
        "niah_pass": niah_pass,
        "niah_total": len(cases),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [json.loads(l) for l in NIAH_FILE.read_text().splitlines() if l.strip()]
    print(f"[bench] Loaded {len(cases)} NIAH cases from {NIAH_FILE}", flush=True)

    all_results = []
    for cond in CONDITIONS:
        res = run_condition(cond, cases)
        all_results.append(res)
        # Save partial results
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n=== RESULTS TABLE ===")
    print(f"{'Condition':<22} {'Drafter':<35} {'keep':>5} {'drafter_p50':>12} {'tail_p50':>10} {'NIAH':>6}")
    print("-" * 100)
    for r in all_results:
        drafter_s = f"{r['drafter_p50_s']:.3f}s" if r["drafter_p50_s"] else "N/A"
        tail_s = f"{r['tail_score_p50_s']:.3f}s" if r["tail_score_p50_s"] else "N/A"
        niah = f"{r['niah_pass']}/{r['niah_total']}"
        print(f"{r['condition']:<22} {r['drafter']:<35} {r['keep']:>5} {drafter_s:>12} {tail_s:>10} {niah:>6}")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[bench] Results saved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
