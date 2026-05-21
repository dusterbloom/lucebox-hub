#!/usr/bin/env python3
"""
NIAH broad-context bench: tests baseline vs ee14 at 1K/4K/8K/16K.
ONE SERVER INSTANCE PER CASE (server crashes on 2nd request due to ggml view bug).
Measures drafter_fwd from server log and answer correctness.
"""
import argparse
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
DRAFTER = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
PORT = 18094
BASE_URL = f"http://127.0.0.1:{PORT}"

CONTEXTS = [1024, 4096, 8192, 16384]
CONDITIONS = ["baseline", "ee14"]


def start_server(condition, ctx, log_path):
    max_ctx = 20000 if ctx >= 8192 else 12000
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env.pop("DFLASH_DRAFTER_EARLY_EXIT_N", None)
    env.pop("DFLASH_DRAFTER_SCORE_LAYERS", None)
    if condition == "ee14":
        env["DFLASH_DRAFTER_EARLY_EXIT_N"] = "14"

    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(max_ctx),
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", str(DRAFTER),
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


def run_one_case_with_server(condition, ctx, case, case_idx, results_dir):
    """Start a fresh server, run one NIAH case, stop server. Returns timing."""
    log_path = results_dir / f"{condition}_{ctx}_case{case_idx}_server.log"
    proc = start_server(condition, ctx, log_path)
    result = {"ttft_s": None, "text": "", "found": False, "error": None,
              "drafter_fwd_s": None}
    try:
        if not wait_server(proc):
            tail = ""
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-20:])
            except Exception:
                pass
            result["error"] = f"server_start_failed: {tail[:300]}"
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
            r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=180)
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

    # Extract drafter time from log
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\[drafter\] forward\+score in ([\d.]+)s", line)
                if m:
                    result["drafter_fwd_s"] = float(m.group(1))
                    break
    except Exception:
        pass

    return result


def run_condition_ctx(condition, ctx, cases, results_dir):
    print(f"\n[bench] condition={condition} ctx={ctx} ({len(cases)} cases, one server per case)", flush=True)
    case_results = []
    for i, case in enumerate(cases):
        print(f"  case {i}: ntok={case.get('n_tokens',ctx)} ans={case.get('answer','?')}", flush=True)
        r = run_one_case_with_server(condition, ctx, case, i, results_dir)
        case_results.append(r)
        status = "OK" if r["found"] else "FAIL"
        drafter_s = f"{r['drafter_fwd_s']:.3f}s" if r['drafter_fwd_s'] else "N/A"
        print(f"  case {i}: ttft={r['ttft_s']:.2f}s drafter={drafter_s} [{status}]", flush=True)
        if r["text"]:
            print(f"  case {i}: text={r['text'][:80]!r}", flush=True)
        if r["error"]:
            print(f"  case {i}: error={r['error'][:100]}", flush=True)

    drafter_times = [r["drafter_fwd_s"] for r in case_results if r["drafter_fwd_s"] is not None]
    ttfts = [r["ttft_s"] for r in case_results if r["ttft_s"] is not None]
    niah_pass = sum(1 for c in case_results if c["found"])

    return {
        "condition": condition, "ctx": ctx,
        "case_results": case_results,
        "drafter_times_s": drafter_times,
        "drafter_p50_s": median(drafter_times) if drafter_times else None,
        "ttft_p50_s": median(ttfts) if ttfts else None,
        "niah_pass": niah_pass,
        "niah_total": len(cases),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="dflash/bench/results/2026-05-21_ee14_broad")
    ap.add_argument("--cases-dir", default="/tmp")
    args = ap.parse_args()

    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases_by_ctx = {}
    for ctx in CONTEXTS:
        f_path = Path(args.cases_dir) / f"niah_{ctx}.jsonl"
        if not f_path.exists():
            print(f"[error] missing {f_path}", flush=True)
            sys.exit(1)
        with open(f_path) as f:
            cases_by_ctx[ctx] = [json.loads(l) for l in f]
        print(f"[init] {len(cases_by_ctx[ctx])} cases for ctx={ctx}", flush=True)

    all_results = []
    for condition in CONDITIONS:
        for ctx in CONTEXTS:
            result = run_condition_ctx(condition, ctx, cases_by_ctx[ctx], results_dir)
            all_results.append(result)
            with open(results_dir / "raw_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Summary
    baseline_times = {r["ctx"]: r["drafter_p50_s"] for r in all_results
                      if r["condition"] == "baseline" and r.get("drafter_p50_s")}

    rows = []
    print("\n=== PASS A TABLE ===")
    print(f"{'ctx':>6}  {'condition':>10}  {'drafter_p50':>12}  {'ttft_p50':>10}  {'NIAH':>6}  {'speedup':>8}")
    for r in all_results:
        ctx, cond = r["ctx"], r["condition"]
        dp50 = r.get("drafter_p50_s")
        tp50 = r.get("ttft_p50_s")
        niah = f"{r.get('niah_pass',0)}/{r.get('niah_total',0)}"
        if dp50 and ctx in baseline_times and cond != "baseline":
            speedup = f"{baseline_times[ctx]/dp50:.2f}x"
        else:
            speedup = "1.00x" if cond == "baseline" else "N/A"
        dp50_s = f"{dp50:.3f}s" if dp50 else "N/A"
        tp50_s = f"{tp50:.2f}s" if tp50 else "N/A"
        print(f"{ctx:>6}  {cond:>10}  {dp50_s:>12}  {tp50_s:>10}  {niah:>6}  {speedup:>8}")
        rows.append({"ctx": ctx, "condition": cond, "drafter_fwd_p50": dp50_s,
                     "ttft_p50": tp50_s, "NIAH": niah, "speedup": speedup})

    with open(results_dir / "SUMMARY_PASS_A.md", "w") as f:
        f.write("# Pass A: NIAH 1K-16K — ee14 vs baseline\n\n")
        f.write("| ctx | condition | drafter_fwd_p50 | ttft_p50 | NIAH | speedup |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(f"| {row['ctx']} | {row['condition']} | {row['drafter_fwd_p50']} | {row['ttft_p50']} | {row['NIAH']} | {row['speedup']} |\n")
    print(f"\n[done] {results_dir}/SUMMARY_PASS_A.md", flush=True)


if __name__ == "__main__":
    main()
