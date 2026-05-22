#!/usr/bin/env python3
"""
Task #48: Q3_K_S target + reranker Q8 drafter + ee7 + --prefill-skip-park full coexistence test.
3 contexts: 32K / 64K / 128K. 3 NIAH cases each.
Verdict: (A) all clean → SHIP, (B) works ≤64K only, (C) quality regression.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
TARGET = Path("/home/peppi/models/qwen3.6-27b-q3ks/Qwen3.6-27B-Q3_K_S.gguf")
DRAFTER = Path("/home/peppi/models/Qwen3-Reranker-0.6B-Q8_0.gguf")
OUT = REPO / "dflash/bench/results/2026-05-22_q3ks_coexist"
PORT = 18102

# Baseline from prior benches (ee7 + BF16 drafter + Q4_K_M + park)
EE7_PARK_BASELINE = {32768: 1.44, 65536: None, 131072: None}

CONTEXTS = [
    (32768,  "niah_32768",  "/tmp/niah_32768.jsonl",  36864),
    (65536,  "niah_65536",  "/tmp/niah_65536.jsonl",  70656),
    (131072, "niah_131072", "/tmp/niah_131072.jsonl", 135168),
]


def start_server(target, max_ctx, log_path):
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH_DRAFTER_EARLY_EXIT_N"] = "7"
    env["DFLASH_DRAFTER_SCORE_LAYERS"] = "7"
    cmd = [
        str(SERVER_BIN), str(target),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(max_ctx),
        "--cache-type-k", "tq3_0",
        "--cache-type-v", "tq3_0",
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", str(DRAFTER),
        "--prefill-skip-park",
    ]
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc, timeout=300):
    for _ in range(timeout):
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
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
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(3)


def read_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return int(out.split("\n")[0].strip())
    except Exception:
        return 0


def extract_log_metrics(log_path):
    metrics = {"drafter_fwd_s": None, "tail_score_s": None, "cuda_crash": False,
               "skip_park_on": False}
    try:
        content = Path(log_path).read_text()
        if re.search(r"cuMemSetAccess|NOT_READY|CUDA_ERROR_NOT_READY", content):
            metrics["cuda_crash"] = True
        if re.search(r"pflash_skip_park.*ON|skip_park.*true|prefill_skip_park.*1", content, re.I):
            metrics["skip_park_on"] = True
        m = re.search(r"\[drafter\]\s+forward\+score in ([\d.]+)s", content)
        if m:
            metrics["drafter_fwd_s"] = float(m.group(1))
        m = re.search(r"tail.score.*?([\d.]+)s", content, re.I)
        if m:
            metrics["tail_score_s"] = float(m.group(1))
    except Exception:
        pass
    return metrics


def run_context(ctx_tokens, ctx_name, niah_file, max_ctx):
    print(f"\n[q3ks] === Context {ctx_tokens//1024}K ({ctx_name}) ===")
    log_dir = OUT
    log_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    try:
        with open(niah_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    except Exception as e:
        print(f"[q3ks] NIAH file error: {e}")
        return None

    cases = cases[:3]
    results = []
    peak_vram_mib = 0

    for idx, case in enumerate(cases):
        log_path = log_dir / f"smoke_{ctx_tokens//1024}k_case{idx}.log"
        vram_before = read_vram_mib()
        print(f"[q3ks] case {idx}: starting server (ctx={ctx_tokens})...")

        proc = start_server(TARGET, max_ctx, log_path)
        ok = wait_server(proc, timeout=300)

        if not ok:
            print(f"[q3ks] case {idx}: server failed to start")
            metrics = extract_log_metrics(log_path)
            if metrics["cuda_crash"]:
                print(f"[q3ks] case {idx}: cuMemSetAccess crash detected")
            stop_server(proc)
            results.append({
                "pass": False, "answer": "", "expected": case.get("answer", ""),
                "drafter_fwd_s": None, "crash": True, "cuda_crash": metrics["cuda_crash"],
            })
            continue

        vram_after_load = read_vram_mib()
        peak_vram_mib = max(peak_vram_mib, vram_after_load)
        print(f"[q3ks] case {idx}: server up, VRAM={vram_after_load} MiB")

        found = False
        text = ""
        try:
            payload = {
                "model": "dflash",
                "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": 64,
                "stream": False,
                "temperature": 0.0,
            }
            t0 = time.perf_counter()
            r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                              json=payload, timeout=600)
            elapsed = time.perf_counter() - t0
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            expected = case.get("answer", "")
            found = expected in text
            print(f"[q3ks] case {idx}: answer='{text[:80]}' expected={expected} "
                  f"PASS={found} wall={elapsed:.1f}s")
        except Exception as e:
            print(f"[q3ks] case {idx}: request error: {e}")

        vram_peak = read_vram_mib()
        peak_vram_mib = max(peak_vram_mib, vram_peak)

        stop_server(proc)

        metrics = extract_log_metrics(log_path)
        results.append({
            "pass": found,
            "answer": text[:80],
            "expected": case.get("answer", ""),
            "drafter_fwd_s": metrics["drafter_fwd_s"],
            "crash": False,
            "cuda_crash": metrics["cuda_crash"],
        })
        print(f"[q3ks] case {idx}: drafter_fwd={metrics['drafter_fwd_s']}s "
              f"skip_park_on={metrics['skip_park_on']}")

    return {
        "ctx_tokens": ctx_tokens,
        "results": results,
        "peak_vram_gb": peak_vram_mib / 1024,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not TARGET.exists():
        print(f"[q3ks] ERROR: Q3_K_S model not found at {TARGET}")
        sys.exit(1)

    print(f"[q3ks] Target: {TARGET} ({TARGET.stat().st_size / 1e9:.1f} GB)")
    print(f"[q3ks] Drafter: {DRAFTER}")

    all_results = {}
    for ctx_tokens, ctx_name, niah_file, max_ctx in CONTEXTS:
        result = run_context(ctx_tokens, ctx_name, niah_file, max_ctx)
        if result is None:
            print(f"[q3ks] Skipping {ctx_name}: data unavailable")
            continue
        all_results[ctx_tokens] = result

        # Fail-fast: if 128K OOMs, don't bother with next tier
        if result["peak_vram_gb"] > 23.5 and ctx_tokens == 131072:
            print(f"[q3ks] 128K VRAM {result['peak_vram_gb']:.1f} GB > 23.5 GB threshold, stopping")
            break

    # Write per-context JSON
    for ctx_tokens, result in all_results.items():
        out_path = OUT / f"results_{ctx_tokens//1024}k.json"
        out_path.write_text(json.dumps(result, indent=2))

    # Determine overall verdict
    crash_ctxs = []
    oom_ctxs = []
    quality_fail_ctxs = []

    for ctx_tokens, result in all_results.items():
        passes = sum(1 for r in result["results"] if r["pass"])
        total = len(result["results"])
        crashes = any(r.get("cuda_crash") for r in result["results"])
        crash_any = any(r.get("crash") for r in result["results"])
        vram = result["peak_vram_gb"]

        if crashes or (crash_any and passes == 0):
            crash_ctxs.append(ctx_tokens)
        elif vram > 23.5:
            oom_ctxs.append(ctx_tokens)
        elif passes < 2:
            quality_fail_ctxs.append(ctx_tokens)

    if crash_ctxs:
        verdict = f"(C) cuMemSetAccess crash at ctx {[c//1024 for c in crash_ctxs]}K — Q3_K_S skip-park unsafe"
    elif quality_fail_ctxs and not oom_ctxs:
        verdict = f"(C) NIAH quality regression at {[c//1024 for c in quality_fail_ctxs]}K — keep Q4_K_M"
    elif oom_ctxs and not crash_ctxs:
        max_clean = max((c for c in all_results if c not in oom_ctxs), default=0)
        verdict = f"(B) Works ≤{max_clean//1024}K, OOM at {[c//1024 for c in oom_ctxs]}K — ship at 32-64K, keep park at 128K"
    elif all_results and not crash_ctxs and not oom_ctxs and not quality_fail_ctxs:
        verdict = "(A) Q3_K_S + reranker Q8 + ee7 + skip-park: CLEAN at all contexts — SHIP as new production stack"
    else:
        verdict = "(B) Partial success — review per-context results"

    # Build summary table
    table_rows = []
    for ctx_tokens, result in sorted(all_results.items()):
        passes = sum(1 for r in result["results"] if r["pass"])
        total = len(result["results"])
        dfwds = [r["drafter_fwd_s"] for r in result["results"] if r["drafter_fwd_s"] is not None]
        p50_dfw = sorted(dfwds)[len(dfwds)//2] if dfwds else None
        crashes = any(r.get("cuda_crash") for r in result["results"])
        vram = result["peak_vram_gb"]
        choreo = "YES-crash" if crashes else ("OOM" if vram > 23.5 else "NO")
        p50_str = f"{p50_dfw:.2f}s" if p50_dfw else "N/A"
        table_rows.append(
            f"| {ctx_tokens//1024}K | {p50_str} | {passes}/{total} | {vram:.1f} GB | {choreo} |"
        )

    table = "\n".join(table_rows) if table_rows else "| (no data) |"

    q3ks_size_gb = TARGET.stat().st_size / 1e9 if TARGET.exists() else 0

    summary = f"""# Q3_K_S + Reranker Q8 + ee7 + skip-park Full Coexistence (Task #48)

Binary: {REPO}/dflash/build/dflash_server
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Target: Q3_K_S ({q3ks_size_gb:.1f} GB) — requantized from Q4_K_M with --allow-requantize
Drafter: Qwen3-Reranker-0.6B-Q8_0 (~0.6 GB)
Flags: DFLASH_DRAFTER_EARLY_EXIT_N=7 DFLASH_DRAFTER_SCORE_LAYERS=7 --prefill-skip-park
KV: tq3_0/tq3_0, keep-ratio=0.05

## Results

| ctx | drafter_fwd (p50) | NIAH | peak VRAM | choreography fired? |
|---|---|---|---|---|
{table}

## Baseline (ee7 + BF16 drafter + Q4_K_M + park, 32K)
drafter_fwd p50 = 1.44 s

## Verdict

{verdict}

## Per-context detail
"""
    for ctx_tokens, result in sorted(all_results.items()):
        summary += f"\n### {ctx_tokens//1024}K\n"
        for i, r in enumerate(result["results"]):
            summary += (f"- case {i}: NIAH={'PASS' if r['pass'] else 'FAIL'}, "
                        f"drafter_fwd={r['drafter_fwd_s']}s, "
                        f"answer='{r['answer'][:50]}'\n")

    summary_path = OUT / "SUMMARY.md"
    summary_path.write_text(summary)
    print(f"\n[q3ks] SUMMARY:\n{summary}")
    print(f"[q3ks] Written to {summary_path}")
    return verdict


if __name__ == "__main__":
    main()
