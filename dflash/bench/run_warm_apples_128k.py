#!/usr/bin/env python3
"""
Apples-to-apples warm bench: prior stack vs new stack at 128K.
Single server per condition — all 3 cases sent to same server.
p50 computed over cases 1+2 only (case 0 = cold warm-up, discarded).
"""
import json, os, re, statistics, subprocess, sys, time
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO / "dflash/build/dflash_server"
OUT = REPO / "dflash/bench/results/2026-05-22_warm_apples_128k"
PORT = 18103
CTX = 131072
MAX_CTX = 139264
NIAH_FILE = Path("/tmp/niah_warm_128k.jsonl")

CONDITIONS = [
    {
        "name": "A_prior",
        "target": "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf",
        "drafter": "/home/peppi/models/Qwen3-0.6B-BF16.gguf",
        "skip_park": False,
        "label": "Prior (Q4_K_M + BF16 drafter + park/unpark)",
    },
    {
        "name": "B_new",
        "target": "/home/peppi/models/qwen3.6-27b-q3ks/Qwen3.6-27B-Q3_K_S.gguf",
        "drafter": "/home/peppi/models/Qwen3-Reranker-0.6B-Q8_0.gguf",
        "skip_park": True,
        "label": "New (Q3_K_S + reranker Q8 + skip-park)",
    },
]


def start_server(target, drafter, skip_park, log_path):
    env = os.environ.copy()
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env["DFLASH_DRAFTER_EARLY_EXIT_N"] = "7"
    env["DFLASH_DRAFTER_SCORE_LAYERS"] = "7"
    cmd = [
        str(SERVER_BIN), target,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--cache-type-k", "tq3_0",
        "--cache-type-v", "tq3_0",
        "--prefill-compression", "always",
        "--prefill-keep-ratio", "0.05",
        "--prefill-drafter", drafter,
    ]
    if skip_park:
        cmd.append("--prefill-skip-park")
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc, timeout=360):
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


def extract_drafter_fwd(log_path):
    try:
        content = Path(log_path).read_text()
        m = re.search(r"\[drafter\]\s+forward\+score in ([\d.]+)s", content)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def run_condition(cond, cases):
    name = cond["name"]
    log_path = OUT / f"{name}_server.log"
    print(f"\n[warm-bench] === Condition {name}: {cond['label']} ===")
    print(f"[warm-bench] Starting server (single instance for all 3 cases)...")

    proc = start_server(cond["target"], cond["drafter"], cond["skip_park"], log_path)
    ok = wait_server(proc, timeout=360)

    if not ok:
        print(f"[warm-bench] Server failed to start for {name}")
        stop_server(proc)
        return None

    vram_loaded = read_vram_mib()
    print(f"[warm-bench] Server up, VRAM={vram_loaded} MiB")

    results = []
    peak_vram = vram_loaded

    for idx, case in enumerate(cases):
        print(f"[warm-bench] Sending case {idx} ({'COLD warm-up' if idx == 0 else 'WARM'})...")
        payload = {
            "model": "dflash",
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 64,
            "stream": False,
            "temperature": 0.0,
        }
        text = ""
        found = False
        wall = None
        try:
            t0 = time.perf_counter()
            r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                              json=payload, timeout=600)
            wall = time.perf_counter() - t0
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            expected = case.get("answer", "")
            found = expected in text
            print(f"[warm-bench] case {idx}: NIAH={'PASS' if found else 'FAIL'} "
                  f"wall={wall:.1f}s ans='{text[:60]}'")
        except Exception as e:
            print(f"[warm-bench] case {idx}: request error: {e}")

        vram_now = read_vram_mib()
        peak_vram = max(peak_vram, vram_now)

        # Read drafter_fwd from log (it appends as server processes)
        # Wait a moment for log flush
        time.sleep(0.5)
        # Find the Nth occurrence of drafter forward (idx-th case)
        drafter_fwd = None
        try:
            content = Path(log_path).read_text()
            matches = re.findall(r"\[drafter\]\s+forward\+score in ([\d.]+)s", content)
            if idx < len(matches):
                drafter_fwd = float(matches[idx])
        except Exception:
            pass

        results.append({
            "case": idx,
            "cold": idx == 0,
            "niah": found,
            "answer": text[:80],
            "expected": case.get("answer", ""),
            "drafter_fwd_s": drafter_fwd,
            "wall_s": wall,
        })
        print(f"[warm-bench] case {idx}: drafter_fwd={drafter_fwd}s")

    stop_server(proc)

    # Compute p50 of cases 1+2 (warm only)
    warm_dfwds = [r["drafter_fwd_s"] for r in results[1:] if r["drafter_fwd_s"] is not None]
    p50_warm = statistics.median(warm_dfwds) if warm_dfwds else None

    niah_warm = sum(1 for r in results[1:] if r["niah"])

    print(f"[warm-bench] {name}: warm p50 drafter_fwd = {p50_warm}s  NIAH warm = {niah_warm}/2  peak VRAM = {peak_vram/1024:.1f} GB")

    return {
        "name": name,
        "label": cond["label"],
        "results": results,
        "p50_warm_s": p50_warm,
        "niah_warm_2": niah_warm,
        "peak_vram_gb": peak_vram / 1024,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not NIAH_FILE.exists():
        print(f"[warm-bench] ERROR: NIAH file not found at {NIAH_FILE}")
        sys.exit(1)

    cases = []
    with open(NIAH_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    cases = cases[:3]
    print(f"[warm-bench] Loaded {len(cases)} NIAH cases from {NIAH_FILE}")

    all_cond_results = {}
    for cond in CONDITIONS:
        result = run_condition(cond, cases)
        if result is not None:
            all_cond_results[cond["name"]] = result
            # Save intermediate
            out_path = OUT / f"{cond['name']}_results.json"
            out_path.write_text(json.dumps(result, indent=2))

    # Build summary
    a = all_cond_results.get("A_prior")
    b = all_cond_results.get("B_new")

    def fmt_s(v):
        return f"{v:.2f} s" if v is not None else "N/A"

    def row(res):
        if res is None:
            return "| N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        r0 = res["results"][0] if len(res["results"]) > 0 else {}
        r1 = res["results"][1] if len(res["results"]) > 1 else {}
        r2 = res["results"][2] if len(res["results"]) > 2 else {}
        case0 = fmt_s(r0.get("drafter_fwd_s"))
        case1 = fmt_s(r1.get("drafter_fwd_s"))
        case2 = fmt_s(r2.get("drafter_fwd_s"))
        p50 = fmt_s(res["p50_warm_s"])
        niah = f"{res['niah_warm_2']}/2"
        vram = f"{res['peak_vram_gb']:.1f} GB"
        return f"| {res['label']} | {case0} | {case1} | {case2} | {p50} | {niah} | {vram} |"

    table = f"""| Stack | case 0 drafter_fwd (cold) | case 1 drafter_fwd (warm) | case 2 drafter_fwd (warm) | p50 cases 1+2 | NIAH | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
{row(a)}
{row(b)}"""

    # Resolution verdict
    verdict = "INCONCLUSIVE (missing data)"
    recommendation = "Insufficient data for recommendation."
    if a and b and a["p50_warm_s"] is not None and b["p50_warm_s"] is not None:
        ratio = b["p50_warm_s"] / a["p50_warm_s"]
        if ratio <= 1.5:
            verdict = f"NEW STACK WARM p50 = {b['p50_warm_s']:.2f}s vs prior {a['p50_warm_s']:.2f}s ({ratio:.2f}x) — within 1.5x threshold"
            recommendation = "SHIP new stack (Q3_K_S + reranker Q8 + skip-park) as production default — choreography elimination win, drafter cost acceptable."
        elif ratio <= 2.0:
            verdict = f"NEW STACK WARM p50 = {b['p50_warm_s']:.2f}s vs prior {a['p50_warm_s']:.2f}s ({ratio:.2f}x) — MARGINAL (1.5-2x)"
            recommendation = "MARGINAL: per-condition wall comparison needed before decision; Q3_K_S VRAM savings may justify the overhead."
        else:
            verdict = f"NEW STACK WARM p50 = {b['p50_warm_s']:.2f}s vs prior {a['p50_warm_s']:.2f}s ({ratio:.2f}x) — REAL OVERHEAD (>2x)"
            recommendation = "KEEP prior stack (Q4_K_M + BF16 drafter + park/unpark) as default; new stack available as opt-in for VRAM-constrained setups."
    elif a and b:
        if a["p50_warm_s"] is None:
            verdict = "PRIOR STACK failed to produce warm timing data"
        if b["p50_warm_s"] is None:
            verdict = "NEW STACK failed to produce warm timing data — view_3d crash likely"

    summary = f"""# Apples-to-Apples Warm 128K Bench

**Date**: 2026-05-22
**Binary**: {SERVER_BIN}
**GPU**: NVIDIA GeForce RTX 3090 (24 GB)
**Context**: 128K (131072 tokens)
**Method**: Single server per condition, 3 NIAH cases sequentially, p50 = median(case1, case2)

## Results

{table}

## Resolution

{verdict}

## Production Recommendation

{recommendation}

## Per-Case Detail

### Condition A: Prior Stack (Q4_K_M + BF16 drafter + park/unpark)
"""
    if a:
        for r in a["results"]:
            summary += (f"- case {r['case']} ({'cold' if r['cold'] else 'warm'}): "
                        f"NIAH={'PASS' if r['niah'] else 'FAIL'} "
                        f"drafter_fwd={r['drafter_fwd_s']}s wall={r['wall_s']}s "
                        f"ans='{r['answer'][:50]}'\n")
    else:
        summary += "- NO DATA\n"

    summary += "\n### Condition B: New Stack (Q3_K_S + reranker Q8 + skip-park)\n"
    if b:
        for r in b["results"]:
            summary += (f"- case {r['case']} ({'cold' if r['cold'] else 'warm'}): "
                        f"NIAH={'PASS' if r['niah'] else 'FAIL'} "
                        f"drafter_fwd={r['drafter_fwd_s']}s wall={r['wall_s']}s "
                        f"ans='{r['answer'][:50]}'\n")
    else:
        summary += "- NO DATA\n"

    summary_path = OUT / "SUMMARY.md"
    summary_path.write_text(summary)
    print(f"\n[warm-bench] === SUMMARY ===\n{summary}")
    print(f"[warm-bench] Written to {summary_path}")


if __name__ == "__main__":
    main()
