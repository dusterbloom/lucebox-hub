#!/usr/bin/env bash
# Anchor parameter co-sweep at ctx=32K, keep=0.05, mode=ALWAYS.
# Grid: DFLASH_COMPRESS_QUERY_TOKENS in {48,96,192}
#       DFLASH_COMPRESS_ANCHOR_RADIUS in {1,2,4}
#       DFLASH_COMPRESS_MAX_ANCHOR_HITS in {4,8,16}
# 27 combos × 3 cases each. Fixed ctx=32768, keep=0.05.
#
# Usage:
#   bash dflash/bench/run_anchor_cosweep.sh
#
# Output: dflash/bench/results/2026-05-21_anchor_cosweep/<qt>_<ar>_<mh>/summary.json
#         dflash/bench/results/2026-05-21_anchor_cosweep/SUMMARY.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVER="${WORKTREE_ROOT}/dflash/build/dflash_server"
MODEL="/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFTER="/home/peppi/models/Qwen3-0.6B-BF16.gguf"
SERVER_LOG="/tmp/anchor_cosweep_server.log"
SERVER_URL="http://127.0.0.1:8080"
PORT=8080
MAX_CTX=36864
N_CASES=3
CTX=32768
KEEP=0.05

OUT_BASE="${WORKTREE_ROOT}/dflash/bench/results/2026-05-21_anchor_cosweep"
CASES_FILE="${OUT_BASE}/cases_32k_n3.jsonl"

SWEEP_PY="${WORKTREE_ROOT}/pflash/tests/niah_gen.py"

START_EPOCH=$SECONDS
WALL_BUDGET=6000  # 100 min in seconds

SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "${OUT_BASE}"

# Generate 32K cases once (seed 42, 3 cases, Qwen3-0.6B tokenizer)
if [[ ! -f "${CASES_FILE}" ]]; then
    echo "[anchor-sweep] generating ${N_CASES} cases at ctx=${CTX}..."
    python3 "${SWEEP_PY}" \
        --n "${N_CASES}" \
        --ctx "${CTX}" \
        --out "${CASES_FILE}" \
        --seed-base 42 \
        --tokenizer "Qwen/Qwen3-0.6B"
else
    echo "[anchor-sweep] reusing existing cases: ${CASES_FILE}"
fi

wait_for_server() {
    local deadline=$((SECONDS + 120))
    while [[ $SECONDS -lt $deadline ]]; do
        if grep -q "listening on http://127.0.0.1:${PORT}" "${SERVER_LOG}" 2>/dev/null; then
            return 0
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[anchor-sweep] server PID=${SERVER_PID} died, last log:"
            tail -20 "${SERVER_LOG}"
            return 1
        fi
        sleep 2
    done
    echo "[anchor-sweep] server timeout after 120s"
    tail -20 "${SERVER_LOG}"
    return 1
}

run_combo() {
    local qt="$1"
    local ar="$2"
    local mh="$3"
    local cell_dir="${OUT_BASE}/${qt}_${ar}_${mh}"
    local cell_summary="${cell_dir}/summary.json"

    # Resume support: skip if summary already exists
    if [[ -f "${cell_summary}" ]]; then
        echo "[anchor-sweep] skip (cached) qt=${qt} ar=${ar} mh=${mh}"
        return 0
    fi

    # Wall budget guard
    local elapsed=$((SECONDS - START_EPOCH))
    if [[ $elapsed -gt $WALL_BUDGET ]]; then
        echo "[anchor-sweep] WALL BUDGET ${WALL_BUDGET}s exceeded at ${elapsed}s — stopping"
        exit 0
    fi

    echo ""
    echo "===== COMBO qt=${qt} ar=${ar} mh=${mh} (elapsed=${elapsed}s) ====="

    # Kill stale server
    pkill -f "dflash_server" 2>/dev/null || true
    sleep 1

    # Start server with anchor params in env
    GGML_CUDA_NO_VMM=1 \
    DFLASH27B_KV_K=tq3_0 \
    DFLASH27B_KV_V=tq3_0 \
    DFLASH_COMPRESS_QUERY_TOKENS="${qt}" \
    DFLASH_COMPRESS_ANCHOR_RADIUS="${ar}" \
    DFLASH_COMPRESS_MAX_ANCHOR_HITS="${mh}" \
    "${SERVER}" "${MODEL}" \
        --host 127.0.0.1 --port "${PORT}" \
        --max-ctx "${MAX_CTX}" \
        --cache-type-k tq3_0 \
        --cache-type-v tq3_0 \
        --prefill-compression always \
        --prefill-keep-ratio "${KEEP}" \
        --prefill-drafter "${DRAFTER}" \
        --prefill-skip-park \
        > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "[anchor-sweep] server PID=${SERVER_PID}"

    if ! wait_for_server; then
        echo "[anchor-sweep] SKIP: server failed for qt=${qt} ar=${ar} mh=${mh}"
        SERVER_PID=""
        return 0
    fi

    mkdir -p "${cell_dir}"

    # Run N_CASES NIAH requests against the server
    python3 - "${SERVER_URL}" "${CASES_FILE}" "${cell_dir}" "${qt}" "${ar}" "${mh}" "${CTX}" "${KEEP}" <<'PYEOF'
import sys, json, time, math, statistics
import urllib.request, urllib.error

server_url = sys.argv[1]
cases_file = sys.argv[2]
cell_dir   = sys.argv[3]
qt, ar, mh = sys.argv[4], sys.argv[5], sys.argv[6]
ctx_tokens = int(sys.argv[7])
keep_ratio = float(sys.argv[8])

with open(cases_file) as f:
    cases = [json.loads(l) for l in f]

scores = []
walls  = []

for i, case in enumerate(cases):
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": 128,
    }).encode()

    req = urllib.request.Request(
        server_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            wall_s = time.time() - t0
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"  [warn] case {i}: {exc}", flush=True)
        continue

    text = ""
    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "").strip()

    sc = 1.0 if case["answer"] in text else 0.0
    scores.append(sc)
    walls.append(wall_s)
    print(f"  case {i}: wall={wall_s:.1f}s ok={sc==1.0} ans={case['answer']!r} got={text[:60]!r}", flush=True)

    raw = {
        "case_idx": i, "prompt_len": case.get("n_tokens", 0),
        "answer": case["answer"], "response_text": text,
        "score": sc, "wall_s": wall_s,
        "qt": qt, "ar": ar, "mh": mh,
        "keep_ratio": keep_ratio, "ctx_tokens": ctx_tokens,
    }
    with open(f"{cell_dir}/case_{i:04d}.raw.json", "w") as f:
        json.dump(raw, f, indent=2)

if not scores:
    print("[error] no results", flush=True)
    sys.exit(1)

accuracy = sum(scores) / len(scores)
walls_sorted = sorted(walls)
mean_wall = statistics.mean(walls)
p50_wall  = statistics.median(walls)
summary = {
    "qt": int(qt), "ar": int(ar), "mh": int(mh),
    "ctx_tokens": ctx_tokens,
    "keep_ratio": keep_ratio,
    "n_cases": len(scores),
    "accuracy": accuracy,
    "mean_wall": mean_wall,
    "p50_wall": p50_wall,
}
with open(f"{cell_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  summary: acc={accuracy:.3f} mean={mean_wall:.1f}s p50={p50_wall:.1f}s", flush=True)
PYEOF
    local exit_code=$?

    # Grab anchor activity line from server log
    local anchor_line
    anchor_line=$(grep "\[drafter-skip\]" "${SERVER_LOG}" | tail -1 || echo "(no anchor log line found)")
    echo "[anchor-sweep] anchor activity: ${anchor_line}"
    echo "${anchor_line}" > "${cell_dir}/anchor_log.txt"

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    sleep 1

    if [[ $exit_code -ne 0 ]]; then
        echo "[anchor-sweep] WARN: python client exited ${exit_code} for qt=${qt} ar=${ar} mh=${mh}"
    fi
}

echo "[anchor-sweep] output: ${OUT_BASE}"
echo "[anchor-sweep] model: ${MODEL}"
echo "[anchor-sweep] start: $(date)"

# 3^3 = 27 combos
for qt in 48 96 192; do
    for ar in 1 2 4; do
        for mh in 4 8 16; do
            run_combo "${qt}" "${ar}" "${mh}"
        done
    done
done

echo ""
echo "===== ALL COMBOS DONE in $((SECONDS - START_EPOCH))s ====="
echo "[anchor-sweep] aggregating results..."

# Aggregate all summary.json files and write SUMMARY.md
python3 - "${OUT_BASE}" <<'AGGPYEOF'
import sys, json, pathlib, statistics

base = pathlib.Path(sys.argv[1])

rows = []
for sfile in sorted(base.glob("*/summary.json")):
    try:
        d = json.loads(sfile.read_text())
        if "qt" not in d:
            continue
        anchor_log_file = sfile.parent / "anchor_log.txt"
        anchor_log = anchor_log_file.read_text().strip() if anchor_log_file.exists() else ""
        rows.append({
            "qt": d["qt"], "ar": d["ar"], "mh": d["mh"],
            "n_cases": d["n_cases"],
            "accuracy": d["accuracy"],
            "mean_wall": d["mean_wall"],
            "p50_wall": d["p50_wall"],
            "anchor_log": anchor_log,
        })
    except Exception as e:
        print(f"  [warn] {sfile}: {e}")

if not rows:
    print("[agg] no rows found")
    sys.exit(0)

# Sort by p50_wall ascending
rows.sort(key=lambda r: r["p50_wall"])

# Default combo (96, 2, 8)
default_row = next((r for r in rows if r["qt"]==96 and r["ar"]==2 and r["mh"]==8), None)
default_p50 = default_row["p50_wall"] if default_row else float("nan")

lines = []
lines.append("# Anchor Param Co-Sweep Summary — 2026-05-21")
lines.append("")
lines.append("Fixed: ctx=32768, keep=0.05, mode=ALWAYS, n=3 cases, seed=42")
lines.append("")
lines.append("Baseline (envelope 2026-05-21): OFF=24.3s, ALWAYS=24.2s p50 at 32K/keep=0.05")
lines.append("")
lines.append("## Frontier Table (sorted by p50_wall)")
lines.append("")
lines.append("| qt | ar | mh | n | accuracy | mean_wall | p50_wall | delta_vs_default |")
lines.append("|----|----|----|----|----------|-----------|----------|-----------------|")

for r in rows:
    delta = r["p50_wall"] - default_p50
    delta_str = f"{delta:+.1f}s"
    marker = " **DEFAULT**" if r["qt"]==96 and r["ar"]==2 and r["mh"]==8 else ""
    acc_str = f"{r['accuracy']:.3f}"
    lines.append(
        f"| {r['qt']} | {r['ar']} | {r['mh']} | {r['n_cases']} | "
        f"{acc_str} | {r['mean_wall']:.1f}s | {r['p50_wall']:.1f}s | "
        f"{delta_str}{marker} |"
    )

# Top-3 fastest
fastest_100 = [r for r in rows if r["accuracy"] >= 1.0 - 1e-6]
fastest_100.sort(key=lambda r: r["p50_wall"])
top3 = fastest_100[:3]

lines.append("")
lines.append("## The Winner")
lines.append("")
if fastest_100:
    winner = fastest_100[0]
    delta_w = winner["p50_wall"] - default_p50
    pct_improvement = -delta_w / default_p50 * 100 if default_p50 > 0 else 0
    lines.append(
        f"Best 100%-accuracy combo: qt={winner['qt']}, ar={winner['ar']}, mh={winner['mh']}"
    )
    lines.append(f"p50_wall={winner['p50_wall']:.1f}s vs default={default_p50:.1f}s ({delta_w:+.1f}s, {pct_improvement:.1f}% improvement)")
    lines.append("")
    lines.append("### Top-3 fastest (100% accuracy)")
    lines.append("")
    lines.append("| rank | qt | ar | mh | p50_wall | delta |")
    lines.append("|------|----|----|----|---------:|------:|")
    for rank, r in enumerate(top3, 1):
        d = r["p50_wall"] - default_p50
        lines.append(f"| {rank} | {r['qt']} | {r['ar']} | {r['mh']} | {r['p50_wall']:.1f}s | {d:+.1f}s |")
else:
    lines.append("No 100%-accuracy combo found.")

lines.append("")
lines.append("## Recommendation")
lines.append("")
CHANGE_THRESHOLD = 0.10  # 10% improvement
if fastest_100 and default_p50 > 0:
    winner = fastest_100[0]
    improvement_pct = (default_p50 - winner["p50_wall"]) / default_p50
    if improvement_pct >= CHANGE_THRESHOLD:
        lines.append(
            f"**Change defaults**: qt={winner['qt']}, ar={winner['ar']}, mh={winner['mh']} "
            f"is {improvement_pct*100:.1f}% faster than default at unchanged accuracy."
        )
    else:
        lines.append(
            f"**Keep current defaults** (qt=96, ar=2, mh=8): best improvement is "
            f"{improvement_pct*100:.1f}% which is below the 10% threshold. Noise."
        )
else:
    lines.append("Insufficient data for recommendation.")

lines.append("")
lines.append("## Accuracy Failures")
lines.append("")
failures = [r for r in rows if r["accuracy"] < 1.0 - 1e-6]
if failures:
    lines.append("| qt | ar | mh | accuracy | n_cases |")
    lines.append("|----|----|----|----------|---------|")
    for r in failures:
        lines.append(f"| {r['qt']} | {r['ar']} | {r['mh']} | {r['accuracy']:.3f} | {r['n_cases']} |")
    lines.append("")
    lines.append("Note: n=3 cases — a single miss = 0.667 accuracy. One case id below.")
    for r in failures:
        case_files = sorted(pathlib.Path(base / f"{r['qt']}_{r['ar']}_{r['mh']}").glob("case_*.raw.json"))
        for cf in case_files:
            raw = json.loads(cf.read_text())
            if raw.get("score", 1.0) < 0.5:
                lines.append(f"  Failure: qt={r['qt']} ar={r['ar']} mh={r['mh']} case={raw['case_idx']} ans={raw['answer']!r} got={raw['response_text'][:80]!r}")
else:
    lines.append("None — all combos preserved 100% accuracy.")

lines.append("")
lines.append(f"## Combos run: {len(rows)} / 27")
lines.append("")

summary_path = base / "SUMMARY.md"
summary_path.write_text("\n".join(lines) + "\n")
print(f"[agg] wrote {summary_path} ({len(rows)} rows)")

# Print brief to stdout too
print(f"\n=== RESULTS ===")
print(f"Combos completed: {len(rows)}/27")
if default_row:
    print(f"Default (96,2,8): p50={default_row['p50_wall']:.1f}s acc={default_row['accuracy']:.3f}")
if top3:
    print("Top-3 (100% acc):")
    for rank, r in enumerate(top3, 1):
        d = r["p50_wall"] - default_p50
        print(f"  {rank}. qt={r['qt']} ar={r['ar']} mh={r['mh']}: p50={r['p50_wall']:.1f}s ({d:+.1f}s vs default)")
if failures:
    print(f"Accuracy failures: {len(failures)} combos")
AGGPYEOF

echo "[anchor-sweep] done $(date)"
