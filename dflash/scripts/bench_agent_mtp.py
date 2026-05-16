"""
Agentic-workload bench for any IMtpModule-backed MTP head.

This script is intentionally MTP-implementation-agnostic: it drives
``test_dflash`` through the ``--mtp-gguf`` CLI, which dispatches to
whatever ``IMtpModule`` the GGUF resolves to (today: Qwen3.6 NextN; any
future MTP module wired into the same factory works without changes
here).  The only requirement is that the GGUF carries both the target
backbone and the MTP head tensors.

Reuses ``bench_agent.py``'s prompt construction (2K / 8K / 24K token
buckets from SWE-bench Verified + Codex system prompts) and reports
per bucket: prefill_s, decode_s, decode tok/s, TTFT, total latency,
decode-only speedup, wall-clock speedup, accept_rate, and the actual
emitted-token count (which may be < n_gen on early EOS).

Env vars:
    MTP_GGUF               path to the MTP GGUF (target + head in one file)
    DFLASH_BIN             test_dflash binary
    DFLASH_TOKENIZER       HF tokenizer id (default: Qwen/Qwen3.5-27B)
    BENCH_KV_TYPE          KV quantization (default: q8_0)
"""
import argparse
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""

MTP_GGUF = os.environ.get("MTP_GGUF")
if not MTP_GGUF:
    sys.stderr.write("set MTP_GGUF to the target+head GGUF path\n")
    sys.exit(2)
TEST_DFLASH = os.environ.get("DFLASH_BIN", str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"))
TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-27B")
KV_TYPE = os.environ.get("BENCH_KV_TYPE", "q8_0")
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

FIX_DIR = ROOT / "scripts" / "fixtures"
SWE_PARQUET = FIX_DIR / "swe_bench" / "swe_bench_verified.parquet"
SYS_PROMPT_SMALL = FIX_DIR / "agent_prompts" / "codex_gpt52_codex.md"
SYS_PROMPT_LARGE = FIX_DIR / "agent_prompts" / "codex_gpt52.md"

# Pull prompt construction helpers from bench_agent.py so the prompts
# in MTP cells are byte-identical to the DFlash drafter bench.
sys.path.insert(0, str(ROOT / "scripts"))
from bench_agent import (  # noqa: E402
    BUCKETS, build_prompt, tokenize_to_file, _load_swe_rows, _run_timed,
)

RESULT_RE = re.compile(
    r"RESULT tok_s=(\S+) prompt=(\S+) gamma=(\S+) tokens=(\S+) "
    r"decode_s=(\S+) prefill_s=(\S+) accepted=(\S+) proposed=(\S+)"
)


def _max_ctx_for(n_prompt: int, n_gen: int) -> int:
    need = n_prompt + n_gen + 128
    for cap in (2048, 4096, 8192, 16384, 32768, 65536):
        if cap >= need:
            return cap
    return need


def run_cell(gamma: int, prompt_path: Path, prompt_id: int,
             n_prompt: int, n_gen: int,
             draft_source: str = "chain",
             draft_topk: int = 1,
             ddtree_budget: int = 0,
             ddtree_chain_seed: bool = True):
    """One test_dflash invocation.

    draft_source:
      "chain"    — existing MtpChainRunner path (K=1, no DDTree).
      "mtp_topk" — experiment C: MTP set_draft_topk(K) → build_ddtree →
                   chain-verify of the DDTree's top-1 path. See
                   run_qwen36_mtp_harness in test_dflash.cpp for the
                   BLOCKER note on true tree-mask verify.
    """
    max_ctx = _max_ctx_for(n_prompt, n_gen)
    cmd = [
        TEST_DFLASH, MTP_GGUF,
        "--mtp-gguf", MTP_GGUF,
        "--prompt-bin", str(prompt_path),
        "--n-gen", str(n_gen),
        "--gamma", str(gamma),
        "--prompt-id", str(prompt_id),
        "--max-ctx", str(max_ctx),
        "-ctk", KV_TYPE, "-ctv", KV_TYPE,
        "--draft-source", draft_source,
    ]
    if draft_source == "mtp_topk":
        cmd += ["--draft-topk", str(max(1, draft_topk))]
        if ddtree_budget > 0:
            cmd += [f"--ddtree-budget={ddtree_budget}"]
        if not ddtree_chain_seed:
            cmd += ["--ddtree-no-chain-seed"]
    # The WSL2 CUDA flake mitigation lives in the shell bench script; we
    # replicate it here so back-to-back invocations don't trip the per-
    # process libggml-cuda.so teardown bug.
    time.sleep(15)
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:/usr/local/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"test_dflash exit {r.returncode} (gamma={gamma}, prompt_id={prompt_id}): "
            f"{(r.stderr or r.stdout)[-2000:]}"
        )
    m = RESULT_RE.search(r.stdout)
    if not m:
        raise RuntimeError(f"no RESULT line in stdout:\n{r.stdout[-2000:]}")
    tok_s, _, _, tokens, decode_s, prefill_s, accepted, proposed = m.groups()
    cell = {
        "draft_source": draft_source,
        "draft_topk": draft_topk,
        "ddtree_budget": ddtree_budget,
        "ddtree_chain_seed": ddtree_chain_seed,
        "tok_s": float(tok_s),
        "tokens": int(tokens),
        "decode_s": float(decode_s),
        "prefill_s": float(prefill_s),
        "ttft_s": float(prefill_s) + (float(decode_s) / int(tokens)),
        "total_s": float(prefill_s) + float(decode_s),
        "accepted": int(accepted),
        "proposed": int(proposed),
        "accept_rate": (int(accepted) / int(proposed)) if int(proposed) > 0 else 0.0,
    }
    # Pass through RESULT_JSON if the binary emitted one (experiment-C wiring).
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                cell["raw_result_json"] = json.loads(line[len("RESULT_JSON "):])
            except Exception:
                pass
            break
    # Emit a single per-cell JSON line to stdout so external bench drivers
    # (the brief's experiment-C runner) can grep without parsing the table.
    print("CELL_JSON " + json.dumps(cell))
    return cell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", choices=list(BUCKETS), action="append")
    p.add_argument("--n-gen", type=int, default=128)
    p.add_argument("--n-sample", type=int, default=2,
                   help="SWE rows per bucket (default 2)")
    p.add_argument("--n-runs", type=int, default=2,
                   help="runs per (bucket, row, gamma) cell (default 2)")
    p.add_argument("--out", type=Path,
                   default=ROOT / "bench" / "results" / "mtp_agent_latest.json")
    # Experiment-C wiring. When set, each MTP cell is ALSO run in the
    # selected mtp_topk configuration alongside the existing chain cell.
    # Brief's recipe: A=chain gamma in {4,5,6,7}; C=mtp_topk K=4 budget=A.gamma.
    p.add_argument("--draft-source", default="chain",
                   choices=("chain", "mtp_topk", "both"),
                   help="chain: MTP argmax chain (default). mtp_topk: experiment "
                        "C (MTP set_draft_topk + DDTree). both: run chain and "
                        "mtp_topk per cell for direct comparison.")
    p.add_argument("--draft-topk", type=int, default=4,
                   help="K for mtp_topk path (default 4)")
    p.add_argument("--ddtree-budget", type=int, default=0,
                   help="DDTree budget for mtp_topk (default 0 = bin default 64)")
    p.add_argument("--ddtree-no-chain-seed", action="store_true",
                   help="Disable DDTree chain_seed (paper's pure best-first). "
                        "Required to get a real tree shape when K>1.")
    args = p.parse_args()

    if not Path(MTP_GGUF).is_file():
        print(f"MTP_GGUF not found: {MTP_GGUF}", file=sys.stderr)
        sys.exit(2)
    if not Path(TEST_DFLASH).is_file():
        print(f"test_dflash binary not found: {TEST_DFLASH}", file=sys.stderr)
        sys.exit(2)

    buckets = args.bucket or list(BUCKETS)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    rows = _load_swe_rows().head(args.n_sample).to_dict("records")

    aggregate = {}
    print(f"GGUF:      {MTP_GGUF}")
    print(f"Tokenizer: {TOKENIZER}")
    print(f"Buckets:   {buckets}")
    print(f"n_gen={args.n_gen}  n_sample={args.n_sample}  n_runs={args.n_runs}")

    for bucket in buckets:
        sys_prompt = BUCKETS[bucket]["sys"]
        target_tokens = BUCKETS[bucket]["target"]
        bucket_records = []
        for row_idx, row in enumerate(rows):
            text, n_prompt = build_prompt(tok, sys_prompt, row, target_tokens)
            prompt_id = row_idx * 100 + 1
            prompt_path = TMPDIR / f"mtp_agent_{bucket}_{row_idx}.bin"
            tokenize_to_file(tok, text, prompt_path)
            print(f"\n[{bucket}] row {row_idx} n_prompt={n_prompt}")

            run_chain  = args.draft_source in ("chain", "both")
            run_topk   = args.draft_source in ("mtp_topk", "both")
            ar  = []
            mtp = []
            topk_runs = []
            for run in range(args.n_runs):
                a = run_cell(0, prompt_path, prompt_id, n_prompt, args.n_gen,
                             draft_source="chain")
                ar.append(a)
                print(f"  AR  run {run + 1}: tok_s={a['tok_s']:.2f}  "
                      f"tokens={a['tokens']}  "
                      f"prefill={a['prefill_s']:.2f}s  TTFT={a['ttft_s']:.2f}s  "
                      f"total={a['total_s']:.2f}s")
                if run_chain:
                    m = run_cell(1, prompt_path, prompt_id, n_prompt, args.n_gen,
                                 draft_source="chain")
                    mtp.append(m)
                    accept = m["accepted"] / max(1, m["proposed"])
                    print(f"  MTP run {run + 1}: tok_s={m['tok_s']:.2f}  "
                          f"tokens={m['tokens']}  "
                          f"prefill={m['prefill_s']:.2f}s  TTFT={m['ttft_s']:.2f}s  "
                          f"total={m['total_s']:.2f}s  accept={accept * 100:.1f}%")
                if run_topk:
                    t = run_cell(1, prompt_path, prompt_id, n_prompt, args.n_gen,
                                 draft_source="mtp_topk",
                                 draft_topk=args.draft_topk,
                                 ddtree_budget=args.ddtree_budget,
                                 ddtree_chain_seed=not args.ddtree_no_chain_seed)
                    topk_runs.append(t)
                    accept = t["accepted"] / max(1, t["proposed"])
                    print(f"  TOPK run {run + 1} (K={args.draft_topk}, budget={args.ddtree_budget}, "
                          f"chain_seed={not args.ddtree_no_chain_seed}): "
                          f"tok_s={t['tok_s']:.2f}  tokens={t['tokens']}  "
                          f"accept={accept * 100:.1f}%")
            if not mtp:
                # Comparison columns need an "mtp" cell; degenerate to AR if
                # the user only ran --draft-source mtp_topk so the summary
                # table still computes.
                mtp = list(ar)

            def med(xs, k):
                return statistics.median(x[k] for x in xs)

            ar_med = {k: med(ar, k) for k in ar[0]}
            mtp_med = {k: med(mtp, k) for k in mtp[0]}
            mtp_accept = sum(x["accepted"] for x in mtp) / max(
                1, sum(x["proposed"] for x in mtp))
            entry = {
                "bucket": bucket,
                "n_prompt": n_prompt,
                "n_gen": args.n_gen,
                "ar": ar_med,
                "mtp": mtp_med,
                "mtp_accept_rate": mtp_accept,
                "decode_speedup": mtp_med["tok_s"] / ar_med["tok_s"],
                "wall_speedup": ar_med["total_s"] / mtp_med["total_s"],
            }
            if topk_runs:
                topk_med = {k: med(topk_runs, k) for k in topk_runs[0]
                            if isinstance(topk_runs[0][k], (int, float))}
                entry["mtp_topk"] = topk_med
                entry["mtp_topk_accept_rate"] = (
                    sum(x["accepted"] for x in topk_runs) /
                    max(1, sum(x["proposed"] for x in topk_runs)))
                entry["topk_decode_speedup"] = topk_med["tok_s"] / ar_med["tok_s"]
            bucket_records.append(entry)
        aggregate[bucket] = bucket_records

    print("\n=== summary (median per bucket) ===")
    print(f"{'bucket':>6}  {'n_prompt':>8}  "
          f"{'AR tok/s':>9}  {'MTP tok/s':>10}  "
          f"{'AR pre':>7}  {'MTP pre':>8}  "
          f"{'AR TTFT':>8}  {'MTP TTFT':>9}  "
          f"{'AR total':>9}  {'MTP total':>10}  "
          f"{'decSp':>6}  {'wallSp':>7}  {'accept':>7}")
    summary_lines = []
    for bucket in buckets:
        recs = aggregate[bucket]
        med_ar_tok = statistics.median(r["ar"]["tok_s"] for r in recs)
        med_mtp_tok = statistics.median(r["mtp"]["tok_s"] for r in recs)
        med_ar_pre = statistics.median(r["ar"]["prefill_s"] for r in recs)
        med_mtp_pre = statistics.median(r["mtp"]["prefill_s"] for r in recs)
        med_ar_ttft = statistics.median(r["ar"]["ttft_s"] for r in recs)
        med_mtp_ttft = statistics.median(r["mtp"]["ttft_s"] for r in recs)
        med_ar_total = statistics.median(r["ar"]["total_s"] for r in recs)
        med_mtp_total = statistics.median(r["mtp"]["total_s"] for r in recs)
        med_dec = med_mtp_tok / med_ar_tok
        med_wall = med_ar_total / med_mtp_total
        med_accept = statistics.median(r["mtp_accept_rate"] for r in recs)
        med_n_prompt = int(statistics.median(r["n_prompt"] for r in recs))
        line = (f"{bucket:>6}  {med_n_prompt:>8}  "
                f"{med_ar_tok:>9.2f}  {med_mtp_tok:>10.2f}  "
                f"{med_ar_pre:>7.3f}  {med_mtp_pre:>8.3f}  "
                f"{med_ar_ttft:>8.3f}  {med_mtp_ttft:>9.3f}  "
                f"{med_ar_total:>9.2f}  {med_mtp_total:>10.2f}  "
                f"{med_dec:>6.2f}  {med_wall:>7.2f}  {100 * med_accept:>6.1f}%")
        print(line)
        summary_lines.append(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
