#!/usr/bin/env python3
"""Decompose steady-state T=1 decode time from a rocprofv3 --kernel-trace CSV.

Usage: python3 analyze.py run_kernel_trace.csv [run_index] [skip_steps] > report.json

run_index: 0-based index of the forward() call to analyze (driver-stack native
  mode schedule: run0=warm mode0, run1=warm mode1, run2=rep0 mode0, run3=rep1
  mode1, run4=rep2 mode0, run5=rep3 mode1, run6=rep4 mode1, run7=rep5 mode0,
  run8=rep6 mode1, run9=rep7 mode0). Default 3 (rep1 mode=1).
skip_steps: number of leading decode steps to exclude as warm-up. Default 16.

Run on lucebox4 (box scratch: ~/qwen4exp-exact-stack/prof/). Produced
bench/exact/PROFILE-38.md on 2026-10-08 against driver-stack (exact-stack,
LUCE_QWEN_* gates per run_stack.sh), model Qwen3.8-Flash-Next-UD-Q4_K_XL,
prompt.ids/follow.ids from pr823-release-20261007/artifacts, args `<gguf>
prompt.ids 256 8` (native decode-census mode; see driver_shared_epilogue.cpp
main() -- 256/2 is NOT a valid arg combo, it returns exit code 2).
"""
import csv, sys, array, json
from collections import defaultdict

RUN_LEN = 256  # tokens per forward() call (driver-stack arg3)
N_RUNS = 10    # 2 warm + 8 measured (driver-stack arg4=8)

def classify(name: str) -> str:
    """weight-streaming (scales with weight bytes read) vs everything-else.
    Per task spec: mul_mat_vec_q/mmvq, mul_mat_id/MoE expert GEMV, mmvf,
    dequant-GEMV, lm_head -> weight_stream. hc_*, norms, quantize_q8_1, gdn,
    attention/QSA, argsort/topk, copies, elementwise -> other.
    """
    if name.startswith("void mul_mat_vec_q") or name.startswith("void mul_mat_vec_f"):
        return "weight_stream"
    if name.startswith("void mmb_") or name.startswith("mmb_") or "mmb_" in name[:40]:
        return "weight_stream"
    return "other"

def main():
    path = sys.argv[1]
    run_index = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    skip_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    name_to_id = {}
    id_to_name = []
    name_ids = array.array('i')
    starts = array.array('q')
    ends = array.array('q')

    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        idx_name = header.index("Kernel_Name")
        idx_start = header.index("Start_Timestamp")
        idx_end = header.index("End_Timestamp")
        for row in r:
            name = row[idx_name]
            nid = name_to_id.get(name)
            if nid is None:
                nid = len(id_to_name)
                name_to_id[name] = nid
                id_to_name.append(name)
            name_ids.append(nid)
            starts.append(int(row[idx_start]))
            ends.append(int(row[idx_end]))

    N = len(starts)
    sys.stderr.write(f"loaded {N} kernel-dispatch rows, {len(id_to_name)} distinct names\n")

    argmax_id = None
    for nm, nid in name_to_id.items():
        if nm.startswith("void argmax_f32") or nm.startswith('"void argmax_f32'):
            argmax_id = nid
            break
    if argmax_id is None:
        sys.stderr.write("ERROR: no argmax_f32 kernel found; names sample:\n")
        for nm in id_to_name[:40]:
            sys.stderr.write(f"  {nm}\n")
        sys.exit(1)

    argmax_rows = [i for i in range(N) if name_ids[i] == argmax_id]
    sys.stderr.write(f"argmax launches total={len(argmax_rows)} (expect {N_RUNS*RUN_LEN})\n")
    if len(argmax_rows) != N_RUNS * RUN_LEN:
        sys.stderr.write("WARNING: argmax count does not match N_RUNS*RUN_LEN; proceeding anyway\n")

    block = argmax_rows[run_index * RUN_LEN:(run_index + 1) * RUN_LEN]
    assert len(block) == RUN_LEN, f"expected {RUN_LEN} argmax events in run {run_index}, got {len(block)}"

    per_token = []
    gap_hist = {"<2us": [0, 0.0], "2-5us": [0, 0.0], "5-20us": [0, 0.0], ">20us": [0, 0.0]}
    kernel_launch = defaultdict(int)
    kernel_dur = defaultdict(float)  # us, sum of raw durations (not de-overlapped)
    cat_dur_token = defaultdict(float)  # us per category, summed over tokens analyzed
    biggest_gaps = []  # (gap_us, token_idx, where)

    n_tok = 0
    for t in range(skip_steps, RUN_LEN):
        prev_row = block[t - 1]
        cur_row = block[t]
        row_lo, row_hi = prev_row + 1, cur_row
        wall_ns = ends[cur_row] - ends[prev_row]
        launches = row_hi - row_lo + 1

        intervals = sorted((starts[i], ends[i], i) for i in range(row_lo, row_hi + 1))
        merged = []
        cur_s, cur_e = intervals[0][0], intervals[0][1]
        for s, e, _ in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        busy_ns = sum(e - s for s, e in merged)
        idle_ns = wall_ns - busy_ns

        token_start = ends[prev_row]
        prev_end = token_start
        for s, e in merged:
            gap = s - prev_end
            if gap > 0:
                gap_us = gap / 1000.0
                if gap_us < 2:
                    b = "<2us"
                elif gap_us < 5:
                    b = "2-5us"
                elif gap_us < 20:
                    b = "5-20us"
                else:
                    b = ">20us"
                gap_hist[b][0] += 1
                gap_hist[b][1] += gap_us
                biggest_gaps.append((gap_us, t, f"before {s}"))
            prev_end = e
        tail_gap = (token_start + wall_ns) - prev_end
        if tail_gap > 0:
            gap_us = tail_gap / 1000.0
            if gap_us < 2:
                b = "<2us"
            elif gap_us < 5:
                b = "2-5us"
            elif gap_us < 20:
                b = "5-20us"
            else:
                b = ">20us"
            gap_hist[b][0] += 1
            gap_hist[b][1] += gap_us
            biggest_gaps.append((gap_us, t, "tail-of-token"))

        for i in range(row_lo, row_hi + 1):
            nm = id_to_name[name_ids[i]]
            dur_us = (ends[i] - starts[i]) / 1000.0
            kernel_launch[nm] += 1
            kernel_dur[nm] += dur_us
            cat_dur_token[classify(nm)] += dur_us

        per_token.append(dict(t=t, wall_us=wall_ns / 1000.0, busy_us=busy_ns / 1000.0,
                               idle_us=idle_ns / 1000.0, launches=launches))
        n_tok += 1

    biggest_gaps.sort(reverse=True)

    wall_tot = sum(p["wall_us"] for p in per_token)
    busy_tot = sum(p["busy_us"] for p in per_token)
    idle_tot = sum(p["idle_us"] for p in per_token)
    launches_tot = sum(p["launches"] for p in per_token)

    out = {
        "run_index": run_index, "skip_steps": skip_steps, "n_tokens_analyzed": n_tok,
        "wall_us_per_token": wall_tot / n_tok,
        "busy_us_per_token": busy_tot / n_tok,
        "idle_us_per_token": idle_tot / n_tok,
        "launches_per_token": launches_tot / n_tok,
        "weight_stream_us_per_token": cat_dur_token["weight_stream"] / n_tok,
        "other_us_per_token": cat_dur_token["other"] / n_tok,
        "gap_hist": {k: {"count": v[0], "count_per_token": v[0] / n_tok, "total_us": v[1], "us_per_token": v[1] / n_tok} for k, v in gap_hist.items()},
        "biggest_gaps_top20": biggest_gaps[:20],
        "per_token_wall_us": [round(p["wall_us"], 2) for p in per_token],
        "kernel_table": sorted(
            [
                {
                    "name": nm,
                    "launches_total": kernel_launch[nm],
                    "launches_per_token": kernel_launch[nm] / n_tok,
                    "us_per_token": kernel_dur[nm] / n_tok,
                    "mean_us_per_launch": kernel_dur[nm] / kernel_launch[nm],
                    "category": classify(nm),
                }
                for nm in kernel_launch
            ],
            key=lambda d: -d["us_per_token"],
        ),
    }
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
