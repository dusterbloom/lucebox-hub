"""Rung A1 driver (PREREG section A1 + Amendments 1 and 2). Machine-agnostic; paths from a1_core.default_paths.
  stage 1 (probes only, decides):   python3 a1_run.py --out OUT --model M [--control-n 8]
  stage 2 (decisions, behind flag): python3 a1_run.py --out OUT --model M --stage2 [--decision-stride 2]
Resume: rerun the same command; finished tasks are skipped per trajectory and arm.
Pilot (Amendment 2): --n 6 --gate3-after 6 --det-repeat 3 (decides nothing)."""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as core  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--arms", default="FULL,SHIP")
    ap.add_argument("--primary", default="SHIP", help="arm the early look reads (A2: WIN18)")
    ap.add_argument("--stage2", action="store_true", help="also run the decision tasks")
    ap.add_argument("--decision-stride", type=int, default=2, help="stage 2 subsample: positions 2, 4, 6, ... (Gate 2)")
    ap.add_argument("--gate3-after", type=int, default=5)
    ap.add_argument("--early-at", type=int, default=core.EARLY_AT)
    ap.add_argument("--control-n", type=int, default=0, help="block-512 control on the first N trajectories (probes)")
    ap.add_argument("--det-repeat", type=int, default=0, help="re-answer the first N items of trajectory 1 and compare")
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    ap.add_argument("--ratio", type=float, default=0.10, help="slots kept per sealed block; 1.0 = D1 identity diagnostic, never for A1")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    import tool_parse
    H = core.Harness(core.default_paths())
    m = core.load_model(a.model, a.mem_cap_gib, log, ratio=a.ratio)
    ctx = SimpleNamespace(S=m.S, R=m.R, H=H, parse=tool_parse.parse)
    kinds = ("probe", "decision") if a.stage2 else ("probe",)
    summary = core.run_stage(ctx, a.out, kinds=kinds, arms=tuple(a.arms.split(",")), start=a.start, n=a.n,
                             gate3_after=a.gate3_after, early_at=a.early_at,
                             decision_stride=a.decision_stride if a.stage2 else 1,
                             det_repeat=a.det_repeat, control_n=a.control_n, log=log, primary=a.primary)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[a1] done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
