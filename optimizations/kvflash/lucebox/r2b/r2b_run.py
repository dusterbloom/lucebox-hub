"""R2b driver (PREREG "Rung R2b"): FULL-CHK vs F-BLK-UNI (the shipping uniform-block design) with k distractor articles in
front of the questioned article, QuALITY, Qwen3.8-27B, lucebox6 iGPU (ROCm torch, HIP_VISIBLE_DEVICES=1).

  source $R1E/code/r1e/env.sh   # PYTHONPATH has the pinned compaction checkout and modal/am_qwen35 (r1b_core, r2_core)
  python r2b_run.py --splits held,sel --ks 1,8 --arms FULL-CHK,F-BLK-UNI --out $R1E/out/r2b

Stage 1 = split `held` (loader indices 20-49), stage 2 = split `sel` (0-19), in loader order, one article at a time through
r2_core.run_r2 (rows <arm>__k<k>__t0.1__<split>.jsonl; finished (arm, k, article) cells are skipped, so a restart resumes).
--repeat N re-runs the shipping arm at k = 0 on the first N held-out articles (repeatability check, written to --out too).
--max_hours stops between articles once the wall budget is spent (PREREG Gate 2: 24 h).
"""
import argparse
import os
import time

DEFAULT_ARMS = "FULL-CHK,F-BLK-UNI"


def stage_plan(splits, start=0, n=None):
    """[(split, loader index)] in run order: splits in the order given, indices in loader order."""
    import r1b_core as core
    return [(sp, ai) for sp in splits for ai in core.split_range(sp, start, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--splits", default="held,sel", help="comma list, run in this order; held = stage 1, sel = stage 2")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="articles per split (default: all)")
    ap.add_argument("--ks", default="1,8")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_questions", type=int, default=None, help="smoke only: first N questions of each article")
    ap.add_argument("--repeat", type=int, default=0, help="after the first split, re-run F-BLK-UNI at k = 0 on its first N articles")
    ap.add_argument("--max_hours", type=float, default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import r1a_core
    import r2_core as r2
    from evaluation.datasets import load_dataset
    from models.qwen3_5 import HFGenerateShim

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    r1a_core.set_attn(model, "sdpa")
    gpu = torch.cuda.get_device_name(0)
    print(f"[r2b] model loaded in {time.time() - t0:.0f}s on {gpu} torch {torch.__version__}", flush=True)
    shim = HFGenerateShim(model, tok, seed=1)
    data = load_dataset("quality")
    ks, arms = [int(x) for x in a.ks.split(",")], a.arms.split(",")
    splits = a.splits.split(",")
    deadline = None if a.max_hours is None else t0 + 3600 * a.max_hours

    def go(split, ai, ks_, arms_):
        ta = time.time()
        r2.run_r2(model, tok, shim, a.out, [ai], ks_, arms_, None, split=split, n_questions=a.n_questions, data=data,
                  gpu=gpu, log=lambda s: print(s, flush=True))
        print(f"[r2b] {split} article {ai} done in {time.time() - ta:.0f}s (elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)

    for si, split in enumerate(splits):
        for _, ai in stage_plan([split], a.start, a.n):
            if deadline and time.time() > deadline:
                print(f"[r2b] wall budget of {a.max_hours} h spent: stopping before {split} article {ai}", flush=True)
                return
            go(split, ai, ks, arms)
        if si == 0 and a.repeat:
            for _, ai in stage_plan([split], a.start, a.repeat):
                go(split, ai, [0], ["F-BLK-UNI"])
    print(f"[r2b] all done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
