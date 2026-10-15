"""Diagnostics E6 (repeat-prefill fit queries) and E7 (system and tool prefix exact) of PREREG "Diagnostics E6, E7".
Probes only, on the A1 pilot trajectories; the shipping arm (ratio 0.10, 640-token blocks) with one change each:
  E6: each block is fitted on the queries of a repeat-prefill pass over a clone (r2_core.repeat_queries)
  E7: the sealed blocks overlapping [0, end of the system message with its tool schema) stay exact (r2_core exact_blocks)
  python3 a1_e67.py --arm E6|E7 --out ~/r1e/out/e6 [--n 6]      (FULL.jsonl, the pilot's FULL-CHK rows, must already be in --out)"""
import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as core  # noqa: E402

BLOCK = 640
INSTRUCTION = "Repeat the previous context verbatim."      # compaction/query_generation/conversation_specs.py, spec "repeat"


def repeat_prompt_text(tok):
    """The repo's repeat spec as self_study.py builds it: the starter as one user turn, generation prompt, thinking off."""
    return tok.apply_chat_template([{"role": "user", "content": INSTRUCTION}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def system_end(stream, im_end):
    """One past the first <|im_end|>: the rendered system message (tool schema included) is the first message."""
    return list(stream).index(im_end) + 1


def prefix_blocks(n_tokens, block=BLOCK):
    return math.ceil(n_tokens / block)


def run_trajectory(ctx, arm, out, tid, traj, tasks, full_rows, prompt_ids, log):
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    path = os.path.join(out, arm + ".jsonl")
    done = {r["id"] for r in core.read_rows(path)}
    ret = core.retained_probe_ids(H, tasks, full_rows)
    todo = {t["id"] for t in core.pending([t for t in tasks if t["id"] in ret], done)}
    if not todo:
        return None
    plan = core.restrict(core.plan_trajectory(H, traj, [t for t in tasks if t["id"] in todo]), todo)
    kw, extra = {}, {}
    if arm == "E6":
        kw["repeat_prompt"] = prompt_ids
    else:
        end = system_end(plan.stream, S.tok.convert_tokens_to_ids("<|im_end|>"))
        head = S.tok.decode(plan.stream[:3], skip_special_tokens=False)
        assert head.startswith("<|im_start|>system"), head
        kw["exact_blocks"] = prefix_blocks(end)
        extra = {"system_end_tok": end, "exact_blocks": kw["exact_blocks"], "exact_tokens": kw["exact_blocks"] * BLOCK}
        log(f"[{arm}] {tid[-8:]} system+tools end at token {end}: {kw['exact_blocks']} exact blocks = {extra['exact_tokens']} tokens")

    def gen(R_, segs, articles, ks, **k):
        return r2.gen_fblk_window(R_, segs, ks, 0, **k)

    st = core.run_unit(S, R, plan, gen, ctx.parse, lambda row: core.append_row(path, row), arm, log=log, **kw)
    return {**st, **extra}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("E6", "E7"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--mem-cap-gib", type=float, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    import tool_parse
    H = core.Harness(core.default_paths())
    m = core.load_model(a.model, a.mem_cap_gib, log)
    ctx = SimpleNamespace(S=m.S, R=m.R, H=H, parse=tool_parse.parse)
    prompt_ids = None
    if a.arm == "E6":
        text = repeat_prompt_text(m.tok)
        prompt_ids = torch.tensor([m.tok(text, add_special_tokens=False)["input_ids"]])
        log(f"[E6] repeat prompt ({prompt_ids.shape[1]} tokens): {text!r}")
    trajs = H.load_trajs()
    tasks = H.load_tasks()
    full_rows = core.read_rows(os.path.join(a.out, "FULL.jsonl"))
    for tid in list(trajs)[: a.n]:
        tp = [t for t in tasks if t["traj"] == tid and t["kind"] == "probe"]
        t0 = time.time()
        st = run_trajectory(ctx, a.arm, a.out, tid, trajs[tid], tp, [r for r in full_rows if r["traj"] == tid], prompt_ids, log)
        if st:
            core.append_row(os.path.join(a.out, "progress.jsonl"), {**st, "arm": a.arm, "wall_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
