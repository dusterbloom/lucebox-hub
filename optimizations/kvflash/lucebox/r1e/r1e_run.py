"""R1(e) driver (PREREG "R1(e) on lucebox6 iGPU"): ORIG vs F-UNI-GLOBAL vs F-UNI-LOCAL on QuALITY, Qwen3.8-27B.

Runs on lucebox6 (ROCm torch, HIP_VISIBLE_DEVICES=1) in the pinned compaction checkout with the modal/am_qwen35 python on
the path (r1b_core, r2_core, r1a_core are reused unchanged):
  PYTHONPATH=$R1E/code/repo:$R1E/code/am_qwen35:<this dir> python r1e_run.py --split sel --out $R1E/out/sel

  ORIG           uncompacted context (R1b `original_triples`).
  F-UNI-GLOBAL   R1b cell FUNI at 0.10: one compaction of the whole article, whole-article context-prefill queries.
  F-UNI-LOCAL    r2_core.gen_fblk with the FUNI compactor: sealed 640-token blocks of the formatted context, each compacted
                 once to 64 slots per head (uniform) from its own prefill queries behind the compacted prefix; open tail exact.
Rows are appended per (arm, split) as jsonl; finished articles are skipped (resume per article).
"""
import argparse
import json
import os
import time

RATIO = 0.10
ARMS = ("ORIG", "F-UNI-GLOBAL", "F-UNI-LOCAL")


def row_path(out, arm, split):
    return f"{out}/{arm}__t{RATIO}__{split}.jsonl"


def block_plan(seq_len, block=640):
    """(sealed (start, end) blocks, exact tail length) for a context of `seq_len` tokens, as gen_fblk cuts it."""
    blocks, end = [], 0
    while end + block <= seq_len:
        blocks.append((end, end + block))
        end += block
    return blocks, seq_len - end


def local_triples(R, S, article):
    """The F-UNI-LOCAL cache for one article: gen_fblk at checkpoint k = 0 over the single segment (the R1b context)."""
    import r2_core as r2
    segs, _ = r2.build_segments(S.tok, [article])
    (_, triples, orig_len, info), = list(r2.gen_fblk(R, segs, [article], [0]))
    return triples, orig_len, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/r1e/models/Qwen3.8-27B"))
    ap.add_argument("--split", default="sel")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_questions", type=int, default=None, help="smoke only: first N questions of each article")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import r1a_core
    import r1b_core as core
    import r2_core as r2
    from evaluation.datasets import load_dataset
    from models.qwen3_5 import HFGenerateShim

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    r1a_core.set_attn(model, "sdpa")
    gpu = torch.cuda.get_device_name(0)
    print(f"[r1e] model loaded in {time.time() - t0:.0f}s on {gpu} torch {torch.__version__}", flush=True)
    S = core.Session(model, tok, HFGenerateShim(model, tok, seed=1))
    R = r2.Runner(S, None, cell="FUNI")
    data = load_dataset("quality")
    arms = a.arms.split(",")
    os.makedirs(a.out, exist_ok=True)
    for ai in core.split_range(a.split, a.start, a.n):
        art = data[ai]
        aid = str(art["article_id"])
        todo = [m for m in arms if aid not in core.done_articles(row_path(a.out, m, a.split))]
        if not todo:
            continue
        qs = art["questions"][:a.n_questions] if a.n_questions else art["questions"]
        ta = time.time()
        for arm in todo:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            if arm == "F-UNI-LOCAL":
                triples, seq_len, info = local_triples(R, S, art["article"])
            else:
                pf = S.prefill(art["article"])
                torch.cuda.synchronize()  # GPU work is async; time what was actually computed
                seq_len = pf["seq_len"]
                info = {"prefill_s": round(time.time() - t1, 2)}
                if arm == "ORIG":
                    triples = S.original_triples(pf)
                else:
                    triples, more = S.compact("FUNI", pf, RATIO, None)
                    info.update(more)
                del pf
            torch.cuda.synchronize()
            build_s = round(time.time() - t1, 1)
            t2 = time.time()
            res = S.answer(triples, seq_len, qs)
            ans_s = round(time.time() - t2, 1)
            rows = core._rows(art, res, qs, arm, {
                "arm": arm, "article_index": ai, "target": RATIO, "seq_len": seq_len, "build_s": build_s,
                "answer_s": ans_s, "n_questions": len(qs), "gpu": gpu,
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), **info})
            with open(row_path(a.out, arm, a.split), "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[r1e] art {ai} {aid} {arm}: acc {sum(r['correct'] for r in rows)}/{len(rows)} seq {seq_len} "
                  f"build {build_s}s answer {ans_s}s peak {rows[0]['peak_gb']}GB {info}", flush=True)
            del triples
        print(f"[r1e] article {ai} done in {time.time() - ta:.0f}s", flush=True)


if __name__ == "__main__":
    main()
