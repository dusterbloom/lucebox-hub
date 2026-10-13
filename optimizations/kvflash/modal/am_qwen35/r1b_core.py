"""R1b/R1c harness (see evidence/am/PREREG.md, sections R1b/R1c and R1b Amendment 1).

Runs inside the Modal image (cwd /repo, compaction repo with patches 0001-0003 and the overlay applied).
  run_cells    QuALITY cells: per article one prefill, then `original` and every requested configuration, per-question rows
               appended to a jsonl per (cell, target, split) on the results volume; finished articles are skipped (resume).
  run_curves   R1(c): per-head influence curves for fit articles (repo HeadInfluenceComputer, hybrid-patched).
  solve_budgets / run_validation   R1(c): ratio-agnostic solver, budget file, paired check on held-out fit articles.
"""
import json
import math
import os
import time

import torch

NAME = "Qwen/Qwen3.8-27B"
N_NEW = 256
MAX_RATIO_PER_HEAD = 0.75
SPLITS = {"sel": (0, 20), "held": (20, 50), "fit": (90, 95), "val": (95, 100)}

HA = "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy"
OMPF = "omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy"
# id -> (algorithm config file, method name, query config, per-head budget)
CELLS = {
    "F": ("best", HA, "context-prefill", True),
    "O1": ("best", HA, "repeat", True),
    "O2": ("best", OMPF, "repeat", True),
    "C": ("best", OMPF, "ss-plus-repeat", True),
    "UAM": ("best", HA, "repeat", False),
    "RND": ("key-selection", "random_subset_keys_nnls2_-3_3_lsq_on-policy", "repeat", False),
    "TRN": ("key-selection", "truncate_nnls2_-3_3_lsq_on-policy", "repeat", False),
}


def split_range(split, start=0, n=None):
    a, b = SPLITS[split]
    return range(a + start, b if n is None else min(b, a + start + n))


def done_articles(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["article_id"] for l in open(path) if l.strip()}


def cell_path(out, cell, target, split):
    return f"{out}/{cell}__t{target}__{split}.jsonl"


def method_for(cell, budget_path):
    """The repo's compaction method object for a cell id."""
    import evaluation  # noqa: F401  (import order: evaluation <-> compaction circular import)
    from compaction.compaction_methods.registry import get_compaction_method
    from evaluation.configs.utils import load_algorithm_config
    cfg_file, name, _, per_head = CELLS[cell]
    kw = dict(load_algorithm_config(cfg_file)[name])
    if per_head:
        kw["precomputed_budget_path"] = budget_path
        kw["max_ratio_per_head"] = MAX_RATIO_PER_HEAD
    return get_compaction_method(name, method_kwargs=kw)


class Session:
    """Model + tokenizer + cache plumbing shared by all cells of a job. Hybrid (Qwen3.5/3.8: DeltaNet layers carried
    through the adapter) or plain full-attention (Qwen3-4B on the repo's own modeling, PREREG R1b Amendment 1)."""

    def __init__(self, model, tok, shim=None, name=NAME):
        self.model, self.tok, self.shim, self.name = model, tok, shim, name
        self.dev = next(model.parameters()).device
        self.layer_types = getattr(model.config.get_text_config(), "layer_types", None)
        self.hybrid = bool(self.layer_types) and "linear_attention" in self.layer_types
        self.linear = {i for i, t in enumerate(self.layer_types or []) if t == "linear_attention"}
        self.eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} - {None}
        if self.hybrid:
            from models.qwen3_5 import adapt_model
            adapt_model(model)

    def prefill(self, article):
        from evaluation.utils import compute_article_indices, format_context
        formatted = format_context(self.tok, article, None, self.name)
        ids = self.tok(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.dev)
        pf = {"formatted": formatted, "seq_len": ids.shape[1], "dyn": None,
              "article_idx": compute_article_indices(self.tok, formatted, article)}
        with torch.no_grad():
            if self.hybrid:
                from models.qwen3_5 import dynamic_cache_to_pseudo_kv
                pf["dyn"] = self.model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
                pf["kv"] = dynamic_cache_to_pseudo_kv(pf["dyn"], self.layer_types)
            else:
                pf["kv"] = self.model(ids, use_cache=True).past_key_values
        return pf

    def original_triples(self, pf):
        trip = tuple((k, k.new_zeros(k.shape[:3]), v) for k, v in pf["kv"])
        if not self.hybrid:
            return trip
        from models.qwen3_5 import inject_linear_state
        return inject_linear_state(trip, pf["dyn"], self.layer_types)

    def _cache(self, triples, seq_len):
        if self.hybrid:
            from models.qwen3_5 import compacted_cache_from_triples
            return compacted_cache_from_triples(triples, self.linear, seq_len)
        from models.cache import CompactedPrefixCache
        return CompactedPrefixCache(tuple((a.clone(), b.clone(), c.clone()) for a, b, c in triples),
                                    original_seq_len=seq_len)

    @torch.no_grad()
    def _greedy(self, cache, first_logits, n_new):
        toks, logits = [], first_logits
        for step in range(n_new):
            nxt = int(logits[0, -1].argmax())
            toks.append(nxt)
            if nxt in self.eos or step == n_new - 1:
                break
            logits = self.model(input_ids=torch.tensor([[nxt]], device=self.dev), past_key_values=cache,
                                use_cache=True).logits
        return toks

    def answer(self, triples, seq_len, questions, max_new=N_NEW, thinking=False):
        """Greedy answers to every question on one prefix. Returns [(text, finite_logits)]."""
        from evaluation.utils import format_question
        res = []
        for q in questions:
            prompt = format_question(self.tok, q["question"], q["options"], self.name, enable_thinking=thinking)
            qids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.dev)
            cache = self._cache(triples, seq_len)
            with torch.no_grad():
                o = self.model(input_ids=qids, past_key_values=cache, use_cache=True, **({'logits_to_keep': 1} if self.hybrid else {}))
            res.append((self.tok.decode(self._greedy(cache, o.logits, max_new), skip_special_tokens=True),
                        bool(torch.isfinite(o.logits[:, -1]).all())))
        return res

    def compact(self, cell, pf, target, budget_path):
        from evaluation.configs.utils import load_query_config
        from models.generate import get_sliding_layer_info
        n_art = len(pf["article_idx"])
        actual = max(1, int(n_art * target)) + (pf["seq_len"] - n_art)
        sliding, _ = get_sliding_layer_info(self.model)
        extra = {"past_key_values_for_queries": pf["dyn"]} if self.hybrid else {}
        t0 = time.time()
        compacted, stats = method_for(cell, budget_path).compact_kv_cache(
            past_key_values=pf["kv"], target_size=actual, indices=pf["article_idx"],
            query_config=load_query_config(CELLS[cell][2]), model=self.model, tokenizer=self.tok,
            formatted_context=pf["formatted"], compute_stats=False, vllm_model=self.shim,
            sliding_layer_indices=sliding, **extra)
        wall = time.time() - t0
        qs = stats.get("query_generation", {}) if isinstance(stats, dict) else {}
        info = {"compaction_s": round(wall, 2), "query_gen_s": round(qs.get("query_generation_time", 0.0), 2)}
        triples = compacted
        if self.hybrid:
            from models.qwen3_5 import inject_linear_state
            triples = inject_linear_state(compacted, pf["dyn"], self.layer_types)
        full = [t[0].shape[2] for i, t in enumerate(triples) if i not in self.linear]
        info["full_layer_len_mean"] = round(sum(full) / len(full), 1)
        info["nan_layers"] = [i for i, t in enumerate(triples) if i not in self.linear and
                              any(torch.isnan(x).any() for x in t)]
        return triples, info


def _rows(art, results, questions, cell, extra):
    from evaluation.utils import parse_model_choice
    out = []
    for qi, ((text, finite), q) in enumerate(zip(results, questions)):
        ch = parse_model_choice(text)
        out.append({"cell": cell, "article_id": str(art["article_id"]), "qi": qi, "gold": q["gold_label"],
                    "parsed": ch, "correct": bool(ch and ch == q["gold_label"]), "finite": finite,
                    "answer": text.strip()[:200], **extra})
    return out


def run_cells(model, tok, shim, out, cells, split, target, budget_path, articles, log=print, data=None, name=NAME):
    """Per article: prefill once, `original` (if requested as 'orig') and each configuration; resume by article."""
    from evaluation.datasets import load_dataset
    data = data or load_dataset("quality")
    S = Session(model, tok, shim, name)
    os.makedirs(out, exist_ok=True)
    for ai in articles:
        art = data[ai]
        aid = str(art["article_id"])
        todo = [c for c in cells if aid not in done_articles(cell_path(out, c, target, split))]
        if not todo:
            continue
        pf = S.prefill(art["article"])
        qs = art["questions"]
        t0 = time.time()
        for c in todo:
            t1 = time.time()
            if c == "orig":
                info = {}
                res = S.answer(S.original_triples(pf), pf["seq_len"], qs)
            else:
                triples, info = S.compact(c, pf, target, budget_path)
                res = S.answer(triples, pf["seq_len"], qs)
            rows = _rows(art, res, qs, c, {"article_index": ai, "target": target, "seq_len": pf["seq_len"],
                                           "article_tokens": len(pf["article_idx"]), **info,
                                           "cell_wall_s": round(time.time() - t1, 1)})
            with open(cell_path(out, c, target, split), "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            log(f"[r1b] art {ai} {aid} {c} t{target}: acc {sum(r['correct'] for r in rows)}/{len(rows)} "
                f"wall {rows[0]['cell_wall_s']}s {info}")
        del pf
        torch.cuda.empty_cache()
        log(f"[r1b] article {ai} done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- R1(c)

def write_reference_answers(S, art, max_new=128, cache_dir="data/reference_answers_cache"):
    """Greedy non-thinking reference answers from the uncompacted model, in the repo's own cache-file format
    (so `get_or_generate_reference_answers` loads them instead of generating with vLLM). PREREG R1b Amendment 1."""
    import hashlib
    pf = S.prefill(art["article"])
    qs = art["questions"]
    res = S.answer(S.original_triples(pf), pf["seq_len"], qs, max_new=max_new)
    qids = [q.get("question_unique_id", f"q_{i}") for i, q in enumerate(qs)]
    h = hashlib.md5("_".join(sorted(qids)).encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    path = f"{cache_dir}/{NAME.split('/')[-1]}_{art['article_id']}_{h}.json"
    json.dump({"model_name": NAME, "article_id": art["article_id"], "question_ids": qids, "question_ids_hash": h,
               "max_new_tokens": max_new, "generations": [{"question_id": i, "generated_text": t}
                                                          for i, (t, _) in zip(qids, res)]}, open(path, "w"), indent=1)
    return pf, path


def uniform_baseline(S):
    heads = [(l, h) for l in range(len(S.layer_types)) if l not in S.linear
             for h in range(S.model.config.get_text_config().num_key_value_heads)]
    return {f"L{l}H{h}": 1.0 / len(heads) for l, h in heads}


def run_curves(model, tok, out, articles, target_ratio=0.05, n_eval=21, max_ratio=0.5, log=print, data=None):
    """Per-article head influence curves -> {out}/per_article_curves/article_{idx}.json (skips finished articles)."""
    import evaluation  # noqa: F401
    from evaluation.configs.utils import load_algorithm_config, load_query_config
    from evaluation.datasets import load_dataset
    from head_budget_optimization.influence import HeadInfluenceComputer, save_head_curves
    import numpy as np
    S = Session(model, tok)
    data = data or load_dataset("quality")
    kw = dict(load_algorithm_config("best")["highest_attn_keys_rms_nnls2_-3_3_lsq"])
    algo = kw.pop("algorithm")
    comp = HeadInfluenceComputer(model=model, tokenizer=tok, query_config=load_query_config("repeat"),
                                 algorithm_kwargs=kw, algorithm_name=algo, device=str(S.dev), enable_thinking=False)
    eval_ratios = np.linspace(0, max_ratio, n_eval).tolist()
    base = uniform_baseline(S)
    os.makedirs(f"{out}/per_article_curves", exist_ok=True)
    for ai in articles:
        path = f"{out}/per_article_curves/article_{ai}.json"
        if os.path.exists(path):
            continue
        art = data[ai]
        t0 = time.time()
        pf, ref = write_reference_answers(S, art)
        comp.query_cache = pf["dyn"]
        curves = comp.compute_all_head_curves_for_article(
            article_data=art, past_key_values=pf["kv"], article_indices=pf["article_idx"],
            formatted_context=pf["formatted"], baseline_proportions=base, target_ratio=target_ratio,
            eval_ratios=eval_ratios, model_name=NAME, max_new_tokens=128)
        save_head_curves(curves, path, {"article_idx": ai, "article_id": art["article_id"], "reference": ref,
                                        "eval_s": round(time.time() - t0)})
        log(f"[r1c] article {ai} curves done in {time.time() - t0:.0f}s")
        del pf
        torch.cuda.empty_cache()


def solve_budgets(curves_dir, out_dir, solve_ratios=(0.05, 0.10), num_layers=64, num_heads=4):
    """Aggregate per-article curves, ratio-agnostic solve; writes head_curves.json, optimized_agnostic.json."""
    from head_budget_optimization.influence import load_and_aggregate_article_curves, save_head_curves
    from head_budget_optimization.solver import HeadBudgetSolver
    curves, meta = load_and_aggregate_article_curves(curves_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_head_curves(curves, f"{out_dir}/head_curves.json", {"n_articles": len(meta)})
    solver = HeadBudgetSolver(head_curves=curves, num_layers=num_layers, num_heads=num_heads)
    props, stats = solver.solve_ratio_agnostic(target_ratios=list(solve_ratios), step_size=0.001, weights=None)
    solver.save_proportions(props, f"{out_dir}/optimized_agnostic.json")
    json.dump({"agnostic_stats": stats}, open(f"{out_dir}/solve_stats.json", "w"), default=str, indent=1)
    return props, stats


def run_validation(model, tok, out, budget_path, articles, target=0.10, log=print):
    """Mean log-perplexity of reference answers after compaction at `target`: per-head budgets vs uniform, paired by
    article. Same method (HA, NNLS/lsq, on-policy) and queries (`repeat`)."""
    from evaluation.datasets import load_dataset
    from evaluation.utils import compute_perplexity_on_compacted_cache, format_question
    S = Session(model, tok)
    data = load_dataset("quality")
    rows = []
    for ai in articles:
        art = data[ai]
        pf, _ = write_reference_answers(S, art)
        from evaluation.utils import get_or_generate_reference_answers
        refs = get_or_generate_reference_answers(article_id=art["article_id"], model=model, tokenizer=tok,
                                                 formatted_context=pf["formatted"], questions=art["questions"],
                                                 model_name=NAME, max_new_tokens=128)
        lp = {}
        for label, cell in (("per_head", "O1"), ("uniform", "UAM")):
            triples, info = S.compact(cell, pf, target, budget_path)
            vals = []
            for (qid, gen_ids, _), q in zip(refs, art["questions"]):
                qp = format_question(tok, q["question"], q["options"], NAME, enable_thinking=False)
                _, l = compute_perplexity_on_compacted_cache(model, tok, triples, gen_ids, qp, device="cuda",
                                                             original_seq_len=pf["seq_len"])
                vals.append(l)
            lp[label] = sum(vals) / len(vals)
        rows.append({"article_index": ai, "article_id": str(art["article_id"]), **lp})
        log(f"[r1c-val] {rows[-1]}")
        del pf
        torch.cuda.empty_cache()
    d = [r["per_head"] - r["uniform"] for r in rows]
    summ = {"target": target, "rows": rows, "mean_diff_per_head_minus_uniform": sum(d) / len(d),
            "n_articles": len(d), "per_head_le_uniform": sum(x <= 0 for x in d)}
    json.dump(summ, open(f"{out}/validation.json", "w"), indent=1)
    return summ
