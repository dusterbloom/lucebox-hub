"""R1a harness: HF transformers vs the Qwen3.5 Attention-Matching adapter without compaction, plus the smoke test.

Runs inside the Modal image (cwd /repo, the compaction repo with the patches and overlay applied).
Pure helpers (`compare_runs`, `first_divergence`) have no torch dependency beyond tensors and are unit tested.
"""
import json
import os
import sys
import time

import torch

SUFFIX_TOKENS = 32
N_NEW = 256
CTX_CHUNK = 4096

PROMPTS = [
    ("P1_short", "Explain in three paragraphs why the sky is blue and why sunsets are red."),
    ("P2_code", "Write a Python function that merges overlapping intervals, with a docstring and three unit tests, "
               "then explain its time complexity."),
    ("P3_qa2k", ("article", ["20041"], "\n\nQuestion: Summarize the story in five sentences.")),
    ("P4_qa8k", ("article", ["51350"], "\n\nQuestion: Summarize the story in five sentences.")),
    ("P5_qa16k", ("article", ["51350", "22876"], "\n\nQuestion: Summarize the two stories in five sentences.")),
]


def load_articles(path="/repo/data/QuALITY.v1.0.1.htmlstripped.dev"):
    arts = {}
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            arts.setdefault(str(d["article_id"]), d["article"].strip())
    return arts


def prompt_text(spec, arts):
    if isinstance(spec, str):
        return spec
    _, ids, tail = spec
    return "\n\n".join(arts[i] for i in ids) + tail


def prompt_ids(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"]


def first_divergence(a, b):
    """Index of the first differing token of two equal-length lists, or None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_runs(tok_a, log_a, tok_b, log_b):
    """Compare two decode runs: identical tokens, first-step and all-step max |logit diff| (float32), and the top-1/top-2
    margin of run b at the first divergence."""
    out = {"tokens_identical": tok_a == tok_b, "n_tokens": [len(tok_a), len(tok_b)]}
    div = first_divergence(tok_a, tok_b)
    out["first_divergence"] = div
    n = min(log_a.shape[0], log_b.shape[0])
    d = (log_a[:n].float() - log_b[:n].float()).abs().amax(dim=-1)  # (n,)
    out["first_step_max_logit_diff"] = float(d[0])
    out["all_steps_max_logit_diff"] = float(d.max())
    if div is not None and div < n:
        top2 = log_b[div].float().topk(2).values
        out["margin_at_divergence"] = float(top2[0] - top2[1])
        out["top1_logit_at_divergence"] = float(top2[0])
    return out


def set_attn(model, impl):
    """Switch the attention backend of an already loaded model."""
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:  # pragma: no cover
        for cfg in (model.config, model.config.get_text_config()):
            cfg._attn_implementation = impl


def ctx_prefill(model, text_cfg, ctx_ids, chunk=CTX_CHUNK):
    """Unmodified-HF prefill of the context into a hybrid DynamicCache, in chunks (same for every arm)."""
    from transformers import DynamicCache
    cache = DynamicCache(config=text_cfg)
    with torch.no_grad():
        for s in range(0, ctx_ids.shape[1], chunk):
            model(input_ids=ctx_ids[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


def run_one_prompt(model, tok, ids, attn_impl, n_new=N_NEW, suffix_tokens=SUFFIX_TOKENS, do_whole=True):
    """Run the three arms on one tokenised prompt. Returns a result dict (no tensors)."""
    from models.qwen3_5 import (adapt_model, compacted_cache_from_triples, dynamic_cache_to_triples,
                                greedy_decode, restore_model)
    from transformers import DynamicCache
    set_attn(model, attn_impl)
    text_cfg = model.config.get_text_config()
    layer_types = text_cfg.layer_types
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    n = ids.shape[1]
    suffix_tokens = min(suffix_tokens, n // 2)  # deviation from PREREG: P1 (29 tokens) is shorter than the 32-token suffix
    ctx, suf = ids[:, :-suffix_tokens], ids[:, -suffix_tokens:]
    res = {"attn_impl": attn_impl, "prompt_tokens": n, "ctx_tokens": ctx.shape[1], "suffix_tokens": suf.shape[1]}
    t0 = time.time()
    with torch.no_grad():
        cache0 = ctx_prefill(model, text_cfg, ctx)
        triples, linear = dynamic_cache_to_triples(cache0, layer_types)
        acache = compacted_cache_from_triples(triples, linear, ctx.shape[1])
        del triples

        out = model(input_ids=suf, past_key_values=cache0, use_cache=True, logits_to_keep=1)
        tok_split, log_split = greedy_decode(model, cache0, out.logits, n_new, keep_logits=True)
        del cache0, out

        adapt_model(model)
        try:
            out = model(input_ids=suf, past_key_values=acache, use_cache=True, logits_to_keep=1)
            tok_ad, log_ad = greedy_decode(model, acache, out.logits, n_new, keep_logits=True)
        finally:
            restore_model(model)
        del acache, out
    res["adapter_vs_split"] = compare_runs(tok_ad, log_ad, tok_split, log_split)
    res["tokens_split_head"] = tok_split[:16]
    res["wall_split_adapter_s"] = round(time.time() - t0, 1)

    if do_whole:
        t1 = time.time()
        try:
            with torch.no_grad():
                cache_w = DynamicCache(config=text_cfg)
                out = model(input_ids=ids, past_key_values=cache_w, use_cache=True, logits_to_keep=1)
                tok_whole, log_whole = greedy_decode(model, cache_w, out.logits, n_new, keep_logits=True)
                del cache_w, out
            res["whole_vs_split"] = compare_runs(tok_whole, log_whole, tok_split, log_split)
            res["adapter_vs_whole_tokens_identical"] = tok_ad == tok_whole
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res["whole_vs_split"] = "oom"
        res["wall_whole_s"] = round(time.time() - t1, 1)
    res["tokens_split_text"] = tok.decode(tok_split, skip_special_tokens=False)
    res["tokens_adapter_text_equal_split_text"] = tok_ad == tok_split
    return res


def passes(res_by_prompt):
    """Gating criterion of the PREREG for one backend: tokens identical on every prompt and first-step logit diff <= 1e-2."""
    ok_tokens = all(r["adapter_vs_split"]["tokens_identical"] for r in res_by_prompt.values())
    ok_logits = all(r["adapter_vs_split"]["first_step_max_logit_diff"] <= 1e-2 for r in res_by_prompt.values())
    return {"tokens_identical_all": ok_tokens, "first_step_logit_diff_ok": ok_logits, "pass": ok_tokens and ok_logits,
            "n_identical": sum(r["adapter_vs_split"]["tokens_identical"] for r in res_by_prompt.values()),
            "n_prompts": len(res_by_prompt)}


def run_r1a(model, tok, backends=("eager", "sdpa"), prompts=PROMPTS, n_new=N_NEW, out_path=None):
    arts = load_articles()
    results = {}
    for impl in backends:
        per = {}
        for name, spec in prompts:
            ids = prompt_ids(tok, prompt_text(spec, arts))
            print(f"[R1a] {impl} {name}: {ids.shape[1]} tokens", flush=True)
            per[name] = run_one_prompt(model, tok, ids, impl, n_new=n_new)
            r = per[name]["adapter_vs_split"]
            print(f"[R1a] {impl} {name}: identical={r['tokens_identical']} first_step_diff={r['first_step_max_logit_diff']:.4g} "
                  f"all_steps_diff={r['all_steps_max_logit_diff']:.4g} div={r['first_divergence']}", flush=True)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if out_path:
                json.dump({"partial": True, "results": {**results, impl: per}}, open(out_path, "w"), indent=1)
        results[impl] = {"per_prompt": per, "verdict": passes(per)}
    return results


# ---------------------------------------------------------------------------------------------------------
# Smoke test: one compaction at target 0.1, global AM highest-attention-keys, query config `repeat`
# ---------------------------------------------------------------------------------------------------------
def run_smoke(model, tok, target=0.1, n_questions=3, article_index=0, max_new=256):
    import evaluation  # noqa: F401  (must come first: evaluation <-> compaction circular import)
    from compaction.compaction_methods.registry import get_compaction_method
    from evaluation.configs.utils import load_query_config
    from evaluation.datasets import load_dataset
    from evaluation.utils import compute_article_indices, format_context, format_question, parse_model_choice
    from models.qwen3_5 import (adapt_model, compacted_cache_from_triples, dynamic_cache_to_kv_tuples,
                                greedy_decode, inject_linear_state, linear_layer_set)

    set_attn(model, "sdpa")
    adapt_model(model)
    name = "Qwen/Qwen3.8-27B"
    text_cfg = model.config.get_text_config()
    layer_types = text_cfg.layer_types
    linear = linear_layer_set(layer_types)
    dev = next(model.parameters()).device

    data = load_dataset("quality")
    art = data[article_index]
    article, questions = art["article"], art["questions"][:n_questions]
    formatted = format_context(tok, article, None, name)
    ids = tok(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    seq_len = ids.shape[1]
    article_idx = compute_article_indices(tok, formatted, article)
    print(f"[smoke] article {art['article_id']} seq_len {seq_len} article tokens {len(article_idx)}", flush=True)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    dyn = out.past_key_values
    kv = dynamic_cache_to_kv_tuples(dyn, layer_types)

    method = get_compaction_method("AM-HighestAttnKeys", method_kwargs={
        "algorithm": "global_highest_attention_keys", "score_method": "rms", "nnls_iters": 2,
        "nnls_lower_bound": 0.05, "nnls_upper_bound": 20.0, "c2_method": "lsq"})
    qcfg = load_query_config("repeat")
    target_article = max(1, int(len(article_idx) * target))
    actual_target = target_article + (seq_len - len(article_idx))
    t0 = time.time()
    compacted, stats = method.compact_kv_cache(
        past_key_values=kv, target_size=actual_target, indices=article_idx, query_config=qcfg, model=model,
        tokenizer=tok, formatted_context=formatted, compute_stats=False,
        past_key_values_for_queries=dyn, linear_layer_indices=linear)
    t_compact = time.time() - t0
    compacted = inject_linear_state(compacted, dyn, layer_types)

    nan_layers = [i for i, (c1, b, c2) in enumerate(compacted)
                  if i not in linear and (torch.isnan(c1).any() or torch.isnan(b).any() or torch.isnan(c2).any())]
    full_lens = [c[0].shape[2] for i, c in enumerate(compacted) if i not in linear]
    info = {"article_id": art["article_id"], "seq_len": seq_len, "article_tokens": len(article_idx),
            "target": target, "target_article_tokens": target_article, "compaction_s": round(t_compact, 1),
            "full_layer_prefix_len_min_max_mean": [min(full_lens), max(full_lens), sum(full_lens) / len(full_lens)],
            "nan_layers": nan_layers, "n_full_layers": len(full_lens), "n_linear_layers": len(linear)}
    print("[smoke]", info, flush=True)

    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} - {None}

    def answer(triples):
        rows = []
        for q in questions:
            prompt = format_question(tok, q["question"], q["options"], name, enable_thinking=False)
            qids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
            cache = compacted_cache_from_triples(triples, linear, seq_len)
            with torch.no_grad():
                o = model(input_ids=qids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                toks, _ = greedy_decode(model, cache, o.logits, max_new, stop_ids=eos)
            text = tok.decode(toks, skip_special_tokens=True)
            ch = parse_model_choice(text)
            rows.append({"question": q["question"], "answer": text.strip()[:300], "parsed": chr(64 + ch) if ch else "?",
                         "gold": chr(64 + q["gold_label"]), "correct": bool(ch and ch == q["gold_label"]),
                         "finite_logits": bool(torch.isfinite(o.logits).all())})
        return rows

    orig_triples = inject_linear_state(
        tuple((k, k.new_zeros(k.shape[:3]), v) for k, v in kv), dyn, layer_types)
    orig = answer(orig_triples)
    comp = answer(compacted)
    restore = __import__("models.qwen3_5", fromlist=["restore_model"]).restore_model
    restore(model)
    return {"info": info, "uncompacted_adapter": orig, "compacted": comp,
            "n_correct_uncompacted": sum(r["correct"] for r in orig), "n_correct_compacted": sum(r["correct"] for r in comp)}
