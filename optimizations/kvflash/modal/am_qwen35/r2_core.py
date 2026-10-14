"""R2 multi-round harness (see evidence/am/PREREG.md, section "R2 multi-round").

Context = probed article A followed by k distractor QuALITY articles; questions about A only, asked after all k.
Arms (all built on the R1b Session: prefill, answer, F compaction through the repo's per-layer-head on-policy method):
  FULL   one-shot prefill of the whole context, no compaction.
  F-SEG  each article compacted once when it arrives (F, 0.10 of its article tokens), earlier compacted articles as
         prefix, never recompacted. Queries = the segment's own prefill under that prefix.
  F-BLK  the token stream cut into sealed 640-token blocks by position; each block compacted once to 64 slots (0.10),
         per-head budgets rescaled to the block by the repo's own formula, earlier blocks as prefix. The open tail
         (< 640 tokens) stays uncompacted.
  F-RE   after every new article the WHOLE cache is recompacted to 0.10 of its current slots (the shape of the repo's
         online loop, run_reasoning_evaluation.py:552-567), F's on-policy method (which drops the previous beta).
One pass per (article, arm) over nested distractors serves every checkpoint k (state after k distractors).
"""
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

from r1b_core import NAME, Session, _rows, done_articles, method_for

RATIO = 0.10
BLOCK = 640
USER_TAG = "<|im_start|>user"
DISTRACTOR_POOL = range(50, 90)  # loader indices outside selection (0-19), held-out (20-49) and R1c fit/val (90-99)
ARMS = ("FULL", "F-SEG", "F-BLK", "F-RE", "F-RE-UNI")  # F-RE-UNI: R2 Amendment 1 diagnostic (uniform budget, no padding)
ARM_CELL = {"F-RE-UNI": "FUNI"}  # arm -> R1b cell that supplies the compactor (default "F")


# ---------------------------------------------------------------- pure helpers (CPU-testable)

def pick_distractors(probe_idx, n=8, seed=0, pool=DISTRACTOR_POOL):
    """Fixed, nested (first k of the list = the k-distractor context) and disjoint from the probe set."""
    return random.Random(f"r2-{seed}-{probe_idx}").sample(list(pool), n)


def sealed_blocks(start, total, block=BLOCK):
    """Block (start, end) pairs sealed between `start` and `total` (stream positions) and the new sealed end."""
    out, pos = [], start
    while pos + block <= total:
        out.append((pos, pos + block))
        pos += block
    return out, pos


def slots_for(n_tokens, ratio=RATIO):
    return max(1, int(n_tokens * ratio))


def pad_layers(triples, linear, length=None):
    """Pad every full-attention layer to a common slot count (zero K/V, beta = -inf), as the repo pads heads in a layer."""
    full = [t[0].shape[2] for i, t in enumerate(triples) if i not in linear]
    L = max(full) if length is None else length
    out = []
    for i, (c1, b, c2) in enumerate(triples):
        if i in linear or c1.shape[2] == L:
            out.append((c1, b, c2))
            continue
        n = L - c1.shape[2]
        out.append((torch.cat([c1, c1.new_zeros(*c1.shape[:2], n, c1.shape[3])], 2),
                    torch.cat([b, b.new_full((*b.shape[:2], n), float("-inf"))], 2),
                    torch.cat([c2, c2.new_zeros(*c2.shape[:2], n, c2.shape[3])], 2)))
    return tuple(out), L


def splice(prefix, seg, linear):
    """Concatenate a compacted segment behind the compacted prefix on every full-attention layer. DeltaNet entries
    are taken from `seg` (the state after the segment)."""
    if prefix is None:
        return seg
    return tuple(s if i in linear else tuple(torch.cat([p, x], 2) for p, x in zip(pt, s))
                 for i, (pt, s) in enumerate(zip(prefix, seg)))


def slot_stats(budgets):
    """{(layer, head): slots} -> how many heads got 0, 1, 2 slots and the totals."""
    v = list(budgets.values())
    return {"heads": len(v), "zero": sum(x == 0 for x in v), "one": sum(x == 1 for x in v),
            "two": sum(x == 2 for x in v), "sum": sum(v), "max": max(v), "mean": sum(v) / len(v)}


def budget_slot_stats(budget_path, slots, seg_len, max_ratio, num_layers=64, num_heads=4, n_attn_layers=16):
    """The repo's own per-head budget formula (per_layer_head_on_policy.py:241-266) for `slots` slots over a segment of
    `seg_len` tokens. Needs the repo on the path."""
    from compaction.compaction_methods.base import apply_max_ratio_cap, load_budgets_from_json
    props = load_budgets_from_json(budget_path, num_layers, num_heads)
    total_heads = n_attn_layers * num_heads
    props = apply_max_ratio_cap(props, max_ratio, slots / seg_len, total_heads)
    return slot_stats({k: int(p * slots * total_heads) for k, p in props.items() if p is not None})


# ---------------------------------------------------------------- segments

@dataclass
class Seg:
    text: str
    ids: torch.Tensor      # (1, n) cpu
    art: range             # article token range inside this segment

    @property
    def n(self):
        return self.ids.shape[1]


def build_segments(tok, articles):
    """Segment 0 is the R1b context (system + user turn with the article); every later article is one more user turn.
    Returns (segments, tokenization_is_additive)."""
    from evaluation.utils import compute_article_indices, format_context
    texts = []
    for i, a in enumerate(articles):
        full = format_context(tok, a, None, NAME)
        texts.append(full if i == 0 else full[full.index(USER_TAG):])
    segs = [Seg(t, tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"],
                compute_article_indices(tok, t, a)) for t, a in zip(texts, articles)]
    joined = tok("".join(texts), return_tensors="pt", add_special_tokens=False)["input_ids"]
    return segs, torch.equal(joined, torch.cat([s.ids for s in segs], 1))


# ---------------------------------------------------------------- cache plumbing

def triples_from_cache(cache, linear):
    """CompactedPrefixCache after a forward -> (C1, beta, C2) per layer. Appended tokens get beta 0; DeltaNet layers
    carry their updated conv/recurrent state."""
    out = []
    for i, layer in enumerate(cache.layers):
        if i in linear:
            conv, rec = layer.conv_states[0], layer.recurrent_states[0]
            out.append((conv.clone(), conv.new_zeros(conv.shape[0], 1, 0), rec.clone()))
        else:
            n_new = layer.keys.shape[2] - layer.beta.shape[2]
            beta = torch.cat([layer.beta, layer.beta.new_zeros(*layer.beta.shape[:2], n_new)], 2)
            out.append((layer.keys, beta, layer.values))
    return tuple(out)


@torch.no_grad()
def extend(model, layer_types, triples, orig_len, ids, want_queries=False):
    """Prefill `ids` behind a (possibly compacted) prefix. `triples=None` means no prefix (plain HF forward).
    Returns (new_triples, new_orig_len, queries or None); queries are (attn_layers, heads, n, head_dim), the queries
    the prefill itself computes (post-norm, post-RoPE at the true positions, under the prefix)."""
    from models.qwen3_5 import (compacted_cache_from_triples, dynamic_cache_to_triples)
    from models.qwen3_5.query_capture import attention_layer_indices, get_decoder_layers, project_queries
    linear = {i for i, t in enumerate(layer_types) if t == "linear_attention"}
    caps, hooks = [], []
    if want_queries:
        layers = get_decoder_layers(model)

        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            caps.append(project_queries(module, hs, kwargs.get("position_embeddings"))[0].detach())

        hooks = [layers[i].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
                 for i in attention_layer_indices(model)]
    try:
        if triples is None:
            cache = model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
            new, _ = dynamic_cache_to_triples(cache, layer_types)
        else:
            cache = compacted_cache_from_triples(triples, linear, orig_len)
            model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            new = triples_from_cache(cache, linear)
    finally:
        for h in hooks:
            h.remove()
    q = torch.stack(caps).permute(0, 2, 1, 3).contiguous() if caps else None
    return new, (orig_len or 0) + ids.shape[1], q


@contextmanager
def fixed_queries(q):
    """Make the repo's context-prefill generator return `q` (captured during our own prefill) instead of running a
    separate uncompacted forward. Restores the original method on exit."""
    from compaction.query_generation.context_prefill import ContextPrefillQueryGenerator as C
    orig = C._extract_query_vectors_from_prefill

    def patched(self, input_ids, head_dim, start_token_idx=None):
        return q if start_token_idx is None else q[:, :, start_token_idx:, :]

    C._extract_query_vectors_from_prefill = patched
    try:
        yield
    finally:
        C._extract_query_vectors_from_prefill = orig


def full_layer_len(triples, linear):
    ls = [t[0].shape[2] for i, t in enumerate(triples) if i not in linear]
    return sum(ls) / len(ls), max(ls)


# ---------------------------------------------------------------- compaction steps

class Runner:
    def __init__(self, S, budget, ratio=RATIO, cell="F"):
        self.S, self.budget, self.ratio = S, budget, ratio
        self.cell, self.method = cell, method_for(cell, budget)

    def _pseudo_kv(self, triples, linear, last=None):
        return tuple((t[0], t[2]) if i in linear else
                     ((t[0][:, :, -last:], t[2][:, :, -last:]) if last else (t[0], t[2]))
                     for i, t in enumerate(triples))

    def _compact(self, kv, target_size, indices, text, queries=None):
        from evaluation.configs.utils import load_query_config
        from models.generate import get_sliding_layer_info
        torch.manual_seed(0)
        sliding, _ = get_sliding_layer_info(self.S.model)
        with (fixed_queries(queries) if queries is not None else nullcontext()):
            return self.method.compact_kv_cache(
                past_key_values=kv, target_size=target_size, indices=indices,
                query_config=load_query_config("context-prefill"), model=self.S.model, tokenizer=self.S.tok,
                formatted_context=text, compute_stats=False, vllm_model=self.S.shim, sliding_layer_indices=sliding)

    def _inject_linear(self, compacted, state):
        return tuple(state[i] if i in self.S.linear else c for i, c in enumerate(compacted))

    def segment_step(self, triples, orig_len, ids, art, text):
        """Prefill a segment behind the compacted prefix, compact only its article tokens (all tokens when `art`
        covers the segment), splice behind the prefix. Returns (triples, orig_len, info)."""
        S = self.S
        t0 = time.time()
        state, new_len, q = extend(S.model, S.layer_types, triples, orig_len, ids.to(S.dev), want_queries=True)
        t1 = time.time()
        n = ids.shape[1]
        target = slots_for(len(art), self.ratio) + (n - len(art))
        seg, _stats = self._compact(self._pseudo_kv(state, S.linear, last=n), target, art, text, queries=q)
        seg = self._inject_linear(seg, state)
        out = splice(triples, seg, S.linear)
        del q, state
        return out, new_len, {"prefill_s": round(t1 - t0, 2), "compact_s": round(time.time() - t1, 2)}

    def recompact_step(self, triples, orig_len, ids, full_text):
        """F-RE: prefill the new article behind the compacted cache, then recompact the WHOLE cache to RATIO of its
        slots (indices = every slot, no protected tail, see PREREG), queries from an uncompacted forward over the text
        so far (the repo's context-prefill, unmodified)."""
        S = self.S
        t0 = time.time()
        state, new_len, _ = extend(S.model, S.layer_types, triples, orig_len, ids.to(S.dev))
        t1 = time.time()
        padded, L = pad_layers(state, S.linear)
        target = slots_for(L, self.ratio)
        comp, _stats = self._compact(self._pseudo_kv(padded, S.linear), target, range(0, L), full_text)
        comp = self._inject_linear(comp, state)
        return comp, new_len, {"prefill_s": round(t1 - t0, 2), "compact_s": round(time.time() - t1, 2),
                               "slots_before": L, "slots_target": target}


# ---------------------------------------------------------------- arms (generators yielding at checkpoints)

def _info(R, triples, orig_len, trace):
    mean, mx = full_layer_len(triples, R.S.linear)
    eff = [torch.isfinite(t[1]).sum(-1).float().mean().item() for i, t in enumerate(triples) if i not in R.S.linear]
    return {"ctx_len": orig_len, "slots_mean": round(mean, 1), "slots_max": mx, "slots_effective_mean": round(sum(eff) / len(eff), 1),
            "prefill_s": round(sum(t.get("prefill_s", 0) for t in trace), 2),
            "compact_s": round(sum(t.get("compact_s", 0) for t in trace), 2),
            "steps": len(trace), "nan_layers": [i for i, t in enumerate(triples) if i not in R.S.linear and
                                                any(torch.isnan(x).any() for x in t)]}


def gen_full(R, segs, articles, ks):
    S = R.S
    for k in ks:
        ids = torch.cat([s.ids for s in segs[:k + 1]], 1).to(S.dev)
        t0 = time.time()
        with torch.no_grad():
            dyn = S.model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
        from models.qwen3_5 import dynamic_cache_to_triples
        triples, _ = dynamic_cache_to_triples(dyn, S.layer_types)
        info = _info(R, triples, ids.shape[1], [{"prefill_s": time.time() - t0}])
        yield k, triples, ids.shape[1], info
        del dyn, triples
        torch.cuda.empty_cache()


def gen_fseg(R, segs, articles, ks):
    S = R.S
    pf = S.prefill(articles[0])  # identical to the k=0 rows of R1b
    t0 = time.time()
    triples, _ = S.compact(R.cell, pf, R.ratio, R.budget)
    orig_len, trace = pf["seq_len"], [{"prefill_s": 0.0, "compact_s": round(time.time() - t0, 2)}]
    del pf
    for i in range(1, max(ks) + 1):
        triples, orig_len, st = R.segment_step(triples, orig_len, segs[i].ids, segs[i].art, segs[i].text)
        trace.append(st)
        if i in ks:
            yield i, triples, orig_len, _info(R, triples, orig_len, trace)


def gen_fblk(R, segs, articles, ks):
    S = R.S
    stream = torch.cat([s.ids for s in segs], 1)
    ends = [sum(s.n for s in segs[:i + 1]) for i in range(len(segs))]
    triples, orig_len, consumed, trace = None, 0, 0, []
    for k in sorted(ks):
        blocks, new_end = sealed_blocks(consumed, ends[k])
        for a, b in blocks:
            ids = stream[:, a:b]
            triples, orig_len, st = R.segment_step(triples, orig_len, ids, range(0, b - a), S.tok.decode(ids[0]))
            trace.append(st)
        consumed = new_end
        tail = stream[:, consumed:ends[k]]
        t_tr, t_len = triples, orig_len
        if tail.shape[1]:
            t_tr, t_len, _ = extend(S.model, S.layer_types, triples, orig_len, tail.to(S.dev))
        yield k, t_tr, t_len, {**_info(R, t_tr, t_len, trace), "tail_tokens": int(tail.shape[1]), "blocks": len(trace)}


def gen_fre(R, segs, articles, ks):
    S = R.S
    pf = S.prefill(articles[0])
    t0 = time.time()
    triples, _ = S.compact(R.cell, pf, R.ratio, R.budget)
    orig_len, trace = pf["seq_len"], [{"prefill_s": 0.0, "compact_s": round(time.time() - t0, 2)}]
    del pf
    text = segs[0].text
    for i in range(1, max(ks) + 1):
        text += segs[i].text
        triples, orig_len, st = R.recompact_step(triples, orig_len, segs[i].ids, text)
        trace.append(st)
        if i in ks:
            yield i, triples, orig_len, {**_info(R, triples, orig_len, trace), "recompactions": i + 1}


ARM_GENS = {"FULL": gen_full, "F-SEG": gen_fseg, "F-BLK": gen_fblk, "F-RE": gen_fre, "F-RE-UNI": gen_fre}


# ---------------------------------------------------------------- driver

def row_path(out, arm, k, split="sel"):
    return f"{out}/{arm}__k{k}__t{RATIO}__{split}.jsonl"


def run_r2(model, tok, shim, out, probes, ks, arms, budget, split="sel", n_questions=None, seed=0, log=print,
           data=None, gpu=""):
    from evaluation.datasets import load_dataset
    data = data or load_dataset("quality")
    S = Session(model, tok, shim)
    runners = {}
    os.makedirs(out, exist_ok=True)
    for ai in probes:
        art = data[ai]
        aid = str(art["article_id"])
        qs = art["questions"][:n_questions] if n_questions else art["questions"]
        dis = pick_distractors(ai, max(ks), seed)
        articles = [art["article"]] + [data[d]["article"] for d in dis]
        segs, additive = build_segments(tok, articles)
        log(f"[r2] probe {ai} {aid} distractors {dis} seg tokens {[s.n for s in segs]} additive_tokenization={additive}")
        for arm in arms:
            todo = [k for k in ks if aid not in done_articles(row_path(out, arm, k, split))]
            if not todo:
                continue
            torch.cuda.reset_peak_memory_stats()
            cell = ARM_CELL.get(arm, "F")
            if cell not in runners:
                runners[cell] = Runner(S, budget, cell=cell)
            for k, triples, orig_len, info in ARM_GENS[arm](runners[cell], segs, articles, todo):
                t1 = time.time()
                res = S.answer(triples, orig_len, qs)
                ans_s = round(time.time() - t1, 1)
                peak = round(torch.cuda.max_memory_allocated() / 2**30, 1)
                extra = {"arm": arm, "k": k, "article_index": ai, "target": RATIO, "seq_len": orig_len,
                         "distractors": dis[:k], "answer_s": ans_s, "peak_gb": peak, "gpu": gpu, **info}
                rows = _rows(art, res, qs, arm, extra)
                with open(row_path(out, arm, k, split), "a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in rows))
                log(f"[r2] art {ai} {arm} k={k}: acc {sum(r['correct'] for r in rows)}/{len(rows)} "
                    f"ctx {orig_len} slots {info['slots_mean']} prefill {info['prefill_s']}s compact {info['compact_s']}s "
                    f"answer {ans_s}s peak {peak}GB")
                del triples
                torch.cuda.empty_cache()
