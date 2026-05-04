"""Generate NIAH single-needle test cases at any context size."""
import argparse, json, random
from transformers import AutoTokenizer

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
NEEDLE_TMPL = "The special magic {key} number is: {value}."
QUESTION_TMPL = "What is the special magic {key} number? Answer in one short sentence."


def gen_one(seed: int, target_tokens: int, tokenizer, tolerance: float = 0.005):
    """Generate a NIAH case whose tokenized length is within `tolerance` of
    `target_tokens` *and never exceeds it*, regardless of which tokenizer is
    passed in.

    Strategy: binary-search the chars-per-token ratio against the actual
    tokenizer until we land within tolerance, then hard-trim chars off the
    filler if we still overshoot by even one token. The hard trim guarantees
    `n_tokens <= target_tokens` so callers can set `--ctx` to a model's RoPE
    limit without silent truncation.
    """
    rng = random.Random(seed)
    key = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    value = "".join(rng.choices("0123456789", k=7))
    needle = NEEDLE_TMPL.format(key=key, value=value)
    question = QUESTION_TMPL.format(key=key)
    insert_frac = rng.uniform(0.25, 0.75)  # capture once for determinism

    def build(target_chars: int) -> str:
        target_chars = max(0, target_chars)
        filler = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
        insert = int(target_chars * insert_frac)
        body = filler[:insert] + " " + needle + " " + filler[insert:]
        return (
            "Below is a long passage. Answer the question at the end based ONLY on information in the passage.\n\n"
            f"{body}\n\nQuestion: {question}\nAnswer:"
        )

    def n_tokens(prompt: str) -> int:
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    # Binary-search chars-per-token. Bounds [2.0, 6.0] cover every common
    # tokenizer (Qwen ~3.7, Llama/Mistral ~4.0, GPT-2 BPE ~4.2, byte-level ~5+).
    lo, hi = 2.0, 6.0
    target_chars = int(target_tokens * (lo + hi) / 2)
    prompt = build(target_chars)
    actual = n_tokens(prompt)
    for _ in range(20):
        if abs(actual - target_tokens) / target_tokens < tolerance:
            break
        if actual > target_tokens:
            hi = (lo + hi) / 2
        else:
            lo = (lo + hi) / 2
        target_chars = int(target_tokens * (lo + hi) / 2)
        prompt = build(target_chars)
        actual = n_tokens(prompt)

    # Hard-trim: shrink filler until actual <= target_tokens. Step by larger
    # chunks first to keep the worst case bounded (~few re-tokenizations).
    for step in (256, 64, 16, 1):
        while actual > target_tokens and target_chars >= step:
            target_chars -= step
            prompt = build(target_chars)
            actual = n_tokens(prompt)

    return {"prompt": prompt, "answer": value, "key": key, "n_tokens": actual}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.out, "w") as f:
        for i in range(args.n):
            ex = gen_one(seed=42 + i, target_tokens=args.ctx, tokenizer=tok)
            assert ex["n_tokens"] <= args.ctx, (
                f"case {i}: n_tokens={ex['n_tokens']} exceeds --ctx={args.ctx}")
            f.write(json.dumps(ex) + "\n")
            print(f"  case {i}: ntok={ex['n_tokens']} key={ex['key']} ans={ex['answer']}")
    print(f"saved {args.n} cases to {args.out}")


if __name__ == "__main__":
    main()
