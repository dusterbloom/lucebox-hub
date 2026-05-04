"""C++-only NIAH bench: daemon compress + generate, no Python drafter."""
import argparse, json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from pflash.dflash_client import DflashClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--bin", default="/home/lucebox/lucebox-hub/dflash/build/test_dflash")
    ap.add_argument("--target", default="/opt/lucebox/models/Qwen3.6-27B-Q4_K_M.gguf")
    ap.add_argument("--draft-spec", default="/home/lucebox/lucebox-hub/dflash/models/draft/model.safetensors",
                    help="draft model used for spec decoding (NOT drafter scorer)")
    ap.add_argument("--drafter-gguf", default="/home/lucebox/lucebox-hub/dflash/models/Qwen3-0.6B-BF16.gguf",
                    help="C++ drafter scorer GGUF (Qwen3-0.6B BF16)")
    ap.add_argument("--target-tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--drafter-tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="daemon KV cache max ctx; sized for compressed prompt+gen, NOT source")
    ap.add_argument("--keep-ratio", type=float, default=0.020)
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--ddtree-budget", type=int, default=16)
    ap.add_argument("--ddtree-temp", type=float, default=None,
                    help="Drafter softmax temperature; lower = sharper")
    ap.add_argument("--no-chain-seed", action="store_true")
    ap.add_argument("--fa-window", type=int, default=0)
    ap.add_argument("--kv-tq3", type=int, choices=[0, 1], default=1,
                    help="3-bit KV cache; saves VRAM")
    ap.add_argument("--auto-max-ctx", action="store_true",
                    help="Auto-size --max-ctx based on src tokens")
    args = ap.parse_args()

    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer)
    drafter_tok = AutoTokenizer.from_pretrained(args.drafter_tokenizer)
    cases = [json.loads(l) for l in open(args.cases)][:args.n]

    max_ctx = args.max_ctx
    if args.auto_max_ctx:
        first = json.loads(open(args.cases).readline())
        src_tokens_est = len(drafter_tok(first["prompt"], return_tensors="pt")["input_ids"][0])
        expected_compressed = int(src_tokens_est * args.keep_ratio * 1.15) + 64
        needed = expected_compressed + args.n_gen + 512
        max_ctx = max(max_ctx, needed)
        max_ctx = ((max_ctx + 1023) // 1024) * 1024
        print(f"[init] auto-max-ctx: src~{src_tokens_est}, compressed~{expected_compressed}, max_ctx={max_ctx}", flush=True)

    print(f"[init] spawning daemon: {args.bin}", flush=True)
    dflash = DflashClient(
        args.bin, args.target, args.draft_spec,
        max_ctx=max_ctx,
        ddtree_budget=args.ddtree_budget,
        ddtree_temp=args.ddtree_temp,
        chain_seed=not args.no_chain_seed,
        fa_window=args.fa_window,
        kv_tq3=bool(args.kv_tq3),
    )

    correct = 0
    for i, case in enumerate(cases):
        prompt = case["prompt"]
        # Drafter tokenizes prompt, daemon scores+compresses, returns drafter ids.
        ids = drafter_tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
        S = len(ids)
        print(f"[case {i}] src={S} keep={args.keep_ratio}", flush=True)

        t0 = time.time()
        compressed_ids = dflash.compress(ids, args.keep_ratio, args.drafter_gguf)
        t_score = time.time() - t0
        comp = len(compressed_ids)
        print(f"[case {i}] compressed={comp} ratio={S/max(comp,1):.1f}x score_s={t_score:.1f}", flush=True)

        # Decode compressed ids with DRAFTER tokenizer, re-encode with TARGET + chat template.
        comp_text = drafter_tok.decode(compressed_ids, skip_special_tokens=True)
        user_msg = comp_text + "\n\nAnswer the user question based on the above context."
        chat = target_tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)
        target_ids = target_tok(chat, return_tensors="pt")["input_ids"][0].tolist()

        # Free drafter (1.2GB), restore target+spec_draft for target gen.
        dflash.free_drafter()
        dflash.unpark_target()
        dflash.unpark_draft()

        t0 = time.time()
        out_ids = dflash.generate(target_ids, args.n_gen)
        t_gen = time.time() - t0
        # Re-park for next iter (drafter scoring).
        dflash.park_draft()
        print(f"[case {i}] raw out_ids ({len(out_ids)}): {out_ids[:20]}", flush=True)
        out_text = target_tok.decode(out_ids, skip_special_tokens=True)
        out_text_keep = target_tok.decode(out_ids, skip_special_tokens=False)
        print(f"[case {i}] out_with_special: {out_text_keep!r}", flush=True)

        ok = case["answer"] in out_text
        if ok:
            correct += 1
        ttft = t_score + t_gen  # rough; gen includes ttft
        print(f"[case {i}] gen_s={t_gen:.1f} ttft={ttft:.1f} ok={ok} ans={case['answer']}", flush=True)
        print(f"[case {i}] out: {out_text!r}", flush=True)

    print(f"\naccuracy: {correct}/{len(cases)}", flush=True)
    dflash.close()


if __name__ == "__main__":
    main()
