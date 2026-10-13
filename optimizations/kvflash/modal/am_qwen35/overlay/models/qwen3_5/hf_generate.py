# models/qwen3_5/hf_generate.py
"""HF `generate` stand-in for the vLLM object the repo's self-study query generation expects.

The repo calls `vllm_model.generate(prompts, sampling_params)` and reads `output.outputs[0].text`, plus
`wake_up()` / `sleep()`. This shim provides exactly that on top of an already loaded HF model (same weights, so no
second copy of the 27B), with left padding and batched sampling. Sampling parameters are the four fields the repo
sets (`temperature`, `top_p`, `top_k`, `max_tokens`).
"""
from types import SimpleNamespace

import torch


class HFGenerateShim:
    def __init__(self, model, tokenizer, seed: int = 1, batch_size: int = 6):
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self._seed = seed
        self.n_generated_tokens = 0

    def wake_up(self):
        pass

    def sleep(self):
        pass

    @torch.no_grad()
    def generate(self, prompts, sampling_params):
        tok, model = self.tokenizer, self.model
        dev = next(model.parameters()).device
        old_side = tok.padding_side
        tok.padding_side = "left"
        outs = [None] * len(prompts)
        # longest first keeps the padding waste of a batch small
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        try:
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                torch.manual_seed(self._seed + s)  # sampling in HF uses the global torch RNG
                gen = model.generate(
                    **enc, do_sample=sampling_params.temperature > 0, temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p, top_k=sampling_params.top_k,
                    max_new_tokens=sampling_params.max_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                for row, i in zip(new, idx):
                    text = tok.decode(row, skip_special_tokens=True)
                    self.n_generated_tokens += int((row != (tok.pad_token_id or tok.eos_token_id)).sum())
                    outs[i] = SimpleNamespace(outputs=[SimpleNamespace(text=text)])
        finally:
            tok.padding_side = old_side
        return outs
