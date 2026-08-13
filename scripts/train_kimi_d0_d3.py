#!/usr/bin/env python3
"""Real Kimi-K3 layer-one D0-D3 shared SiTU/FiLM screen.

The shared nonlinear core is trained exactly once.  It is then frozen while
the shift-only, scale-only, and full scale-plus-shift cards are trained on the
same sequence-disjoint fitting split.  The original validation sequences are
reserved for final reporting and causal controls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file
from torch import nn

from train_kimi_panel_directional import load_data, sha256, summarize


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> torch.Tensor:
    return (
        beta_gate * torch.tanh(gate / beta_gate) * torch.sigmoid(gate)
        * beta_up * torch.tanh(up / beta_up)
    )


class SharedSituCore(nn.Module):
    def __init__(self, dimension: int, hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dimension, hidden_size, bias=False)
        self.up = nn.Linear(dimension, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, dimension, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.down(situ_glu(self.gate(latent), self.up(latent)))


def apply_cards(
    shared: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
) -> torch.Tensor:
    output = shared
    if scale is not None:
        mixed_scale = (
            scale[expert_ids] * router_weights[:, :, None]
        ).sum(dim=1)
        output = output * (1.0 + mixed_scale)
    if shift is not None:
        mixed_shift = (
            shift[expert_ids] * router_weights[:, :, None]
        ).sum(dim=1)
        output = output + mixed_shift
    return output


def metric(exact: torch.Tensor, estimate: torch.Tensor) -> dict[str, object]:
    cosine = functional.cosine_similarity(exact, estimate, dim=1)
    relative = (
        torch.linalg.vector_norm(estimate - exact, dim=1)
        / torch.linalg.vector_norm(exact, dim=1).clamp_min(1e-12)
    )
    rms_ratio = (
        estimate.square().mean(dim=1).sqrt()
        / exact.square().mean(dim=1).sqrt().clamp_min(1e-12)
    )
    return {
        "cosine": summarize(cosine.float().cpu().numpy()),
        "relative_l2": summarize(relative.float().cpu().numpy()),
        "rms_ratio": summarize(rms_ratio.float().cpu().numpy()),
    }


@torch.no_grad()
def evaluate(
    indices: torch.Tensor,
    latent: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    teacher: torch.Tensor,
    core: SharedSituCore,
    scale: torch.Tensor | None,
    shift: torch.Tensor | None,
    batch_size: int,
    *,
    card_permutation: torch.Tensor | None = None,
    uniform_routing: bool = False,
) -> dict[str, object]:
    exact_parts: list[torch.Tensor] = []
    estimate_parts: list[torch.Tensor] = []
    for begin in range(0, indices.numel(), batch_size):
        selected = indices[begin : begin + batch_size]
        ids = expert_ids[selected]
        if card_permutation is not None:
            ids = card_permutation[ids]
        weights = router_weights[selected]
        if uniform_routing:
            weights = torch.full_like(weights, 1.0 / weights.shape[1])
        shared = core(latent[selected])
        estimate_parts.append(apply_cards(shared, ids, weights, scale, shift))
        exact_parts.append(teacher[selected])
    return metric(torch.cat(exact_parts), torch.cat(estimate_parts))


def train_core(
    core: SharedSituCore,
    train_indices: torch.Tensor,
    development_indices: torch.Tensor,
    latent: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    teacher: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    evaluate_every: int,
    patience: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float | int]]]:
    optimizer = torch.optim.AdamW(core.parameters(), lr=learning_rate)
    generator = torch.Generator(device=latent.device)
    generator.manual_seed(seed)
    best_state = {name: value.detach().cpu().clone()
                  for name, value in core.state_dict().items()}
    best_mean = -float("inf")
    best_step = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        sampled = torch.randint(
            train_indices.numel(), (batch_size,),
            generator=generator, device=latent.device
        )
        indices = train_indices[sampled]
        estimate = core(latent[indices])
        loss = (1.0 - functional.cosine_similarity(
            estimate, teacher[indices], dim=1
        )).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        optimizer.step()
        if step % evaluate_every == 0 or step == steps:
            development = evaluate(
                development_indices, latent, expert_ids, router_weights,
                teacher, core, None, None, evaluation_batch_size
            )
            mean = development["cosine"]["mean"]
            history.append({
                "step": step,
                "training_batch_loss": float(loss.detach().cpu()),
                "development_mean_cosine": mean,
            })
            print(
                f"D0 step={step} loss={float(loss.detach().cpu()):.7f} "
                f"development_cosine={mean:.9f}", flush=True
            )
            if mean > best_mean + 1e-7:
                best_mean = mean
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in core.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    core.load_state_dict({name: value.to(latent.device)
                          for name, value in best_state.items()})
    return best_state, best_step, history


def train_cards(
    arm: str,
    core: SharedSituCore,
    expert_count: int,
    dimension: int,
    train_indices: torch.Tensor,
    development_indices: torch.Tensor,
    latent: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    teacher: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    evaluate_every: int,
    patience: int,
    seed: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int,
           list[dict[str, float | int]]]:
    use_scale = arm in ("D2", "D3")
    use_shift = arm in ("D1", "D3")
    scale = nn.Parameter(torch.zeros(
        expert_count, dimension, device=latent.device
    )) if use_scale else None
    shift = nn.Parameter(torch.zeros(
        expert_count, dimension, device=latent.device
    )) if use_shift else None
    parameters = [value for value in (scale, shift) if value is not None]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device=latent.device)
    generator.manual_seed(seed)
    best_mean = evaluate(
        development_indices, latent, expert_ids, router_weights, teacher,
        core, scale, shift, evaluation_batch_size
    )["cosine"]["mean"]
    best_step = 0
    best_scale = scale.detach().cpu().clone() if scale is not None else None
    best_shift = shift.detach().cpu().clone() if shift is not None else None
    stale = 0
    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        sampled = torch.randint(
            train_indices.numel(), (batch_size,),
            generator=generator, device=latent.device
        )
        indices = train_indices[sampled]
        with torch.no_grad():
            shared = core(latent[indices])
        estimate = apply_cards(
            shared, expert_ids[indices], router_weights[indices], scale, shift
        )
        loss = (1.0 - functional.cosine_similarity(
            estimate, teacher[indices], dim=1
        )).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % evaluate_every == 0 or step == steps:
            development = evaluate(
                development_indices, latent, expert_ids, router_weights,
                teacher, core, scale, shift, evaluation_batch_size
            )
            mean = development["cosine"]["mean"]
            history.append({
                "step": step,
                "training_batch_loss": float(loss.detach().cpu()),
                "development_mean_cosine": mean,
            })
            print(
                f"{arm} step={step} loss={float(loss.detach().cpu()):.7f} "
                f"development_cosine={mean:.9f}", flush=True
            )
            if mean > best_mean + 1e-7:
                best_mean = mean
                best_step = step
                best_scale = (
                    scale.detach().cpu().clone() if scale is not None else None
                )
                best_shift = (
                    shift.detach().cpu().clone() if shift is not None else None
                )
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if scale is not None and best_scale is not None:
        scale.data.copy_(best_scale.to(latent.device))
    if shift is not None and best_shift is not None:
        shift.data.copy_(best_shift.to(latent.device))
    return scale, shift, best_step, history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--core-steps", type=int, default=1500)
    parser.add_argument("--card-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--core-learning-rate", type=float, default=3e-4)
    parser.add_argument("--card-learning-rate", type=float, default=3e-5)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260731)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.hidden_size <= 0 or args.core_steps <= 0 or args.card_steps <= 0:
        parser.error("training sizes and steps must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.monotonic()
    data = load_data(args.capture, args.teacher)
    latent = torch.from_numpy(data.latent.copy()).to(device)
    expert_ids = torch.from_numpy(data.expert_ids.copy()).to(device)
    router_weights = torch.from_numpy(data.router_weights.copy()).to(device)
    teacher = torch.from_numpy(data.teacher.copy()).to(device)
    train_indices = torch.from_numpy(data.train_indices).to(device)
    development_indices = torch.from_numpy(data.development_indices).to(device)
    validation_indices = torch.from_numpy(data.validation_indices).to(device)
    expert_count = int(data.expert_ids.max()) + 1

    core = SharedSituCore(data.dimension, args.hidden_size).to(device)
    core_state, core_best_step, core_history = train_core(
        core, train_indices, development_indices, latent, expert_ids,
        router_weights, teacher, steps=args.core_steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        learning_rate=args.core_learning_rate,
        evaluate_every=args.evaluate_every, patience=args.patience,
        seed=args.seed
    )
    for parameter in core.parameters():
        parameter.requires_grad_(False)

    arms: dict[str, dict[str, object]] = {}
    trained_cards: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    d0_development = evaluate(
        development_indices, latent, expert_ids, router_weights, teacher,
        core, None, None, args.evaluation_batch_size
    )
    d0_validation = evaluate(
        validation_indices, latent, expert_ids, router_weights, teacher,
        core, None, None, args.evaluation_batch_size
    )
    arms["D0"] = {
        "description": "shared SiTU-GLU core only",
        "best_step": core_best_step,
        "development": d0_development,
        "held_out_validation": d0_validation,
        "history": core_history,
    }
    trained_cards["D0"] = (None, None)

    for ordinal, arm in enumerate(("D1", "D2", "D3"), start=1):
        scale, shift, best_step, history = train_cards(
            arm, core, expert_count, data.dimension,
            train_indices, development_indices, latent, expert_ids,
            router_weights, teacher, steps=args.card_steps,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            learning_rate=args.card_learning_rate,
            evaluate_every=args.evaluate_every, patience=args.patience,
            seed=args.seed + ordinal
        )
        trained_cards[arm] = (scale, shift)
        arms[arm] = {
            "description": {
                "D1": "shared core plus router-mixed shift cards",
                "D2": "shared core plus router-mixed scale cards",
                "D3": "shared core plus router-mixed scale and shift cards",
            }[arm],
            "best_step": best_step,
            "development": evaluate(
                development_indices, latent, expert_ids, router_weights,
                teacher, core, scale, shift, args.evaluation_batch_size
            ),
            "held_out_validation": evaluate(
                validation_indices, latent, expert_ids, router_weights,
                teacher, core, scale, shift, args.evaluation_batch_size
            ),
            "history": history,
        }

    d3_scale, d3_shift = trained_cards["D3"]
    permutation_generator = torch.Generator(device=device)
    permutation_generator.manual_seed(args.seed + 1000)
    permutation = torch.randperm(
        expert_count, generator=permutation_generator, device=device
    )
    controls = {
        "cards_zeroed": d0_validation,
        "card_identities_permuted": evaluate(
            validation_indices, latent, expert_ids, router_weights, teacher,
            core, d3_scale, d3_shift, args.evaluation_batch_size,
            card_permutation=permutation
        ),
        "uniform_weights_over_teacher_selected_experts": evaluate(
            validation_indices, latent, expert_ids, router_weights, teacher,
            core, d3_scale, d3_shift, args.evaluation_batch_size,
            uniform_routing=True
        ),
    }

    tensors = {
        f"core.{name}": value.to(torch.bfloat16).contiguous()
        for name, value in core_state.items()
    }
    for arm, (scale, shift) in trained_cards.items():
        if scale is not None:
            tensors[f"{arm}.expert_scale"] = (
                scale.detach().cpu().to(torch.bfloat16).contiguous()
            )
        if shift is not None:
            tensors[f"{arm}.expert_shift"] = (
                shift.detach().cpu().to(torch.bfloat16).contiguous()
            )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_prefix.with_suffix(".safetensors")
    save_file(tensors, str(artifact_path), metadata={
        "schema": "kimi-k3-real-d0-d3-v1",
        "model_layer": str(data.model_layer),
        "hidden_size": str(args.hidden_size),
        "seed": str(args.seed),
        "objective": "aggregate_cosine",
    })

    for arm, values in arms.items():
        mean = values["held_out_validation"]["cosine"]["mean"]
        values["verdict"] = (
            "GREEN" if mean >= 0.9998 else "YELLOW" if mean >= 0.99 else "RED"
        )
    best_arm = max(
        arms,
        key=lambda name: arms[name]["held_out_validation"]["cosine"]["mean"],
    )
    core_parameters = 3 * data.dimension * args.hidden_size
    card_parameters = expert_count * data.dimension
    result = {
        "schema": "kimi-k3-real-d0-d3-v1",
        "status": "exploratory_followup_not_registered_primary_gate",
        "model_layer": data.model_layer,
        "capture_sha256": sha256(args.capture),
        "teacher_sha256": sha256(args.teacher),
        "seed": args.seed,
        "objective": "mean aggregate cosine",
        "hidden_size": args.hidden_size,
        "tokens": {
            "training": int(train_indices.numel()),
            "development": int(development_indices.numel()),
            "held_out_validation": int(validation_indices.numel()),
        },
        "arms": arms,
        "D3_causal_controls": controls,
        "best_arm": best_arm,
        "parameter_budget": {
            "D0_parameters": core_parameters,
            "one_card_type_parameters": card_parameters,
            "D3_parameters": core_parameters + 2 * card_parameters,
            "D3_bfloat16_bytes_per_layer": 2 * (
                core_parameters + 2 * card_parameters
            ),
            "D3_bfloat16_bytes_all_92_moe_layers": 92 * 2 * (
                core_parameters + 2 * card_parameters
            ),
        },
        "artifact": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "final_logit_kl": None,
        "final_logit_kl_unavailable_reason": (
            "This bounded experiment stops at the first routed layer; "
            "logit divergence requires the full remaining model."
        ),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0
        ),
    }
    result_path = args.output_prefix.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print("D0-D3 held-out mean cosine")
    for arm in ("D0", "D1", "D2", "D3"):
        print(
            f"{arm}: "
            f"{arms[arm]['held_out_validation']['cosine']['mean']:.9f}"
        )
    print(f"best arm: {best_arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
