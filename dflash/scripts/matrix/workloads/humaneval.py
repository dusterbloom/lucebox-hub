"""HumanEval workload — stub for next iteration.

Real implementation will load humaneval problems, tokenise them with the
same chat template as swe_bench.py, and yield WorkloadPrompts.

Grading will call the HumanEval execution harness (human-eval package or
evalplus) against the generated completions.
"""
from __future__ import annotations

from typing import Iterable, Dict, Any

from matrix.workload import Workload, WorkloadPrompt


class HumanEvalWorkload(Workload):
    """HumanEval+ workload (stub — raises NotImplementedError)."""

    name = "humaneval"

    def __init__(self, n_sample: int = 8, seed: int = 42) -> None:
        self.n_sample = n_sample
        self.seed = seed

    def prompts(self) -> Iterable[WorkloadPrompt]:
        raise NotImplementedError(
            "HumanEvalWorkload is not yet implemented. "
            "Use SweBenchWorkload for now."
        )

    def config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_sample": self.n_sample,
            "seed": self.seed,
        }
