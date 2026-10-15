"""Stub test for E8 (a1_core.select_only): the selection of the fit's own scoring, rows copied verbatim, bias 0."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core  # noqa: E402


class FitAlgo:
    """Stands in for the shipping fit: ranks the rows by key norm, then returns a fitted key, a nonzero bias and fitted values."""
    idx_as_tensor = False

    def __init__(self, **kw):
        pass

    def compute_compacted_cache(self, K, V, queries, t, *a, **kw):
        idx = torch.topk(K.float().norm(dim=-1), t).indices
        return K[idx] * 0.5, torch.full((t,), -1.0, dtype=K.dtype), V[idx] * 2 + 1, idx if self.idx_as_tensor else idx.tolist()


class TensorIdxAlgo(FitAlgo):
    idx_as_tensor = True


@pytest.mark.parametrize("cls", [FitAlgo, TensorIdxAlgo])
def test_select_only_keeps_the_selected_rows_verbatim_with_bias_zero(cls):
    torch.manual_seed(0)
    K, V = torch.randn(40, 8).bfloat16(), torch.randn(40, 8).bfloat16()
    fit = cls().compute_compacted_cache(K, V, None, 6)
    c1, beta, c2, idx = a1_core.select_only(cls)().compute_compacted_cache(K, V, None, 6)
    sel = torch.as_tensor(fit[3])
    assert torch.equal(torch.as_tensor(idx), sel)                # the same selection and rank order as the fit's own scoring
    assert torch.equal(c1, K[sel]) and torch.equal(c2, V[sel])   # byte-equal source rows
    assert not beta.any() and beta.shape == fit[1].shape and beta.dtype == fit[1].dtype
    assert not torch.equal(c1, fit[0]) and not torch.equal(c2, fit[2])   # the fitted tensors are gone
