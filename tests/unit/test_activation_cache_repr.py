"""Regression tests for ActivationCache's compact representation."""

import torch

from transformer_lens.ActivationCache import ActivationCache


class _TensorWithForbiddenRepr(torch.Tensor):
    """A real Tensor that fails if a representation attempts to print its value."""

    @staticmethod
    def __new__(cls) -> "_TensorWithForbiddenRepr":
        return torch.Tensor._make_subclass(cls, torch.zeros(1), require_grad=False)

    def __repr__(self) -> str:
        raise AssertionError("ActivationCache.__repr__ must not render cached tensors")


def test_repr_lists_keys_without_rendering_cached_tensors() -> None:
    keys = ("hook_embed", "blocks.0.hook_resid_pre")
    cache = ActivationCache(
        {key: _TensorWithForbiddenRepr() for key in keys},
        model=None,
    )

    representation = repr(cache)

    for key in keys:
        assert key in representation
