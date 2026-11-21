import pytest
import torch

from transformer_lens import ActivationCache


class KeysOnlyDict(dict):
    """Fail if representation starts reading cached values."""

    def __getitem__(self, key):
        raise AssertionError("ActivationCache repr accessed a cached tensor")

    def values(self):
        raise AssertionError("ActivationCache repr requested tensor values")

    def items(self):
        raise AssertionError("ActivationCache repr requested tensor items")


@pytest.mark.parametrize(
    "keys",
    [[], ["hook_embed"], ["blocks.1.hook_resid_pre", "hook_embed", "quoted'key"]],
)
def test_repr_lists_keys_without_touching_tensors(keys, monkeypatch):
    cache_dict = {key: torch.zeros(2, 3) for key in keys}
    cache = ActivationCache(cache_dict, model=None)
    # Arm the guard after constructor argument type checking has inspected values.
    cache.cache_dict = KeysOnlyDict(cache_dict)

    def fail_tensor_formatting(*args, **kwargs):
        raise AssertionError("ActivationCache repr formatted a cached tensor")

    # dict.__repr__ can bypass the overridden mapping accessors.
    monkeypatch.setattr(torch.Tensor, "__repr__", fail_tensor_formatting)
    monkeypatch.setattr(torch.Tensor, "__str__", fail_tensor_formatting)
    monkeypatch.setattr(torch.Tensor, "__format__", fail_tensor_formatting)

    assert repr(cache) == f"ActivationCache with keys {keys!r}"
