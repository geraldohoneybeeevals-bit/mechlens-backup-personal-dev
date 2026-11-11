#!/usr/bin/env python3
"""Measure ActivationCache repr cost along two independent dimensions.

Run from any directory with:

    python debugging/activation_cache_repr_probe.py

The probe prepends this checkout to sys.path and then asserts that the imported
ActivationCache implementation is exactly transformer_lens/ActivationCache.py
from this repository. It will fail instead of silently using an installed copy.

It measures only repr(cache), not cache construction, over repeated trials:
1. fixed number of keys while each cached tensor gets larger;
2. fixed tensor size while the number of cache keys grows.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULE = (REPO_ROOT / "transformer_lens" / "ActivationCache.py").resolve()


class ValueMustNotBeRendered:
    """A cache value whose repr makes accidental value rendering obvious."""

    def __repr__(self) -> str:
        raise AssertionError("ActivationCache.__repr__ attempted to render a cached value")


def load_checkout_activation_cache() -> type[Any]:
    """Import ActivationCache and prove it came from this checkout."""
    repo_root = str(REPO_ROOT)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

    module = importlib.import_module("transformer_lens.ActivationCache")
    module_path = Path(module.__file__).resolve()
    assert module_path == EXPECTED_MODULE, (
        "ActivationCache was not loaded from this checkout. "
        f"Expected {EXPECTED_MODULE}, got {module_path}."
    )

    activation_cache = module.ActivationCache
    class_path = Path(inspect.getfile(activation_cache)).resolve()
    assert class_path == EXPECTED_MODULE, (
        "ActivationCache class was not defined by this checkout. "
        f"Expected {EXPECTED_MODULE}, got {class_path}."
    )
    return activation_cache


def make_cache(
    activation_cache: type[Any],
    key_count: int,
    make_value: Callable[[], Any],
) -> Any:
    """Build a real ActivationCache with the requested keys and values."""
    return activation_cache(
        {
            f"blocks.{index}.hook_resid_pre": make_value()
            for index in range(key_count)
        },
        model=None,
    )


def time_repr(cache: Any, repeats: int) -> dict[str, Any]:
    """Time only repr(cache), retaining each result for a consistency check."""
    samples = []
    representation = ""
    for _ in range(repeats):
        started = perf_counter()
        representation = repr(cache)
        samples.append(perf_counter() - started)

    return {
        "repr_characters": len(representation),
        "seconds": samples,
        "min_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
    }


def comma_separated_positive_ints(value: str) -> list[int]:
    try:
        parsed = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fixed-keys", type=int, default=100)
    parser.add_argument("--fixed-tensor-elements", type=int, default=1)
    parser.add_argument(
        "--tensor-elements",
        type=comma_separated_positive_ints,
        default=[1, 1_024, 65_536],
        help="Tensor-element sweep at fixed keys (default: 1,1024,65536).",
    )
    parser.add_argument(
        "--key-counts",
        type=comma_separated_positive_ints,
        default=[1, 10, 100, 1_000, 10_000],
        help="Key-count sweep at fixed tensor size (default: 1,10,100,1000,10000).",
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.fixed_keys <= 0 or args.fixed_tensor_elements <= 0:
        parser.error("--repeats, --fixed-keys, and --fixed-tensor-elements must be positive")

    activation_cache = load_checkout_activation_cache()

    # Establish directly that the real class does not render cache values.
    sentinel_cache = make_cache(
        activation_cache, 1, lambda: ValueMustNotBeRendered()
    )
    assert "ValueMustNotBeRendered" not in repr(sentinel_cache)

    tensor_size_sweep = []
    for tensor_elements in args.tensor_elements:
        cache = make_cache(
            activation_cache,
            args.fixed_keys,
            lambda: torch.empty(tensor_elements),
        )
        result = time_repr(cache, args.repeats)
        result.update(
            {
                "key_count": args.fixed_keys,
                "tensor_elements_per_key": tensor_elements,
                "tensor_bytes_per_key": tensor_elements * torch.empty(0).element_size(),
                "total_tensor_bytes": (
                    args.fixed_keys
                    * tensor_elements
                    * torch.empty(0).element_size()
                ),
            }
        )
        tensor_size_sweep.append(result)

    key_count_sweep = []
    for key_count in args.key_counts:
        cache = make_cache(
            activation_cache,
            key_count,
            lambda: torch.empty(args.fixed_tensor_elements),
        )
        result = time_repr(cache, args.repeats)
        result.update(
            {
                "key_count": key_count,
                "tensor_elements_per_key": args.fixed_tensor_elements,
                "tensor_bytes_per_key": (
                    args.fixed_tensor_elements * torch.empty(0).element_size()
                ),
                "total_tensor_bytes": (
                    key_count
                    * args.fixed_tensor_elements
                    * torch.empty(0).element_size()
                ),
            }
        )
        key_count_sweep.append(result)

    print(
        json.dumps(
            {
                "activation_cache_file": str(EXPECTED_MODULE),
                "repeats": args.repeats,
                "fixed_key_count": args.fixed_keys,
                "fixed_tensor_elements": args.fixed_tensor_elements,
                "tensor_size_sweep": tensor_size_sweep,
                "key_count_sweep": key_count_sweep,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
