#!/usr/bin/env python3
"""Probe the real checkout's ActivationCache repr without inspecting cached values.

Run from any directory with:

    python debugging/activation_cache_repr_probe.py --keys 10000

The probe prepends this checkout to sys.path and then asserts that the imported
ActivationCache implementation is exactly transformer_lens/ActivationCache.py
from this repository. It will fail instead of silently using an installed copy.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULE = (REPO_ROOT / "transformer_lens" / "ActivationCache.py").resolve()


class ValueMustNotBeRendered:
    """A cache value whose repr would make accidental value rendering obvious."""

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


def make_cache(activation_cache: type[Any], key_count: int) -> Any:
    """Build a real ActivationCache whose values must not be repr'd."""
    cache_dict = {
        f"blocks.{index}.hook_resid_pre": ValueMustNotBeRendered()
        for index in range(key_count)
    }
    return activation_cache(cache_dict, model=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keys",
        type=int,
        default=10_000,
        help="Number of synthetic cache keys to include (default: 10,000).",
    )
    args = parser.parse_args()
    if args.keys < 0:
        parser.error("--keys must be non-negative")

    activation_cache = load_checkout_activation_cache()
    cache = make_cache(activation_cache, args.keys)

    started = perf_counter()
    representation = repr(cache)
    elapsed_seconds = perf_counter() - started

    assert len(cache) == args.keys
    assert representation.startswith("ActivationCache with keys [")
    assert "ValueMustNotBeRendered" not in representation

    print(
        json.dumps(
            {
                "activation_cache_file": str(EXPECTED_MODULE),
                "key_count": len(cache),
                "repr_characters": len(representation),
                "repr_seconds": elapsed_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
