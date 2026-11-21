"""Time the checkout's real ActivationCache.__repr__, without notebook rendering.

Run from an installed checkout:
    uv run python debugging/activation_cache_repr_probe.py

CPU tensor allocation, cache construction, and output are outside timed regions.
No model or weights are loaded. This is a diagnostic probe, not a regression test.
"""

import argparse
import inspect
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def measure(cache, repeats: int, warmups: int) -> tuple[int, float, float]:
    for _ in range(warmups):
        repr(cache)
    samples = []
    characters = 0
    for _ in range(repeats):
        start = perf_counter_ns()
        rendered = repr(cache)
        elapsed = perf_counter_ns() - start
        samples.append(elapsed / 1_000_000)
        characters = len(rendered)
        del rendered
    return characters, statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-sizes",
        nargs="+",
        type=positive_int,
        default=[1, 1024, 1_048_576],
        help="elements per tensor in size sweep",
    )
    parser.add_argument("--fixed-keys", type=positive_int, default=8)
    parser.add_argument(
        "--key-counts", nargs="+", type=positive_int, default=[1, 100, 1000, 10_000]
    )
    parser.add_argument("--fixed-tensor-size", type=positive_int, default=16)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--warmups", type=positive_int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    expected = root / "transformer_lens" / "ActivationCache.py"
    if not expected.is_file():
        parser.error("place this probe in debugging/ inside the TransformerLens checkout")
    sys.path.insert(0, str(root))

    import torch

    from transformer_lens.ActivationCache import ActivationCache

    actual = Path(inspect.getfile(ActivationCache)).resolve()
    if actual != expected.resolve():
        raise RuntimeError(f"Wrong ActivationCache import: {actual}; expected {expected}")

    print(f"ActivationCache source: {actual}")
    print(f"Python: {sys.version.split()[0]}; PyTorch: {torch.__version__}")
    print(f"CPU float32; repeats={args.repeats}; warmups={args.warmups}")
    print("Times cover repr only; allocation, cache construction, and display are excluded.")
    print("sweep,keys,elements_per_tensor,payload_bytes,repr_chars,median_ms,min_ms")

    def run_case(sweep: str, count: int, elements: int) -> None:
        # Fixed-width names avoid a name-length change at decimal boundaries.
        cache_dict = {
            f"probe.{i:012d}": torch.zeros(elements, dtype=torch.float32, device="cpu")
            for i in range(count)
        }
        cache = ActivationCache(cache_dict, model=None)
        characters, median_ms, min_ms = measure(cache, args.repeats, args.warmups)
        payload_bytes = sum(t.numel() * t.element_size() for t in cache_dict.values())
        print(
            f"{sweep},{count},{elements},{payload_bytes},{characters},"
            f"{median_ms:.6f},{min_ms:.6f}"
        )

    for elements in args.tensor_sizes:
        run_case("tensor_size", args.fixed_keys, elements)
    for count in args.key_counts:
        run_case("key_count", count, args.fixed_tensor_size)


if __name__ == "__main__":
    main()
