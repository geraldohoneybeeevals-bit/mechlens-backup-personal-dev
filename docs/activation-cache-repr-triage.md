# ActivationCache repr triage (#1126)

**Is `ActivationCache.__repr__` itself the problem in this checkout? No.** The source and CPU measurements do not support the report's implied tensor-size-driven slowdown: repr formats only keys and never touches cached tensor values. With eight fixed keys, increasing tensor payload from 32 bytes to 512 MiB kept repr at about 0.9 microseconds and 202 characters.

This verdict does not establish the notebook hang's actual cause. Key count still matters: 100,000 keys took about 6.24 ms to format and produced 2.2 million characters; notebook transport and rendering were not measured. Reproduce the exact failing expression before changing behavior.

Investigation branch: `investigate/activation-cache-repr-1126`. Runtime behavior is unchanged. Source inspected at base commit `33387b7def1ead3a8cba532a729da5599e875873`; the real-tensor probe was run at `6580eefe4138069b50fbbb797a6323e586ca8517`.

## What the report says

The emailed #1126 report says: "Printing a large cache hangs the notebook. Could repr be truncated?" It supplies no exact expression, reproduction, versions, device, key count, timings, or traceback. The symptom is reported; attribution to `ActivationCache.__repr__` is unproven. Public upstream #1126 is a different item and is not evidence for this report.

## What the current code does

[`ActivationCache.__repr__`](../transformer_lens/ActivationCache.py#L176) returns:

```python
return f"ActivationCache with keys {list(self.cache_dict.keys())}"
```

It builds a list of keys and formats those keys. It does not traverse tensor values or the model. Its cost scales with key count and total formatted key length, not the tensors' element count. The wrapper has no custom `__str__`, `_repr_pretty_`, HTML, or MIME representation.

| Expression | Normal formatting path |
| --- | --- |
| `repr(cache)` | Keys-only `__repr__` |
| `str(cache)`, `print(cache)`, `f"{cache}"` | Same representation; printing additionally writes the string |
| Bare `cache` or `display(cache)` in IPython | Standard text/plain formatter falls back to `__repr__` |
| `cache.cache_dict`, `dict(cache)`, or `cache[key]` | Exposes tensor values and can invoke tensor formatting |
| Bare `model.run_with_cache(...)` | Displays the tuple, including model output (usually logits) as well as the cache |

Dictionary-like methods do not make IPython expand the wrapper automatically. Custom formatters, inspectors, subclasses, or monkey patches can change the notebook path. An isolated check using the extracted method and IPython 9.18.1 confirmed the standard repr/str/print/MIME paths did not format sentinel values; that check was separate from the real-package measurements below.

[`HookedTransformer.run_with_cache`](../transformer_lens/HookedTransformer.py#L735) runs the model/caching hooks before constructing the wrapper. `return_cache_object=False` returns the raw dictionary. [`HookedRootModule.save_hook`](../transformer_lens/HookedRootModule.py#L483) detaches, optionally moves tensors to another device, slices, and stores them. Cache production, transfers, retained memory, and optional backward execution are separate from representation. `TransformerBridge` likewise returns a wrapper by default and can return a raw dictionary.

Large key output can still be expensive to construct, transport, or render. Tensor formatting through other expressions can perform numerical formatting and device synchronization/value transfers; PyTorch normally summarizes large tensors, so print settings matter. None of these possibilities establishes the reporter's cause.

## CPU measurements and exact command

Ran the committed [probe](../debugging/activation_cache_repr_probe.py), importing the real `ActivationCache` from this checkout; its source-path check passed. No extracted class, mocked tensor, or model weights were used. Environment: Python 3.12.14, PyTorch 2.15.0+cpu, CPU float32 tensors. Dependencies were installed in a separate environment from package requirements, not a frozen `uv.lock` sync.

Exact command executed from `/workspace/scratch/7d699218bb50/mechlens` (the environment was the sibling `.probe-venv` directory):

```bash
HF_HUB_OFFLINE=1 uv run --no-project --python ../.probe-venv/bin/python debugging/activation_cache_repr_probe.py --repeats 200 --warmups 20 --tensor-sizes 1 1024 1048576 16777216 --key-counts 1 100 1000 10000 100000
```

Defaults not overridden: `--fixed-keys 8` and `--fixed-tensor-size 16`. Every case used 20 warmups followed by 200 measurements. Allocation, cache construction, output-length measurement, and printing were outside the timed region. Keys had fixed-width names. Times below are milliseconds, copied from probe output.

| Sweep | Keys | Elements per tensor | Tensor payload (bytes) | Repr characters | Median ms | Minimum ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tensor size | 8 | 1 | 32 | 202 | 0.000901 | 0.000861 |
| Tensor size | 8 | 1,024 | 32,768 | 202 | 0.000911 | 0.000861 |
| Tensor size | 8 | 1,048,576 | 33,554,432 | 202 | 0.000901 | 0.000861 |
| Tensor size | 8 | 16,777,216 | 536,870,912 | 202 | 0.000891 | 0.000861 |
| Key count | 1 | 16 | 64 | 48 | 0.000381 | 0.000351 |
| Key count | 100 | 16 | 6,400 | 2,226 | 0.006710 | 0.006439 |
| Key count | 1,000 | 16 | 64,000 | 22,026 | 0.061396 | 0.059609 |
| Key count | 10,000 | 16 | 640,000 | 220,026 | 0.612476 | 0.576548 |
| Key count | 100,000 | 16 | 6,400,000 | 2,200,026 | 6.236173 | 5.995250 |

Increasing tensor payload from 32 bytes to 512 MiB under the same keys did not increase representation time or output length. Increasing key count increased both. These are measurements from one environment, not universal latency guarantees. The experiment does not measure notebook transport/rendering, GPU synchronization, cache creation, or reproduce the reported hang. It supersedes the earlier illustrative timing of an extracted method with sentinel values.

## Separate note: functools.partial repr in hook_points.py

[`HookPoint.add_hook`](../transformer_lens/hook_points.py#L248) names registered hooks. A `functools.partial` representation can recursively format its bound arguments, potentially including a large dictionary of tensors. The current guard avoids formatting those bound arguments:

```python
full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
```

There is an analogous guard for the backward-hook wrapper. This happens during hook registration. Printing an `ActivationCache` does not call `add_hook`. The comment documents a related performance concern on a separate code path; no evidence links it specifically to the emailed #1126 report, and the current code already guards that path.

## Regression coverage and acceptance status

[`tests/unit/test_activation_cache_repr.py`](../tests/unit/test_activation_cache_repr.py) checks empty, single-key, and multiple-key output, including insertion order and quoting. Guards fail if repr accesses dictionary values or formats tensors. They are armed after construction because runtime argument type checking legitimately inspects values.

Actual results on CPU:

```text
tests/unit/test_activation_cache_repr.py
3 passed, 1 warning in 5.76s

tests/acceptance/test_activation_cache.py
FAILED tests/acceptance/test_activation_cache.py::test_stack_neuron_results_project_incl_remainder
1 failed, 41 passed, 2 warnings in 93.44s (0:01:33)
```

The acceptance failure reproduced in a diagnostic rerun (41 passed, 1 failed). One float32 element in the remainder component differed: actual `0.15668678283691406`, expected `0.15666961669921875`, absolute difference `1.71661376953125e-05`. No elements before the remainder exceeded the test tolerance. This is a projection assertion failure, not a repr failure; its root cause remains unresolved. No runtime calculation or tolerance was changed. The initial environment-only model-loading failure was resolved by installing missing `socksio` before these results. Warnings concern SVDInterpreter instrumentation and the deprecated utils import.

## What to reproduce before changing behavior

1. Capture the exact expression, object type, imported source path, versions, device, key count, tensor shapes, print settings, and notebook extensions.
2. Assign model output and cache in a separate cell; time `s = repr(cache)` without displaying `s`, then inspect its length. Account for queued accelerator work when measuring device operations.
3. Separate string formatting, standard kernel formatting, output transport, and browser rendering. Compare wrapper, dictionary, selected tensor, and output tuple independently.
4. Capture a stack/profile during a kernel stall. Truncate keys only if excessive key output reproduces the problem; the report alone does not justify changing representation behavior.
