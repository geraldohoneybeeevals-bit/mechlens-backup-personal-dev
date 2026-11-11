# ActivationCache representation triage

## Issue report / scope

The reported concern is that displaying a large `ActivationCache` hangs a notebook, with an
initial assumption that printing cached activation tensors is responsible.

The referenced public TransformerLens item [#1126][public-1126] is a closed pull request titled
“Attention cleanup,” not an issue report. Its body is the unfilled pull-request template, and its
diff changes neither `ActivationCache.py` nor `hook_points.py`. It therefore does not establish a
reproduction, a root cause, or a proposed fix for this behavior.

This triage applies to `mechlens-backup-personal-dev`, branch
`investigate/activation-cache-repr-1126`.

## What the code actually does

### The normal `HookedTransformer.run_with_cache` path returns an `ActivationCache`

`HookedTransformer.run_with_cache` defaults `return_cache_object=True`. It calls the inherited
implementation to get `cache_dict`, then wraps that dictionary in `ActivationCache`:

```python
out, cache_dict = super().run_with_cache(...)
if return_cache_object:
    cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
    return out, cache
return out, cache_dict
```

Passing `return_cache_object=False` is the explicit opt-out that returns the plain dictionary.

### `ActivationCache.__repr__` formats keys, not cached values

The current implementation is:

```python
return f"ActivationCache with keys {list(self.cache_dict.keys())}"
```

It copies the mapping’s key references into a list and formats the resulting list of key strings.
It does not iterate over `cache_dict.values()`, call `repr()` on a cached tensor, transfer tensor
data between devices, or serialize tensor contents.

For normal Python `print(cache)`, the object has no `__str__`, so this representation is used.
This checkout supplies no `_repr_html_` or `_repr_mimebundle_` method for `ActivationCache`.
A notebook’s frontend/output formatter can still add transport and rendering cost for the resulting
text output; that is outside this class’s implementation and must be measured in a notebook
reproduction.

### The `partial.__repr__` guard is separate

`HookPoint.add_hook` assigns a generated name to an internal hook wrapper. For `functools.partial`
hooks it deliberately avoids `partial.__repr__()`:

```python
if isinstance(hook, partial):
    # partial.__repr__() can be extremely slow if arguments contain large objects,
    # which is common when caching tensors.
    full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
```

This occurs during hook registration, before the model forward pass and before a cache is returned.
It is not part of `ActivationCache.__repr__`.

The `run_with_cache` setup path creates the caching callbacks as
`partial(save_hook, is_backward=False)` (and, optionally, the backward equivalent). The cache is
captured by the nested `save_hook` function rather than passed as a `partial` argument. The
existing guard nevertheless avoids the general costly-partial-repr failure mode.

## Measurements

**Status: pending.** No probe timings have been collected from this branch. Do not interpret the
absence of values below as zero cost or use estimated timings.

The branch contains `debugging/activation_cache_repr_probe.py`. It asserts that it imports
`transformer_lens/ActivationCache.py` from this checkout, then times `repr(cache)` repeatedly
over two independent sweeps:

1. Fixed key count, increasing tensor size per cached value.
2. Fixed tensor size, increasing key count.

It also uses a real `torch.Tensor` subclass whose `__repr__` raises, confirming that the
`ActivationCache` representation does not render cached values.

Run:

```bash
uv run python debugging/activation_cache_repr_probe.py
```

Expected measurement fields are `repr_characters`, raw timing samples, minimum, median, mean,
key count, and tensor-byte totals. Results should be added here only after that command runs in the
actual checkout.

## Diagnosis

### Established

- The default `HookedTransformer.run_with_cache` result is an `ActivationCache`.
- Its representation contains cache key names only; it does not format activation tensors.
- Its direct Python-side work grows with the number of keys and the total length of their names.
- The repository already avoids a different potential repr cost: calling `partial.__repr__()`
  while registering hooks.

### Not established

- That printing or displaying the `ActivationCache` is the source of the reported hang.
- That the notebook kernel is blocked, rather than the frontend rendering or receiving a large
  `text/plain` result.
- That `partial.__repr__` occurs on, or causes slowness in, the reported execution path.
- Any numerical relationship between activation tensor size and `repr(cache)` time.

### More plausible cost centers to isolate

1. **Cache construction:** `run_with_cache` installs callbacks and stores selected activations
   during the forward pass. The default cache captures every selected hook output.
2. **Device transfers:** requesting a cache device different from the model device can add
   per-hook transfer cost.
3. **Memory pressure:** retaining all activations can exhaust or pressure accelerator/host memory.
4. **Notebook output handling:** rendering a very large key-list string can stall a frontend even
   if kernel-side `repr(cache)` is quick.

These are hypotheses, not conclusions from the current report.

## Next steps

1. Run the focused representation regression test:

   ```bash
   uv run pytest -q tests/unit/test_activation_cache_repr.py
   ```

2. Run the existing ActivationCache acceptance suite (it loads `solu-2l` and can require a model
   download):

   ```bash
   uv run pytest -q tests/acceptance/test_activation_cache.py
   ```

3. Run the probe and record its JSON output in the Measurements section:

   ```bash
   uv run python debugging/activation_cache_repr_probe.py
   ```

4. Reproduce in a notebook with separate cells for:

   ```python
   repr(cache)  # kernel-side representation generation
   cache        # notebook output transport and frontend rendering
   ```

   Record separate wall-clock measurements for cache construction, `repr(cache)`, and notebook
   display. Only change `ActivationCache` behavior if the reproduction identifies its
   representation as the material cost.

[public-1126]: https://github.com/TransformerLensOrg/TransformerLens/pull/1126
