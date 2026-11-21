# ActivationCache representation investigation (#1126)

Status: investigation only; no runtime behavior changes.
Base: main at 33387b7def1ead3a8cba532a729da5599e875873.
Branch: investigate/activation-cache-repr-1126.

## Report versus evidence

The emailed report says: "Printing a large cache hangs the notebook. Could repr be truncated?" It contains no reproduction, exact expression, versions, device, key count, timing, or traceback. The symptom is reported; attribution to ActivationCache.__repr__ is unproven. The public upstream #1126 is a different item, so it must not be treated as this report.

## Source trace at the base commit

ActivationCache.py:176-183 defines __repr__ as:

```python
return f"ActivationCache with keys {list(self.cache_dict.keys())}"
```

ActivationCache is a plain wrapper, not a dict subclass. It defines no __str__, _repr_pretty_, HTML representation, or MIME representation. Its constructor stores the dictionary/model references and flags; representation does not traverse the model or tensor values.

| Expression | Normal path |
| --- | --- |
| repr(cache) | __repr__ builds and formats a list of keys |
| str(cache), print(cache), f"{cache}" | Same representation; print also writes the string |
| Bare cache or display(cache) in IPython | Standard text/plain formatting falls back to __repr__ |
| cache.cache_dict or dict(cache) | Dictionary formatting exposes tensor values |
| cache[key] | Displays the selected tensor |
| Bare model.run_with_cache(...) | Displays the returned tuple, including model output (usually logits) |

Dictionary-like methods do not automatically cause IPython to expand the wrapper. Custom formatters, inspectors, subclasses, or monkey patches remain environment-specific possibilities.

HookedTransformer.py:735-742 calls the parent run_with_cache first, then wraps the returned dictionary when return_cache_object=True (the default); False returns the raw dictionary. HookedRootModule.py:417-435 installs caching hooks and executes the forward pass, optionally backward, before returning. Its save_hook at lines 483-512 uses tensor.detach().to(device), optional batch removal and position slicing, then dictionary insertion. Device movement and retained activation memory belong to cache production, not representation; detach is not a blanket tensor copy.

TransformerBridge also returns (output, ActivationCache) by default (model_bridge/bridge.py:2235-2239), and can return a raw dictionary. Compatibility aliases can add keys referencing existing tensors.

HookPoint.add_hook (hook_points.py:248-254 and 273-276) explicitly avoids representing functools.partial bound arguments when naming hooks, because they can contain large objects. The current code already has that guard. This is a separate representation path, not evidence that displaying ActivationCache calls it.

## Where cost could arise

1. Many keys or long names: __repr__ formats every key. Cost scales with key count and formatted key length, not tensor element count. List allocation, string construction, output transport, and frontend rendering are distinct costs. Truncation could help if key output is the actual problem.
2. Tensor formatting through raw dictionaries, individual tensors, or the output tuple: PyTorch formatting performs numerical formatting and value extraction. Device-backed values can introduce synchronization and transfer costs. Normal print settings summarize large tensors; large tensors do not automatically print every element. Inspect print settings, including full-output settings. The PyTorch v2.6.0 torch/_tensor_str.py source was inspected as a reference, not as proof of the reporter's installed version.
3. Work before display: forward execution, caching, transfers, memory pressure, and optional backward work can dominate an entire cell. Assign outputs in a separate cell to isolate representation.
4. Notebook-specific handling: registered formatters, variable inspectors, output transport, and browser rendering need separate measurement if repr is fast.

## Checks performed

An isolated Python check extracted the exact __repr__ method via AST from the inspected source into a minimal class. Dictionary values and the model were sentinels whose __repr__ raises. repr, str, print to StringIO, and IPython 9.18.1 DisplayFormatter.format all succeeded and emitted only the key list. Representing the raw dictionary raised as expected. The installed IPython pretty-printer source was inspected: normal custom __repr__ dispatch calls _repr_pprint, which calls repr(obj).

A single synthetic timing run used keys blocks.{i}.hook_resid_pre and sentinel values, timing only repr after dictionary construction:

| Keys | Output characters | Time (ms) |
| ---: | ---: | ---: |
| 100 | 2,816 | 0.010 |
| 10,000 | 298,916 | 2.393 |
| 100,000 | 3,088,916 | 25.562 |

These are illustrative single-run measurements, not a benchmark or reproduction of the hang. No frontend rendering was timed. PyTorch was unavailable in the execution environment; the full library, GPU behavior, actual notebook, and reporter environment were not tested. No project test suite or docs build was run for this investigation.

## Reproduce before changing behavior

1. Record the exact failing expression, type(cache), imported source path, commit/package versions, Python/IPython/PyTorch versions, notebook frontend/extensions, device, key count, representative shapes, and tensor print options. Confirm the running kernel imports this checkout.
2. Run and assign out, cache = model.run_with_cache(...) separately from any display. For accelerator timing, account for queued work with appropriate explicit synchronization around measurements.
3. Time s = repr(cache) without displaying s; inspect len(s). Separately time print(cache) into StringIO and the kernel formatter (get_ipython().display_formatter.format(cache)), assigning their results. Then compare real print/display and browser responsiveness.
4. Compare wrapper, raw dictionary, selected tensor, and model-output display independently. Do not accidentally benchmark the returned tuple when investigating the cache wrapper.
5. Hold keys fixed while varying tensor sizes; separately vary key count/name length with cheap placeholders. Use a sentinel representation to detect accidental value formatting.
6. Capture a stack/profile during a kernel stall. If string formatting finishes quickly but the UI stalls, investigate output transport and frontend rendering instead.

Decision: change truncation only after excessive key output is shown to cause the problem. If tensor formatting, cache production, or an external formatter is responsible, changing this wrapper's representation does not address that mechanism.
