# Context-Length Rounding and QLoRA Training Windows

## Summary

`llama.cpp` rounds the effective context length up to a multiple of 256 when a context is created. For example, requesting `n_ctx = 32` results in an effective `n_ctx` of 256. This is an alignment optimization for the KV cache, not a rule that training sequences must contain 256 meaningful tokens.

The behavior was introduced upstream in [PR #17046: kv-cache: pad the cache size to 256 for performance](https://github.com/ggml-org/llama.cpp/pull/17046). The PR cites CUDA Flash Attention's preference for K/V tensor sizes that are rounded, with possible benefits for other backends as well. The change addressed performance reports including [Issue #17037](https://github.com/ggml-org/llama.cpp/issues/17037); the PR also links [Issue #17058](https://github.com/ggml-org/llama.cpp/issues/17058).

## Where the rounding happens

In `src/llama-context.cpp`, `cparams.n_ctx` is assigned from the requested context and `cparams.n_batch` / `cparams.n_ubatch` are derived before this line:

```cpp
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

Consequently, requesting a small context can leave the effective context larger than the batch and micro-batch capacities calculated earlier. With a requested context of 32, the effective context can become 256 while `n_batch` and `n_ubatch` remain 32. Code that needs the effective values should query the initialized context using `llama_n_ctx()`, `llama_n_batch()`, and `llama_n_ubatch()` rather than assuming they equal the requested values.

This ordering matters to QLoRA because `llama_context::opt_epoch()` uses the effective context length to fetch a complete training window from `ggml_opt_dataset_t`. A dataset built with a 32-token window is not compatible with an optimizer epoch that attempts to read a 256-token window. The shared runtime should not be changed just to work around a test fixture that was built with the wrong length.

## Does rounding add padding tokens?

No. The context-size rounding changes the context/KV-cache capacity; it does not automatically append token IDs to every `llama_decode()` batch. Ordinary inference can still submit fewer tokens than the context capacity.

QLoRA has a separate, explicit notion of window padding. In `dataset-window.cpp`, each training window is allocated at the requested dataset window length. If the sample does not fill the window, the remaining input positions are filled with a padding token. Those positions have no next-token target, so their labels are set to `-1`.

In `llama_context::opt_epoch_iter()`, the label tensor is zeroed first and a one-hot target is written only for labels with a non-negative target and a positive supervision weight. The `-1` positions and zero-weight targets therefore contribute zero to the cross-entropy numerator. The raw `ggml_cross_entropy_loss` reduction divides by the total number of rows, including rows whose label sum is zero, so the uncorrected cross-entropy value is diluted by masked positions. The CPU implementation skips zero-label rows in the numerator and divides by the full row count; the CUDA implementation gives those rows zero loss and uses the full row count in each row contribution.

The current QLoRA path compensates for that raw reduction. Before each optimizer accumulation group, `llama_context::opt_epoch()` sums the supervision weights for the group and sets that value with `ggml_opt_set_loss_normalizer()`. In `ggml-opt.cpp`, the optimizer rescales the accumulated gradient to undo the per-microbatch and accumulation-period averages, then normalizes it by the group's total supervision weight. QLoRA's progress and epoch summaries likewise convert the raw loss sum to a mean over supervised target weight. Therefore, the raw cross-entropy tensor's denominator includes masked rows, but the current QLoRA optimizer gradient and reported mean loss are normalized over supervised weight; adding masked positions alone should not be described as shrinking those normalized results.

For the current causal-language-model window builder, padding is appended at the end of a sample window and the corresponding targets are masked. Earlier supervised positions cannot attend to future padding tokens under causal attention. This should not be generalized to architectures or attention modes with different masking semantics without verification.

## Implications for the model-backed gradient test

A correct model-backed test must make the effective model context, the dataset window length, and the sequence used by the independent loss calculation agree. It must also ensure that `n_batch` and `n_ubatch` can accommodate the intended training window. Setting `n_ctx` alone before context creation is not sufficient when the runtime rounds the value after deriving those capacities.

For a full-window finite-difference check, a simple fixture is to request a context of 256, set both batch capacities to 256, and provide 257 sample tokens: 256 input tokens plus the next-token target for the final position. That creates one 256-position training window without trailing padding labels. The test should still query the resulting context values and assert the invariants instead of relying only on requested settings.

## Practical guidance

- Keep the 256 rounding for ordinary inference, where it can improve KV-cache performance.
- Full-context QLoRA training sets `no_alloc_kv_cache` and uses one microbatch for the requested context. In that mode, the runtime keeps the requested context and per-sequence capacity exact instead of rounding them to 256; models with shared KV still allocate their required cache, but at the requested capacity.
- When adding tests around training, query actual context and batch values after creating the context.
- Keep logical dataset-window length and runtime context capacity explicit; do not assume they are interchangeable for all APIs.
- Test raw cross-entropy reduction separately from the QLoRA normalization applied to optimizer gradients and reported losses. The raw reduction includes masked rows in its denominator; the QLoRA path normalizes by supervised target weight.
