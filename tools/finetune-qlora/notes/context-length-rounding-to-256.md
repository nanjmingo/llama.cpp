# Context Length Rounding to 256

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

In `llama_context::opt_epoch_iter()`, the label tensor is zeroed first and a one-hot target is written only for labels greater than or equal to zero. The `-1` positions therefore contribute zero cross-entropy. The CPU implementation of `ggml_cross_entropy_loss` skips rows whose label sum is zero, but its final reduction still divides by the total number of rows, not the number of supervised rows. Masked positions are excluded from the loss numerator, but they remain part of the denominator. As a result, increasing the number of masked positions can reduce the reported mean loss and the gradient scale. That behavior is separate from the context-capacity rounding and deserves its own treatment when changing loss normalization.

For the current causal-language-model window builder, padding is appended at the end of a sample window and the corresponding targets are masked. Earlier supervised positions cannot attend to future padding tokens under causal attention. This should not be generalized to architectures or attention modes with different masking semantics without verification.

## Implications for the model-backed gradient test

A correct model-backed test must make the effective model context, the dataset window length, and the sequence used by the independent loss calculation agree. It must also ensure that `n_batch` and `n_ubatch` can accommodate the intended training window. Setting `n_ctx` alone before context creation is not sufficient when the runtime rounds the value after deriving those capacities.

For a full-window finite-difference check, a simple fixture is to request a context of 256, set both batch capacities to 256, and provide 257 sample tokens: 256 input tokens plus the next-token target for the final position. That creates one 256-position training window without trailing padding labels. The test should still query the resulting context values and assert the invariants instead of relying only on requested settings.

## Practical guidance

- Treat the 256 rounding as an upstream performance optimization unless benchmarks show a reason to revisit it; do not remove it solely to make a small test context work.
- When adding tests around training, query actual context and batch values after creating the context.
- Keep logical dataset-window length and runtime context capacity explicit; do not assume they are interchangeable for all APIs.
- Test the loss and gradient paths separately. A zeroed label masks a loss row, but does not by itself guarantee that loss normalization is independent of the masked-token count.
- If support for logical windows smaller than the effective context is needed, design it explicitly in the training path rather than changing shared context initialization without backend-wide validation.
