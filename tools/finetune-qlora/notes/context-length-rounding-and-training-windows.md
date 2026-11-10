# Context-Length Rounding and QLoRA Training Windows

## Context capacity in the runtime

`src/llama-context.cpp` rounds `cparams.n_ctx` up to a multiple of 256 unless the
context satisfies the runtime's full-context-training condition. For non-unified KV
cache configurations, `n_ctx_seq` is also rounded unless full-context training is
active. The padding is applied after the effective batch and micro-batch sizes have
been derived, so those values do not necessarily grow with the padded context.

The full-context-training condition requires `no_alloc_kv_cache`, an effective
`n_ubatch` equal to `n_ctx`, an effective `n_batch` at least `n_ctx`, no secondary
context, and either a unified KV cache or a context length divisible by the sequence
count. QLoRA sets `no_alloc_kv_cache` and requires a full-context microbatch. Under
these conditions, the requested context capacity is retained instead of rounded to
256. Code that depends on effective values should query the initialized context with
`llama_n_ctx()`, `llama_n_batch()`, and `llama_n_ubatch()` rather than assuming they
match requested settings.

`no_alloc_kv_cache` does not mean that every model's KV cache is omitted. In
full-context training, the runtime omits KV data-buffer allocation for models without
shared KV. Models with shared KV still allocate the cache they need, while retaining
the requested full-context capacity.

## Dataset windows and padding

Context-capacity rounding does not append token IDs to inference batches. QLoRA's
window builder has a separate padding operation: a sample shorter than the context
length is padded to one full window, using the BOS token when available (otherwise
token ID 0), and positions without a next-token target receive label `-1`. For
samples longer than the context length, only complete context windows are created;
a trailing fragment is not emitted as a separate padded window. Targets already
covered by an earlier overlapping window are masked to avoid duplicate training.

For `text` samples, the first half of the first window is also masked as context.
This initial mask is applied only to the first window. Chat and prompt/completion
samples use their rendered supervision weights instead of this additional text
initial-context mask.

## Loss normalization with masked positions

The raw `ggml_cross_entropy_loss` reduction divides by the number of rows, including
rows whose labels are masked. The current QLoRA path applies a separate normalization:
`llama_context::opt_epoch()` sums supervision weights over the accumulation group and
passes the sum to `ggml_opt_set_loss_normalizer()`. The optimizer uses that normalizer
when scaling the accumulated gradient. QLoRA's progress and epoch loss summaries are
also converted to means over supervised target weight.

Keep these quantities distinct when inspecting loss values: the raw cross-entropy
reduction includes masked rows in its denominator, while the optimizer gradient and
QLoRA-reported mean loss are normalized by supervised target weight.

## Model-backed gradient tests

A model-backed test should assert the effective context and batch values returned by
the initialized context, and use the same window length in the dataset and independent
loss calculation. For a single full-window finite-difference check, a fixture can use
a 256-token context with batch and micro-batch capacities of 256 and provide 257
sample tokens (256 input positions plus the final next-token target). The test should
still query the initialized context and assert these invariants rather than relying
only on requested settings.
