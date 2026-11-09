# Thread Safety of `std::localtime()` in `common_chat_templates_apply()`

## Summary

`tools/finetune-qlora/dataset-template.cpp` calls `common_chat_templates_apply()`. When the Jinja renderer is used, the call reaches `common_chat_extra_context()` in `common/chat.cpp`, which formats the current date using `format_time()`. `format_time()` calls `std::localtime()`.

`std::localtime()` may use shared static storage for its returned `std::tm` object. POSIX explicitly says that `localtime()` need not be thread-safe and that calls to related time-conversion functions may overwrite the same storage. Concurrent calls can therefore interfere with one another, so this call path must not be assumed to be thread-safe.

## Call path

1. [`apply_chat_template()` in `dataset-template.cpp`](https://github.com/nanjmingo/llama.cpp/blob/develop2/tools/finetune-qlora/dataset-template.cpp#L13-L25) calls `common_chat_templates_apply()`.
2. [`common_chat_templates_inputs::use_jinja`](https://github.com/nanjmingo/llama.cpp/blob/develop2/common/chat.h#L256-L263) defaults to `true`. `common_chat_templates_apply()` dispatches to `common_chat_templates_apply_jinja()` when this flag is enabled.
3. [`common_chat_templates_apply_jinja()` in `common/chat.cpp`](https://github.com/nanjmingo/llama.cpp/blob/develop2/common/chat.cpp#L1299-L1368) assigns `common_chat_extra_context()` to `params.extra_context`.
4. [`common_chat_extra_context()`](https://github.com/nanjmingo/llama.cpp/blob/develop2/common/chat.cpp#L1065-L1072) calls `format_time()` for both `datetime` and `date_string`.
5. [`format_time()`](https://github.com/nanjmingo/llama.cpp/blob/develop2/common/chat.cpp#L37-L44) calls `std::localtime(&time)` and copies the pointed-to `std::tm` value.

There is another relevant call site: the Jinja builtin [`strftime_now` in `common/jinja/value.cpp`](https://github.com/nanjmingo/llama.cpp/blob/develop2/common/jinja/value.cpp#L434-L443) passes `std::localtime()` directly to `std::strftime()`. A change limited to `format_time()` would not address this separate call path.

## Current QLoRA dataset-loading path

In [`append_tokenized_batch()` in `dataset.cpp`](https://github.com/nanjmingo/llama.cpp/blob/develop2/tools/finetune-qlora/dataset.cpp#L45-L83), the loop that calls `render_dataset_example()` is sequential. The code calls `pool->parallel_for()` afterwards for tokenization. Template rendering is therefore serial in this particular loading path today.

That is a property of the current caller, not a thread-safety guarantee provided by `common_chat_templates_apply()` or `format_time()`: neither function acquires a lock to serialize concurrent calls. The `Keep this call serial` comment in `dataset-template.cpp` documents the constraint; it does not implement synchronization. If rendering is parallelized later, or another caller invokes this path concurrently, the shared `std::localtime()` storage becomes relevant.

## Related upstream pull requests

- [PR #24047 — `common: centralise datetime formatting`](https://github.com/ggml-org/llama.cpp/pull/24047) proposes a shared `time_format` function in `common.h` / `common.cpp` for formatting C++ chrono time points. Its overview explicitly identifies scattered formatting code that uses threading-unsafe functions such as `localtime` and `ctime`. The PR is closed, not merged. In the review discussion, the guidance for `common/jinja/value.cpp` was to leave the existing `strftime_now` code as-is and make `chat.cpp` responsible for calling `common_time_format`; this PR should not be treated as evidence that all call sites were converted.
- [PR #23473 — `server: make get_datetime return a locale-independent format`](https://github.com/ggml-org/llama.cpp/pull/23473) proposes a locale-independent format for the server's `get_datetime` tool. Its discussion includes a `localtime_r` / `localtime_s` approach, but the change concerns the server tool and does not directly fix the QLoRA template-rendering path described above.
- [PR #26117 — `server: add format arg to datetime tool`](https://github.com/ggml-org/llama.cpp/pull/26117) was merged and describes itself as an alternative to a comment on PR #24047. It changes the server datetime tool so the model can choose the output format. This server-specific change does not, by itself, demonstrate that `std::localtime()` was removed from `common_chat_templates_apply()` or from `strftime_now`.

## Development notes

- Before making template rendering concurrent, review both `format_time()` in `common/chat.cpp` and `strftime_now` in `common/jinja/value.cpp`.
- A mutex used by only one wrapper is insufficient if other call sites continue to call `std::localtime()` without using the same mutex. Prefer a thread-safe time-conversion API with caller-owned storage, or a formatting implementation that does not depend on shared static storage.
- Do not infer that this fork's call path is fixed merely because a related upstream PR exists. Check the actual `common/chat.cpp` and `common/jinja/value.cpp` implementations in the target revision.

## References

- [POSIX `localtime()` / `localtime_r()` specification](https://www.man7.org/linux/man-pages/man3/localtime_r.3p.html) — documents that `localtime()` need not be thread-safe and describes `localtime_r()` as writing the result to caller-provided storage.
- [Upstream PR #24047](https://github.com/ggml-org/llama.cpp/pull/24047)
- [Upstream PR #23473](https://github.com/ggml-org/llama.cpp/pull/23473)
- [Upstream PR #26117](https://github.com/ggml-org/llama.cpp/pull/26117)
