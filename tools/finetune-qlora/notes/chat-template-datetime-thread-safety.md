# Date/Time Formatting and Chat-Template Thread Safety

## Relevant call path

`tools/finetune-qlora/dataset-template.cpp` calls
`common_chat_templates_apply()`. The Jinja path obtains `datetime` and `date_string`
from `common_chat_extra_context()` in `common/chat.cpp`; its `format_time()` helper
calls `std::localtime()` and copies the returned `std::tm` value.

There is a second call site in `common/jinja/value.cpp`: the Jinja builtin
`strftime_now` passes the result of `std::localtime()` directly to `std::strftime()`.
Both call sites therefore depend on the thread-safety properties of `std::localtime()`.

POSIX specifies that `localtime()` need not be thread-safe and that related time
conversion functions may use shared storage. Concurrent calls can therefore
interfere on implementations that use the shared storage. The POSIX `localtime_r()`
variant writes the result to caller-provided storage.

## Current QLoRA dataset-loading path

`append_tokenized_batch()` in `tools/finetune-qlora/dataset.cpp` calls
`render_dataset_example()` in a sequential loop. It then calls `parallel_for()` for
tokenization. Thus template rendering is serial in the current QLoRA dataset-loading
path; tokenization is the parallel stage.

This serialization is a property of the current caller, not a thread-safety guarantee
provided by `common_chat_templates_apply()`, `format_time()`, or `strftime_now()`.
The comment in `dataset-template.cpp` to keep rendering serial documents this
constraint but does not add synchronization. Revisit both `std::localtime()` call
sites before making template rendering concurrent or allowing concurrent callers.

## Maintenance requirement

If date/time formatting is changed, address both `format_time()` and `strftime_now()`.
Changing only one wrapper does not remove the other `std::localtime()` call. A mutex
around one call site is insufficient if other call sites do not use the same mutex.

## Reference

- [POSIX `localtime()` / `localtime_r()` specification](https://www.man7.org/linux/man-pages/man3/localtime_r.3p.html)
