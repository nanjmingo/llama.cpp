# Chat-Template Date/Time Thread Safety

## Summary

`std::localtime()` is **not thread-safe** under POSIX because it uses shared internal storage. Concurrent calls can overwrite each other's state.

To prevent race conditions, **chat-template rendering in QLoRA is strictly sequential**. Only the subsequent tokenization step is parallelized.

---

## Call Sites Using `std::localtime()`

Two locations in llama.cpp rely on `std::localtime()`:

| Location | Caller | Usage |
|---|---|---|
| `common/chat.cpp` | `common_chat_extra_context()` via `format_time()` | Sets `datetime` and `date_string` fields in Jinja context. |
| `common/jinja/value.cpp` | `strftime_now` (Jinja builtin function) | Formats current timestamp inside Jinja expressions. |

Both call sites share the same underlying storage on platforms where `std::localtime()` is not thread-safe.

---

## Current QLoRA Implementation

In `tools/finetune-qlora/dataset.cpp`:

```text
JSONL Batch (up to 128 lines)
       |
       v
1. Parse JSON records          [Parallel: dataset_thread_pool::parallel_for()]
       |
       v
2. Render chat templates       [SERIAL / Single-Threaded]  <-- Protected here!
       |
       v
3. Tokenize rendered samples   [Parallel: parallel_for()]
```

- Template rendering is deliberately serialized in `append_tokenized_batch()`.
- Tokenization (which does not call `std::localtime()`) runs in parallel.

---

## Invariants & Rules for Developers

1. **Do not parallelize template rendering** without fixing both call sites first.
2. **Fixing only one helper is insufficient:** Wrapping `format_time()` with a mutex does not protect against concurrent calls from `strftime_now`. Both call sites must be migrated to reentrant storage (such as POSIX `localtime_r()` or Windows `localtime_s()`) or share a common lock across `common/`.
3. **Preserve serial rendering:** Keep the serial loop in `dataset.cpp` unless a comprehensive thread-safe time library is integrated project-wide.

---

## Reference

- [POSIX `localtime()` / `localtime_r()` Specification](https://www.man7.org/linux/man-pages/man3/localtime_r.3p.html)
