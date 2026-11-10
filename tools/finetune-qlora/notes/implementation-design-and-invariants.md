# QLoRA Implementation Design and Internal Invariants

This note records implementation details that are useful when modifying or debugging
the QLoRA integration. It is not a user guide; user-facing setup and runtime behavior
are documented in the parent `README.md`.

## Dataset-loading implementation

The JSONL reader processes up to 128 records per batch. `dataset_thread_pool::parallel_for()`
executes a one-task batch synchronously on the calling thread, without dispatching work
to workers. Reducing the batch size to one would therefore remove parallelism from JSON
parsing and tokenization. Reducing memory further would require a bounded asynchronous
pipeline rather than simply reducing the batch size to one.

The QLoRA-specific command-line options are parsed in `args.cpp`; shared llama.cpp
options continue to use `common/arg.cpp`. JSONL parsing and tokenization use the shared
`-t` / `--threads` setting and can run in parallel; chat-template rendering remains
sequential.

## Selected changed llama.cpp files

This is a selection of significant changes, not a complete file inventory.

| File | Change |
|---|---|
| `ggml/src/ggml.c` | Adds backward support for 3D `GET_ROWS_BACK`, `MUL_MAT_ID`, and `FLASH_ATTN_EXT` (including Q/K/V and optional attention sinks). `SET_ROWS` and `SSM_CONV` / `SSM_SCAN` stop gradient propagation through their sources; they do not gain backward implementations. The in-place/view assertion remains. |
| `src/llama-context.cpp` | Preserves exact context capacity for full-context training; sizes the training scheduler before `ggml_opt_init`; warns when trainable LoRA targets match shared-KV source projections; reuses tensor metadata context during window iteration and selects accumulation groups. |
| `src/llama-adapter.cpp` | Adds a repack-buffer fallback for LoRA tensors, trying a device-native buffer type before CPU. |
| `common/common.h`, `common/common.cpp` | Adds `no_alloc_kv_cache` to common parameters and maps it to `llama_context_params`. `common/arg.cpp` is unchanged by this addition. |
| `tools/finetune-qlora/args.cpp` | Parses QLoRA-only CLI flags locally; no QLoRA option registration is added to `common/arg.cpp`. |
| `include/llama.h` | Exposes `llama_opt_dataset_shuffle()` so callers control per-epoch shuffling; adds `llama_opt_epoch_range()` with an explicit evaluation-range end. |
| `ggml/src/ggml-cuda/out-prod.cu` | Adds CUDA `OUT_PROD` handling for quantized `src0` and `OUT_PROD_ID` for MoE backward. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Adds `supports_op` handling for quantized `OUT_PROD` and `OUT_PROD_ID`; includes a CPU-resident IDs fix in `mul_mat_id`. |

## Key invariants

- `params.load_mode = LLAMA_LOAD_MODE_NONE` is set before `common_init_from_params()`. The model loader maps this mode to `use_mmap = false`. The code establishes the selected loading mode, but does not document why QLoRA requires it; do not infer that frozen base tensors are written back during training.
- `params.warmup = false` is set before `common_init_from_params()`. The code establishes that warmup is disabled, but does not establish the previously suggested claim that warmup would run inference with uninitialized LoRA PARAM tensors or cause a segmentation fault. A reproduction or historical failure report is needed to justify that rationale.
- `params.no_alloc_kv_cache = true` is set for training. When the requested context and batch settings satisfy the full-context conditions, the context preserves the requested capacity; it skips KV data-buffer allocation only when the model has no shared KV. Shared-KV models still require their cache.
- FlashAttention follows the common `--flash-attn` setting (default `auto`); CPU and CUDA backward implementations are available.
- QLoRA defaults `params.cache_type_k` and `params.cache_type_v` to F32 before common argument parsing; `--cache-type-k` and `--cache-type-v` override these defaults.
- LoRA A/B tensors are marked `PARAM` with `ggml_set_param()` on tensors loaded by `llama_adapter_lora_init()`, not on the pre-init scratch tensors in `lt.buf`.
- `common_init_from_params()` creates the base model and context before the temporary adapter GGUF is created. The adapter is then loaded with `llama_adapter_lora_init()`, attached with `common_set_adapter_lora()`, and its trainable A/B tensors are marked `PARAM`. `train_adapter()` subsequently calls `llama_opt_init()`, so the optimizer graph is initialized after the adapter is attached.

## Why `opt_init` sizes the scheduler before creating the optimizer

`llama_context::opt_init()` measures a training forward graph when possible. If it
cannot measure the graph, it falls back to `max(1024, 8 * model->n_tensors())` nodes.
It then calculates:

```cpp
inflated = max(train_fwd_nodes, 1024) * 4;
sched_size = inflated * 2;
```

The graph-result capacity is `inflated`; the backend scheduler is created with
`sched_size`. The scheduler must be created before `ggml_opt_init()` because the
optimizer context retains the scheduler pointer. The current source uses this capacity
heuristic; it does not establish a general measured ratio such as “the backward graph
is 3–5 times the forward graph.”
