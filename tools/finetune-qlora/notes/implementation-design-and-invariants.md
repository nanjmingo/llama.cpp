# QLoRA Implementation Design and Internal Invariants

This note records source-level details that are useful when modifying or debugging
QLoRA. User-facing behavior is documented in the parent `README.md`.

## Relationship to `examples/training`

QLoRA reuses llama.cpp's `llama_opt_*` / `ggml_opt_*` optimizer infrastructure, but
it does not use the data-loading or model-saving path in
`examples/training/finetune.cpp`. QLoRA reads JSONL records through its own dataset
reader, template renderer, tokenizer, and window builder. It does not build an
optimizer dataset from `params.prompt` with `common_opt_dataset_init()`, use the
example's `val_split` partition, or save a fully fine-tuned model with
`llama_model_save_to_file()`. It marks selected adapter A/B tensors as optimizer
parameters and writes adapter GGUF files and QLoRA checkpoints; base-model weights
remain frozen.

Do not carry over two properties of `examples/training`: `finetune.cpp` forces
K/V cache types to F32, and its README says Flash Attention is disabled during
training. QLoRA initializes K/V cache defaults to F32 but allows `--cache-type-k`
and `--cache-type-v` to override them. It also honors the common `--flash-attn`
setting, including its `auto`, `on`, and `off` modes; CPU and CUDA backward
implementations are present.

`common_params_parse()` is called with `LLAMA_EXAMPLE_FINETUNE` to reuse common
llama.cpp options. Consequently, example-oriented options such as `--prompt`,
`--file`, and `--val-split` can still be parsed, but QLoRA's training path does not
consume them. Use `--train-file` for dataset input and `--val-windows` to control
validation windows. The presence of a parsed common option does not make its
`examples/training` behavior part of QLoRA.

## Dataset-loading implementation

The JSONL reader accumulates up to 128 non-empty lines per batch and parses the
records with `dataset_thread_pool::parallel_for()`. A one-task call, or a call
without workers, runs synchronously on the calling thread.

In `append_tokenized_batch()`, records are rendered sequentially. The resulting
rendered samples are then tokenized with `parallel_for()`. Therefore, a batch with
one input record does not parallelize record parsing, but tokenization can still
run in parallel when that record produces multiple rendered samples (for example,
multiple assistant turns). It is incorrect to state that a one-record batch always
removes tokenization parallelism.

The QLoRA-specific command-line options are parsed in `args.cpp`; shared llama.cpp
options continue to use `common/arg.cpp`.

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

- `params.load_mode = LLAMA_LOAD_MODE_NONE` is set before `common_init_from_params()`. The model loader maps this mode to `use_mmap = false`. The code does not explain why QLoRA requires this setting. Do not infer that frozen base tensors are written back during training.
- `params.warmup = false` is set before `common_init_from_params()`. The code does not document why warmup is disabled. In particular, the adapter is loaded after `common_init_from_params()`, so the source does not support the claim that initialization warmup runs inference with uninitialized LoRA PARAM tensors. The reason for disabling warmup remains undocumented in the source.
- `params.no_alloc_kv_cache = true` is set for training. The runtime preserves the requested context capacity when its full-context-training conditions are met. It skips KV data-buffer allocation only when the model has no shared KV; Shared-KV models still require their cache.
- FlashAttention follows the common `--flash-attn` setting (default `auto`); CPU and CUDA backward implementations are present.
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
heuristic; it does not establish a general measured ratio between the backward and
forward graph sizes.
