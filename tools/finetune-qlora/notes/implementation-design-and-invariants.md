# QLoRA Implementation Design and Internal Invariants

This note documents architecture choices, lifecycle invariants, and code modifications for developers and AI agents working on `llama-finetune-qlora`.

---

## 1. Differences from `examples/training`

While QLoRA uses the shared `ggml_opt_*` optimizer engine, its architecture differs substantially from `examples/training/finetune.cpp`:

| Component | `examples/training/finetune.cpp` | `tools/finetune-qlora` |
|---|---|---|
| **Weights Updated** | Full model fine-tuning (updates base model weights). | LoRA fine-tuning (base weights are frozen; updates only adapter A/B tensors). |
| **Output Format** | Full base model file (`llama_model_save_to_file`). | GGUF LoRA adapter (`adapter.gguf`) and training checkpoints. |
| **KV Cache Precision** | Hardcoded to F32. | Defaults to F32, configurable via `--cache-type-k` and `--cache-type-v`. |
| **Flash Attention** | Disabled during training. | Fully supported in training via `--flash-attn` (CPU and CUDA backward). |
| **Dataset Loading** | Raw prompt text via `common_opt_dataset_init()`. | Multi-format JSONL reader, Jinja chat templating, and window packer. |
| **Validation** | Splits dataset by ratio (`--val-split`). | Evaluates remaining windows after `--max-steps` via `--val-windows`. |

> [!NOTE]
> QLoRA calls `common_params_parse(..., LLAMA_EXAMPLE_FINETUNE)` to reuse common CLI flags. Flags like `--prompt`, `--file`, and `--val-split` may be parsed, but QLoRA ignores them.

---

## 2. Dataset Pipeline Architecture

Data processing runs in three stages inside `dataset.cpp`:

```text
JSONL Lines (up to 128 lines per batch)
                 |
                 v
Stage 1: Parse JSON records         --> [Parallel: dataset_thread_pool::parallel_for()]
                 |
                 v
Stage 2: Render chat templates      --> [SERIAL / Single-Threaded] (thread-safety constraint)
                 |
                 v
Stage 3: Tokenize rendered samples  --> [Parallel: parallel_for()]
                 |
                 v
Stage 4: Build training windows     --> [Window Packer: BOS padding & loss masking]
```

- **Thread safety:** Template rendering is kept serial because POSIX `std::localtime()` uses shared memory.
- **Parallel tokenization:** Even if a batch has only one record, tokenization runs in parallel if that record produces multiple turns/samples.

---

## 3. Key Runtime Invariants

Keep these invariants in mind when editing the initialization sequence:

1. **`params.load_mode = LLAMA_LOAD_MODE_NONE`**:
   Passed to `common_init_from_params()`, which maps to `use_mmap = false`.
2. **`params.warmup = false`**:
   Inference warmup is disabled during initialization.
3. **`params.no_alloc_kv_cache = true`**:
   Tells the runtime not to allocate KV cache buffers for layers that compute self-attention directly within the window. Shared-KV layers still allocate the cache buffers they require.
4. **Parameter Tagging (`PARAM`)**:
   Trainable adapter tensors are marked `PARAM` using `ggml_set_param()` on tensors loaded by `llama_adapter_lora_init()`, **not** on temporary scratch buffers.
5. **Initialization Order**:
   The execution lifecycle must strictly follow this order:
   ```text
   1. Create base model and context: common_init_from_params()
   2. Create temporary adapter GGUF
   3. Load adapter: llama_adapter_lora_init()
   4. Attach adapter: common_set_adapter_lora()
   5. Mark adapter A/B tensors as PARAM: ggml_set_param()
   6. Initialize optimizer graph: llama_opt_init()
   ```

---

## 4. Backend Scheduler Capacity Heuristic

`llama_context::opt_init()` calculates the backend scheduler size before creating the optimizer graph:

```cpp
// 1. Measure or estimate forward graph node count:
int64_t train_fwd_nodes = measured ? measured_nodes : std::max<int64_t>(1024, 8 * model->n_tensors());

// 2. Compute capacity for backward and optimizer graph:
int64_t inflated   = std::max<int64_t>(train_fwd_nodes, 1024) * 4;
int64_t sched_size = inflated * 2;
```

The backend scheduler must be instantiated with `sched_size` before `ggml_opt_init()` because the optimizer context holds a pointer to the scheduler.

---

## 5. Overview of Core llama.cpp Modifications

| File | Purpose / Change |
|---|---|
| `ggml/src/ggml.c` | Backward support for 3D `GET_ROWS_BACK`, `MUL_MAT_ID`, and `FLASH_ATTN_EXT` (with attention sinks). |
| `src/llama-context.cpp` | Preserves unrounded context capacity for full-context training; sizes scheduler before `ggml_opt_init`; warns on Shared-KV source targets; manages gradient accumulation groups. |
| `src/llama-adapter.cpp` | Adds repack-buffer fallback for LoRA tensors (tries device-native buffer before CPU). |
| `common/common.h`, `common/common.cpp` | Adds `no_alloc_kv_cache` to context params. |
| `tools/finetune-qlora/args.cpp` | Local CLI parser for QLoRA-specific parameters. |
| `include/llama.h` | Exposes `llama_opt_dataset_shuffle()` and `llama_opt_epoch_range()`. |
| `ggml/src/ggml-cuda/` | Quantized `OUT_PROD` and `OUT_PROD_ID` kernels for MoE backward support on CUDA. |
