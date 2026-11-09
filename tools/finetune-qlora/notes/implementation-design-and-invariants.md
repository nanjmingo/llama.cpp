# QLoRA Implementation Design and Internal Invariants

This note records implementation details that are useful when modifying or debugging
the QLoRA integration. It is not a user guide; user-facing setup and runtime behavior
are documented in the parent `README.md`.

## Dataset-loading implementation

The JSONL reader processes up to 128 records per batch. This is deliberately not
one blocking `parallel_for()` call per record: the thread pool executes a single
task synchronously, so one-record batches would usually remove parallelism from
JSON parsing and tokenization while increasing synchronization overhead. Reducing
memory further would require a bounded asynchronous pipeline rather than simply
reducing the batch size to one.

The QLoRA-specific command-line options are parsed in `args.cpp`, rather than
registered in `common/arg.cpp`. JSONL parsing and tokenization use the shared
`-t` / `--threads` setting and can run in parallel; chat-template rendering
remains sequential.

## Modified llama.cpp files

| File | Change |
|---|---|
| `ggml/src/ggml.c` | Backward graph fixes: `GET_ROWS` 3D, `SET_ROWS`, `MUL_MAT_ID`, `SSM_SCAN/CONV`, `FLASH_ATTN_EXT` now propagates gradients through Q/K/V and optional attention sinks; inplace-op assert → warn+skip |
| `src/llama-context.cpp` | Preserves exact context capacity for full-context training; `opt_init`: scheduler and graph sizing, shared-KV LoRA gradient warning; `opt_epoch_iter`: metadata-context reuse and accumulation-group selection |
| `src/llama-adapter.cpp` | Repack-buft fallback for LoRA tensors: tries device-native buft before CPU |
| `common/common.h`, `common/arg.cpp` | Removed the old QLoRA-only parameter fields and CLI registrations; QLoRA options are now local to the example |
| `tools/finetune-qlora/args.cpp` | Parses QLoRA-only CLI flags locally; no QLoRA option registration is needed in `common/arg.cpp` |
| `include/llama.h` | Exposes `llama_opt_dataset_shuffle()` so callers control per-epoch shuffling; `llama_opt_epoch()` no longer owns shuffle policy; added `llama_opt_epoch_range()` with an explicit evaluation-range end |
| `ggml/src/ggml-cuda/out-prod.cu` | CUDA `OUT_PROD` with quantized src0 (dequantize on GPU + cuBLAS); `OUT_PROD_ID` for MoE backward |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_op` for quantized `OUT_PROD` and `OUT_PROD_ID`; CPU-resident ids fix in `mul_mat_id` |

## Key invariants

- `params.use_mmap = false` — forced; mmap'd tensors cannot have data written back.
- FlashAttention follows the common `--flash-attn` setting (default `AUTO`); CPU and CUDA backward implementations are available.
- `params.warmup = false` — warmup runs inference with PARAM tensors and can segfault.
- QLoRA defaults `params.cache_type_k` and `params.cache_type_v` to F32 before common argument parsing; `--cache-type-k` and `--cache-type-v` override these defaults.
- LoRA A/B tensors are marked `PARAM` via `ggml_set_param` on the tensors loaded by `llama_adapter_lora_init`, not on the pre-init scratch tensors in `lt.buf`.
- The adapter GGUF is pre-saved and loaded via `params.lora_adapters` before `common_init_from_params` so that `sched_reserve` includes LoRA graph nodes in its sizing.

## Why `opt_init` inflates the scheduler hash set

`ggml_opt_init` captures `sched.get()` at construction time. The backward graph
(`gb_grad`, `gb_opt`) is roughly 3–5 times larger than the forward graph in node
count. If the scheduler hash set is sized only for the forward graph,
`ggml_backend_sched_alloc_graph` on the backward graph can overflow it. The current
implementation recreates `sched` with `inflated = fwd_nodes * 4` slots before calling
`ggml_opt_init`.
