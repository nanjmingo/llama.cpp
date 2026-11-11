# QLoRA Implementation Status and Operator Gaps

This note records the current implementation status and operator-level limitations in `llama-finetune-qlora`.

---

## 1. Implemented Features

| Feature | Description |
|---|---|
| **Layer Freezing** | `--freeze-layers N` excludes LoRA adapters from layers $0$ to $N-1$ and prunes backward paths. |
| **Dataset Shuffling** | `--shuffle-dataset` performs pseudo-random window shuffling at each epoch boundary. |
| **MoE LoRA Backward** | `MUL_MAT_ID` and `OUT_PROD_ID` enable backward passes through routed MoE experts. |
| **Quantized CUDA Out-Prod** | Supports quantized `OUT_PROD` operations in CUDA kernels. |
| **Special Token Wrapping** | Automatically inserts tokenizer special tokens once per rendered sample (not duplicated across turns). |
| **Memory Reuse** | Reuses the `ctx_compute_opt` metadata context across training windows via `ggml_reset()`. |
| **Epoch Loss Reporting** | Calculates and logs true token-weighted average loss across epochs. |

---

## 2. Operator Limitations and Gaps

| Operator / Component | Current Status | Impact on Training |
|---|---|---|
| **`SSM_CONV` / `SSM_SCAN`** | No backward implementation. | Gradients stop at SSM layers. Upstream projections (e.g. `ssm_in`) receive no gradients. Downstream projections (`ssm_out`) work normally. |
| **`GGML_OP_GLU`** | Backward only supports split `SWIGLU` and `GEGLU`. | Other GLU variants cannot be trained with backward autodiff. (Unary GELU / GELU-quick are supported via standard ops). |
| **`SET_ROWS`** | No backward path from cache destination to source values. | Gradients cannot flow through KV cache views back to producer projections. Affects Shared-KV models. |
| **MTP / NextN Models** | Training graph optimizes main model only. | Multi-token prediction layers are ignored during training and kept frozen in adapter checkpoints. |
