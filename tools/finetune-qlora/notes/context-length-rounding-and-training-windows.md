# Context-Length Rounding and Training Windows

## Summary

This note explains how `llama-finetune-qlora` handles context dimensions, dataset window packing, and loss normalization.

---

## 1. Context Capacity in the Runtime

By default, `src/llama-context.cpp` rounds context capacity (`cparams.n_ctx`) up to a multiple of 256.

### The Full-Context Training Exception

QLoRA bypasses this 256-token rounding when all of the following conditions are met:

| Condition | Setting in QLoRA |
|---|---|
| No KV cache allocation | `cparams.no_alloc_kv_cache = true` |
| Microbatch equals context | `n_ubatch == n_ctx` |
| Batch size is large enough | `n_batch >= n_ctx` |
| No secondary context | `cparams.n_ctx_seq == 1` (or unified KV cache) |

When these conditions hold:
- **Requested context size is exact:** A context of `900` creates a 900-token graph, not 1024.
- **KV data buffers are omitted:** For standard models, forward passes read the input window directly; KV buffers are not allocated. (Shared-KV models still allocate their required cache).

> [!TIP]
> Always query effective runtime values using `llama_n_ctx()`, `llama_n_batch()`, and `llama_n_ubatch()` rather than assuming they match the raw CLI parameters.

---

## 2. Dataset Window Construction and Padding

Context capacity is an internal buffer size. Dataset windowing is a separate process in `dataset.cpp`:

| Sample Length | Packing Behavior |
|---|---|
| **Shorter than `n_ctx`** | Padded to exactly `n_ctx` tokens using the BOS token (or token ID 0). Padding positions receive target label `-1` (loss masked). |
| **Longer than `n_ctx`** | Split into multiple full-sized windows. Any trailing fragment that does not fill a full window is discarded. |
| **Overlapping tokens** | Targets already trained in an earlier window are masked with label `-1` to avoid double-counting. |
| **Plain `text` format** | The first half of the first window is masked as prompt context (`weight = 0`). Subsequent tokens and windows are prediction targets. |
| **Chat / Prompt formats** | Prompt and non-assistant tokens are masked (`weight = 0`). Assistant / completion tokens are trained (`weight = 1`). No half-window mask is applied. |

---

## 3. Loss Normalization with Masked Positions

Do not confuse raw GGML loss calculation with the optimizer's effective gradient scaling:

```text
Raw GGML Operator:
  loss = -sum(labels * log_softmax(logits)) / total_rows
  (total_rows includes masked positions where label == -1)

QLoRA Optimizer Normalization:
  normalized_grad = accumulated_grad / sum(supervision_weights)
  (Normalizes only by active target tokens across the accumulation group)
```

1. **Raw GGML Cross-Entropy:** `ggml_cross_entropy_loss` divides by all rows, including masked rows.
2. **QLoRA Optimizer Normalizer:** `llama_context::opt_epoch()` computes the sum of active target weights across all windows in the gradient accumulation group, then calls `ggml_opt_set_loss_normalizer()`.
3. **Reported Loss:** Progress logs and epoch summaries report the mean loss over active supervised target tokens, not over all context tokens.

---

## 4. Invariants for Tests

When writing tests or finite-difference gradient checks:
- Verify effective context values using `llama_n_ctx()`.
- Ensure dataset samples match the context length: for a 256-token context, provide 257 tokens (256 input tokens + 1 target token).
- Distinguish between per-token cross-entropy and group-normalized loss when comparing numerical values against independent references.
