# Shared-KV Attention and Incomplete QLoRA Gradients

## Summary

In models with Shared-KV attention (such as Gemma 3N and Gemma 4 prefix layers), consumer layers reuse key/value states computed by an earlier layer via the KV cache.

Because `GGML_OP_SET_ROWS` lacks a backward implementation, **gradients from consumer layers do not propagate back to the original K/V projection adapters**.

---

## Technical Cause: Disconnected Backward Graph

```text
Producer Layer:
  [Tokens] --> [attn_k / attn_v] --> k_cur, v_cur
                                           |
                                           v (Cached via GGML_OP_SET_ROWS)
                                     [ KV Cache ]   <-- NO BACKWARD PATH
                                           |
Consumer Layer:                            v (Read via ggml_view_4d)
  [Attention] <--------------------- k_view, v_view
```

1. **Forward Path:** The producer layer computes `k_cur` and `v_cur`, which are written into the KV cache using `GGML_OP_SET_ROWS`. The consumer layer reads those states through `ggml_view_4d()`.
2. **Backward Path:** `GGML_OP_SET_ROWS` has no backward operator. The autodiff graph terminates at the KV cache buffer.
3. **Consequence:** Gradients from the consumer layer cannot reach the producer's `attn_k` and `attn_v` adapters. Those adapters only receive gradients from the producer layer's own self-attention.

> [!NOTE]
> Setting `-ub == -c` (full-context training) does **not** bypass this issue. Shared-KV models still allocate and route through KV cache buffers.

---

## Current Mitigation: Runtime Warning

In `src/llama-context.cpp`, the trainer inspects model architecture and active LoRA targets.

If a trainable adapter targets `attn_k` or `attn_v` on a known Shared-KV model (e.g. Gemma 3N or Gemma 4):
- The trainer logs a warning explaining that gradients will be incomplete.
- Training is allowed to continue, but the resulting adapter weights may be sub-optimal on affected layers.

---

## Recommendations for Contributors

- **Target selection:** When fine-tuning Shared-KV models, consider excluding `attn_k` and `attn_v` from `--lora-targets` (e.g. train `attn_q,attn_output,ffn_gate,ffn_up,ffn_down`).
- **Graph improvements:** To properly support end-to-end autodiff through Shared-KV, `GGML_OP_SET_ROWS` backward or an explicit direct tensor-routing graph (without intermediate cache mutation) must be implemented.
