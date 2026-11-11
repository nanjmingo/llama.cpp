# QLoRA Developer Notes

Technical notes and design invariants for contributors and AI agents working on `llama-finetune-qlora`.

| Document | Topic | Key Takeaway |
|---|---|---|
| [Implementation Design and Invariants](implementation-design-and-invariants.md) | Architecture, GGML changes, execution lifecycle | Explains how QLoRA integrates with GGML/llama runtime, differences from `examples/training`, and key runtime invariants. |
| [Context-Length Rounding and Training Windows](context-length-rounding-and-training-windows.md) | Window packing, context sizing, loss normalization | Details why context size is unpadded in full-context mode, how dataset windows are constructed, and how loss is normalized. |
| [Shared-KV Attention Limitations](shared-kv-attention-gradient-limitation.md) | Graph backward limitation in Shared-KV | Explains why gradients do not propagate through `SET_ROWS` in Shared-KV models (e.g. Gemma 3N / 4 prefix), causing incomplete K/V gradients. |
| [Chat-Template Date/Time Thread Safety](chat-template-datetime-thread-safety.md) | Concurrency constraint in dataset loading | Explains why template rendering must remain serial due to POSIX `std::localtime()` shared storage in Jinja context rendering. |
| [Optimization Status and Operator Gaps](optimization-roadmap.md) | Operator support and backward coverage | Summarizes currently implemented features vs missing backward operators (SSM, specific GLU variants, MTP). |
