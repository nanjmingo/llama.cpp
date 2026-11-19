# QLoRA Developer Notes

Technical notes and design invariants for contributors and AI agents working on `llama-finetune-qlora`.

| Document | Topic | Key Takeaway |
|---|---|---|
| [Implementation Design and Invariants](implementation-design-and-invariants.md) | Architecture, GGML changes, execution lifecycle | Explains how QLoRA integrates with GGML/llama runtime, differences from `examples/training`, and key runtime invariants. |
| [Context-Length Rounding and Training Windows](context-length-rounding-and-training-windows.md) | Window packing, context sizing, loss normalization | Details why context size is unpadded in full-context mode, how dataset windows are constructed, and how loss is normalized. |
| [Shared-KV Attention Gradient Support](shared-kv-attention-gradient-limitation.md) | Gemma 4 full-context Shared-KV backpropagation | Documents direct K/V tensor routing during Gemma 4 full-context training, cross-device scheduling, and the remaining cache-backed limitation. |
| [Chat-Template Date/Time Thread Safety](chat-template-datetime-thread-safety.md) | Concurrency constraint in dataset loading | Explains why template rendering must remain serial due to POSIX `std::localtime()` shared storage in Jinja context rendering. |
| [Optimization Status and Operator Gaps](optimization-roadmap.md) | Operator support and backward coverage | Summarizes currently implemented features vs missing backward operators (SSM, specific GLU variants, MTP). |
