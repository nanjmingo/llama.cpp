# QLoRA Optimization Roadmap

Development tracking notes for potential and completed optimizations. Status labels
reflect the state recorded when this note was written and should be revalidated
against the current source before planning work.

| Priority | Optimization | Expected gain | Recorded status |
|---|---|---|---|
| Done | `--freeze-layers N` — no LoRA on first N layers; backward auto-pruned | Proportional to N/total | Implemented |
| Done | `--shuffle-dataset` — shuffle windows each epoch | Better convergence | Implemented |
| Done | BOS separators — insert BOS between concatenated samples | Correct cross-sample boundaries | Implemented |
| Done | Per-epoch loss summary — log train loss after each epoch | Observability | Implemented |
| Done | `MUL_MAT_ID` / `OUT_PROD_ID` backward — routed expert LoRA gradients | Enables explicit MoE expert targets | Implemented |
| Done | Quantized `OUT_PROD` — dequantize on GPU + cuBLAS for backward matmul | Full GPU training (no CPU fallback) | Implemented |
| Done | Reuse `ctx_compute_opt` — allocate tensor metadata context once, `ggml_reset()` across windows | Eliminate approximately 0.5 s/window overhead | Implemented |
| Out of scope | Static training graphs — current optimizer and graph lifecycle rebuild the graph for each window | Would need a dedicated graph-lifecycle redesign | Skipped |
| Low | `SSM_SCAN/CONV` backward — enable LoRA on Mamba SSM layers | Unlocks NemotronH SSM layers | Planned |
| Low | GELU backward — implement `ggml_gelu_back` kernel (UNARY + GLU) | Support GPT-2/Phi-style models | Planned; requires new CUDA/CPU kernels |
