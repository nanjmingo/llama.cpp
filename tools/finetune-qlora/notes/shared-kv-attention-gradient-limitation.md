# Shared-KV Attention Gradient Support

## Full-Context Training

The common `llm_graph_context` retains the K/V tensors produced by each source layer. When an attention layer reuses an earlier layer's K/V, the common attention builders connect it directly to those source tensors during full-context training. The source mapping comes from `llama_hparams::kv_source()`; model graph implementations do not need to add their own K/V arrays or route tensors by layer. This covers the Shared-KV paths currently defined by Gemma 3N, Gemma 4, and Mistral Large 4.

This avoids writing K/V through `GGML_OP_SET_ROWS` and reading them back through the KV cache. The resulting graph contains normal tensor dependencies from every consumer to the source K/V projections, so autodiff accumulates gradients from all consumers at the source. The backend scheduler handles dependencies between tensors assigned to different devices; no host-side tensor copy or device-specific routing is needed.

This path is enabled when `cparams.training` is true, which requires the full sequence to fit in the training microbatch. QLoRA enforces this by requiring `-ub` to equal `-c`. Inference continues to use the existing KV-cache path.

## Remaining Limitation

`GGML_OP_SET_ROWS` still has no backward implementation. Training configurations that do not fit the full sequence in one microbatch continue to use the cache-backed path, so gradients from consumer layers do not propagate through the cache to the original K/V projection adapters. The runtime warning is retained for these configurations.
