# Shared-KV Attention Gradient Support

## Gemma 4 Full-Context Training

Gemma 4 consumer layers reuse key/value states computed by an earlier layer. During full-context training, the graph retains each KV-producing layer's normalized and position-encoded K/V tensors and connects consumer attention directly to the source tensors.

This avoids writing K/V through `GGML_OP_SET_ROWS` and reading them back through the KV cache. The resulting graph contains normal tensor dependencies from every consumer to the source K/V projections, so autodiff accumulates gradients from all consumers at the source. The backend scheduler handles dependencies between tensors assigned to different devices; no host-side tensor copy or device-specific routing is needed.

This path is enabled only when `cparams.training` is true, which requires the full sequence to fit in the training microbatch. QLoRA enforces this by requiring `-ub` to equal `-c`. Inference continues to use the existing KV-cache path.

## Remaining Limitation

`GGML_OP_SET_ROWS` still has no backward implementation. Shared-KV models that use the cache-backed training path therefore do not propagate consumer-layer gradients to the original K/V projection adapters. This remains the case for Gemma 3N and for Gemma 4 training where the full sequence does not fit in one microbatch.

The runtime warning is retained for these unsupported configurations. It is suppressed for Gemma 4 full-context training, where the direct tensor-routing path is used.
