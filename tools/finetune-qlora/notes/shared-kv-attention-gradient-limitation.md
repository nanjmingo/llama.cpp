# Shared-KV Attention and Incomplete QLoRA Gradients

## Summary

QLoRA cannot compute complete gradients for trainable LoRA adapters attached to shared
K/V source projections when later attention layers reuse those projections. The current
warning logic covers prefix Shared-KV configurations in Gemma 3N and Gemma 4. Gemma 3N
sets `n_layer_kv_from_start` to 20; Gemma 4 derives the start layer from its shared-KV
configuration.

The forward pass can execute, but the current graph does not preserve the complete
differentiable dependency from shared-KV consumers back to the layer that produced
the K/V tensors. Consequently, LoRA adapters on those source projections receive
incomplete gradients from consumer layers. The runtime warning is emitted only when
a loaded adapter has a trainable A or B tensor matching an affected K/V source
projection. The warning is diagnostic only; it does not make the gradient complete.
This limitation should not be generalized to every LoRA target in the model without
checking the graph path for that target.

## Graph path

In `llm_graph_input_attn_kv_iswa`, a layer that produces K/V uses its current
`k_cur` and `v_cur` tensors during training. A layer that reuses K/V has no such
tensors and reads the KV-cache backing tensors through `ggml_view_4d()`. The cache
is updated with `GGML_OP_SET_ROWS`, which writes source K/V values into cache rows
but has no backward path from the cache destination to those source values. The cache
read is a view of the cache tensor, not a `GET_ROWS` operation whose backward pass
could reconnect it to the `SET_ROWS` source.

The warning helper in `src/llama-context.cpp` explicitly recognizes `LLM_ARCH_GEMMA3N`
and `LLM_ARCH_GEMMA4`, finds K/V producers with downstream shared-KV consumers, and
warns when a trainable adapter matches one of those producer weights.

Using the full-context setting (`-ub == -c`) does not resolve this limitation. A
correct fix must preserve the differentiable K/V dependency across layers. Simply
retaining and reusing producer tensors is not sufficient for layer-split multi-GPU
execution, where producer and consumer layers may be placed on different devices.
A backend-aware cross-device path has not been implemented or validated.

## Validation guidance

- Keep trainable LoRA on shared K/V source projections documented as unsupported for Gemma 3N and Gemma 4 Shared-KV configurations until the graph dependency is fixed.
- Treat the runtime warning as a diagnostic for the affected source projections, not as proof that the adapter is valid or that every other target is affected.
- Test gradients that cross shared-KV layer boundaries, including multi-GPU layer splits, before claiming support.
