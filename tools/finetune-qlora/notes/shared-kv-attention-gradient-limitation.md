# Shared-KV Attention and Incomplete QLoRA Gradients

## Summary

QLoRA training is unsupported for Gemma 4 E2B/E4B models that use Shared KV.
The forward pass can execute, but the current graph does not preserve the complete
differentiable dependency from shared-KV consumers back to the layer that produced
the K/V tensors. Consequently, some gradients for upstream trainable adapters can
be missing. A runtime warning is diagnostic only; it does not make the gradients
complete.

## Graph path

In `llm_graph_input_attn_kv_iswa`, a layer that produces K/V uses its current
`k_cur` and `v_cur` tensors during training. A layer that reuses K/V has no such
tensors and reads the KV-cache backing tensors through `ggml_view_4d()`. The cache
is updated with `GGML_OP_SET_ROWS`, which writes source K/V values into cache rows
but has no backward path from the cache destination to those source values. The
cache read is a view of the cache tensor, not a `GET_ROWS` operation whose backward
pass could reconnect it to the `SET_ROWS` source.

Using the full-context setting (`-ub == -c`) does not resolve this limitation.
A correct fix must preserve the differentiable K/V dependency across layers. Simply
retaining and reusing producer tensors is not sufficient for layer-split multi-GPU
execution, where producer and consumer layers may be placed on different devices.
A backend-aware cross-device path has not been implemented or validated.

## Validation guidance

- Keep Gemma 4 E2B/E4B Shared-KV training documented as unsupported until the graph dependency is fixed.
- Do not treat the runtime warning as proof that the adapter is valid.
- Any proposed fix must be tested with gradients that cross shared-KV layer boundaries, including multi-GPU layer splits.
