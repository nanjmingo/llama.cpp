# Shared-KV Attention and Incomplete QLoRA Gradients

## Summary

The current graph does not preserve a differentiable dependency from Shared-KV
consumer layers through the KV cache to the K/V source projections. As a result,
trainable LoRA adapters on affected source projections receive incomplete gradients
from consumer layers. The forward pass can still execute; successful execution does
not establish that these adapter gradients are complete.

The warning helper in `src/llama-context.cpp` explicitly handles prefix Shared-KV
configurations in Gemma 3N and Gemma 4. It warns when a loaded adapter has a trainable
A or B tensor matching an affected K/V source projection. This is a targeted warning,
not a guarantee that every Shared-KV architecture or every LoRA target has been
identified. Do not generalize the limitation to unrelated targets without examining
their graph paths.

## Graph path

In `llm_graph_input_attn_kv_iswa`, a layer that produces K/V uses its current
`k_cur` and `v_cur` tensors during training. A layer that reuses K/V reads the KV-cache
backing tensors through `ggml_view_4d()`. The cache is updated with `GGML_OP_SET_ROWS`,
which has no backward path from the cache destination to the source values. The later
cache view therefore does not reconnect a consumer's gradient to the original K/V
producer.

Using full-context training (`-ub == -c`) does not remove this graph limitation.
The warning is diagnostic only; it does not modify the graph or repair gradients.

## Scope of unverified behavior

The source warning covers the Gemma 3N and Gemma 4 prefix Shared-KV cases described
above. Other model configurations and multi-GPU layer-split arrangements require
separate graph inspection and gradient tests before their support can be claimed.
The source code does not provide a validated general fix for those cases.
