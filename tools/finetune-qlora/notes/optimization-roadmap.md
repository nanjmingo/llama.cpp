# QLoRA Implementation Status and Operator Gaps

This note records behavior visible in the current source snapshot. It does not assign
future priorities or claim performance improvements; those require project decisions
or measurements.

## Implemented in the current source

- `--freeze-layers N` excludes LoRA targets from the first N transformer layers.
- `--shuffle-dataset` shuffles training windows at epoch boundaries.
- Automatic tokenizer special tokens are added once per rendered sample when the
  selected input format requests them.
- Training reports an epoch loss summary.
- `MUL_MAT_ID` / `OUT_PROD_ID` provide the routed expert gradient path used by MoE
  LoRA targets.
- Quantized `OUT_PROD` has CUDA handling and backend support checks in the current
  CUDA source. This is not a statement that every backend supports the operation.
- The optimizer reuses the `ctx_compute_opt` metadata context across windows and
  resets it with `ggml_reset()`.

## Operator limitations visible in the current source

- `SSM_CONV` and `SSM_SCAN` have no backward implementation. Gradients do not
  propagate through these operators to upstream inputs such as the `ssm_in`
  projection. A downstream projection such as `ssm_out` is outside that stopped
  path and can receive gradients from later operations.
- `GGML_OP_GLU` backward handles the split `SWIGLU` and `GEGLU` cases. Other GLU
  variants are not handled. Ordinary unary GELU and GELU-quick backward paths are
  implemented; they are distinct from the `GGML_OP_GLU` cases.
- `SET_ROWS` does not propagate gradients to the source values or indices. This is
  relevant to graph paths that write K/V into a cache and later read the cache via
  views; see [the Shared-KV note](shared-kv-attention-gradient-limitation.md).
- The current QLoRA training graph optimizes the main model, not the MTP/NextN
  prediction graph.

No priority or delivery date is recorded here because it is not inferable from the
implementation.
