# QLoRA Optimization Roadmap

Development tracking notes for implemented behavior and potential follow-up work. Status
labels reflect the current source snapshot; expected effects that depend on hardware or
training data require measurement rather than inference from implementation alone.

| Priority | Optimization | Expected effect or scope | Recorded status |
|---|---|---|---|
| Done | `--freeze-layers N` — omit LoRA from the first N layers and prune unused backward paths | Reduces trainable adapter parameters and the graph work associated with them; runtime improvement depends on model, targets, and backend | Implemented |
| Done | `--shuffle-dataset` — shuffle windows each epoch | Changes training-window order; any convergence effect is dataset- and seed-dependent | Implemented |
| Done | Per-sample special-token handling — add tokenizer special tokens once per rendered sample when requested | Avoids treating prompt and completion pieces as separate samples for automatic special-token insertion | Implemented |
| Done | Per-epoch loss summary — log train loss after each epoch | Training observability | Implemented |
| Done | `MUL_MAT_ID` / `OUT_PROD_ID` backward — routed expert LoRA gradients | Enables explicit MoE expert targets | Implemented |
| Done | Quantized `OUT_PROD` — backend-supported quantized weights for backward outer products | Avoids requiring a CPU fallback when the selected backend supports the operation | Implemented |
| Done | Reuse `ctx_compute_opt` — allocate tensor metadata context once and call `ggml_reset()` across windows | Avoids repeated metadata-context destruction and allocation; no portable per-window speedup is established by the source alone | Implemented |
| Out of scope | Static training graphs — current optimizer and graph lifecycle rebuild the graph for each window | Would need a dedicated graph-lifecycle redesign | Skipped |
| Low | `SSM_CONV` / `SSM_SCAN` backward — propagate gradients through the Mamba state-space path | Would allow upstream projections such as `ssm_in` to receive gradients through the SSM path | Planned |
| Low | Additional `GGML_OP_GLU` backward variants — extend beyond the currently implemented split `SWIGLU` and `GEGLU` cases | Would cover architectures that use currently unsupported GLU variants; ordinary unary GELU and GELU-quick backward are already implemented | Planned; model coverage should be driven by an identified unsupported graph |
