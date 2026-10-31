# llama.cpp — Native QLoRA Training

Native QLoRA SFT training pipeline for quantized GGUF models.

The base model weights remain **frozen** (quantized tensors are skipped by `llama_set_param` because they are not `GGML_TYPE_F32`). LoRA A/B tensors are stored in the selected type (`F32`, `F16`, or `BF16`) and updated with F32 gradients. The saved adapter GGUF is directly compatible with the existing `llama_adapter_lora_init` loader and `llama-export-lora` merge tool.

**Status:** QLoRA SFT is implemented and functional. Training speed is currently limited by full backprop through quantized weights — see [Known Limitations](#known-limitations).

---

## Build

Run these commands from the repository root.

```bash
# First time (CUDA build):
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build -j $(nproc)

# Incremental rebuild (after code changes):
cmake --build build --target llama-finetune-qlora -j $(nproc)
```

Resumable checkpoints store the adapter, optimizer state, dataset permutation,
and training progress separately. AdamW moments and LoRA A/B tensors can use F32,
F16, or BF16 storage; the default is F32. CPU and CUDA update paths support these
storage types. Non-F32 LoRA storage requires AdamW; SGD remains F32-only.

### Tests

For test setup, the aggregate build target, test execution, fixtures, and model-backed template checks, see [the QLoRA test guide](tests/README.md).

---

## QLoRA SFT (`llama-finetune-qlora`)

Trains LoRA adapters on a quantized GGUF model.

### Recommended command (1.7B model; memory-dependent full-context run)

```bash
./build/bin/llama-finetune-qlora \
  --model ~/qwen3-1.7b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --output-dir ./output \
  --lora-rank 16 --lora-alpha 16 \
  -c 4096 -b 4096 -ub 4096 --gradient-accumulation-steps 4 \
  --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 -lr-min 1e-6 \
  --save-checkpoint-steps 100 --save-checkpoint-limit 3 \
  --save-snapshot-steps 100 --save-snapshot-limit 10 \
  --epochs 3 --seed 42
```

### Recommended command (15B model with partial offload; memory-dependent)

```bash
./build/bin/llama-finetune-qlora \
  --model ~/nemotron-15b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --output-dir ./nemotron-output \
  --lora-rank 16 --lora-alpha 16 \
  -ngl 13 -c 4096 -b 4096 -ub 4096 --gradient-accumulation-steps 4 \
  --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 -lr-min 1e-6 \
  --save-checkpoint-steps 100 --save-checkpoint-limit 3 \
  --save-snapshot-steps 200 --save-snapshot-limit 10 \
  --epochs 3 --seed 42
```

### All params

#### Common params

These are shared llama.cpp model and execution parameters.

| Parameter | Default | Description |
|---|---|---|
| `-m` / `--model` | *(required)* | Path to the quantized GGUF base model |
| `-t` / `--threads` | common default | CPU thread count; also used for JSONL parsing and tokenization |
| `-c` / `--ctx-size` | `512` | Training context window (tokens) |
| `-b` / `--batch-size` | `2048` | Must be at least `-c`; causal models clamp the effective batch size to `-c` |
| `-ub` / `--ubatch-size` | `512` | Must equal `-c`; with causal attention, the effective sizes satisfy `n_ubatch == n_batch == n_ctx` |
| `-ngl` / `--gpu-layers` / `--n-gpu-layers` | `999` | Number of GPU layers to offload |
| `-ctk` / `--cache-type-k` | `f32` | K cache storage type; accepts the KV cache types supported by llama.cpp, including `q8_0` |
| `-ctv` / `--cache-type-v` | `f32` | V cache storage type; accepts the KV cache types supported by llama.cpp, including `q8_0` |

#### Input params

| Parameter | Default | Description |
|---|---|---|
| `--train-file` | *(required)* | Training dataset path; may be repeated to load multiple files |
| `--fail-on-invalid-records` | off | Stop on the first invalid record instead of skipping invalid records |

#### LoRA params

| Parameter | Default | Description |
|---|---|---|
| `--lora-rank` | `16` | LoRA rank r |
| `--lora-alpha` | `0` (= rank) | LoRA alpha; effective scale = alpha/rank |
| `--lora-type` | `f32` | Storage type of trainable LoRA A/B tensors: `f32`, `f16`, or `bf16` |
| `--lora-targets` | `attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down` | Comma-separated internal tensor name substrings |
| `--freeze-layers` | `0` | Skip LoRA on the first N transformer layers (blk.0..N-1); backward is pruned automatically |

#### Training params

| Parameter | Default | Description |
|---|---|---|
| `--optimizer-state-type` | `f32` | AdamW first/second moment storage: `f32`, `f16`, or `bf16` |
| `--gradient-accumulation-steps` | `1` | Accumulate gradients from up to N complete windows before updating LoRA weights |
| `--max-steps` | *(disabled)* | Stop after N optimizer updates, repeating the dataset as needed; overrides `--epochs` |
| `--val-windows` | all remaining windows | Maximum number of windows to evaluate after the training range; `0` disables validation |
| `--shuffle-dataset` | off | Shuffle dataset windows at the start of each epoch |
| `-epochs` / `--epochs` | `3` | Training epochs |
| `-lr` / `--learning-rate` | `1e-5` | AdamW learning rate |
| `-wd` / `--weight-decay` | `0` | AdamW or SGD weight decay applied to trainable LoRA tensors; `0` disables weight decay |
| `-opt` / `--optimizer` | `adamw` | Optimizer used to update LoRA tensors: `adamw` or `sgd` |
| `--lr-scheduler` | `constant` | Learning-rate schedule: `constant` or `cosine` |
| `--warmup-steps` | `0` | Linear warmup over N optimizer updates |
| `--warmup-init-ratio` | `0.1` | Initial warmup LR as a fraction of peak LR; must be in `(0, 1]` |
| `-lr-min` / `--learning-rate-min` | `-1` | Cosine floor; values below 0 use 0 |
| `-decay-epochs` / `--learning-rate-decay-epochs` | `-1` | For the constant scheduler, decay toward `--learning-rate-min` over N epochs; takes effect when the minimum is positive and below the base learning rate; ignored by the cosine scheduler |
| `-s` / `--seed` | `42` | Random seed for LoRA initialization |

#### Output params

| Parameter | Default | Description |
|---|---|---|
| `--output-dir` | *(required)* | Output directory for checkpoint directories, adapter snapshots, and the final `adapter.gguf` |

#### Checkpoint params

| Parameter | Default | Description |
|---|---|---|
| `--save-checkpoint-steps` | `0` | Save checkpoints every N optimizer steps; `0` saves only the final checkpoint when checkpoint saving is enabled |
| `--save-checkpoint-limit` | `0` | Enable checkpoint saving and retain at most N checkpoints; `0` disables checkpoint saving |
| `--save-snapshot-steps` | `0` | Save adapter-only snapshots every N optimizer steps; `0` disables periodic snapshots |
| `--save-snapshot-limit` | `0` | Enable snapshot saving and retain at most N snapshots; `0` disables snapshot saving |
| `--resume-from` | *(disabled)* | Resume from a checkpoint directory or an output directory containing checkpoints; the checkpoint supplies the adapter |

Use `--max-steps N` to keep profiling or training runs to the same number of optimizer updates even when
the datasets contain different numbers of windows. The dataset is repeated as needed, and `--epochs` is
ignored when `--max-steps` is set. `N` counts optimizer updates, not dataset windows; with gradient
accumulation enabled, one update can consume multiple windows. The final pass stops on an optimizer-update
boundary and may therefore end partway through an epoch. When this leaves windows after the training range,
those windows are evaluated by default. Use `--val-windows 0` to skip validation entirely during profiling,
or `--val-windows N` to evaluate at most N of the remaining windows.

JSONL parsing and tokenization use the shared `-t` / `--threads` setting. Chat-template rendering remains sequential.

QLoRA requires the effective batch sizes to satisfy `n_ubatch == n_batch == n_ctx`. For causal
models, llama.cpp clamps the effective batch size to the context size, so at the command line set
`-ub == -c` and `-b >= -c` (a larger `-b` is accepted but clamped to `-c`). Each dataset window
is processed as one complete forward/backward graph so gradients can flow through the current
window's K/V projections without using the KV cache. Splitting a window into smaller micro-batches
is rejected. This can require substantially more activation memory; reduce `-c` or offload more
layers if the full window does not fit.

`--cache-type-k` and `--cache-type-v` select the K and V cache storage types independently.
Both default to F32 for QLoRA. They accept the KV cache types supported by the common
llama.cpp argument parser, such as `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`,
`q5_0`, and `q5_1`. Backend and model constraints still apply; in particular, quantized
V cache requires Flash Attention, which llama.cpp enables automatically when its setting
is `auto`. For architectures that require matching K/V cache types, specify the same type
for both options. For example, use `--cache-type-k q8_0 --cache-type-v q8_0` to select Q8_0
for both caches.

For full-context QLoRA training, ordinary attention layers use the current window's K/V
directly instead of reading the KV cache. Their K/V data buffers are therefore not allocated.
The cache metadata is retained, but K/V data buffers and cache-quantization rotation matrices
are not allocated. Architectures that read cached K/V during training, including Shared KV
models, keep their K/V data buffers and any required rotation matrices allocated. The cache type
options affect those retained buffers, but have no effect on ordinary full-context training layers.

`--lora-type` selects the stored type for trainable adapter tensors. F32 is the default
and reference setting; F16 and BF16 reduce adapter storage while keeping F32 gradient
accumulation.

`--optimizer-state-type` independently selects AdamW's first and second moment types.
F16 and BF16 reduce optimizer-state memory compared with F32, but alter moment rounding
and may change convergence. This option is only used by AdamW; SGD requires its default
`f32` value.

`--gradient-accumulation-steps N` accumulates gradients over up to N independent dataset windows before
updating the LoRA weights. Activations are released between windows, so increasing N does not retain all
windows' activations simultaneously. A shorter final group at each epoch is also applied as one optimizer
step. The scheduler advances once per optimizer update, and `--warmup-steps` and cosine decay are measured
in optimizer updates rather than windows. During warmup, step 0 starts at
`peak_lr * warmup_init_ratio`; each subsequent logical step linearly interpolates toward peak LR using
`progress = step / warmup_steps`. The ratio is ignored when warmup is disabled.

All save intervals use the cumulative optimizer step, not dataset windows or
the number of save operations. A positive `--save-checkpoint-limit` enables
checkpoint saving and retains at most that many checkpoints. With
`--save-checkpoint-steps N`, periodic checkpoints are written at steps `N`,
`2N`, `3N`, and so on. A final checkpoint is also written when checkpoint
saving is enabled, even when the final step is not divisible by N. When
`--save-checkpoint-steps` is `0`, only the final checkpoint is written. When
`--save-checkpoint-limit` is `0`, checkpoint saving is disabled entirely,
including the final checkpoint, regardless of the step interval.

The output directory has this layout:

```text
output/
    checkpoint-000100/
        adapter.gguf
        optimizer.gguf
        training-state.json
        manifest.json
    snapshot-000100.gguf
    snapshot-000200.gguf
    adapter.gguf
```

`--save-checkpoint-limit N` enables checkpoint saving and retains the newest
N checkpoints. A value of `0` disables all checkpoint saving and retention,
including the final checkpoint. A new checkpoint is fully written and published
before older checkpoints are removed. The checkpoint is first written in a
temporary directory; its manifest is written last. Replacing an existing
checkpoint uses a backup directory so an interrupted replacement can be
recovered at startup.

`--save-snapshot-limit N` enables snapshot saving and retains the newest N
snapshots. A value of `0` disables all snapshot saving. When enabled,
`--save-snapshot-steps N` writes adapter-only files at optimizer steps divisible
by N, for example `snapshot-000100.gguf`; a value of `0` disables periodic
snapshots. Snapshots are not forced at the final step because the final adapter
is always saved as `adapter.gguf`. Existing files with the same step number are
replaced.

When a snapshot and checkpoint are saved at the same step, the adapter data is
stored once and shared with a hard link when supported. When checkpoint saving
is enabled, the final `adapter.gguf` also shares its data with the final
checkpoint adapter. If the final adapter matches a snapshot instead, it can
share that file. If hard links are not available, the files are copied instead.
Replacing or pruning one link does not modify the data referenced by the other
links. The final `adapter.gguf` is saved regardless of whether checkpoint or
snapshot saving is enabled.

Resume with `--resume-from DIR`, where `DIR` is either a `checkpoint-NNNNNN`
directory or an output directory containing checkpoint directories. The newest
valid checkpoint is selected when an output directory is specified. The
checkpoint supplies the adapter, so do not combine `--resume-from` with `--lora`.
The current checkpoint format is required; older checkpoint layouts are not
supported. Preserve the original training
configuration: the base model file, dataset files, architecture, LoRA rank/alpha
targets, frozen layers, context length, accumulation steps, optimizer, learning
rate and scheduler settings, `--max-steps`, `--val-windows`, and effective epoch count are validated before resume. The
dataset permutation, optimizer RNG, AdamW moments, optimizer iteration, scheduler
step, and next epoch/window are restored. Resume is supported at optimizer-update
boundaries only; incomplete accumulated gradients are not serialized. Reduced-precision optimizer-state storage can change the optimizer's numeric
trajectory and should be benchmarked against F32. F16 and BF16 use 16-bit values.

The JSONL reader processes up to 128 records per batch. This is deliberately not
one blocking `parallel_for()` call per record: the thread pool executes a single
task synchronously, so one-record batches would usually remove parallelism from
JSON parsing and tokenization while increasing synchronization overhead. Reducing
memory further would require a bounded asynchronous pipeline rather than simply
reducing the batch size to one.

The QLoRA-specific command-line options are parsed in `args.cpp`, rather
than registered in `common/arg.cpp`. JSONL parsing and tokenization use the
shared `-t` / `--threads` setting and can run in parallel; chat-template
rendering remains sequential. Chat records use the template embedded in the
model metadata. If the model has no chat template, chat records cannot be
rendered; no fallback template is applied.

### Full-context memory requirements

Peak activation memory is driven primarily by the full context length because the effective micro-batch
and batch sizes both equal `-c` for causal models. Set `-ub == -c`; `-b` must be at least `-c`, and
llama.cpp clamps its effective value to `-c`. Gradient accumulation reduces the frequency of optimizer
updates, not the activation memory needed by an individual window. If a full window does not fit, reduce
`-c` or increase CPU offload rather than lowering `-ub`. More context also changes the windowing of the
dataset and the number of windows per epoch.

### Default LoRA targets

When using `--lora`, only one input adapter is supported; multiple or repeated
`--lora` inputs are rejected. Every `--lora-targets` entry must match at least
one base-model tensor. Matching existing adapter tensors are trained,
missing matching tensors are added with zero-initialized B tensors, and existing
non-target tensors are retained but frozen.

Existing adapter tensors are converted to `--lora-rank` and `--lora-alpha` while
preserving their initial weight delta. Increasing the rank preserves existing
factor components, initializes new A components, and zero-initializes new B
components. Reducing any existing tensor's rank is rejected. Changing the
alpha/rank ratio rescales B so loading the converted adapter initially retains
the same effect. Frozen tensors are retained in periodic snapshots and the final
adapter output.

llama.cpp uses **internal GGUF tensor names**, not HuggingFace names:

| llama.cpp internal | HuggingFace equivalent | Status |
|---|---|---|
| `attn_q` | `q_proj` | ✅ default target, trainable |
| `attn_k` | `k_proj` | ✅ default target, trainable |
| `attn_v` | `v_proj` | ✅ default target, trainable |
| `attn_output` | `o_proj` | ✅ default target, trainable |
| `ffn_gate` | `gate_proj` | ✅ default target, trainable |
| `ffn_up` | `up_proj` | ✅ default target, trainable |
| `ffn_down` | `down_proj` | ✅ default target, trainable |
| `ffn_gate_exps` | per-expert `gate_proj` | ⚠️ opt-in; MoE expert LoRA |
| `ffn_up_exps` | per-expert `up_proj` | ⚠️ opt-in; MoE expert LoRA |
| `ffn_down_exps` | per-expert `down_proj` | ⚠️ opt-in; MoE expert LoRA |
| `ffn_gate_up_exps` | fused per-expert gate/up | ⚠️ opt-in; use instead of separate gate/up tensors |
| `ssm_in` | `in_proj` | ❌ not in defaults — zero gradient (SSM_SCAN no backward) |
| `ssm_out` | `out_proj` | ❌ not in defaults — zero gradient (SSM_SCAN no backward) |

The default target set includes both K/V projections as ordinary trainable targets.
No K/V-specific target option is required. As described above, each QLoRA training window is
processed as one full-context microbatch. With causal attention, the effective sizes satisfy
`n_ubatch == n_batch == n_ctx`; set `-ub == -c`, while `-b` may be greater than or equal to `-c`.
This allows gradients to flow through the current window's K/V projections.

**MoE expert targets:** Expert LoRA is opt-in. Use `ffn_gate_exps`, `ffn_up_exps`, and
`ffn_down_exps` for separate expert matrices, or `ffn_gate_up_exps` for architectures
that store gate and up weights in one tensor. These names refer to the model's internal
GGUF tensors, not Hugging Face module names. For example, to train attention Q/O and
the routed experts:

```bash
./build/bin/llama-finetune-qlora \
  --model moe-q4_k_m.gguf --train-file data/train.jsonl \
  --lora-targets attn_q,attn_output,ffn_gate_exps,ffn_up_exps,ffn_down_exps \
  --lora-rank 16 --lora-alpha 16 \
  -c 512 -b 512 -ub 512 \
  --output-dir ./adapter-moe-output --epochs 3
```

For models with a fused `ffn_gate_up_exps` tensor, select that target instead of the
separate gate/up targets. Each expert has its own LoRA A/B matrices, and the model's
existing routing selects which expert adapters contribute to each token. The base
expert weights and router remain frozen; only the selected LoRA parameters are trained.
Expert targets are not included in the default target list.

**MTP/NextN models:** LoRA targets are discovered from tensors registered by the loaded base model, not by scanning every tensor in the GGUF. Tensors whose `blk.N` index is at or beyond the main model's layer count are excluded from training. Existing MTP/NextN tensors in an input adapter are retained in the output adapter but kept frozen. The current QLoRA training graph optimizes the main model only, not the MTP prediction graph; MTP-aware training is not supported yet.

### Dataset formats

QLoRA accepts JSONL files only. Each line must be a JSON object using one of the
following commonly used record formats. The format is detected independently
for each record, so different formats can be mixed in the same file or across
multiple `--train-file` inputs. Input records are validated and normalized
before template application and tokenization.

- `messages`: a `messages` array containing `role` and `content` string fields.
- `prompt-completion`: `prompt` and `completion` string fields.
- `text`: a `text` string field.
- `alpaca`: `instruction`, optional `input`, and `output` string fields. A
  non-empty `input` is joined to `instruction` with two newline characters.
- `sharegpt`: a `conversations` array containing `from` and `value` fields.
  `human` maps to `user`; `gpt` maps to `assistant`.

Column-name customization is not supported. If one record matches multiple
formats, it is rejected as ambiguous; remove or rename the overlapping fields
so that the intended format is clear. Multiple JSONL files may be loaded by
repeating `--train-file`.

An invalid record has malformed JSON, is not a JSON object, does not match a
supported format, matches more than one format, or has missing fields, wrong
field types, or an invalid message structure. Invalid records are skipped by
default; `--fail-on-invalid-records` stops loading at the first such record.
Template-rendering and tokenization failures are reported separately and are
still skipped.

Internally each token has a float weight. At this stage, weight `0` is converted
to the existing `-1` label mask; any positive weight is treated as ordinary
weight `1` by the current backend. Fractional weighted loss is reserved for a
future backend change. Samples shorter than the context length are padded, and
overlap labels are masked only where the target was already covered by an
earlier window.


**Chat format** (each assistant turn is trained as its own sample; earlier conversation turns remain masked context):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

**Prompt/completion** (loss on completion only):
```json
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

**Plain text** (all text tokens receive non-zero weight):
```json
{"text": "The quick brown fox."}
```

### Verify and use the adapter

```bash
# Hot-load for inference (no merge needed)
./build/bin/llama-cli --model base.gguf --lora adapter.gguf -p "Hello"

# Merge into base model
./build/bin/llama-export-lora \
  --model base.gguf --lora adapter.gguf --output merged.gguf
```

---

## Known Limitations & Optimization Roadmap

### Current limitations

**1. Gemma 4 E2B/E4B Shared KV**
QLoRA training is currently unsupported for Gemma 4 E2B/E4B models that use Shared KV. In
`llm_graph_input_attn_kv_iswa`, a layer that produces K/V uses its current `k_cur` and
`v_cur` tensors during training. A layer that reuses K/V has no such tensors and reads the
KV-cache backing tensors through `ggml_view_4d()`. The cache is updated with
`GGML_OP_SET_ROWS`, which writes the source K/V values into cache rows but has no backward
path from the cache destination to those source values. The cache read is a view of the
cache tensor, not a `GET_ROWS` operation whose backward pass could reconnect it to the
`SET_ROWS` source. Consequently, gradients contributed by shared-KV attention layers do
not reach the layer that produced those K/V tensors, and upstream trainable adapters can
miss part of their gradient. The forward pass may run, but the resulting training graph
has incomplete gradients and must be treated as unsupported. QLoRA detects trainable LoRA
K/V projections used by shared attention layers during optimizer initialization and emits
a warning for each affected projection. This warning is diagnostic only; it does not make
the gradients complete. Using the full-context setting (`-ub == -c`) does not resolve this
limitation.

A correct fix must preserve the differentiable K/V dependency across layers. Simply
retaining and reusing the producer tensors is not sufficient for layer-split multi-GPU
execution, where producer and consumer layers may be placed on different devices. A
backend-aware cross-device path has not been implemented or validated.

**2. Full backprop through frozen quantized layers**
Every backward step dequantizes all frozen Q4_K_M weight tensors to compute activation gradients (needed to propagate loss from the output back to each LoRA layer). For a 28-layer 1.7B model at `-c 512`, this is ~280 dequantizing matmuls per update window → step time is 3–5× slower than inference.

**3. Activation VRAM**
All forward activations are kept in VRAM throughout the backward pass. VRAM ≈ `model + KV + n_layers × hidden × n_ctx × 10 × 4B + 2 × lora_params × 4B`. Reducing `-c` reduces activation memory; gradient accumulation does not lower the memory required by one window.

**4. Full backprop through all layers** *(partially addressed by `--freeze-layers`)*
Gradients propagate through all layers that have LoRA adapters. Use `--freeze-layers N` to skip LoRA allocation for blk.0..N-1 — those layers receive no gradient (the `grads_needed` pruner already skips their backward ops automatically). Only the top (total_layers - N) layers are trained.

### Optimization roadmap

| Priority | Optimization | Expected gain | Status |
|---|---|---|---|
| ✅ Done | **`--freeze-layers N`** — no LoRA on first N layers; backward auto-pruned | Proportional to N/total | Implemented |
| ✅ Done | **`--shuffle-dataset`** — shuffle windows each epoch | Better convergence | Implemented |
| ✅ Done | **BOS separators** — insert BOS between concatenated samples | Correct cross-sample boundaries | Implemented |
| ✅ Done | **Per-epoch loss summary** — log train loss after each epoch | Observability | Implemented |
| ✅ Done | **`MUL_MAT_ID` / `OUT_PROD_ID` backward** — routed expert LoRA gradients | Enables explicit MoE expert targets | Implemented |
| Done | **Quantized `OUT_PROD`** - dequantize on GPU + cuBLAS for backward matmul | Full GPU training (no CPU fallback) | Implemented |
| ✅ Done | **Reuse `ctx_compute_opt`** — allocate tensor metadata context once, `ggml_reset()` across windows | Eliminate ~0.5 s/window overhead | Implemented |
| ❌ Skip | **Static training graphs** — the current optimizer and training graph lifecycle rebuild the graph for each window | Would need a dedicated graph-lifecycle redesign | Not in scope |
| Low | **`SSM_SCAN/CONV` backward** — enable LoRA on Mamba SSM layers | Unlocks NemotronH SSM layers | Planned |
| Low | **GELU backward** — implement `ggml_gelu_back` kernel (UNARY + GLU) | Support GPT-2/Phi-style models | Planned (needs new CUDA/CPU kernels) |

---

## Implementation notes (for developers)

Development investigations are documented separately in [notes/](notes/), one report per topic. See [Context Length Rounding to 256](notes/Context-Length-Rounding-to-256.md).

### Modified llama.cpp files

| File | Change |
|---|---|
| `ggml/src/ggml.c` | Backward graph fixes: `GET_ROWS` 3D, `SET_ROWS`, `MUL_MAT_ID`, `SSM_SCAN/CONV`, `FLASH_ATTN_EXT` now propagates gradients through Q/K/V and optional attention sinks; inplace-op assert → warn+skip |
| `src/llama-context.cpp` | `opt_init`: scheduler and graph sized with inflated capacity before `ggml_opt_init`; warns when trainable LoRA K/V projections are shared by downstream attention layers; `opt_epoch_iter`: per-window timing instrumentation and accumulation-group period selection |
| `src/llama-adapter.cpp` | Repack-buft fallback for LoRA tensors: tries device-native buft before CPU |
| `common/common.h`, `common/arg.cpp` | Removed the old QLoRA-only parameter fields and CLI registrations; QLoRA options are now local to the example |
| `tools/finetune-qlora/args.cpp` | Parses QLoRA-only CLI flags locally; no QLoRA option registration is needed in `common/arg.cpp` |
| `include/llama.h` | Exposes `llama_opt_dataset_shuffle()` so callers control per-epoch shuffling; `llama_opt_epoch()` no longer owns shuffle policy; added `llama_opt_epoch_range()` |
| `ggml/src/ggml-cuda/out-prod.cu` | CUDA `OUT_PROD` with quantized src0 (dequantize on GPU + cuBLAS); `OUT_PROD_ID` for MoE backward |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_op` for quantized `OUT_PROD` and `OUT_PROD_ID`; CPU-resident ids fix in `mul_mat_id` |

### Key invariants

- `params.use_mmap = false` — forced; mmap'd tensors can't have data written back
- FlashAttention follows the common `--flash-attn` setting (default `AUTO`); CPU and CUDA backward implementations are available
- `params.warmup = false` — warmup runs inference with PARAM tensors → segfault
- QLoRA defaults `params.cache_type_k` and `params.cache_type_v` to F32 before common argument parsing; `--cache-type-k` and `--cache-type-v` override these defaults
- LoRA A/B tensors are marked `PARAM` via `ggml_set_param` on the tensors loaded by `llama_adapter_lora_init`, not on the pre-init scratch tensors in `lt.buf`
- The adapter GGUF is pre-saved and loaded via `params.lora_adapters` BEFORE `common_init_from_params` so that `sched_reserve` includes LoRA graph nodes in its sizing

### Why opt_init inflation matters

`ggml_opt_init` captures `sched.get()` at construction time. The backward graph (`gb_grad`, `gb_opt`) is ~3–5× larger than the forward graph in node count. If the scheduler hash_set is sized only for the forward graph, `ggml_backend_sched_alloc_graph` on the backward graph will overflow it. We recreate `sched` with `inflated = fwd_nodes × 4` slots BEFORE calling `ggml_opt_init`.
