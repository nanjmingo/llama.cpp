# llama.cpp — Native QLoRA Training

Native QLoRA SFT training pipeline for quantized GGUF models.

The base model weights remain **frozen**; training updates only the selected LoRA adapter tensors. LoRA A/B tensors can be stored as `F32`, `F16`, or `BF16`; gradients are accumulated in F32. The output is a GGUF LoRA adapter loaded through llama.cpp adapter APIs. The repository also provides `llama-export-lora` for merging adapters, but the QLoRA test guide does not list an automated merge-and-reload regression test.

**Implementation status:** The QLoRA SFT training path, adapter save/load, gradient accumulation, and checkpoint/resume paths are implemented. Model and backend coverage depends on the operators used by each training graph; see [Known Limitations](#known-limitations).

---

## Build

Run these commands from the repository root.

Choose the backend at configure time. A CPU build does not require a backend;
add `-DGGML_CUDA=ON` for a CUDA build.

```bash
# CPU build
cmake -B build -DLLAMA_CURL=OFF

# Or, configure a CUDA build instead:
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF

# Build only the QLoRA executable
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

### Basic invocation

The following shows the required arguments and full-context batch settings. It is a
syntax example, not a model, memory, or backend compatibility recommendation. Choose
`-c` for the intended training window and set `-ub` to the same value; `-b` must be
at least that large.

```bash
./build/bin/llama-finetune-qlora \
  --model /path/to/base.gguf \
  --train-file /path/to/train.jsonl \
  --output-dir ./output \
  --lora-rank 16 --lora-alpha 16 \
  -c 512 -b 512 -ub 512
```

### All params

#### Common params

These are shared llama.cpp model and execution parameters.

| Parameter | Default | Description |
|---|---|---|
| `-m` / `--model` | *(required)* | Path to the quantized GGUF base model |
| `-t` / `--threads` | common default | CPU thread count; also used for JSONL parsing and tokenization |
| `-c` / `--ctx-size` | `0` (use the model context) | Training context window (tokens) |
| `-b` / `--batch-size` | `2048` | Must be at least `-c`; causal models clamp the effective batch size to `-c` |
| `-ub` / `--ubatch-size` | `512` | Must equal `-c`; with causal attention, the effective sizes satisfy `n_ubatch == n_batch == n_ctx` |
| `-ngl` / `--gpu-layers` / `--n-gpu-layers` | `auto` | Number of GPU layers to offload; accepts an integer, `auto`, or `all` |
| `-ctk` / `--cache-type-k` | `f32` | K cache storage type; accepts the KV cache types supported by llama.cpp, including `q8_0` |
| `-ctv` / `--cache-type-v` | `f32` | V cache storage type; accepts the KV cache types supported by llama.cpp, including `q8_0` |
| `-fa` / `--flash-attn` | `auto` | Flash Attention mode: `on`, `off`, or `auto`; backward propagation is supported on CPU and CUDA |
| `-ncmoe` / `--n-cpu-moe N` | `0` | Keep MoE expert weights in the first N layers in CPU memory; QLoRA training supports CPU-offloaded experts |
| `--moe-cache-mib N` | `0` (disabled) | GPU cache capacity in MiB for MoE experts kept on CPU; with multiple GPUs, it is split according to `--tensor-split` |
| `--op-offload` / `--no-op-offload` | enabled | Enable or disable offloading host-tensor operations to devices |
| `--repack` / `--no-repack` (`-nr`) | enabled | Enable or disable weight repacking |

`-ncmoe N` / `--n-cpu-moe N` keeps the MoE expert weights in the first N layers in CPU memory. This is supported during QLoRA training, including backward propagation through CPU-offloaded experts. `--moe-cache-mib N` optionally reserves N MiB of GPU cache for those CPU-resident experts; `0` disables the cache. With multiple GPUs, the cache capacity is distributed according to `--tensor-split`.

`--op-offload` is enabled by default and allows operations on host-resident tensors to be offloaded to devices. Use `--no-op-offload` to disable it.

When `--flash-attn` is `on`, Flash Attention is used in the training graph as well as inference. Its backward implementations are available on CPU and CUDA. `auto` lets llama.cpp select the setting based on model and backend support.

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
| `--forward-precision` | `f32` | Requested input precision for forward matrix multiplications, clamped to each weight tensor storage type: `f32`, `f16`, `bf16`, or `q8` |
| `--backward-precision` | `f32` | Requested input precision for backward matrix multiplications, clamped to each weight tensor storage type: `f32`, `f16`, or `bf16` |
| `--gradient-accumulation-steps` | `1` | Accumulate gradients from up to N complete windows before updating LoRA weights |
| `--grad-stats-interval` | `0` | Log global LoRA gradient statistics every N optimizer updates; `0` disables logging |
| `--max-steps` | *(disabled)* | Stop after N optimizer updates, repeating the dataset as needed; overrides `--epochs` |
| `--val-windows` | all remaining windows | Maximum number of windows to evaluate after the training range; `0` disables validation |
| `--shuffle-dataset` | off | Shuffle dataset windows at the start of each epoch using `-s` / `--seed` |
| `-epochs` / `--epochs` | `2` | Training epochs |
| `-lr` / `--learning-rate` | `1e-5` | AdamW learning rate |
| `-wd` / `--weight-decay` | `0` | AdamW or SGD weight decay applied to trainable LoRA tensors; `0` disables weight decay |
| `-opt` / `--optimizer` | `adamw` | Optimizer used to update LoRA tensors: `adamw` or `sgd` |
| `--lr-scheduler` | `constant` | Learning-rate schedule: `constant` or `cosine` |
| `--warmup-steps` | `0` | Linear warmup over N optimizer updates |
| `--warmup-init-ratio` | `0.1` | Initial warmup LR as a fraction of peak LR; must be in `(0, 1]` |
| `-lr-min` / `--learning-rate-min` | `-1` | Cosine floor; values below 0 use 0 |
| `-decay-epochs` / `--learning-rate-decay-epochs` | `-1` | For the constant scheduler, decay toward `--learning-rate-min` over N epochs; takes effect when the minimum is positive and below the base learning rate; ignored by the cosine scheduler |
| `-s` / `--seed` | random | Seed the optimizer RNG used for dataset shuffling when `--shuffle-dataset` is enabled; set a fixed value for reproducible permutations |

`--forward-precision` controls the requested input precision for forward matrix multiplications, and `--backward-precision` controls the requested input precision for backward matrix multiplications, including input-gradient and LoRA-gradient calculations. Each request is clamped independently for the storage type of the corresponding weight tensor: F32 weights always use F32; F16 weights use F16 for F16/Q8 requests and F32 for BF16/F32 requests; BF16 weights use BF16 for BF16/Q8 requests and F32 for F16/F32 requests; Q8_0 and lower-bit quantized weights use the requested precision unchanged. This policy also applies to LoRA A/B weights: the requested precision is clamped according to each adapter tensor's actual storage type, selected by `--lora-type`. F32 adapter tensors use F32; F16 and BF16 adapter tensors follow the corresponding rules above. Backward precision defaults to `f32`. `q8` is not supported as a backward precision. Both options leave F32 gradient tensors and optimizer moment storage (`--optimizer-state-type`) unchanged and do not change LoRA tensor storage (`--lora-type`). For CUDA MoE, explicit forward precision uses a synchronization fallback, and backward precision is applied to expert input- and weight-gradient GEMMs through the routing fallback. These paths can introduce synchronization and may disable graph capture for affected nodes.

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
is processed as one complete forward/backward graph. This keeps the current window's K/V dependencies
inside one graph for ordinary attention layers; Shared-KV cache reuse has the separate gradient
limitation described below. In this full-context training
mode, the requested `-c` is used as-is instead of being rounded up to a multiple of 256, so a context
such as `900` uses a 900-token training graph rather than a 1024-token one. Splitting a window into smaller
micro-batches is rejected. This can require substantially more activation memory; reduce `-c` or
offload more layers if the full window does not fit.

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
F16 and BF16 store each AdamW moment in a 16-bit type rather than F32. This option is only used by AdamW; SGD requires its default `f32` value.

`--grad-stats-interval N` reports the global norm, RMS, maximum finite absolute value, exact-zero ratio,
and non-finite element count for trainable LoRA gradients after the same loss normalization used by the optimizer,
every N optimizer updates. The default `0`
disables collection. When enabled, collection copies the gradient accumulators to host memory and synchronizes
with the backend at the requested interval; use a larger interval to reduce measurement overhead.

`--gradient-accumulation-steps N` accumulates gradients over up to N independent dataset windows before
updating the LoRA weights. Activations are released between windows, so increasing N does not retain all
windows' activations simultaneously. A shorter final group at each epoch is also applied as one optimizer
step. The scheduler advances once per optimizer update, and `--warmup-steps` and cosine decay are measured
in optimizer updates rather than windows. During warmup, step 0 starts at
`peak_lr * warmup_init_ratio`; each subsequent logical step linearly interpolates toward peak LR using
`progress = step / warmup_steps`. The ratio is ignored when warmup is disabled.

For cross-entropy training, the optimizer normalizes the accumulated gradient by the total supervision
weight in the complete accumulation group immediately before each optimizer update. Thus a window with
few assistant-target tokens does not receive the same total update weight as a window with many targets.
The final partial group uses its actual window count and supervision-weight sum. Progress and epoch loss
values are reported as means over supervised target weight, not means over context positions or windows.

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
Checkpoint format version 4 stores the current epoch's accumulated loss, data
count, and supervision-weight sum, so progress averages and the final epoch loss
continue across a mid-epoch resume. Version 3 checkpoints remain readable, but
they do not contain these aggregates; their resumed epoch statistics start at
the checkpoint position. The base model file, dataset files, architecture, LoRA
rank/alpha/targets, frozen layers, context length, gradient accumulation steps,
invalid-record handling, optimizer, and optimizer-state type must match the
checkpoint. These settings determine the model, dataset, trainable parameters,
or optimizer-state layout. `--learning-rate`, `--learning-rate-min`,
`--weight-decay`, `--learning-rate-decay-epochs`, `--lr-scheduler`,
`--warmup-steps`, `--warmup-init-ratio`, `--epochs`, `--max-steps`,
`--val-windows`, `--shuffle-dataset`, `--lora-type`, `--forward-precision`,
`--backward-precision`, and checkpoint/snapshot intervals and retention limits
may be changed when resuming. A shorter training limit is accepted only when
the checkpoint still has optimizer steps remaining. The current dataset
permutation is restored from
`training-state.json`. The optimizer RNG state, AdamW moments, optimizer
iteration, scheduler step, and next epoch/window are restored from the
checkpoint; the RNG state includes the state used for future dataset shuffles.
`-s` / `--seed` initializes the optimizer RNG for a new run and does not override
the RNG state restored from a checkpoint. Resume is supported at
optimizer-update boundaries only; incomplete accumulated gradients are not
serialized. F16 and BF16 optimizer-state storage uses a different numeric representation from F32.

Chat records require a chat template in the model metadata. If the model does
not provide one, chat records cannot be rendered; no fallback template is applied.

### Full-context memory requirements

The training graph processes one complete context window per microbatch. For causal models,
`-ub` must equal `-c`, and `-b` must be at least `-c` (the effective batch is clamped to `-c`).
Gradient accumulation combines updates across windows but does not split an individual window or
reduce the activation memory needed by that window. If a window does not fit, reduce `-c`.
Changing `-c` also changes dataset window construction and the number of windows per epoch.

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

QLoRA matches **internal GGUF tensor-name substrings**. The default target list is
`attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down`. These names do not
map one-to-one to Hugging Face module names across every architecture.

| Internal tensor-name substring | Default | Notes |
|---|---:|---|
| `attn_q` | Yes | Attention query projection |
| `attn_k` | Yes | Attention key projection |
| `attn_v` | Yes | Attention value projection |
| `attn_output` | Yes | Attention output projection |
| `ffn_gate` | Yes | Dense FFN gate projection |
| `ffn_up` | Yes | Dense FFN up projection |
| `ffn_down` | Yes | Dense FFN down projection |
| `ffn_gate_exps` | No | Explicit opt-in for separate MoE expert gate tensors |
| `ffn_up_exps` | No | Explicit opt-in for separate MoE expert up tensors |
| `ffn_down_exps` | No | Explicit opt-in for separate MoE expert down tensors |
| `ffn_gate_up_exps` | No | Explicit opt-in for architectures with a fused expert gate/up tensor |
| `ssm_in` | No | Gradient propagation stops at `SSM_CONV` / `SSM_SCAN`; this upstream projection does not receive gradients through that path |
| `ssm_out` | No | Downstream of `SSM_SCAN`; its LoRA weights can receive gradients from downstream operations |

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

Each token has a float supervision weight. A weight of `0` masks that token as
a target; positive weights contribute to the loss. For each training window,
weights are kept unchanged. During training, gradients are accumulated across
the configured number of windows and normalized once, immediately before the
optimizer update, by the sum of supervision weights across that entire
accumulation group. This computes a token-weighted mean cross-entropy instead
of giving each window equal weight regardless of how many targets it contains.
The final incomplete accumulation group uses its actual number of windows and
its own total supervision weight. Targets already covered by an earlier
overlapping window are masked to avoid duplicate training.


**Chat format** (each assistant turn is trained as its own sample; earlier conversation turns remain masked context):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

**Prompt/completion** (loss on completion only):
```json
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

**Plain text** (the first half of the first context window is context; the
remaining tokens are prediction targets. Later windows train only targets not
already covered by an earlier window):
```json
{"text": "The quick brown fox."}
```

For `text` records, the initial context mask is applied only to the first
window. For `prompt/completion` and chat records, the prompt or non-assistant
content is masked and all completion or assistant-target tokens are trained;
there is no additional first-half mask. For text and prompt/completion records,
automatic tokenizer special tokens are added once per rendered sample, not
once per prompt/completion piece. Chat records use the special tokens emitted
by the model chat template.

Long samples are split into complete context windows. A trailing fragment that
does not form a complete context window is not added as a separately padded
training window. Samples shorter than the context length are padded to one
context window.

### Inspect rendered dataset samples

`eval-qlora-dataset` uses the same JSONL reader, model chat-template renderer,
and tokenizer functions as training. It loads model metadata and vocabulary
only. By default it writes the decoded sample text, token count, target-token
count, and compact effective-weight ranges. It omits per-token rows because
they can make large datasets' output files many times larger than the input.
Special tokens such as BOS, EOS, and EOT are shown with readable markers.

For a large dataset, use the compact default:

```bash
cmake --build build-qlora-test --target eval-qlora-dataset --parallel 2
./build-qlora-test/bin/eval-qlora-dataset \
  --model /path/to/model.gguf \
  --input data/train.jsonl \
  --output data/train-rendered.txt \
  --context-size 4096
```

The `effective_weight_ranges` field describes the weights actually eligible for
target construction at sample level. For `text` records it includes the initial
half-context mask; window-overlap masking and trailing partial-window handling
are applied later by the training window builder. A range such as `0..2048:0,2048..10000:1` means token indices
`[0, 2048)` have zero target weight and `[2048, 10000)` have unit target weight.

Use `--token-details` only when investigating individual token IDs or byte
sequences. It adds one row per token with both the renderer-assigned
`rendered_weight` and the `effective_weight` after the `text` initial-context
mask, plus each token's raw piece bytes in hexadecimal. For `text` records,
`rendered_weight` can be `1` throughout the text (except automatically added
BOS) while the first half has `effective_weight=0`; the initial-context mask is
applied by the window builder rather than stored in the tokenizer's raw weight
vector.

### Verify and use the adapter

```bash
# Hot-load for inference (no merge needed)
./build/bin/llama-cli --model base.gguf --lora adapter.gguf -p "Hello"

# Merge into base model
./build/bin/llama-export-lora \
  --model base.gguf --lora adapter.gguf --output merged.gguf
```

The QLoRA tests exercise adapter serialization and loading. The test guide does not
list a regression test for `llama-export-lora` followed by inference with the merged
model; verify that workflow separately for the model and build being used.

---

## Known Limitations

**1. Shared-KV attention**

The current graph does not propagate gradients from Shared-KV consumer layers back
through the KV cache to the K/V source projections. A warning is emitted when a
trainable adapter matches an affected source projection. Training can continue, but
those adapters receive incomplete gradients. Full-context training (`-ub == -c`)
does not remove this limitation. The warning logic explicitly handles Gemma 3N and
Gemma 4 prefix Shared-KV configurations; see [the implementation note](notes/shared-kv-attention-gradient-limitation.md).

**2. SSM operators**

`SSM_CONV` and `SSM_SCAN` have no backward implementation in the current graph.
Gradient propagation stops at these operators. Consequently, an adapter on an
upstream projection such as `ssm_in` does not receive gradients through the SSM path.
A downstream projection such as `ssm_out` is outside that stopped path and can
receive gradients.

**3. GLU variants**

`GGML_OP_GLU` backward currently handles split `SWIGLU` and `GEGLU` cases. Other
GLU variants are not handled. Ordinary unary GELU and GELU-quick backward paths are
implemented; this limitation is about the other `GGML_OP_GLU` variants, not GELU
in general.

**4. MTP / NextN**

The training graph optimizes the main model, not the MTP prediction graph. MTP-aware
training is not supported. Existing MTP/NextN adapter tensors can be retained in the
output adapter, but are not trainable through the current main-model graph.

**5. Activation memory**

The training graph processes one complete context window per microbatch and retains
forward activations needed by backward. Gradient accumulation combines updates across
windows but does not reduce the activation memory required by an individual window.
See [Full-context memory requirements](#full-context-memory-requirements).

**6. Model and backend coverage**

The existence of a model loader or a forward implementation does not establish that
all operators in that model's training graph have backward support on every backend.
Check the operator paths and the model-backed tests before treating an untested
architecture/backend combination as supported. The [test guide](tests/README.md)
describes which paths are covered by automated tests.

