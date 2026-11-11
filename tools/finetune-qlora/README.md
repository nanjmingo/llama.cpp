# llama.cpp - Native QLoRA Training

Native QLoRA SFT (Supervised Fine-Tuning) training tool for quantized GGUF models.

Base model weights stay **frozen**. Training updates only the LoRA adapter weights. The output is a GGUF LoRA adapter that you can load directly in `llama-cli` or merge with `llama-export-lora`.

- **Supported precision**: Trainable LoRA weights support F32, F16, and BF16 storage (gradients accumulate in F32).
- **Optimizers**: AdamW (F32, F16, or BF16 moment states) and SGD (F32 only).
- **Features**: Full-context causal training, gradient accumulation, Flash Attention (CPU and CUDA), MoE expert training, and crash-safe checkpoint/resume.

---

## Build

Run CMake from the repository root:

```bash
# CPU build
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-finetune-qlora -j

# Or, CUDA build
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-finetune-qlora -j
```

---

## Quick Start

Train a LoRA adapter with context size 512:

```bash
./build/bin/llama-finetune-qlora \
  --model /path/to/base-model.gguf \
  --train-file /path/to/train.jsonl \
  --output-dir ./output \
  --lora-rank 16 \
  --lora-alpha 16 \
  -c 512 -b 512 -ub 512
```

> [!IMPORTANT]
> **Context and Batch Size Rule:**
> QLoRA processes each training window as one full-context graph. You must set:
> - `-c <N>` : Context window length in tokens.
> - `-ub <N>`: Must equal `-c` (`-ub == -c`).
> - `-b <N>` : Must be greater than or equal to `-c` (`-b >= -c`). Causal models clamp effective batch size to `-c`.
>
> Windows cannot be split into smaller microbatches. If you run out of VRAM/memory, reduce `-c` or offload more layers to GPU.

---

## CLI Options

### Model and Context Options

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Path to the base GGUF model file. |
| `-c`, `--ctx-size` | `0` (model default) | Training context window size (tokens). Used as-is without rounding up. |
| `-b`, `--batch-size` | `2048` | Physical batch size. Must be `>= -c`. Clamped to `-c`. |
| `-ub`, `--ubatch-size` | `512` | Microbatch size. Must equal `-c`. |
| `-t`, `--threads` | auto | CPU threads for training, JSONL reading, and tokenization. |
| `-ngl`, `--n-gpu-layers` | `auto` | Number of layers to offload to GPU (`auto`, `all`, or integer). |
| `-fa`, `--flash-attn` | `auto` | Flash Attention mode (`on`, `off`, `auto`). CPU and CUDA backward supported. |
| `-ctk`, `--cache-type-k` | `f32` | KV cache K type (`f32`, `f16`, `bf16`, `q8_0`, etc.). |
| `-ctv`, `--cache-type-v` | `f32` | KV cache V type (`f32`, `f16`, `bf16`, `q8_0`, etc.). |
| `-ncmoe`, `--n-cpu-moe N` | `0` | Keep MoE expert weights in CPU memory for the first N layers. |
| `--moe-cache-mib N` | `0` (disabled) | GPU cache capacity (in MiB) for CPU-resident MoE experts. |
| `--op-offload` / `--no-op-offload` | enabled | Offload host-tensor operations to devices. |

### Dataset Options

| Option | Default | Description |
|---|---|---|
| `--train-file` | *(required)* | Path to JSONL dataset. Repeat this flag to load multiple files. |
| `--fail-on-invalid-records` | off | Exit with an error on the first bad record instead of skipping it. |
| `--shuffle-dataset` | off | Shuffle dataset windows at the start of each epoch. |
| `-s`, `--seed` | random | Random seed for dataset shuffling. |

### LoRA Options

| Option | Default | Description |
|---|---|---|
| `--lora-rank` | `16` | LoRA rank $r$. |
| `--lora-alpha` | `0` (= rank) | LoRA scaling factor $\alpha$. Effective scale is $\alpha / r$. |
| `--lora-type` | `f32` | Storage type for trainable LoRA A/B tensors: `f32`, `f16`, or `bf16`. |
| `--lora-targets` | (standard targets) | Comma-separated substrings matching internal GGUF tensor names. |
| `--freeze-layers` | `0` | Skip LoRA adapters on the first N layers (`blk.0` ... `blk.N-1`). |

### Training & Optimizer Options

| Option | Default | Description |
|---|---|---|
| `-opt`, `--optimizer` | `adamw` | Optimizer: `adamw` or `sgd`. |
| `-lr`, `--learning-rate` | `1e-5` | Peak learning rate. |
| `-wd`, `--weight-decay` | `0` | Weight decay for LoRA tensors. |
| `-epochs`, `--epochs` | `2` | Number of training epochs (ignored when `--max-steps` is set). |
| `--max-steps` | disabled | Stop after N optimizer updates. Loops the dataset if needed. |
| `--gradient-accumulation-steps` | `1` | Accumulate gradients across N windows before each optimizer update. |
| `--optimizer-state-type` | `f32` | AdamW moment storage type: `f32`, `f16`, or `bf16`. (SGD is F32 only). |
| `--lr-scheduler` | `constant` | Schedule: `constant` or `cosine`. |
| `--warmup-steps` | `0` | Linear warmup duration in optimizer updates. |
| `--warmup-init-ratio` | `0.1` | Starting LR ratio during warmup (`start_lr = peak_lr * ratio`). |
| `-lr-min`, `--learning-rate-min` | `-1` (0) | Floor LR for cosine schedule or constant decay. |
| `-decay-epochs` | `-1` | Epochs to decay constant LR toward `--learning-rate-min`. |
| `--forward-precision` | `f32` | Requested input precision for forward GEMM: `f32`, `f16`, `bf16`, `q8`. |
| `--backward-precision` | `f32` | Requested input precision for backward GEMM: `f32`, `f16`, `bf16`. |
| `--val-windows` | all remaining | Max windows evaluated after training range (`0` disables validation). |
| `--grad-stats-interval` | `0` | Log gradient norm/stats every N updates (`0` disables logging). |

#### Precision Clamping Rule

`--forward-precision` and `--backward-precision` are clamped according to each weight tensor's storage type:
- **F32 weights**: Always compute in F32.
- **F16 weights**: Use F16 for F16/Q8 requests; use F32 for BF16/F32 requests.
- **BF16 weights**: Use BF16 for BF16/Q8 requests; use F32 for F16/F32 requests.
- **Quantized weights (Q8_0, Q4_K, etc.)**: Use the requested precision as-is.

---

## Checkpointing and Resuming

### Checkpoint Options

| Option | Default | Description |
|---|---|---|
| `--output-dir` | *(required)* | Directory to save checkpoints, snapshots, and final `adapter.gguf`. |
| `--save-checkpoint-steps` | `0` | Save a full checkpoint every N updates (`0` = only save final). |
| `--save-checkpoint-limit` | `0` (disabled) | Keep at most N checkpoints (`0` disables all checkpoints). |
| `--save-snapshot-steps` | `0` | Save adapter-only snapshot every N updates (`0` = disabled). |
| `--save-snapshot-limit` | `0` (disabled) | Keep at most N snapshots (`0` disables snapshot retention). |
| `--resume-from` | disabled | Path to a checkpoint directory or output directory to resume from. |

### Output Directory Layout

```text
output/
    checkpoint-000100/      # Full state (resumable)
        adapter.gguf        # Adapter weights
        optimizer.gguf      # Optimizer states (AdamW moments)
        training-state.json # Progress, step count, dataset permutation, loss aggregates
        manifest.json       # Checkpoint metadata
    snapshot-000100.gguf    # Adapter-only weights at step 100
    adapter.gguf            # Final completed adapter
```

- **Checkpoints** contain everything needed to resume training exactly where it stopped.
- **Snapshots** contain only adapter weights for intermediate testing.
- Hard links are used when supported so duplicate files do not consume extra disk space.

### Resuming Training

To resume from the latest checkpoint in `./output`:

```bash
./build/bin/llama-finetune-qlora \
  --model /path/to/base-model.gguf \
  --train-file /path/to/train.jsonl \
  --output-dir ./output \
  --resume-from ./output \
  -c 512 -b 512 -ub 512
```

> [!NOTE]
> When resuming, model architecture, base model path, dataset files, LoRA rank/alpha/targets, context size, and optimizer type must match the checkpoint. You can safely change learning rate, scheduler, epochs, or snapshot limits.

---

## LoRA Target Modules

Target modules are specified as internal GGUF tensor-name substrings with `--lora-targets`.

### Standard Targets (Default)

By default, `--lora-targets` trains:
`attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down`

| Substring | Module | Description |
|---|---|---|
| `attn_q` | Self-attention | Query projection |
| `attn_k` | Self-attention | Key projection |
| `attn_v` | Self-attention | Value projection |
| `attn_output` | Self-attention | Output projection |
| `ffn_gate` | Feed-Forward | Gate projection (dense models) |
| `ffn_up` | Feed-Forward | Up projection (dense models) |
| `ffn_down` | Feed-Forward | Down projection (dense models) |

### Mixture of Experts (MoE) Targets

MoE expert weights are **opt-in**. Add expert substrings to `--lora-targets`:

| Substring | Description |
|---|---|
| `ffn_gate_exps` | Separate MoE expert gate projections |
| `ffn_up_exps` | Separate MoE expert up projections |
| `ffn_down_exps` | Separate MoE expert down projections |
| `ffn_gate_up_exps` | Fused MoE expert gate/up projections |

Example for MoE model:

```bash
./build/bin/llama-finetune-qlora \
  --model moe-model.gguf \
  --train-file train.jsonl \
  --lora-targets attn_q,attn_output,ffn_gate_exps,ffn_up_exps,ffn_down_exps \
  --output-dir ./output-moe \
  -c 512 -b 512 -ub 512
```

---

## Dataset Formats

The dataset must be in **JSONL** format (one JSON object per line). Format is auto-detected per line, so multiple formats can be mixed.

### Supported Record Formats

| Format | Required Fields | Loss Masking Policy |
|---|---|---|
| `messages` | `messages`: list of `{"role": "...", "content": "..."}` | Trains assistant messages only. User and system messages are masked as context. |
| `prompt-completion` | `prompt`: string, `completion`: string | Trains `completion` tokens only. `prompt` is masked. |
| `alpaca` | `instruction`: string, `output`: string, optional `input` | Trains `output` tokens only. Instruction/input are masked. |
| `sharegpt` | `conversations`: list of `{"from": "...", "value": "..."}` | Trains `gpt` turns only. `human` turns are masked. |
| `text` | `text`: string | Unsupervised pretraining style. First half of the first window is masked; remaining tokens are targets. |

### JSON Examples

**1. Chat format (recommended for SFT):**
```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4."}]}
```

**2. Prompt / Completion:**
```json
{"prompt": "Write a haiku about code:\n", "completion": "Bugs in the morning\nCoffee brews and tests all pass\nShip to production"}
```

**3. Raw text:**
```json
{"text": "The quick brown fox jumps over the lazy dog."}
```

### Windowing and Target Normalization

- Samples longer than `-c` are split into multiple complete windows. Incomplete trailing tokens are discarded.
- Samples shorter than `-c` are padded with BOS tokens up to length `-c` (padded positions receive label `-1`).
- Gradient accumulation normalizes the total gradient by the **sum of supervised target tokens** across all windows in the group, ensuring fair weighting across short and long answers.

---

## Inspecting Rendered Datasets (`eval-qlora-dataset`)

To verify how your dataset is tokenized and masked before training:

```bash
cmake --build build -t eval-qlora-dataset -j
./build/bin/eval-qlora-dataset \
  --model /path/to/model.gguf \
  --input data/train.jsonl \
  --output data/rendered.txt \
  --context-size 512
```

The output file shows token counts, target token counts, and target ranges (e.g., `0..32:0,32..128:1` means tokens 0-31 are prompt/masked, and tokens 32-127 are trained). Use `--token-details` if you need byte-level token inspect logs.

---

## Inference and Export

### 1. Test Adapter with `llama-cli` (Hot-load)

```bash
./build/bin/llama-cli \
  --model /path/to/base-model.gguf \
  --lora ./output/adapter.gguf \
  -p "User: What is 2+2?\nAssistant:"
```

### 2. Merge Adapter into Base Model

```bash
./build/bin/llama-export-lora \
  --model /path/to/base-model.gguf \
  --lora ./output/adapter.gguf \
  --output ./output/merged-model.gguf
```

---

## Known Limitations

| Area | Limitation | Workaround / Detail |
|---|---|---|
| **Shared-KV Attention** | Consumer layers read KV cache via `SET_ROWS`, which lacks a backward path. Gradients do not flow back to source K/V projections. | A warning is logged on Gemma 3N / Gemma 4 prefix configurations. Avoid targeting `attn_k`/`attn_v` on affected shared layers. |
| **SSM / Mamba** | `SSM_CONV` and `SSM_SCAN` do not have backward implementations. | Gradients stop at SSM layers. Upstream `ssm_in` cannot receive gradients; downstream `ssm_out` works. |
| **GLU Variants** | Backward supports `SWIGLU` and `GEGLU` splits. Other GLU variants are not supported. | Standard models (Llama, Mistral, Qwen) use supported SwiGLU. |
| **MTP / NextN** | Multi-Token Prediction heads are not optimized. | MTP prediction layers are skipped during training and kept frozen in output adapters. |
| **Activation Memory** | Each training window is processed as a complete graph. | Cannot split a window into smaller microbatches. If out of memory, reduce context size `-c`. |
