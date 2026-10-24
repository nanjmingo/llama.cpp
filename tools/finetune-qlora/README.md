# llama.cpp — Native QLoRA Training

Native QLoRA SFT training pipeline for quantized GGUF models.

The base model weights remain **frozen** (quantized tensors are skipped by `llama_set_param` because they are not `GGML_TYPE_F32`). Only freshly-allocated F32 LoRA A/B tensors are trained. The saved adapter GGUF is directly compatible with the existing `llama_adapter_lora_init` loader and `llama-export-lora` merge tool.

**Status:** QLoRA SFT is implemented and functional. Training speed is currently limited by full backprop through quantized weights — see [Known Limitations](#known-limitations).

---

## Build

```bash
cd /mnt/w/llm-trading-arena/unsloth-api/llama.cpp

# First time (CUDA build):
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build -j$(nproc)

# Incremental rebuild (after code changes):
cmake --build build --target llama-finetune-qlora -j$(nproc)
# If llama-adapter.cpp or llama-context.cpp changed, rebuild all:
cmake --build build -j$(nproc)

# ROCm build:
cmake -B build-rocm -DGGML_HIP=ON -DLLAMA_CURL=OFF
cmake --build build-rocm --target llama-finetune-qlora -j$(nproc)
```

The ROCm build supports the same SFT, partial offload, and optimizer modes as
the CUDA build. Periodic saves are adapter-weight snapshots only; optimizer-state
checkpoint/resume is not currently implemented. HIP compiles the shared
`ggml-cuda` training kernels, including quantized `OUT_PROD` and `OUT_PROD_ID`.

### Tests

For test setup, the aggregate build target, test execution, fixtures, and model-backed template checks, see [the QLoRA test guide](tests/README.md).

---

## QLoRA SFT (`llama-finetune-qlora`)

Trains LoRA adapters on a quantized GGUF model.

### Recommended command (1.7B model, 16 GB card)

```bash
./build/bin/llama-finetune-qlora \
  --model ~/qwen3-1.7b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --lora-rank 16 --lora-alpha 16 \
  -c 4096 -b 4096 -ub 512 \
  --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 -lr-min 1e-6 \
  --save-every 10 \
  --lora-out ~/adapter.gguf \
  --epochs 3 --seed 42
```

### Recommended command (15B model, 16 GB card, partial offload)

```bash
./build/bin/llama-finetune-qlora \
  --model ~/nemotron-15b-q4_k_m.gguf \
  --train-file data/train.jsonl \
  --lora-rank 16 --lora-alpha 16 \
  -ngl 13 -c 14336 -b 14336 -ub 1024 \
  --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 -lr-min 1e-6 \
  --save-every 8 \
  --lora-out ~/nemotron-lora.gguf \
  --epochs 3 --seed 42
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to quantized GGUF model |
| `-t` / `--threads` | common default | CPU thread count, also used for JSONL parsing and tokenization |
| `--train-file` | *(required)* | Training dataset path; may be repeated to load multiple files |
| `--fail-on-invalid-records` | off | Stop on the first invalid record instead of skipping invalid records |
| `--lora-rank` | `16` | LoRA rank r |
| `--lora-alpha` | `0` (= rank) | LoRA alpha; effective scale = alpha/rank |
| `--lora-targets` | see below | Comma-separated internal tensor name substrings |
| `--lora-out` | `adapter.gguf` | Final adapter GGUF path |
| `--save-every` | `0` | Save a numbered adapter-weight snapshot every N dataset windows (0 = disabled) |
| `--freeze-layers` | `0` | Skip LoRA on first N transformer layers (blk.0..N-1); backward already pruned automatically |
| `--shuffle-dataset` | off | Shuffle dataset windows at the start of each epoch |
| `-epochs` / `--epochs` | `3` | Training epochs |
| `-c` / `--ctx-size` | `512` | Training context window (tokens) |
| `-b` / `--batch-size` | `2048` | Tokens per `llama_decode` call; set equal to `-c` |
| `-ub` / `--ubatch-size` | `512` | GPU micro-batch tokens; controls VRAM vs. step time |
| `-ngl` | `999` | GPU layers to offload |
| `-lr` / `--learning-rate` | `1e-5` | AdamW learning rate |
| `--lr-scheduler` | `constant` | Learning-rate schedule: `constant` or `cosine` |
| `--warmup-steps` | `0` | Linear warmup over N logical training steps |
| `--warmup-init-ratio` | `0.1` | Initial warmup LR as a fraction of peak LR; must be in `(0, 1]` |
| `-lr-min` / `--learning-rate-min` | `-1` | Cosine floor; values below 0 use 0 |
| `--seed` | `42` | Random seed for LoRA init |

JSONL parsing and tokenization use the shared `-t` / `--threads` setting. Chat-template rendering remains sequential.

One scheduler step is one completed dataset window. The recommended `-b == -c` configuration performs
one optimizer update per SFT scheduler step. Micro-batches used for gradient
accumulation do not advance the schedule. During warmup, step 0 starts at
`peak_lr * warmup_init_ratio`; each subsequent logical step linearly interpolates
toward peak LR using `progress = step / warmup_steps`. The ratio is ignored when
warmup is disabled.

`--save-every N` writes numbered adapter-only files alongside the final output,
for example `adapter.step00000010.gguf`. A failed periodic snapshot is reported
but does not interrupt training. Optimizer moments and training position are not
saved, so automatic checkpoint resume is intentionally not supported yet. To
start another run from saved adapter weights, pass the adapter using `--lora`;
the optimizer starts fresh.

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

### VRAM vs. step-time tradeoff

Step time and VRAM both scale linearly with `-ub`:

| Model | `-ub` | VRAM | Step time (approx) |
|---|---|---|---|
| 1.7B Q4_K_M | 512 | ~18 GB | ~120 s (OOM on 16 GB) |
| 1.7B Q4_K_M | 128 | ~6 GB | ~30 s |
| 15B Q4_K_M | 1024 | ~11 GB | ~60 s |

Use `-c` equal to your target sequence length. More context = more windows per sample = more steps per epoch. Reducing `-c` reduces total training time proportionally.

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
| `attn_output` | `o_proj` | ✅ default target, trainable |
| `ffn_gate` | `gate_proj` | ✅ default target, trainable |
| `ffn_up` | `up_proj` | ✅ default target, trainable |
| `ffn_down` | `down_proj` | ✅ default target, trainable |
| `attn_k` | `k_proj` | ⚠️ opt-in; trainable when `-ub` equals `-c` |
| `attn_v` | `v_proj` | ⚠️ opt-in; trainable when `-ub` equals `-c` |
| `ssm_in` | `in_proj` | ❌ not in defaults — zero gradient (SSM_SCAN no backward) |
| `ssm_out` | `out_proj` | ❌ not in defaults — zero gradient (SSM_SCAN no backward) |

To train K/V projections, explicitly include them in `--lora-targets` and process
the whole context in one microbatch. For example, this uses a 512-token context:

```bash
./build/bin/llama-finetune-qlora \
  --model base-q4_k_m.gguf --train-file data/train.jsonl \
  --lora-targets attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down \
  --lora-rank 16 --lora-alpha 16 \
  -c 512 -b 512 -ub 512 \
  --lora-out adapter-kv.gguf --epochs 3
```

The full-context microbatch can require substantially more VRAM than smaller
microbatches. For a 4096-token context, use `-ub 4096`; `-ub 512` with `-c 4096`
cannot train K/V projections. The trainer now stops with a clear error when K/V
LoRA tensors are selected but `n_ubatch != n_ctx`, rather than saving them with
zero gradients. K/V targets remain opt-in to keep the low-VRAM default usable.

**MoE models:** Input LoRA adapters containing expert tensors (`*_exps`) are rejected because expert LoRA training is not supported. Newly initialized adapters exclude expert tensors regardless of `--lora-targets`. LoRA on supported dense FFN tensors (`ffn_gate`, `ffn_up`, `ffn_down`) remains supported.

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

**1. Full backprop through frozen quantized layers**
Every backward step dequantizes all frozen Q4_K_M weight tensors to compute activation gradients (needed to propagate loss from the output back to each LoRA layer). For a 28-layer 1.7B model at `-ub 512`, this is ~280 dequantizing matmuls per step → step time is 3–5× slower than inference.

**2. Activation VRAM**
All forward activations are kept in VRAM throughout the backward pass. VRAM ≈ `model + KV + n_layers × hidden × n_ubatch × 10 × 4B + 2 × lora_params × 4B`. Reducing `-ub` reduces VRAM linearly.

**3. Full backprop through all layers** *(partially addressed by `--freeze-layers`)*
Gradients propagate through all layers that have LoRA adapters. Use `--freeze-layers N` to skip LoRA allocation for blk.0..N-1 — those layers receive no gradient (the `grads_needed` pruner already skips their backward ops automatically). Only the top (total_layers - N) layers are trained.

### Optimization roadmap

| Priority | Optimization | Expected gain | Status |
|---|---|---|---|
| ✅ Done | **`--freeze-layers N`** — no LoRA on first N layers; backward auto-pruned | Proportional to N/total | Implemented |
Implemented |
| ✅ Done | **`--shuffle-dataset`** — shuffle windows each epoch | Better convergence | Implemented |
| ✅ Done | **BOS separators** — insert BOS between concatenated samples | Correct cross-sample boundaries | Implemented |
| ✅ Done | **Per-epoch loss summary** — log train loss after each epoch | Observability | Implemented |
| ✅ Done | **`MUL_MAT_ID` backward** — LoRA on MoE dense FFN layers; `OUT_PROD_ID` for scattered outer product | Unlocks Mixtral/Nemotron-MoE | Implemented |
| Done | **Quantized `OUT_PROD`** - dequantize on GPU + cuBLAS/hipBLAS for backward matmul | Full GPU training (no CPU fallback) | Implemented |
| ✅ Done | **Reuse `ctx_compute_opt`** — allocate tensor metadata context once, `ggml_reset()` across ubatches | Eliminate ~0.5 s/step overhead | Implemented |
| ❌ Skip | **Static training graphs** — KV mask shape changes per ubatch (`n_kv` grows); graph topology not static | Would need KV cache redesign | Not feasible |
| Low | **`SSM_SCAN/CONV` backward** — enable LoRA on Mamba SSM layers | Unlocks NemotronH SSM layers | Planned |
| Low | **GELU backward** — implement `ggml_gelu_back` kernel (UNARY + GLU) | Support GPT-2/Phi-style models | Planned (needs new CUDA/CPU kernels) |

---

## Implementation notes (for developers)

Development investigations are documented separately in [notes/](notes/), one report per topic. See [Context Length Rounding to 256](notes/Context-Length-Rounding-to-256.md).

### Modified llama.cpp files

| File | Change |
|---|---|
| `ggml/src/ggml.c` | Backward graph fixes: `GET_ROWS` 3D, `SET_ROWS`, `MUL_MAT_ID`, `SSM_SCAN/CONV`, `FLASH_ATTN_EXT` all stop gradient; inplace-op assert → warn+skip |
| `src/llama-context.cpp` | `opt_init`: scheduler and graph sized with inflated capacity before `ggml_opt_init`; `opt_epoch_iter`: per-ubatch timing instrumentation |
| `src/llama-adapter.cpp` | Repack-buft fallback for LoRA tensors: tries device-native buft before CPU |
| `common/common.h`, `common/arg.cpp` | Removed the old QLoRA-only parameter fields and CLI registrations; QLoRA options are now local to the example |
| `tools/finetune-qlora/args.cpp` | Parses QLoRA-only CLI flags locally; no QLoRA option registration is needed in `common/arg.cpp` |
| `include/llama.h` | Exposes `llama_opt_dataset_shuffle()` so callers control per-epoch shuffling; `llama_opt_epoch()` no longer owns shuffle policy; added `llama_opt_epoch_range()` |
| `ggml/src/ggml-cuda/out-prod.cu` | Shared CUDA/HIP `OUT_PROD` with quantized src0 (dequantize on GPU + cuBLAS/hipBLAS); `OUT_PROD_ID` for MoE backward |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_op` for quantized `OUT_PROD` and `OUT_PROD_ID`; CPU-resident ids fix in `mul_mat_id` |

### Key invariants

- `params.use_mmap = false` — forced; mmap'd tensors can't have data written back
- `params.flash_attn_type = DISABLED` — no backward impl for flash attention
- `params.warmup = false` — warmup runs inference with PARAM tensors → segfault
- `params.cache_type_k = F32` — training requires F32 KV (or BF16 with `--cache-type-k bf16`)
- LoRA A/B tensors are marked `PARAM` via `ggml_set_param` on the tensors loaded by `llama_adapter_lora_init`, not on the pre-init scratch tensors in `lt.buf`
- The adapter GGUF is pre-saved and loaded via `params.lora_adapters` BEFORE `common_init_from_params` so that `sched_reserve` includes LoRA graph nodes in its sizing

### Why opt_init inflation matters

`ggml_opt_init` captures `sched.get()` at construction time. The backward graph (`gb_grad`, `gb_opt`) is ~3–5× larger than the forward graph in node count. If the scheduler hash_set is sized only for the forward graph, `ggml_backend_sched_alloc_graph` on the backward graph will overflow it. We recreate `sched` with `inflated = fwd_nodes × 4` slots BEFORE calling `ggml_opt_init`.
