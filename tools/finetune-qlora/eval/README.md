# QLoRA Precision Evaluation Tool

`eval-qlora-model-precision` evaluates how the selected compute type affects training loss convergence across different learning rates.

---

## How It Works

1. Measures initial baseline loss on a fixed evaluation window using F32 compute type.
2. For each specified learning rate, trains a fresh LoRA adapter for N optimizer updates (`--epoch`).
3. Evaluates inference loss after each update (always evaluated with F32 compute type).
4. Prints per-step loss changes and elapsed time.

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `-m`, `--model` | *(required)* | Path to GGUF model file. |
| `--compute-type` | `f32` | Matrix multiplication compute type (`f32`, `f16`, `bf16`, `q8`). `q8` uses Q8 for forward and F16 for backward. |
| `--learning-rates` | *(required)* | Comma-separated learning rates (e.g. `1e-2,1e-3,1e-4`). |
| `--epoch` | `10` | Number of optimizer updates per learning rate. |
| `--n-gpu-layers` | `0` | GPU offload layers (`0` = CPU only, `-1` = all layers). |
| `--flash-attn` | `off` | Flash Attention mode (`on` or `off`). |
| `--lora-targets` | (standard) | Comma-separated GGUF tensor-name components; missing boundary periods are added automatically. |
| `-ctk`, `-ctv` | `f32` | KV cache storage types for K and V. |
| `--verbose-logs` | disabled | Show verbose library logging. |

---

## Output Format

```text
[initial_loss]=1.278270
[lr=1e-2] [1]=-3.00e-05 [2]=+1.10e-04 [3]=-0 [sum]=+8.00e-05 [time]=1.234s
[lr=1e-3] [1]=-1.25e-04 [2]=-2.10e-04 [3]=-1.50e-04 [sum]=-4.85e-04 [time]=1.189s
```

- `[initial_loss]`: Baseline loss before training.
- `[N]`: Loss change at update $N$ relative to update $N-1$ (negative means loss decreased).
- `[sum]`: Net total loss change from baseline to final step.
- `[time]`: Elapsed time for that learning rate.

---

## Build and Run

### 1. Build the Tool

```bash
cmake --build build --target llamango-eval-tools -j
```

### 2. Standard Dense Model Example

```bash
./build/bin/eval-qlora-model-precision \
  --model /path/to/SmolLM2-135M-Instruct-Q8_0.gguf \
  --compute-type q8 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --learning-rates 1e-2,1e-3,1e-4,1e-5 \
  --epoch 10
```

### 3. MoE Model with Fused Expert Targets

```bash
./build/bin/eval-qlora-model-precision \
  --model /path/to/moe-model.gguf \
  --lora-targets ffn_gate_up_exps,ffn_down_exps \
  --n-gpu-layers 99 \
  --flash-attn on \
  --learning-rates 1e-3,1e-4 \
  --epoch 10
```
