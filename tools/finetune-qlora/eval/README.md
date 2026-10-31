# QLoRA Evaluation Tools

## Inspect rendered dataset samples

`eval-qlora-dataset` reads one or more JSONL files using the training dataset
reader, applies the model's embedded chat template using the training renderer,
and tokenizes each rendered sample through the training tokenizer. It loads
only model metadata and vocabulary (`vocab_only`); model weights and GPU layers
are not loaded.

The output file contains one section for each source record and rendered
sample. It groups tokens into prompt/context and target-candidate sections,
prints token IDs and sample-level supervision weights, and renders common
special tokens as `<BOS>`, `<EOS>`, `<EOT>`, or `<SEP>`. Other control tokens use
their vocabulary spelling. For `text` records, the first half of the first
context window is shown as initial context; for prompt/completion and chat
records, the token weights show the prompt/assistant-only mask used by training.
The printed weights and roles are shown before overlap-window masking and
per-window normalization.

Build and run:

```bash
cmake --build build-qlora-test --target eval-qlora-dataset --parallel 2
./build-qlora-test/bin/eval-qlora-dataset \
  --model /path/to/model.gguf \
  --input data/train.jsonl \
  --input data/validation.jsonl \
  --output data/rendered.txt \
  --context-size 4096
```

The model argument is required because chat records use the template embedded
in that GGUF. Non-chat records are rendered without a chat template, but still
use the model vocabulary and tokenizer. `--input` may be repeated.

## Compare training activation precision

`eval-qlora-model-precision` measures inference loss before training and after each optimizer update for every requested learning rate. The `--activation-precision` option selects the activation precision used for training. Losses printed in the result are always measured with F32 activation precision; the selected training precision is restored before the next optimizer update.

Each learning rate starts from the same initial LoRA adapter and a new optimizer context. Within a learning-rate run, LoRA parameters and optimizer state are retained across updates. `--epoch` specifies the number of optimizer updates (default: `10`). The initial F32 evaluation loss is printed once on a separate `[initial_loss]` line with six digits after the decimal point, before the learning-rate results. Entries `[1]` through `[N]` on each learning-rate line are loss deltas: `[1]` is the loss after update 1 minus the initial loss, and `[N]` is the loss after update `N` minus the loss after update `N-1`. The final `[sum]` is the net loss change from the initial loss to the final loss (equivalent to the sum of all per-update deltas before formatting). A negative delta means the loss decreased; a positive delta means it increased. This probe uses one fixed training window; `--epoch 10` does not mean ten passes over a representative dataset.

The `--n-gpu-layers N` option controls how many model layers are requested for GPU offload. It defaults to `0` to preserve CPU-only behavior; use a positive number to offload that many layers, or `-1` to request all layers. When GPU offload is requested, KQV and operation offload are enabled for both training and inference contexts. Add `--verbose-logs` to inspect the backend logs and verify how many layers were actually offloaded. A GPU-enabled build and an available GPU backend are required.

The `--flash-attn on|off` option controls Flash Attention in both training and inference contexts. It defaults to `off`, preserving the previous behavior. Some model architectures or backend configurations may not support forcing Flash Attention on. The selected setting is printed in the `[CONFIG]` line.

The `--lora-targets CSV` option selects the model tensor-name substrings for which the evaluation adapter is created and trained. Its default preserves the original attention and dense-FFN targets. To evaluate expert LoRA, explicitly pass the expert tensor names used by the model, for example `--lora-targets ffn_gate_up_exps,ffn_down_exps` for models with fused gate/up expert weights, or `--lora-targets ffn_gate_exps,ffn_up_exps,ffn_down_exps` for models with separate expert weights. Target names are comma-separated, whitespace around names is ignored, and every requested target must match a supported tensor in the model. The selected targets are printed in the `[CONFIG]` line.

The loss measurement uses the configured inference KV-cache types (`-ctk` and `-ctv`), while the activation precision is fixed to F32. Keep the model, target tensors, cache types, sample, and all other settings identical when comparing runs. The base model's GGUF weight types are unchanged. The reported loss values are measurements on a fixed probe sample, not a substitute for held-out task evaluation.

Results are printed as one line per learning rate. Learning rates use compact scientific notation with an explicit exponent sign for nonnegative exponents. Loss deltas use scientific notation with three significant digits; positive deltas have a leading `+`, negative deltas have a leading `-`, and zero is printed as `-0`. The initial loss is printed once on a separate `[initial_loss]` line. Learning-rate lines contain per-update deltas `[1]` through `[N]`, the net `[sum]`, and `[time]` with elapsed seconds to millisecond precision. The timer starts before creating that learning-rate's experiment and includes experiment setup, optimizer updates, and per-update loss evaluation; it excludes model loading and the initial baseline evaluation. For example:

```text
[initial_loss]=1.278270
[lr=1e-2] [1]=-3.00e-05 [2]=+1.10e-04 [3]=-0 [sum]=+8.00e-05 [time]=1.234s
[lr=3e+0] [1]=+2.45e-06 [2]=-1.23e-04 ... [sum]=-1.20e-04 [time]=1.087s
```

Build and run an evaluation:

```bash
cmake --build build-qlora-test --target qlora-eval-tools --parallel 2
./build-qlora-test/bin/eval-qlora-model-precision \
  --model /path/to/test-models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --activation-precision q8 \
  --n-gpu-layers 99 \
  --flash-attn off \
  --verbose-logs \
  --cache-type-k f32 --cache-type-v f32 \
  --learning-rates 1e-2,1e-3,1e-4,1e-5 \
  --epoch 10
```

Evaluate expert LoRA with a model that has fused gate/up expert tensors:

```bash
./build-qlora-test/bin/eval-qlora-model-precision \
  --model /path/to/moe-model.gguf \
  --lora-targets ffn_gate_up_exps,ffn_down_exps \
  --n-gpu-layers 99 \
  --flash-attn on \
  --verbose-logs \
  --learning-rates 1e-3,1e-4 \
  --epoch 10
```

Supported activation precision values are `f32`, `f16`, `bf16`, and `q8`. `--epoch` must be a positive integer. `--verbose-logs` enables library logs. Runtime/configuration errors are reported as errors; numerical results are reported as measurements, not judged as pass or fail.
