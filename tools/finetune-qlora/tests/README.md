# QLoRA Tests

This directory contains the QLoRA test programs, chat-template fixtures, and their build instructions. The default unit-test suite does not load model weights or run long model-backed gradient checks. Tests report pass/fail results and failure reasons; numeric diagnostics are shown only when needed to explain a failure. Model-backed regression checks are separate, manually invoked executables so only the required check needs to be run. The model-backed template utility checks only metadata and vocabulary; a separate opt-in CPU integration test runs actual QLoRA training and evaluates the saved adapter with `llama-perplexity`.

## Test inventory

| CTest name / executable | Coverage | Model required |
|---|---|---|
| `test-qlora-adapter` | Existing LoRA rank/alpha conversion, target addition, initial-delta preservation, F32/F16/BF16 adapter save/load conversion, and invalid target/rank/type rejection. | No |
| `test-qlora-dataset-format` | Supported JSONL record formats, per-record auto-detection, normalization, invalid records, and ambiguous-format diagnostics. | No |
| `test-qlora-dataset-thread-pool` | Worker-count limits, task completion, single-task execution, exception propagation, and pool reuse after an exception. | No |
| `test-qlora-dataset-template` | Actual llama.cpp Jinja rendering with the checked-in ChatML template, expected prompt/target boundaries, multi-turn masking, literal special-token-looking text, empty assistant responses, and template diagnostics. | No |
| `test-qlora-dataset-template-hf` | Smoke-tests the checked-in Gemma 4 and Qwen Jinja templates and checks prompt/target boundaries. | No |
| `test-qlora-dataset-template-model` | Optional model integration check using the model's embedded chat template and vocabulary. It loads metadata and vocabulary only, not model weights. | Local GGUF file |
| `test-qlora-dataset-reader` | Serial and parallel JSONL loading, bounded batch sizes, stable record ordering, diagnostics, strict error handling, and streaming callbacks. | No |
| `test-qlora-dataset-window` | Window construction, padding, overlap masking, loss masks, and packed dataset layout. | No |
| `test-qlora-training-math` | CPU-only synthetic GGML graph checks for masked cross-entropy normalization; weighted-label cross-entropy and logits gradients against independent analytic and finite-difference references; zero-initialized LoRA forward invariance; initial A/B gradients; LoRA A/B gradients against finite differences; loss-reducing gradient steps; logical-batch versus microbatch loss and LoRA-gradient accumulation under uneven supervision masks; static- and dynamic-graph `ggml-opt` loss/gradient/SGD-update comparisons; dynamic-graph gradient reset between optimizer steps; first-update invariants across LoRA ranks; AdamW first/second update reference values; F32/F16/BF16 weight/state update combinations; and low-precision optimizer-state checkpoint roundtrips. | No |
| `test-qlora-activation-precision` | CPU-only synthetic Q8_0-weight matrix multiplication. Checks default Q8 activation quantization and explicit Q8, F32, F16, and BF16 activation precision against independent reference calculations. | No |
| `test-qlora-schedule` | Constant-schedule warmup, cosine-schedule boundaries, optimizer callback learning-rate values, and weight-decay propagation. | No |
| `test-qlora-training-integration` (opt-in) | Runs a short CPU training job with a real GGUF model, verifies finite epoch loss, the final adapter, periodic snapshots, and checkpoint files, then checks that both the base model and trained adapter produce finite positive perplexity values. It does not invoke the long gradient checks or require one tiny training run to improve perplexity. | Yes, 135M GGUF |

`data/chat-template-input.jsonl` and `data/chat-template-expected.jsonl` define deterministic ChatML cases. `data/templates/chatml.jinja` is used by the core template test; the HF-template smoke test reads the checked-in Gemma 4 and Qwen `chat_template.jinja` files. Keep fixtures deterministic and do not require a downloaded model for the default test suite.

The training-math test is intentionally model-independent. Its cross-entropy fixture validates the same `ggml_cross_entropy_loss` operator used by the training graph against the formula

`loss = -sum(labels * log_softmax(logits)) / number_of_positions`

and checks its derivative

`d_logits = (softmax(logits) * sum(labels, classes) - labels) / number_of_positions`.

The fixtures include fully masked positions and fractional label weights. Separate tests compare analytic gradients with central finite differences and compare accumulated LoRA A/B gradients across microbatches with a single logical-batch reference. The `ggml-opt` checks cover both static graphs and the dynamic-graph path used by model training: two consecutive accumulation windows are compared with fresh logical-batch gradients and SGD parameter updates, guarding against stale gradient accumulators crossing optimizer steps. These tests validate the GGML loss/autodiff and small LoRA chain rule; they do not by themselves prove that every model architecture's graph is correct. The opt-in CPU integration test exercises the full model-backed training and adapter-loading path.

## Configure a CPU-only test build

Run these commands from the repository root. The dedicated build directory keeps test configuration separate from an existing CPU or CUDA build.

```bash
cmake -S . -B build-qlora-test \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF
```

Re-run the configuration command if you change CMake options. CMake automatically reconfigures when a relevant `CMakeLists.txt` changes.

## Build all QLoRA test tools

Build the default QLoRA unit-test executables, including the adapter conversion test, HF-template smoke test, and model-backed template utility, with one command:

```bash
cmake --build build-qlora-test --target qlora-tests --parallel 2
```

The `qlora-tests` target contains only the default suite and deliberately excludes all model-backed gradient executables. Required libraries are built as dependencies, but unrelated tool executables are not requested by this target. Increase `--parallel` if the machine has spare memory and CPU capacity. Model-backed gradient programs are excluded from the default build as well as CTest.

## Run the default test suite

The default CTest suite does not need a model and does not load model weights:

```bash
ctest --test-dir build-qlora-test --output-on-failure -R '^test-qlora-'
```

To show per-case output even when tests succeed, add `--verbose`. The template test prints a `[PASS]` line for each fixture so failures can be associated with a specific case. To run one test while debugging, use CTest's regular-expression filter, for example:

```bash
ctest --test-dir build-qlora-test --output-on-failure -R '^test-qlora-dataset-template$'
```

After changing source code, repeat the aggregate build command and then the CTest command. The build system recompiles only files that need it.

## Check a model's actual chat template

The ordinary `test-qlora-dataset-template` test loads `data/templates/chatml.jinja` directly and passes it to llama.cpp's real chat-template engine. The `test-qlora-dataset-template-hf` test does the same for the checked-in Gemma 4 and Qwen templates, checking prompt/target boundaries without requiring model weights.

To additionally check the template stored in a real GGUF file, run the separate utility manually:

```bash
./build-qlora-test/bin/test-qlora-dataset-template-model \
  --model /path/to/model.gguf
```

The utility uses only the template embedded in the GGUF metadata; it does not accept template overrides.

This utility enables `vocab_only`, so it loads model metadata and vocabulary but not the model weights. It uses the GGUF vocabulary to test tokenization and token/weight alignment. Build it directly with `cmake --build build-qlora-test --target test-qlora-dataset-template-model`; it is excluded from the default `all` and `qlora-tests` targets, included in `qlora-model-tests`, and not registered with CTest because it requires a user-supplied local model file.

The model-backed utility prints a `[PASS]` line for each fixture. Its output can include metadata and vocabulary loading logs; use the model-free test when changing or debugging template rendering itself.


## Run the model-backed CPU training integration test

The integration test is opt-in so the normal test suite remains fast and does not download external model files. The compact default fixture is **SmolLM2-135M-Instruct Q8_0** (about 145 MB). For gradient-check diagnosis, also use the **F16** version (about 271 MB), which retains full-precision model weights and helps separate quantized-weight effects from other gradient-path problems. All variants are dense Llama-architecture SmolLM2-135M-Instruct models and use the Apache-2.0 license. See the [existing Q8_0 file page](https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF/blob/main/SmolLM2-135M-Instruct-Q8_0.gguf), the [bartowski F16 file page](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/blob/main/SmolLM2-135M-Instruct-f16.gguf), and its [matching Q8_0 file page](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/blob/main/SmolLM2-135M-Instruct-Q8_0.gguf).

Download a pinned fixture and verify its SHA-256 checksum. The default `q8_0` keeps the existing integration-test fixture. For a controlled precision comparison, `f16` and `q8_0_bartowski` come from the same publisher's GGUF repository:

```bash
# Existing fixture used by previous reproductions:
python3 tools/finetune-qlora/tests/download-test-model.py \
  --variant q8_0 \
  --output /path/to/test-models/SmolLM2-135M-Instruct-Q8_0.gguf

# Full-F16 weights:
python3 tools/finetune-qlora/tests/download-test-model.py \
  --variant f16 \
  --output /path/to/test-models/SmolLM2-135M-Instruct-f16.gguf

# Matching Q8_0 counterpart from the same repository as the F16 file:
python3 tools/finetune-qlora/tests/download-test-model.py \
  --variant q8_0_bartowski \
  --output /path/to/test-models/SmolLM2-135M-Instruct-Q8_0-bartowski.gguf
```

## Run model-backed training regression checks

The model-backed QLoRA regression checks are implemented as one executable and are not registered with CTest because they require a local GGUF model. The runner uses a fixed F32 activation policy and F32 K/V caches so changes in precision settings do not alter the regression baseline. It checks training/inference loss consistency, performs one fixed-learning-rate update, and compares an update-derived gradient with a finite-difference reference. The implementation is split by operation, and each branch uses an independent adapter and context. The base model is loaded once; adapter state is not reset and reused.

Build and run the regression checks:

```bash
cmake --build build-qlora-test --target qlora-model-tests --parallel 2
./build-qlora-test/bin/test-qlora-model-training \
  --model /path/to/test-models/SmolLM2-135M-Instruct-Q8_0.gguf
```

For the full integration test, configure `QLORA_TEST_MODEL` with the selected GGUF path; it is a CMake cache path, so reconfigure when switching between F16 and Q8_0.

The integration wrapper captures subprocess output and prints diagnostics, errors, training summaries, and the final perplexity row by default rather than reproducing all model-loading logs. Set `QLORA_TEST_VERBOSE_LOGS=1` when running CTest, or pass `--verbose-subprocess-logs` when invoking `test-training-integration.py` directly, to print the complete subprocess logs.

```bash
cmake -S . -B build-qlora-test \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DQLORA_TEST_MODEL=/path/to/test-models/SmolLM2-135M-Instruct-Q8_0.gguf

cmake --build build-qlora-test --target qlora-tests --parallel 2
ctest --test-dir build-qlora-test --output-on-failure \
  -R '^test-qlora-training-integration$'
```

When `QLORA_TEST_MODEL` is set, `qlora-tests` also builds `llama-finetune-qlora` and `llama-perplexity`. The integration test creates a temporary JSONL training dataset, runs one epoch with `-ngl 0`, checks that the loss is finite, `adapter.gguf`, periodic `snapshot-NNNNNN.gguf` files, and the final checkpoint adapter are valid, then invokes `llama-perplexity` both without and with the trained adapter. It verifies that baseline and adapter perplexity outputs contain finite positive values, but does not print those measurements or assert that a one-epoch smoke test must reduce perplexity. It does not invoke any of the model-backed gradient executables. The temporary dataset and adapters are removed after the test.

Model files are not downloaded automatically during CMake configuration or CTest execution. The downloader pins all three supported files by SHA-256 and fails rather than accepting a changed or incomplete file.

## Adding a test

1. Add the test source and any small deterministic fixtures under this `tests/` directory.
2. In this directory's `CMakeLists.txt`, add the executable and register model-independent tests with `add_test()`.
3. Add ordinary unit-test executables to the `qlora-tests` target. Keep slow, model-backed diagnostics out of `qlora-tests` and CTest; expose a separate opt-in target if convenient.
4. Update the test inventory and execution instructions in this README.

Prefer testing observable behavior over internal implementation details. Default tests should not depend on network access, a particular local model path, CUDA availability, or nondeterministic timing. Keep model-backed checks opt-in.
