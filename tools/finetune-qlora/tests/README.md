# QLoRA Tests

This directory contains unit tests, integration tests, and chat-template fixtures for `llama-finetune-qlora`.

- **Unit tests:** Fast, deterministic, and model-independent. No weights or GPU required.
- **Model-backed tests:** Opt-in regression tests using local GGUF models (e.g. SmolLM2-135M).

---

## Test Inventory

| Test Target | Type | Model Needed? | Description |
|---|---|---|---|
| `test-qlora-adapter` | Unit | No | LoRA rank/alpha conversion, target filtering, F32/F16/BF16 serialization, non-finite tensor rejection, and immutable adapter snapshot capture. |
| `test-qlora-save-worker` | Unit | No | Background save execution, single-job backpressure, and result/error propagation. |
| `test-qlora-dataset-format` | Unit | No | JSONL parsing, format auto-detection, and bad record handling. |
| `test-qlora-dataset-thread-pool` | Unit | No | Worker thread limits, task completion, and error propagation. |
| `test-qlora-dataset-template` | Unit | No | Real Jinja template rendering with ChatML fixture, multi-turn masking. |
| `test-qlora-dataset-template-hf` | Unit | No | Smoke tests for Gemma 4 and Qwen Jinja templates. |
| `test-qlora-dataset-reader` | Unit | No | JSONL streaming, batching, and ordering guarantees. |
| `test-qlora-dataset-window` | Unit | No | Window packing, BOS padding, and overlap masking. |
| `test-qlora-optimizer-state` | Unit | No | Serialized optimizer-state finite-value validation for F32/F16/BF16 and version 1 compatibility. |
| `test-qlora-training-math` | Unit | No | Cross-entropy autodiff, LoRA gradients against finite differences, optimizer steps, and optimizer-state export, size calculation, and CPU-buffer finite-value checks. |
| `test-qlora-schedule` | Unit | No | LR warmup, cosine decay, and weight decay logic. |
| `test-qlora-dataset-template-model` | Integration | GGUF vocab | Validates model metadata and embedded Jinja chat template (no weights loaded). |
| `test-qlora-training-integration` | Integration | 135M GGUF | End-to-end CPU training, mid-epoch checkpoint/resume, and perplexity verification. |
| `llamango-cuda-tests` | Integration | 135M GGUF + GPU | CUDA resume equivalence test with low-precision backward pass. |

---

## 1. Fast Unit Tests (No Model Required)

### Configure and Build

Configure a dedicated test build from the repository root:

```bash
cmake -S . -B build-qlora-test \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF

cmake --build build-qlora-test --target llamango-tests -j
```

### Run All Unit Tests

```bash
ctest --test-dir build-qlora-test --output-on-failure -R '^test-qlora-'
```

To run a single test with verbose output:

```bash
ctest --test-dir build-qlora-test --verbose -R '^test-qlora-adapter$'
```

---

## 2. Model Template Verification

Test an actual GGUF model's embedded chat template without loading weights (`vocab_only` mode):

```bash
cmake --build build-qlora-test --target test-qlora-dataset-template-model -j

./build-qlora-test/bin/test-qlora-dataset-template-model \
  --model /path/to/model.gguf
```

---

## 3. Opt-in CPU Training Integration Test

Runs two real optimizer updates on CPU, saves a checkpoint, resumes from it, and verifies perplexity with `llama-perplexity`.

### Step 1: Download Test Model

Download the lightweight SmolLM2-135M fixture (~145 MB):

```bash
python3 tools/finetune-qlora/tests/download-test-model.py \
  --variant q8_0 \
  --output /path/to/SmolLM2-135M-Instruct-Q8_0.gguf
```

### Step 2: Configure and Run

```bash
cmake -S . -B build-qlora-test \
  -DLLAMA_BUILD_TESTS=ON \
  -DQLORA_TEST_MODEL=/path/to/SmolLM2-135M-Instruct-Q8_0.gguf

cmake --build build-qlora-test --target llamango-tests -j

ctest --test-dir build-qlora-test --output-on-failure \
  -R '^test-qlora-training-integration$'
```

---

## 4. Opt-in CUDA Integration Test

Verifies training resume equivalence on a CUDA device using `--backward-precision f16`.

```bash
# Configure in your existing CUDA build directory:
cmake -S . -B build \
  -DQLORA_ENABLE_CUDA_RESUME_TESTS=ON \
  -DQLORA_CUDA_TEST_MODEL=/path/to/SmolLM2-135M-Instruct-f16.gguf

# Build and run the dedicated target (excluded from standard ctest):
cmake --build build --target llamango-cuda-tests -j
```

---

## 5. Guidelines for Adding Tests

1. Put new test sources and small JSONL fixtures under `tools/finetune-qlora/tests/`.
2. Register unit tests in `tools/finetune-qlora/tests/CMakeLists.txt` using `add_test()`.
3. Add unit test targets to the aggregate `llamango-tests` build target.
4. Keep the default test suite fast and deterministic:
   - Do not require external network access.
   - Do not require local model files for unit tests.
   - Use small synthetic tensors or checked-in fixtures (`data/`).
