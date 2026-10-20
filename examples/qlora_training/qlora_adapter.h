#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Holds the LoRA A/B tensor pairs and the memory backing them.
struct lora_tensors {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer * buf = nullptr;
    // Map from base tensor name to its {LoRA A, LoRA B} tensors.
    std::unordered_map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> ab;
};

// Resume metadata stored alongside adapter weights in a checkpoint GGUF.
struct checkpoint_state {
    std::string mode;
    int64_t     epoch           = 0;
    int64_t     window          = 0;
    int64_t     step            = 0;
    int64_t     dataset_windows = -1;
    int64_t     context_length  = -1;
    bool        shuffle         = false;
    bool        has_metadata    = false;
    float       alpha           = 0.0f;
    bool        has_alpha       = false;
    int64_t     schedule_step   = -1;
};

// Allocate trainable LoRA tensors for matching 2-D model weights.
lora_tensors alloc_lora_tensors(const std::string & model_path,
                                const std::vector<std::string> & targets,
                                int32_t rank,
                                std::mt19937 & rng,
                                int32_t freeze_layers = 0);

// Keep the optimizer restricted to LoRA A/B tensors.
bool lora_param_filter(const ggml_tensor * tensor, void * userdata);

// Read checkpoint metadata and validate that it can be resumed in this mode.
bool load_checkpoint_state(const std::string & path,
                           const std::string & expected_mode,
                           checkpoint_state & state);

// Write a LoRA adapter or a checkpoint in GGUF format.
void save_adapter(const lora_tensors & tensors,
                  const std::string & out_path,
                  const std::string & arch,
                  float alpha,
                  const std::string & base_model_path,
                  const checkpoint_state * checkpoint = nullptr);
