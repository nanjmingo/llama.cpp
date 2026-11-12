#pragma once

#include "ggml-cpp.h"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using lora_tensor_refs = std::unordered_map<std::string, std::pair<ggml_tensor *, ggml_tensor *>>;

struct lora_tensor_snapshot {
    std::string name;
    enum ggml_type type_a = GGML_TYPE_F32;
    enum ggml_type type_b = GGML_TYPE_F32;
    int32_t n_dims_a = 0;
    int32_t n_dims_b = 0;
    int64_t ne_a[GGML_MAX_DIMS] = {};
    int64_t ne_b[GGML_MAX_DIMS] = {};
    std::vector<uint8_t> data_a;
    std::vector<uint8_t> data_b;
};

struct lora_adapter_snapshot {
    std::string arch;
    std::string base_model;
    float alpha = 0.0f;
    std::vector<lora_tensor_snapshot> tensors;
};

struct lora_tensors {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    lora_tensor_refs refs;
};

lora_tensors alloc_lora_tensors(const std::vector<std::pair<std::string, ggml_tensor *>> & model_tensors,
                                const std::vector<std::string> & targets,
                                int32_t rank,
                                std::mt19937 & rng,
                                int32_t model_n_layer,
                                int32_t freeze_layers = 0,
                                enum ggml_type lora_type = GGML_TYPE_F32);

lora_tensors transform_lora_tensors(const std::string & input_path,
                                    const std::string & expected_arch,
                                    const std::vector<std::pair<std::string, ggml_tensor *>> & model_tensors,
                                    const std::vector<std::string> & targets,
                                    int32_t rank,
                                    float alpha,
                                    float input_scale,
                                    std::mt19937 & rng,
                                    int32_t model_n_layer,
                                    int32_t freeze_layers,
                                    enum ggml_type lora_type = GGML_TYPE_F32);

bool validate_lora_adapter_rank(const std::string & input_path,
                                const std::string & expected_arch,
                                int32_t requested_rank);

bool validate_lora_targets(const std::vector<std::pair<std::string, ggml_tensor *>> & model_tensors,
                           const std::vector<std::string> & targets,
                           int32_t model_n_layer,
                           int32_t freeze_layers);

bool lora_tensor_matches_targets(const std::string & name,
                                 const std::vector<std::string> & targets,
                                 int32_t freeze_layers,
                                 int32_t model_n_layer = 0);

bool lora_param_filter(const ggml_tensor * tensor, void * userdata);

bool capture_lora_adapter_snapshot(const lora_tensor_refs & tensors,
                                   const std::string & arch,
                                   float alpha,
                                   const std::string & base_model_path,
                                   lora_adapter_snapshot & snapshot,
                                   std::string * error = nullptr);

bool save_lora_adapter_snapshot(const lora_adapter_snapshot & snapshot,
                                const std::string & out_path,
                                bool log_success = true);

bool lora_tensors_are_finite(const lora_tensor_refs & tensors, std::string * error = nullptr);

bool save_adapter(const lora_tensor_refs & tensors,
                  const std::string & out_path,
                  const std::string & arch,
                  float alpha,
                  const std::string & base_model_path,
                  bool log_success = true,
                  bool * validation_failed = nullptr);
