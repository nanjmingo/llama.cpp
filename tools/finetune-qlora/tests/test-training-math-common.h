#pragma once

#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qlora_training_math {

void require(bool condition, const std::string & message);
ggml_context_ptr make_context(std::size_t size = 8 * 1024 * 1024);
void compute_graph(ggml_context * ctx, ggml_cgraph * graph);
float evaluate_loss(ggml_context * ctx, ggml_cgraph * graph, ggml_tensor * loss);
std::vector<float> copy_tensor(const ggml_tensor * tensor);

struct tiny_lora_graph {
    ggml_context_ptr ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * labels = nullptr;
    ggml_tensor * base_logits = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * loss = nullptr;
    ggml_cgraph * forward = nullptr;
    ggml_cgraph * backward = nullptr;
    float alpha_over_rank = 1.0f;
};

tiny_lora_graph make_tiny_lora_graph(int64_t rank = 2, float alpha_over_rank = 1.0f);

struct logits_loss_graph {
    ggml_context_ptr ctx;
    ggml_tensor * logits = nullptr;
    ggml_tensor * loss = nullptr;
    ggml_cgraph * forward = nullptr;
    ggml_cgraph * backward = nullptr;
};

logits_loss_graph make_logits_loss_graph(const std::vector<float> & logits_values,
                                         const std::vector<float> & labels_values,
                                         int64_t positions,
                                         float loss_scale = 1.0f);

struct lora_batch_graph {
    ggml_context_ptr ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * base_logits = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * loss = nullptr;
    ggml_cgraph * forward = nullptr;
    ggml_cgraph * backward = nullptr;
};

lora_batch_graph make_lora_batch_graph(const std::vector<float> & input_values,
                                       const std::vector<float> & labels_values,
                                       const std::vector<float> & base_logits_values,
                                       int64_t positions,
                                       const std::vector<float> & a_values,
                                       const std::vector<float> & b_values,
                                       float alpha_over_rank,
                                       float loss_scale = 1.0f);

void run_training_math_loss_tests();
void run_training_math_lora_rank_tests();
void run_training_math_accumulation_tests();
void run_training_math_ggml_opt_tests();
void run_training_math_lora_gradient_tests();
void run_training_math_adamw_test();
void run_training_math_quantized_test();

} // namespace qlora_training_math
