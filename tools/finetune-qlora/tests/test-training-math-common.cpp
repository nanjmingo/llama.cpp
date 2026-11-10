#include "test-training-math-common.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qlora_training_math {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ggml_context_ptr make_context(std::size_t size) {
    return ggml_context_ptr(ggml_init({ size, nullptr, false }));
}

void compute_graph(ggml_context * ctx, ggml_cgraph * graph) {
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
    require(status == GGML_STATUS_SUCCESS,
            std::string("GGML CPU graph execution failed: ") + ggml_status_to_string(status));
}

float evaluate_loss(ggml_context * ctx, ggml_cgraph * graph, ggml_tensor * loss) {
    compute_graph(ctx, graph);
    const float value = ggml_get_f32_1d(loss, 0);
    require(std::isfinite(value), "loss is not finite");
    return value;
}

std::vector<float> copy_tensor(const ggml_tensor * tensor) {
    const float * data = static_cast<const float *>(tensor->data);
    return std::vector<float>(data, data + ggml_nelements(tensor));
}

tiny_lora_graph make_tiny_lora_graph(int64_t rank, float alpha_over_rank) {
    tiny_lora_graph t;
    t.ctx = make_context();
    require(static_cast<bool>(t.ctx), "failed to allocate LoRA gradient test context");

    constexpr int64_t input_dim = 2;
    constexpr int64_t classes = 3;
    constexpr int64_t batch = 2;
    require(rank > 0, "LoRA test rank must be positive");
    t.alpha_over_rank = alpha_over_rank;
    t.input = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, input_dim, batch);
    t.a = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, input_dim, rank);
    t.b = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, rank, classes);
    t.labels = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, classes, batch);
    t.base_logits = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, classes, batch);

    float input_values[] = { 0.5f, -1.0f, 1.5f, 0.25f };
    float labels_values[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float base_logits_values[] = { 0.1f, -0.2f, 0.3f, -0.4f, 0.5f, 0.2f };
    std::copy(std::begin(input_values), std::end(input_values), static_cast<float *>(t.input->data));
    float * a_data = static_cast<float *>(t.a->data);
    std::mt19937 rng(42);
    const float bound = 1.0f / std::sqrt(static_cast<float>(input_dim));
    std::uniform_real_distribution<float> dist_a(-bound, bound);
    for (int64_t i = 0; i < ggml_nelements(t.a); ++i) {
        a_data[i] = dist_a(rng);
    }
    std::fill(static_cast<float *>(t.b->data),
              static_cast<float *>(t.b->data) + ggml_nelements(t.b), 0.0f);
    std::copy(std::begin(labels_values), std::end(labels_values), static_cast<float *>(t.labels->data));
    std::copy(std::begin(base_logits_values), std::end(base_logits_values), static_cast<float *>(t.base_logits->data));
    ggml_set_param(t.a);
    ggml_set_param(t.b);

    ggml_tensor * hidden = ggml_mul_mat(t.ctx.get(), t.a, t.input);
    ggml_tensor * lora_logits = ggml_mul_mat(t.ctx.get(), t.b, hidden);
    lora_logits = ggml_scale(t.ctx.get(), lora_logits, t.alpha_over_rank);
    t.logits = ggml_add(t.ctx.get(), t.base_logits, lora_logits);
    t.loss = ggml_cross_entropy_loss(t.ctx.get(), t.logits, t.labels);
    ggml_set_loss(t.loss);
    t.forward = ggml_new_graph(t.ctx.get());
    ggml_build_forward_expand(t.forward, t.loss);
    t.backward = ggml_graph_dup(t.ctx.get(), t.forward, true);
    ggml_build_backward_expand(t.ctx.get(), t.backward, nullptr);
    return t;
}


logits_loss_graph make_logits_loss_graph(const std::vector<float> & logits_values,
                                         const std::vector<float> & labels_values,
                                         int64_t positions,
                                         float loss_scale) {
    constexpr int64_t classes = 3;
    require(static_cast<int64_t>(logits_values.size()) == classes * positions,
            "logits fixture has an invalid size");
    require(labels_values.size() == logits_values.size(), "labels fixture has an invalid size");

    logits_loss_graph t;
    t.ctx = make_context();
    require(static_cast<bool>(t.ctx), "failed to allocate logits loss context");
    t.logits = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, classes, positions);
    ggml_tensor * labels = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, classes, positions);
    require(t.logits && labels, "failed to allocate logits loss tensors");
    std::copy(logits_values.begin(), logits_values.end(), static_cast<float *>(t.logits->data));
    std::copy(labels_values.begin(), labels_values.end(), static_cast<float *>(labels->data));
    ggml_set_param(t.logits);
    t.loss = ggml_cross_entropy_loss(t.ctx.get(), t.logits, labels);
    if (loss_scale != 1.0f) {
        t.loss = ggml_scale(t.ctx.get(), t.loss, loss_scale);
    }
    ggml_set_loss(t.loss);
    t.forward = ggml_new_graph(t.ctx.get());
    ggml_build_forward_expand(t.forward, t.loss);
    t.backward = ggml_graph_dup(t.ctx.get(), t.forward, true);
    ggml_build_backward_expand(t.ctx.get(), t.backward, nullptr);
    return t;
}


lora_batch_graph make_lora_batch_graph(const std::vector<float> & input_values,
                                       const std::vector<float> & labels_values,
                                       const std::vector<float> & base_logits_values,
                                       int64_t positions,
                                       const std::vector<float> & a_values,
                                       const std::vector<float> & b_values,
                                       float alpha_over_rank,
                                       float loss_scale) {
    constexpr int64_t input_dim = 2;
    constexpr int64_t classes = 3;
    const int64_t rank = static_cast<int64_t>(a_values.size()) / input_dim;
    require(rank > 0 && a_values.size() == static_cast<size_t>(input_dim * rank),
            "LoRA batch A fixture has an invalid shape");
    require(b_values.size() == static_cast<size_t>(rank * classes),
            "LoRA batch B fixture has an invalid shape");
    require(input_values.size() == static_cast<size_t>(input_dim * positions),
            "LoRA batch input fixture has an invalid shape");
    require(labels_values.size() == static_cast<size_t>(classes * positions),
            "LoRA batch labels fixture has an invalid shape");
    require(base_logits_values.size() == labels_values.size(),
            "LoRA batch base logits fixture has an invalid shape");

    lora_batch_graph t;
    t.ctx = make_context();
    require(static_cast<bool>(t.ctx), "failed to allocate LoRA batch context");
    t.input = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, input_dim, positions);
    t.a = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, input_dim, rank);
    t.b = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, rank, classes);
    t.base_logits = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, classes, positions);
    ggml_tensor * labels = ggml_new_tensor_2d(t.ctx.get(), GGML_TYPE_F32, classes, positions);
    require(t.input && t.a && t.b && t.base_logits && labels,
            "failed to allocate LoRA batch graph tensors");
    std::copy(input_values.begin(), input_values.end(), static_cast<float *>(t.input->data));
    std::copy(a_values.begin(), a_values.end(), static_cast<float *>(t.a->data));
    std::copy(b_values.begin(), b_values.end(), static_cast<float *>(t.b->data));
    std::copy(base_logits_values.begin(), base_logits_values.end(), static_cast<float *>(t.base_logits->data));
    std::copy(labels_values.begin(), labels_values.end(), static_cast<float *>(labels->data));
    ggml_set_param(t.a);
    ggml_set_param(t.b);

    ggml_tensor * hidden = ggml_mul_mat(t.ctx.get(), t.a, t.input);
    ggml_tensor * lora_logits = ggml_mul_mat(t.ctx.get(), t.b, hidden);
    lora_logits = ggml_scale(t.ctx.get(), lora_logits, alpha_over_rank);
    t.logits = ggml_add(t.ctx.get(), t.base_logits, lora_logits);
    t.loss = ggml_cross_entropy_loss(t.ctx.get(), t.logits, labels);
    if (loss_scale != 1.0f) {
        t.loss = ggml_scale(t.ctx.get(), t.loss, loss_scale);
    }
    ggml_set_loss(t.loss);
    t.forward = ggml_new_graph(t.ctx.get());
    ggml_build_forward_expand(t.forward, t.loss);
    t.backward = ggml_graph_dup(t.ctx.get(), t.forward, true);
    ggml_build_backward_expand(t.ctx.get(), t.backward, nullptr);
    return t;
}


} // namespace qlora_training_math
