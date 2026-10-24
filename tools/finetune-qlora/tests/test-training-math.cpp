#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ggml_context_ptr make_context(size_t size = 8 * 1024 * 1024) {
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

void test_cross_entropy_masking_and_normalization() {
    ggml_context_ptr ctx = make_context();
    require(static_cast<bool>(ctx), "failed to allocate cross-entropy test context");

    ggml_tensor * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_tensor * labels = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    require(logits && labels, "failed to allocate cross-entropy test tensors");

    float * logits_data = static_cast<float *>(logits->data);
    float * labels_data = static_cast<float *>(labels->data);
    logits_data[0] = 2.0f;
    logits_data[1] = 0.0f;
    logits_data[2] = -1.0f;
    logits_data[3] = -2.0f;
    logits_data[4] = 1.0f;
    logits_data[5] = 0.5f;
    labels_data[0] = 1.0f;
    labels_data[1] = 0.0f;
    labels_data[2] = 0.0f;
    labels_data[3] = 0.0f;
    labels_data[4] = 0.0f;
    labels_data[5] = 0.0f;

    ggml_tensor * loss = ggml_cross_entropy_loss(ctx.get(), logits, labels);
    ggml_set_loss(loss);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, loss);

    const float max_logit = 2.0f;
    const double log_sum_exp = max_logit + std::log(
        std::exp(2.0 - max_logit) + std::exp(0.0 - max_logit) + std::exp(-1.0 - max_logit));
    const float expected_masked_loss = static_cast<float>((log_sum_exp - 2.0) / 2.0);
    const float actual_masked_loss = evaluate_loss(ctx.get(), graph, loss);
    require(std::fabs(actual_masked_loss - expected_masked_loss) < 1e-5f,
            "masked cross-entropy must skip zero-label rows and divide by all sequence positions");

    std::fill(labels_data, labels_data + 6, 0.0f);
    const float all_masked_loss = evaluate_loss(ctx.get(), graph, loss);
    require(std::fabs(all_masked_loss) < 1e-7f, "all-masked cross-entropy must be zero");

    labels_data[3] = 1.0f;
    const float one_of_two_loss = evaluate_loss(ctx.get(), graph, loss);
    require(one_of_two_loss > 0.0f, "a supervised row must contribute to cross-entropy");
    labels_data[0] = 1.0f;
    const float two_of_two_loss = evaluate_loss(ctx.get(), graph, loss);
    require(std::fabs(two_of_two_loss - (actual_masked_loss + one_of_two_loss)) < 1e-5f,
            "cross-entropy should add each supervised row's contribution before dividing by all positions");
}

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

tiny_lora_graph make_tiny_lora_graph(int64_t rank = 2, float alpha_over_rank = 1.0f) {
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

std::vector<float> copy_tensor(const ggml_tensor * tensor) {
    const float * data = static_cast<const float *>(tensor->data);
    return std::vector<float>(data, data + ggml_nelements(tensor));
}

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
                                         float loss_scale = 1.0f) {
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

void test_cross_entropy_gradient_matches_reference() {
    constexpr int64_t classes = 3;
    constexpr int64_t positions = 3;
    const std::vector<float> logits_values = {
        1.2f, -0.4f, 0.8f,
        0.1f,  0.2f, 0.3f,
       -1.0f,  2.0f, 0.5f,
    };
    // Include fractional token weights and a fully masked position. The loss
    // implementation accepts weighted one-hot labels, not just {0, 1} labels.
    const std::vector<float> labels_values = {
        0.75f, 0.0f, 0.0f,
        0.0f,  0.0f, 0.0f,
        0.0f,  0.4f, 0.0f,
    };

    logits_loss_graph t = make_logits_loss_graph(logits_values, labels_values, positions);
    const float actual_loss = evaluate_loss(t.ctx.get(), t.forward, t.loss);
    ggml_graph_reset(t.backward);
    compute_graph(t.ctx.get(), t.backward);
    const ggml_tensor * grad_logits_tensor = ggml_graph_get_grad(t.backward, t.logits);
    require(grad_logits_tensor != nullptr, "cross-entropy backward did not produce logits gradients");
    const std::vector<float> actual_gradient = copy_tensor(grad_logits_tensor);

    double expected_loss = 0.0;
    std::vector<float> expected_gradient(static_cast<size_t>(classes * positions), 0.0f);
    for (int64_t position = 0; position < positions; ++position) {
        const size_t offset = static_cast<size_t>(position * classes);
        float labels_sum = 0.0f;
        for (int64_t token = 0; token < classes; ++token) {
            labels_sum += labels_values[offset + static_cast<size_t>(token)];
        }
        if (labels_sum == 0.0f) {
            continue;
        }

        const float max_logit = *std::max_element(
            logits_values.begin() + static_cast<std::ptrdiff_t>(offset),
            logits_values.begin() + static_cast<std::ptrdiff_t>(offset + classes));
        double exp_sum = 0.0;
        for (int64_t token = 0; token < classes; ++token) {
            exp_sum += std::exp(logits_values[offset + static_cast<size_t>(token)] - max_logit);
        }
        const double log_sum_exp = max_logit + std::log(exp_sum);
        for (int64_t token = 0; token < classes; ++token) {
            const size_t index = offset + static_cast<size_t>(token);
            const double label = labels_values[index];
            const double probability = std::exp(logits_values[index] - log_sum_exp);
            expected_loss -= label * (logits_values[index] - log_sum_exp) / positions;
            expected_gradient[index] = static_cast<float>(
                (probability * labels_sum - label) / positions);
        }
    }

    require(std::fabs(actual_loss - static_cast<float>(expected_loss)) < 2e-6f,
            "cross-entropy loss differs from the independent weighted-label reference");
    require(actual_gradient.size() == expected_gradient.size(), "cross-entropy gradient shape mismatch");
    for (size_t i = 0; i < expected_gradient.size(); ++i) {
        require(std::isfinite(actual_gradient[i]), "cross-entropy gradient is not finite");
        require(std::fabs(actual_gradient[i] - expected_gradient[i]) < 2e-6f,
                "cross-entropy logits gradient differs from (softmax * sum(labels) - labels) / positions");
    }

    // Check the CPU backward implementation against central finite differences
    // for every logit, including logits in the masked position.
    float * logits_data = static_cast<float *>(t.logits->data);
    constexpr float epsilon = 1e-3f;
    for (size_t i = 0; i < logits_values.size(); ++i) {
        logits_data[i] = logits_values[i] + epsilon;
        const float loss_plus = evaluate_loss(t.ctx.get(), t.forward, t.loss);
        logits_data[i] = logits_values[i] - epsilon;
        const float loss_minus = evaluate_loss(t.ctx.get(), t.forward, t.loss);
        logits_data[i] = logits_values[i];
        const float numeric = (loss_plus - loss_minus) / (2.0f * epsilon);
        const float tolerance = 2e-4f + 2e-3f * std::max(std::fabs(numeric), std::fabs(actual_gradient[i]));
        require(std::fabs(numeric - actual_gradient[i]) <= tolerance,
                "cross-entropy logits gradient disagrees with central finite differences at index " +
                    std::to_string(i));
    }
}

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
                                       float loss_scale = 1.0f) {
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

void test_lora_parameter_gradients_accumulate_across_microbatches() {
    constexpr int64_t logical_positions = 4;
    constexpr int64_t microbatch_positions = 2;
    constexpr float alpha_over_rank = 0.75f;
    const std::vector<float> input_values = {
        0.5f, -1.0f,
        1.5f,  0.25f,
       -0.2f,  0.7f,
        0.8f, -0.4f,
    };
    const std::vector<float> base_logits_values = {
         0.1f, -0.2f,  0.3f,
        -0.4f,  0.5f,  0.2f,
         0.7f, -0.1f,  0.2f,
         0.0f,  0.4f, -0.3f,
    };
    const std::vector<float> labels_values = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.6f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };
    const std::vector<float> a_values = { -0.3f, 0.1f, 0.4f, -0.2f };
    const std::vector<float> b_values = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f };

    lora_batch_graph logical = make_lora_batch_graph(
        input_values, labels_values, base_logits_values, logical_positions,
        a_values, b_values, alpha_over_rank);
    const float logical_loss = evaluate_loss(logical.ctx.get(), logical.forward, logical.loss);
    ggml_graph_reset(logical.backward);
    compute_graph(logical.ctx.get(), logical.backward);
    const std::vector<float> logical_grad_a = copy_tensor(ggml_graph_get_grad(logical.backward, logical.a));
    const std::vector<float> logical_grad_b = copy_tensor(ggml_graph_get_grad(logical.backward, logical.b));

    const auto slice = [](const std::vector<float> & values, size_t offset, size_t count) {
        return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                                  values.begin() + static_cast<std::ptrdiff_t>(offset + count));
    };
    constexpr size_t input_per_microbatch = 2 * microbatch_positions;
    constexpr size_t logits_per_microbatch = 3 * microbatch_positions;
    lora_batch_graph first = make_lora_batch_graph(
        slice(input_values, 0, input_per_microbatch), slice(labels_values, 0, logits_per_microbatch),
        slice(base_logits_values, 0, logits_per_microbatch), microbatch_positions,
        a_values, b_values, alpha_over_rank, 0.5f);
    lora_batch_graph second = make_lora_batch_graph(
        slice(input_values, input_per_microbatch, input_per_microbatch),
        slice(labels_values, logits_per_microbatch, logits_per_microbatch),
        slice(base_logits_values, logits_per_microbatch, logits_per_microbatch), microbatch_positions,
        a_values, b_values, alpha_over_rank, 0.5f);
    const float accumulated_loss = evaluate_loss(first.ctx.get(), first.forward, first.loss) +
                                   evaluate_loss(second.ctx.get(), second.forward, second.loss);
    ggml_graph_reset(first.backward);
    ggml_graph_reset(second.backward);
    compute_graph(first.ctx.get(), first.backward);
    compute_graph(second.ctx.get(), second.backward);
    const std::vector<float> first_grad_a = copy_tensor(ggml_graph_get_grad(first.backward, first.a));
    const std::vector<float> first_grad_b = copy_tensor(ggml_graph_get_grad(first.backward, first.b));
    const std::vector<float> second_grad_a = copy_tensor(ggml_graph_get_grad(second.backward, second.a));
    const std::vector<float> second_grad_b = copy_tensor(ggml_graph_get_grad(second.backward, second.b));

    require(std::fabs(logical_loss - accumulated_loss) < 2e-6f,
            "microbatch loss accumulation differs from the logical-batch loss");
    auto compare_accumulated = [](const std::vector<float> & logical_gradient,
                                  const std::vector<float> & first_gradient,
                                  const std::vector<float> & second_gradient,
                                  const char * name) {
        require(logical_gradient.size() == first_gradient.size() &&
                logical_gradient.size() == second_gradient.size(),
                std::string(name) + " gradient shape mismatch");
        for (size_t i = 0; i < logical_gradient.size(); ++i) {
            const float accumulated = first_gradient[i] + second_gradient[i];
            require(std::isfinite(accumulated), std::string(name) + " accumulated gradient is not finite");
            const float tolerance = 2e-6f + 2e-5f * std::fabs(logical_gradient[i]);
            require(std::fabs(logical_gradient[i] - accumulated) < tolerance,
                    std::string(name) + " microbatch gradients do not sum to logical-batch gradient");
        }
    };
    compare_accumulated(logical_grad_a, first_grad_a, second_grad_a, "LoRA A");
    compare_accumulated(logical_grad_b, first_grad_b, second_grad_b, "LoRA B");
}

void test_microbatch_gradient_accumulation_matches_logical_batch() {
    constexpr int64_t classes = 3;
    constexpr int64_t logical_positions = 4;
    constexpr int64_t physical_positions = 2;
    constexpr float accumulation_scale = 0.5f;
    const std::vector<float> logits_values = {
         2.0f, 0.0f, -1.0f,
        -2.0f, 1.0f,  0.5f,
         0.4f, 1.2f, -0.7f,
         0.1f, 0.2f,  0.3f,
    };
    // The first physical microbatch is fully masked; the second is fully supervised.
    // This verifies that gradient accumulation does not accidentally normalize by
    // the number of supervised tokens in each microbatch.
    const std::vector<float> labels_values = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };

    logits_loss_graph logical = make_logits_loss_graph(logits_values, labels_values, logical_positions);
    const float logical_loss = evaluate_loss(logical.ctx.get(), logical.forward, logical.loss);
    ggml_graph_reset(logical.backward);
    compute_graph(logical.ctx.get(), logical.backward);
    const std::vector<float> logical_gradient = copy_tensor(ggml_graph_get_grad(logical.backward, logical.logits));

    const std::vector<float> first_logits(logits_values.begin(), logits_values.begin() + classes * physical_positions);
    const std::vector<float> first_labels(labels_values.begin(), labels_values.begin() + classes * physical_positions);
    const std::vector<float> second_logits(logits_values.begin() + classes * physical_positions, logits_values.end());
    const std::vector<float> second_labels(labels_values.begin() + classes * physical_positions, labels_values.end());

    logits_loss_graph first = make_logits_loss_graph(first_logits, first_labels, physical_positions, accumulation_scale);
    logits_loss_graph second = make_logits_loss_graph(second_logits, second_labels, physical_positions, accumulation_scale);
    const float accumulated_loss =
        evaluate_loss(first.ctx.get(), first.forward, first.loss) +
        evaluate_loss(second.ctx.get(), second.forward, second.loss);
    ggml_graph_reset(first.backward);
    ggml_graph_reset(second.backward);
    compute_graph(first.ctx.get(), first.backward);
    compute_graph(second.ctx.get(), second.backward);
    const std::vector<float> first_gradient = copy_tensor(ggml_graph_get_grad(first.backward, first.logits));
    const std::vector<float> second_gradient = copy_tensor(ggml_graph_get_grad(second.backward, second.logits));

    require(std::fabs(logical_loss - accumulated_loss) < 1e-6f,
            "scaled microbatch losses must sum to the logical-batch loss");
    require(logical_gradient.size() == first_gradient.size() + second_gradient.size(),
            "logical and accumulated gradient sizes do not match");
    for (size_t i = 0; i < first_gradient.size(); ++i) {
        require(std::fabs(logical_gradient[i] - first_gradient[i]) < 1e-6f,
                "first microbatch gradient differs from the logical-batch gradient");
    }
    for (size_t i = 0; i < second_gradient.size(); ++i) {
        require(std::fabs(logical_gradient[first_gradient.size() + i] - second_gradient[i]) < 1e-6f,
                "second microbatch gradient differs from the logical-batch gradient");
    }
}

ggml_opt_optimizer_params test_sgd_optimizer_params(void *) {
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.sgd.alpha = 1e-2f;
    params.sgd.wd = 0.0f;
    return params;
}

struct ggml_opt_lora_fixture {
    ggml_context_ptr model_ctx;
    ggml_context_ptr compute_ctx;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    ggml_backend_buffer_t model_buffer = nullptr;
    ggml_opt_context_t opt = nullptr;
    ggml_opt_result_t result = nullptr;
    ggml_tensor * inputs = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * logits = nullptr;

    ~ggml_opt_lora_fixture() {
        ggml_opt_result_free(result);
        ggml_opt_free(opt);
        ggml_backend_buffer_free(model_buffer);
        ggml_backend_sched_free(scheduler);
        ggml_backend_free(backend);
    }
};

std::unique_ptr<ggml_opt_lora_fixture> make_ggml_opt_lora_fixture(int64_t physical_positions, int32_t opt_period,
                                                                  const std::vector<float> & a_values,
                                                                  const std::vector<float> & b_values,
                                                                  float alpha_over_rank,
                                                                  bool dynamic_graphs = false) {
    constexpr int64_t input_dim = 2;
    constexpr int64_t classes = 3;
    constexpr int64_t rank = 2;
    require(a_values.size() == input_dim * rank && b_values.size() == rank * classes,
            "ggml-opt LoRA fixture has invalid parameter dimensions");

    auto fixture = std::make_unique<ggml_opt_lora_fixture>();
    fixture->model_ctx = ggml_context_ptr(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    const size_t graph_size = GGML_DEFAULT_GRAPH_SIZE;
    const size_t compute_meta_size = 4 * graph_size * ggml_tensor_overhead() +
        3 * ggml_graph_overhead_custom(graph_size, true);
    fixture->compute_ctx = ggml_context_ptr(ggml_init({ compute_meta_size, nullptr, true }));
    require(fixture->model_ctx && fixture->compute_ctx, "failed to allocate ggml-opt contexts");

    fixture->backend = ggml_backend_cpu_init();
    require(fixture->backend != nullptr, "failed to initialize CPU backend for ggml-opt test");
    ggml_backend_t backends[] = { fixture->backend };
    fixture->scheduler = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(fixture->scheduler != nullptr, "failed to create ggml-opt CPU scheduler");

    fixture->inputs = ggml_new_tensor_2d(fixture->model_ctx.get(), GGML_TYPE_F32, input_dim, physical_positions);
    fixture->a = ggml_new_tensor_2d(fixture->model_ctx.get(), GGML_TYPE_F32, input_dim, rank);
    fixture->b = ggml_new_tensor_2d(fixture->model_ctx.get(), GGML_TYPE_F32, rank, classes);
    require(fixture->inputs && fixture->a && fixture->b, "failed to allocate ggml-opt LoRA parameters");
    ggml_set_param(fixture->a);
    ggml_set_param(fixture->b);
    fixture->model_buffer = ggml_backend_alloc_ctx_tensors(fixture->model_ctx.get(), fixture->backend);
    require(fixture->model_buffer != nullptr, "failed to allocate ggml-opt model buffer");
    ggml_backend_tensor_set(fixture->a, a_values.data(), 0, a_values.size() * sizeof(float));
    ggml_backend_tensor_set(fixture->b, b_values.data(), 0, b_values.size() * sizeof(float));

    if (!dynamic_graphs) {
        ggml_tensor * hidden = ggml_mul_mat(fixture->compute_ctx.get(), fixture->a, fixture->inputs);
        fixture->logits = ggml_mul_mat(fixture->compute_ctx.get(), fixture->b, hidden);
        fixture->logits = ggml_scale(fixture->compute_ctx.get(), fixture->logits, alpha_over_rank);
        ggml_cgraph * forward = ggml_new_graph(fixture->compute_ctx.get());
        ggml_build_forward_expand(forward, fixture->logits);
    }

    ggml_opt_params opt_params = ggml_opt_default_params(fixture->scheduler, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    if (!dynamic_graphs) {
        opt_params.ctx_compute = fixture->compute_ctx.get();
        opt_params.inputs = fixture->inputs;
        opt_params.outputs = fixture->logits;
    }
    opt_params.opt_period = opt_period;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_SGD;
    opt_params.get_opt_pars = test_sgd_optimizer_params;
    fixture->opt = ggml_opt_init(opt_params);
    require(fixture->opt != nullptr, "failed to initialize ggml-opt training context");
    fixture->result = ggml_opt_result_init();
    require(fixture->result != nullptr, "failed to allocate ggml-opt result");
    return fixture;
}

void run_ggml_opt_microbatch(ggml_opt_lora_fixture & fixture,
                             const std::vector<float> & input_values,
                             const std::vector<float> & labels_values,
                             float alpha_over_rank = 0.75f) {
    require(input_values.size() == static_cast<size_t>(ggml_nelements(fixture.inputs)),
            "ggml-opt microbatch input size mismatch");

    if (!ggml_opt_static_graphs(fixture.opt)) {
        ggml_reset(fixture.compute_ctx.get());
        ggml_tensor * hidden = ggml_mul_mat(fixture.compute_ctx.get(), fixture.a, fixture.inputs);
        fixture.logits = ggml_mul_mat(fixture.compute_ctx.get(), fixture.b, hidden);
        fixture.logits = ggml_scale(fixture.compute_ctx.get(), fixture.logits, alpha_over_rank);
        ggml_cgraph * forward = ggml_new_graph_custom(fixture.compute_ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(forward, fixture.logits);
        ggml_opt_prepare_alloc(fixture.opt, fixture.compute_ctx.get(), forward, fixture.inputs, fixture.logits);
    }

    ggml_opt_alloc(fixture.opt, true);
    ggml_backend_tensor_set(fixture.inputs, input_values.data(), 0, input_values.size() * sizeof(float));
    ggml_tensor * labels = ggml_opt_labels(fixture.opt);
    require(labels && labels_values.size() == static_cast<size_t>(ggml_nelements(labels)),
            "ggml-opt microbatch label size mismatch");
    ggml_backend_tensor_set(labels, labels_values.data(), 0, labels_values.size() * sizeof(float));
    ggml_opt_eval(fixture.opt, fixture.result);
}

void test_ggml_opt_cross_entropy_gradient_accumulation() {
    constexpr int64_t logical_positions = 4;
    constexpr int64_t physical_positions = 2;
    constexpr float alpha_over_rank = 0.75f;
    const std::vector<float> input_values = {
        0.5f, -1.0f,
        1.5f,  0.25f,
       -0.2f,  0.7f,
        0.8f, -0.4f,
    };
    const std::vector<float> labels_values = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.6f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };
    const std::vector<float> a_values = { -0.3f, 0.1f, 0.4f, -0.2f };
    const std::vector<float> b_values = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f };

    // The direct logical-batch graph is the reference for ggml-opt's internal
    // scaled-loss and gradient-accumulation path.
    const std::vector<float> base_logits_values(logical_positions * 3, 0.0f);
    lora_batch_graph logical = make_lora_batch_graph(
        input_values, labels_values, base_logits_values, logical_positions,
        a_values, b_values, alpha_over_rank);
    const float logical_loss = evaluate_loss(logical.ctx.get(), logical.forward, logical.loss);
    ggml_graph_reset(logical.backward);
    compute_graph(logical.ctx.get(), logical.backward);
    const std::vector<float> expected_grad_a = copy_tensor(ggml_graph_get_grad(logical.backward, logical.a));
    const std::vector<float> expected_grad_b = copy_tensor(ggml_graph_get_grad(logical.backward, logical.b));

    std::unique_ptr<ggml_opt_lora_fixture> fixture = make_ggml_opt_lora_fixture(
        physical_positions, 2, a_values, b_values, alpha_over_rank);
    const std::vector<float> first_inputs(input_values.begin(), input_values.begin() + 4);
    const std::vector<float> second_inputs(input_values.begin() + 4, input_values.end());
    const std::vector<float> first_labels(labels_values.begin(), labels_values.begin() + 6);
    const std::vector<float> second_labels(labels_values.begin() + 6, labels_values.end());
    run_ggml_opt_microbatch(*fixture, first_inputs, first_labels);
    run_ggml_opt_microbatch(*fixture, second_inputs, second_labels);

    std::vector<float> actual_grad_a(expected_grad_a.size());
    std::vector<float> actual_grad_b(expected_grad_b.size());
    ggml_tensor * grad_a = ggml_opt_grad_acc(fixture->opt, fixture->a);
    ggml_tensor * grad_b = ggml_opt_grad_acc(fixture->opt, fixture->b);
    require(grad_a && grad_b, "ggml-opt did not expose accumulated LoRA gradients");
    ggml_backend_tensor_get(grad_a, actual_grad_a.data(), 0, actual_grad_a.size() * sizeof(float));
    ggml_backend_tensor_get(grad_b, actual_grad_b.data(), 0, actual_grad_b.size() * sizeof(float));

    for (size_t i = 0; i < expected_grad_a.size(); ++i) {
        require(std::isfinite(actual_grad_a[i]), "ggml-opt accumulated A gradient is not finite");
        require(std::fabs(actual_grad_a[i] - expected_grad_a[i]) < 3e-6f,
                "ggml-opt accumulated A gradient differs from logical-batch reference");
    }
    for (size_t i = 0; i < expected_grad_b.size(); ++i) {
        require(std::isfinite(actual_grad_b[i]), "ggml-opt accumulated B gradient is not finite");
        require(std::fabs(actual_grad_b[i] - expected_grad_b[i]) < 3e-6f,
                "ggml-opt accumulated B gradient differs from logical-batch reference");
    }

    double accumulated_loss = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(fixture->result, &accumulated_loss, &uncertainty);
    require(std::isfinite(accumulated_loss), "ggml-opt accumulated training loss is not finite");
    require(std::fabs(accumulated_loss - logical_loss) < 3e-6,
            "ggml-opt reported loss differs from the logical-batch reference");
}


std::vector<float> get_backend_tensor_values(const ggml_tensor * tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

void compare_dynamic_ggml_opt_step_to_logical_batch(
        ggml_opt_lora_fixture & fixture,
        const std::vector<float> & input_values,
        const std::vector<float> & labels_values,
        float alpha_over_rank) {
    constexpr int64_t logical_positions = 4;
    constexpr int64_t physical_positions = 2;
    constexpr int64_t input_dim = 2;
    constexpr int64_t classes = 3;
    constexpr float learning_rate = 1e-2f;

    const std::vector<float> a_before = get_backend_tensor_values(fixture.a);
    const std::vector<float> b_before = get_backend_tensor_values(fixture.b);
    const std::vector<float> base_logits(static_cast<size_t>(classes * logical_positions), 0.0f);
    lora_batch_graph reference = make_lora_batch_graph(
        input_values, labels_values, base_logits, logical_positions,
        a_before, b_before, alpha_over_rank);
    const float expected_loss = evaluate_loss(reference.ctx.get(), reference.forward, reference.loss);
    ggml_graph_reset(reference.backward);
    compute_graph(reference.ctx.get(), reference.backward);
    const std::vector<float> gradient_a = copy_tensor(ggml_graph_get_grad(reference.backward, reference.a));
    const std::vector<float> gradient_b = copy_tensor(ggml_graph_get_grad(reference.backward, reference.b));

    const auto slice = [](const std::vector<float> & values, size_t offset, size_t count) {
        return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                                  values.begin() + static_cast<std::ptrdiff_t>(offset + count));
    };
    constexpr size_t inputs_per_microbatch = static_cast<size_t>(input_dim * physical_positions);
    constexpr size_t labels_per_microbatch = static_cast<size_t>(classes * physical_positions);
    ggml_opt_result_reset(fixture.result);
    run_ggml_opt_microbatch(fixture,
        slice(input_values, 0, inputs_per_microbatch),
        slice(labels_values, 0, labels_per_microbatch), alpha_over_rank);
    run_ggml_opt_microbatch(fixture,
        slice(input_values, inputs_per_microbatch, inputs_per_microbatch),
        slice(labels_values, labels_per_microbatch, labels_per_microbatch), alpha_over_rank);

    double actual_loss = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(fixture.result, &actual_loss, &uncertainty);
    require(std::isfinite(actual_loss) && std::fabs(actual_loss - expected_loss) < 3e-5,
            "dynamic ggml-opt loss differs from the logical-batch reference");

    const std::vector<float> a_after = get_backend_tensor_values(fixture.a);
    const std::vector<float> b_after = get_backend_tensor_values(fixture.b);
    for (size_t i = 0; i < a_before.size(); ++i) {
        const float expected = a_before[i] - learning_rate * gradient_a[i];
        require(std::fabs(a_after[i] - expected) < 3e-5f,
                "dynamic ggml-opt A update differs from fresh logical-batch SGD gradient");
    }
    for (size_t i = 0; i < b_before.size(); ++i) {
        const float expected = b_before[i] - learning_rate * gradient_b[i];
        require(std::fabs(b_after[i] - expected) < 3e-5f,
                "dynamic ggml-opt B update differs from fresh logical-batch SGD gradient");
    }
}

void test_dynamic_ggml_opt_resets_gradients_between_optimizer_steps() {
    constexpr float alpha_over_rank = 0.75f;
    const std::vector<float> a_values = { -0.3f, 0.1f, 0.4f, -0.2f };
    const std::vector<float> b_values = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f };
    const std::vector<float> first_inputs = {
        0.5f, -1.0f,
        1.5f,  0.25f,
        -0.2f, 0.7f,
        0.8f, -0.4f,
    };
    const std::vector<float> first_labels = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
    };
    const std::vector<float> second_inputs = {
        -0.7f, 0.9f,
         0.2f, 1.1f,
         1.4f, -0.3f,
        -0.5f, -0.8f,
    };
    const std::vector<float> second_labels = {
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    std::unique_ptr<ggml_opt_lora_fixture> fixture = make_ggml_opt_lora_fixture(
        2, 2, a_values, b_values, alpha_over_rank, true);
    require(!ggml_opt_static_graphs(fixture->opt), "dynamic regression fixture unexpectedly uses static graphs");
    compare_dynamic_ggml_opt_step_to_logical_batch(*fixture, first_inputs, first_labels, alpha_over_rank);
    compare_dynamic_ggml_opt_step_to_logical_batch(*fixture, second_inputs, second_labels, alpha_over_rank);
}

void test_lora_initial_backward_and_finite_difference() {
    tiny_lora_graph t = make_tiny_lora_graph();

    const float initial_loss = evaluate_loss(t.ctx.get(), t.forward, t.loss);
    const std::vector<float> initial_logits = copy_tensor(t.logits);
    const std::vector<float> base_logits = copy_tensor(t.base_logits);
    require(initial_logits.size() == base_logits.size(), "base/logit shape mismatch");
    for (size_t i = 0; i < initial_logits.size(); ++i) {
        require(std::fabs(initial_logits[i] - base_logits[i]) < 1e-7f,
                "zero-initialized LoRA changed the base model logits before training");
    }

    float * input_data = static_cast<float *>(t.input->data);
    const float saved_masked_input0 = input_data[2];
    const float saved_masked_input1 = input_data[3];
    input_data[2] = 100.0f;
    input_data[3] = -100.0f;
    const float loss_with_changed_masked_input = evaluate_loss(t.ctx.get(), t.forward, t.loss);
    require(std::fabs(loss_with_changed_masked_input - initial_loss) < 1e-7f,
            "input positions with all-zero labels must not contribute to the loss");
    input_data[2] = saved_masked_input0;
    input_data[3] = saved_masked_input1;

    ggml_graph_reset(t.backward);
    compute_graph(t.ctx.get(), t.backward);
    const ggml_tensor * grad_a = ggml_graph_get_grad(t.backward, t.a);
    const ggml_tensor * grad_b = ggml_graph_get_grad(t.backward, t.b);
    require(grad_a && grad_b, "GGML backward graph did not produce LoRA A/B gradients");

    std::vector<float> grad_a_values = copy_tensor(grad_a);
    std::vector<float> grad_b_values = copy_tensor(grad_b);
    float grad_a_max = 0.0f;
    float grad_b_norm2 = 0.0f;
    for (float value : grad_a_values) {
        require(std::isfinite(value), "initial A gradient contains a non-finite value");
        grad_a_max = std::max(grad_a_max, std::fabs(value));
    }
    for (float value : grad_b_values) {
        require(std::isfinite(value), "initial B gradient contains a non-finite value");
        grad_b_norm2 += value * value;
    }
    require(grad_a_max < 1e-7f,
            "A gradient should be zero on the first backward pass when B is initialized to zero");
    require(grad_b_norm2 > 1e-8f,
            "B gradient should be nonzero on the first backward pass for this fixture");

    const float loss_before_gradient_step = evaluate_loss(t.ctx.get(), t.forward, t.loss);
    float * b_data = static_cast<float *>(t.b->data);
    for (size_t i = 0; i < grad_b_values.size(); ++i) {
        b_data[i] = -0.05f * grad_b_values[i];
    }
    const float loss_after_gradient_step = evaluate_loss(t.ctx.get(), t.forward, t.loss);
    require(loss_after_gradient_step < loss_before_gradient_step,
            "a small negative-gradient step should lower the fixed-batch loss");
    std::fill(b_data, b_data + ggml_nelements(t.b), 0.0f);

    // Move away from B=0 so both A and B have nontrivial gradients for finite-difference comparison.
    const float b_initial[] = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f };
    std::copy(std::begin(b_initial), std::end(b_initial), b_data);
    ggml_graph_reset(t.backward);
    compute_graph(t.ctx.get(), t.backward);
    grad_a = ggml_graph_get_grad(t.backward, t.a);
    grad_b = ggml_graph_get_grad(t.backward, t.b);
    grad_a_values = copy_tensor(grad_a);
    grad_b_values = copy_tensor(grad_b);

    const float epsilon = 1e-3f;
    auto compare_numeric_gradient = [&](ggml_tensor * parameter, const std::vector<float> & analytic,
                                        const char * parameter_name) {
        float * values = static_cast<float *>(parameter->data);
        const std::vector<float> original = copy_tensor(parameter);
        require(original.size() == analytic.size(), "gradient shape mismatch");
        for (size_t i = 0; i < original.size(); ++i) {
            values[i] = original[i] + epsilon;
            const float loss_plus = evaluate_loss(t.ctx.get(), t.forward, t.loss);
            values[i] = original[i] - epsilon;
            const float loss_minus = evaluate_loss(t.ctx.get(), t.forward, t.loss);
            values[i] = original[i];
            const float numeric = (loss_plus - loss_minus) / (2.0f * epsilon);
            const float tolerance = 2e-3f + 2e-2f * std::max(std::fabs(numeric), std::fabs(analytic[i]));
            if (std::fabs(numeric - analytic[i]) > tolerance) {
                throw std::runtime_error(std::string(parameter_name) + " gradient disagrees with central finite differences at index " +
                                         std::to_string(i) + ": analytic=" + std::to_string(analytic[i]) +
                                         " numeric=" + std::to_string(numeric));
            }
        }
    };
    compare_numeric_gradient(t.a, grad_a_values, "LoRA A");
    compare_numeric_gradient(t.b, grad_b_values, "LoRA B");
}

void test_rank_sweep_first_update_diagnostics() {
    constexpr float learning_rate = 1e-2f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-8f;

    for (int64_t rank : { 1, 2, 4, 8 }) {
        tiny_lora_graph t = make_tiny_lora_graph(rank, 1.0f);
        const float loss_before = evaluate_loss(t.ctx.get(), t.forward, t.loss);
        ggml_graph_reset(t.backward);
        compute_graph(t.ctx.get(), t.backward);
        const ggml_tensor * grad_a = ggml_graph_get_grad(t.backward, t.a);
        const ggml_tensor * grad_b = ggml_graph_get_grad(t.backward, t.b);
        require(grad_a && grad_b, "rank sweep did not produce A/B gradients");
        const std::vector<float> a_gradients = copy_tensor(grad_a);
        const std::vector<float> gradients = copy_tensor(grad_b);
        float grad_a_norm2 = 0.0f;
        float grad_b_norm2 = 0.0f;
        for (float value : a_gradients) {
            require(std::isfinite(value), "rank sweep produced a non-finite A gradient");
            grad_a_norm2 += value * value;
        }
        for (float value : gradients) {
            require(std::isfinite(value), "rank sweep produced a non-finite B gradient");
            grad_b_norm2 += value * value;
        }
        require(grad_a_norm2 < 1e-12f, "initial A gradient should be zero for zero-initialized B");
        require(grad_b_norm2 > 1e-12f, "rank sweep should have nonzero B gradients");

        float * b_data = static_cast<float *>(t.b->data);
        float b_update_norm2 = 0.0f;
        for (size_t i = 0; i < gradients.size(); ++i) {
            const float g = gradients[i];
            const float first_moment = (1.0f - beta1) * g;
            const float second_moment = (1.0f - beta2) * g * g;
            const float corrected_moment = first_moment / (1.0f - beta1);
            const float corrected_variance = second_moment / (1.0f - beta2);
            b_data[i] = -learning_rate * corrected_moment / (std::sqrt(corrected_variance) + epsilon);
            b_update_norm2 += b_data[i] * b_data[i];
        }
        const float loss_after = evaluate_loss(t.ctx.get(), t.forward, t.loss);
        const std::vector<float> logits_after = copy_tensor(t.logits);
        const std::vector<float> base_logits = copy_tensor(t.base_logits);
        float logits_delta_norm2 = 0.0f;
        for (size_t i = 0; i < logits_after.size(); ++i) {
            const float delta = logits_after[i] - base_logits[i];
            logits_delta_norm2 += delta * delta;
        }
        const float logits_delta_norm = std::sqrt(logits_delta_norm2);
        require(std::isfinite(loss_before) && std::isfinite(loss_after) && std::isfinite(logits_delta_norm),
                "rank sweep produced non-finite diagnostics");
        std::cout << "rank-sweep rank=" << rank
                  << " loss_before=" << loss_before
                  << " loss_after=" << loss_after
                  << " loss_delta=" << (loss_after - loss_before)
                  << " grad_b_l2=" << std::sqrt(grad_b_norm2)
                  << " b_update_l2=" << std::sqrt(b_update_norm2)
                  << " logits_delta_l2=" << logits_delta_norm << '\n';
    }
}

void test_adamw_first_update_matches_reference() {
    ggml_context_ptr ctx = make_context();
    require(static_cast<bool>(ctx), "failed to allocate AdamW test context");
    ggml_tensor * weights = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * gradients = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * m = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * v = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * params = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 7);
    require(weights && gradients && m && v && params, "failed to allocate AdamW test tensors");

    const float initial_weights[] = { 0.5f, -1.0f, 2.0f };
    const float gradient_values[] = { 2.0f, -3.0f, 0.0f };
    std::copy(std::begin(initial_weights), std::end(initial_weights), static_cast<float *>(weights->data));
    std::copy(std::begin(gradient_values), std::end(gradient_values), static_cast<float *>(gradients->data));
    std::fill(static_cast<float *>(m->data), static_cast<float *>(m->data) + 3, 0.0f);
    std::fill(static_cast<float *>(v->data), static_cast<float *>(v->data) + 3, 0.0f);

    constexpr float lr = 0.01f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-8f;
    constexpr float wd = 0.02f;
    const float params_values[] = { lr, beta1, beta2, epsilon, wd,
                                    1.0f / (1.0f - beta1), 1.0f / (1.0f - beta2) };
    std::copy(std::begin(params_values), std::end(params_values), static_cast<float *>(params->data));
    ggml_set_param(weights);
    ggml_tensor * updated = ggml_opt_step_adamw(ctx.get(), weights, gradients, m, v, params);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, updated);
    compute_graph(ctx.get(), graph);

    const float * actual = static_cast<const float *>(weights->data);
    float first_update[3];
    for (int i = 0; i < 3; ++i) {
        const float g = gradient_values[i];
        const float m1 = (1.0f - beta1) * g;
        const float v1 = (1.0f - beta2) * g * g;
        const float corrected_m = m1 / (1.0f - beta1);
        const float corrected_v = v1 / (1.0f - beta2);
        const float expected = initial_weights[i] * (1.0f - lr * wd) -
            lr * corrected_m / (std::sqrt(corrected_v) + epsilon);
        require(std::fabs(actual[i] - expected) < 2e-6f,
                "AdamW first update disagrees with the scalar reference calculation");
        first_update[i] = actual[i];
    }

    const float params_values_step2[] = { lr, beta1, beta2, epsilon, wd,
                                          1.0f / (1.0f - beta1 * beta1),
                                          1.0f / (1.0f - beta2 * beta2) };
    std::copy(std::begin(params_values_step2), std::end(params_values_step2),
              static_cast<float *>(params->data));
    compute_graph(ctx.get(), graph);
    actual = static_cast<const float *>(weights->data);
    for (int i = 0; i < 3; ++i) {
        const float g = gradient_values[i];
        const float m1 = (1.0f - beta1) * g;
        const float v1 = (1.0f - beta2) * g * g;
        const float m2 = beta1 * m1 + (1.0f - beta1) * g;
        const float v2 = beta2 * v1 + (1.0f - beta2) * g * g;
        const float corrected_m = m2 / (1.0f - beta1 * beta1);
        const float corrected_v = v2 / (1.0f - beta2 * beta2);
        const float expected = first_update[i] * (1.0f - lr * wd) -
            lr * corrected_m / (std::sqrt(corrected_v) + epsilon);
        require(std::fabs(actual[i] - expected) < 2e-6f,
                "AdamW second update disagrees with the scalar reference calculation or lost optimizer state");
    }
}


void test_quantized_matmul_activation_gradient() {
    constexpr int64_t input_dim = 32;
    constexpr int64_t output_dim = 7;
    constexpr int64_t batch = 3;

    ggml_context_ptr ctx = make_context();
    require(static_cast<bool>(ctx), "failed to allocate quantized matmul gradient test context");

    ggml_context_ptr weights_ctx(ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(weights_ctx), "failed to allocate quantized weight context");
    ggml_tensor * weights = ggml_new_tensor_2d(weights_ctx.get(), GGML_TYPE_Q8_0, input_dim, output_dim);
    require(weights != nullptr, "failed to allocate quantized matmul gradient test weights");
    ggml_backend_buffer_ptr weights_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(), ggml_backend_cpu_buffer_type()));
    require(static_cast<bool>(weights_buffer), "failed to allocate CPU buffer for quantized test weights");

    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, input_dim, batch);
    ggml_tensor * upstream = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, output_dim, batch);
    require(input && upstream, "failed to allocate quantized matmul gradient test tensors");

    std::vector<float> source_weights(static_cast<size_t>(input_dim * output_dim));
    for (int64_t row = 0; row < output_dim; ++row) {
        for (int64_t col = 0; col < input_dim; ++col) {
            source_weights[static_cast<size_t>(row * input_dim + col)] =
                std::sin(static_cast<float>((row + 1) * (col + 3)) * 0.17f) +
                0.25f * std::cos(static_cast<float>((row + 2) * (col + 1)) * 0.11f);
        }
    }

    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    std::vector<uint8_t> quantized_weights(row_size * static_cast<size_t>(output_dim));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, source_weights.data(), quantized_weights.data(),
                        0, output_dim, input_dim, nullptr);
    std::memcpy(weights->data, quantized_weights.data(), quantized_weights.size());

    float * input_data = static_cast<float *>(input->data);
    float * upstream_data = static_cast<float *>(upstream->data);
    for (int64_t i = 0; i < ggml_nelements(input); ++i) {
        input_data[i] = std::sin(static_cast<float>(i + 1) * 0.23f);
    }
    for (int64_t i = 0; i < ggml_nelements(upstream); ++i) {
        upstream_data[i] = std::cos(static_cast<float>(i + 2) * 0.19f) * 0.7f;
    }

    ggml_set_param(input);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, input);
    ggml_tensor * loss = ggml_sum(ctx.get(), ggml_mul(ctx.get(), output, upstream));
    ggml_set_loss(loss);

    ggml_cgraph * forward = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(forward, loss);
    ggml_cgraph * backward = ggml_graph_dup(ctx.get(), forward, true);
    ggml_build_backward_expand(ctx.get(), backward, nullptr);

    bool found_quantized_out_prod = false;
    for (int i = 0; i < ggml_graph_n_nodes(backward); ++i) {
        const ggml_tensor * node = ggml_graph_node(backward, i);
        if (node->op == GGML_OP_OUT_PROD && node->src[0] == weights &&
            node->src[0]->type == GGML_TYPE_Q8_0 && node->src[1]->type == GGML_TYPE_F32) {
            found_quantized_out_prod = true;
            break;
        }
    }
    require(found_quantized_out_prod,
            "backward graph did not contain Q8_0 OUT_PROD; the quantized CPU kernel was not exercised");

    (void) evaluate_loss(ctx.get(), forward, loss);
    ggml_graph_reset(backward);
    compute_graph(ctx.get(), backward);
    const ggml_tensor * gradient = ggml_graph_get_grad(backward, input);
    require(gradient != nullptr, "quantized matrix multiplication did not produce input gradients");

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    require(traits && traits->to_float, "Q8_0 dequantization is unavailable for the reference calculation");
    std::vector<float> dequantized_weights(static_cast<size_t>(input_dim * output_dim));
    const auto * quantized_data = static_cast<const uint8_t *>(weights->data);
    for (int64_t row = 0; row < output_dim; ++row) {
        traits->to_float(quantized_data + static_cast<size_t>(row) * row_size,
                         dequantized_weights.data() + static_cast<size_t>(row * input_dim), input_dim);
    }

    const std::vector<float> actual_gradient = copy_tensor(gradient);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t k = 0; k < input_dim; ++k) {
            float expected = 0.0f;
            for (int64_t row = 0; row < output_dim; ++row) {
                expected += dequantized_weights[static_cast<size_t>(row * input_dim + k)] *
                    upstream_data[b * output_dim + row];
            }
            const size_t index = static_cast<size_t>(b * input_dim + k);
            require(std::isfinite(actual_gradient[index]), "quantized matmul input gradient is not finite");
            require(std::fabs(actual_gradient[index] - expected) <= 2e-5f,
                    "quantized matmul input gradient differs from the explicit transposed-weight product at index " +
                        std::to_string(index) + ": actual=" + std::to_string(actual_gradient[index]) +
                        " expected=" + std::to_string(expected));
        }
    }
}

} // namespace

int main() {
    try {
        test_cross_entropy_masking_and_normalization();
        test_cross_entropy_gradient_matches_reference();
        test_rank_sweep_first_update_diagnostics();
        test_microbatch_gradient_accumulation_matches_logical_batch();
        test_lora_parameter_gradients_accumulate_across_microbatches();
        test_ggml_opt_cross_entropy_gradient_accumulation();
        test_dynamic_ggml_opt_resets_gradients_between_optimizer_steps();
        test_lora_initial_backward_and_finite_difference();
        test_adamw_first_update_matches_reference();
        test_quantized_matmul_activation_gradient();
        std::cout << "training math diagnostics passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
