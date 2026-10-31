#include "test-training-math-common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace qlora_training_math {
namespace {

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


} // namespace

void run_training_math_accumulation_tests() {
    test_microbatch_gradient_accumulation_matches_logical_batch();
    test_lora_parameter_gradients_accumulate_across_microbatches();
}

} // namespace qlora_training_math
