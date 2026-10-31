#include "test-training-math-common.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

namespace qlora_training_math {
namespace {

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

void test_rank_sweep_first_update_invariants() {
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
                "rank sweep produced non-finite values");
        require(std::isfinite(b_update_norm2) && b_update_norm2 > 0.0f,
                "rank sweep produced no finite B-parameter update");
        require(logits_delta_norm > 0.0f,
                "rank sweep B-parameter update did not change the output logits");
    }
}


} // namespace

void run_training_math_lora_gradient_tests() {
    test_lora_initial_backward_and_finite_difference();
}

void run_training_math_lora_rank_tests() {
    test_rank_sweep_first_update_invariants();
}

} // namespace qlora_training_math
