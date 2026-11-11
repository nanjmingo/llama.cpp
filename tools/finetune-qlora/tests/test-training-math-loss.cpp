#include "test-training-math-common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace qlora_training_math {
namespace {

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


} // namespace

void run_training_math_loss_tests() {
    test_cross_entropy_masking_and_normalization();
    test_cross_entropy_gradient_matches_reference();
}

} // namespace qlora_training_math
