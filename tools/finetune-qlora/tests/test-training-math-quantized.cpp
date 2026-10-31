#include "test-training-math-common.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace qlora_training_math {
namespace {

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

void test_quantized_mul_mat_id_activation_gradient_with_slots(const int64_t n_slots) {
    constexpr int64_t input_dim = 32;
    constexpr int64_t output_dim = 7;
    constexpr int64_t n_experts = 3;
    constexpr int64_t n_used = 2;
    constexpr int64_t n_tokens = 2;
    require(n_slots > 0 && n_used % n_slots == 0, "invalid slot count for quantized MUL_MAT_ID gradient test");

    ggml_context_ptr ctx = make_context();
    require(static_cast<bool>(ctx), "failed to allocate quantized MUL_MAT_ID gradient test context");
    ggml_context_ptr weights_ctx(ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(weights_ctx), "failed to allocate quantized MUL_MAT_ID weight context");

    ggml_tensor * weights = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_Q8_0, input_dim, output_dim, n_experts);
    ggml_backend_buffer_ptr weights_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(), ggml_backend_cpu_buffer_type()));
    require(weights && weights_buffer, "failed to allocate quantized MUL_MAT_ID expert weights");

    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_dim, n_slots, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * upstream = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, output_dim, n_used, n_tokens);
    require(input && ids && upstream, "failed to allocate quantized MUL_MAT_ID test tensors");

    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    const size_t expert_size = row_size * static_cast<size_t>(output_dim);
    std::vector<float> source_weights(static_cast<size_t>(input_dim * output_dim));
    std::vector<uint8_t> quantized_expert(expert_size);
    std::vector<float> reference_weights(static_cast<size_t>(input_dim * output_dim * n_experts));
    for (int64_t expert = 0; expert < n_experts; ++expert) {
        for (int64_t row = 0; row < output_dim; ++row) {
            for (int64_t col = 0; col < input_dim; ++col) {
                source_weights[static_cast<size_t>(row * input_dim + col)] =
                    std::sin(static_cast<float>((expert + 1) * (row + 2) * (col + 1)) * 0.071f) +
                    0.2f * std::cos(static_cast<float>((expert + 3) * (row + 1) * (col + 4)) * 0.037f);
            }
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, source_weights.data(), quantized_expert.data(),
                            0, output_dim, input_dim, nullptr);
        std::memcpy((char *) weights->data + expert * expert_size, quantized_expert.data(), expert_size);
        const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        require(traits && traits->to_float, "Q8_0 dequantization is unavailable for the MUL_MAT_ID reference calculation");
        for (int64_t row = 0; row < output_dim; ++row) {
            traits->to_float(quantized_expert.data() + static_cast<size_t>(row) * row_size,
                reference_weights.data() + static_cast<size_t>(expert * input_dim * output_dim + row * input_dim), input_dim);
        }
    }

    float * input_data = static_cast<float *>(input->data);
    float * upstream_data = static_cast<float *>(upstream->data);
    int32_t * ids_data = static_cast<int32_t *>(ids->data);
    for (int64_t i = 0; i < ggml_nelements(input); ++i) {
        input_data[i] = std::sin(static_cast<float>(i + 1) * 0.13f);
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            ids_data[token * n_used + used] = static_cast<int32_t>((used + 2 * token) % n_experts);
            for (int64_t row = 0; row < output_dim; ++row) {
                upstream_data[token * n_used * output_dim + used * output_dim + row] =
                    std::cos(static_cast<float>(1 + row + 3 * used + 5 * token) * 0.17f);
            }
        }
    }

    ggml_set_param(input);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_tensor * loss = ggml_sum(ctx.get(), ggml_mul(ctx.get(), output, upstream));
    ggml_set_loss(loss);

    ggml_cgraph * forward = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(forward, loss);
    ggml_cgraph * backward = ggml_graph_dup(ctx.get(), forward, true);
    ggml_build_backward_expand(ctx.get(), backward, nullptr);

    bool found_id_back = false;
    for (int i = 0; i < ggml_graph_n_nodes(backward); ++i) {
        if (ggml_graph_node(backward, i)->op == GGML_OP_MUL_MAT_ID_BACK) {
            found_id_back = true;
            break;
        }
    }
    require(found_id_back, "backward graph did not contain the quantized MUL_MAT_ID activation-gradient op");

    (void) evaluate_loss(ctx.get(), forward, loss);
    ggml_graph_reset(backward);
    compute_graph(ctx.get(), backward);
    const ggml_tensor * gradient = ggml_graph_get_grad(backward, input);
    require(gradient != nullptr, "quantized MUL_MAT_ID did not produce input gradients");
    require(ggml_are_same_shape(gradient, input), "broadcast MUL_MAT_ID gradient did not reduce to the input shape");

    const std::vector<float> actual = copy_tensor(gradient);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < n_slots; ++slot) {
            for (int64_t col = 0; col < input_dim; ++col) {
                float expected = 0.0f;
                for (int64_t used = slot; used < n_used; used += n_slots) {
                    const int64_t expert = ids_data[token * n_used + used];
                    for (int64_t row = 0; row < output_dim; ++row) {
                        expected += reference_weights[static_cast<size_t>(expert * input_dim * output_dim + row * input_dim + col)] *
                            upstream_data[token * n_used * output_dim + used * output_dim + row];
                    }
                }
                const size_t index = static_cast<size_t>((token * n_slots + slot) * input_dim + col);
                require(std::isfinite(actual[index]), "quantized MUL_MAT_ID input gradient is not finite");
                require(std::fabs(actual[index] - expected) <= 2e-5f,
                    "quantized MUL_MAT_ID gradient mismatch for slot count " + std::to_string(n_slots) +
                    " at index " + std::to_string(index) + ": actual=" + std::to_string(actual[index]) +
                    " expected=" + std::to_string(expected));
            }
        }
    }
}

void test_trainable_mul_mat_id_weight_gradient_with_broadcast() {
    constexpr int64_t input_dim = 8;
    constexpr int64_t output_dim = 5;
    constexpr int64_t n_experts = 3;
    constexpr int64_t n_used = 2;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t n_slots = 1;

    ggml_context_ptr ctx = make_context();
    require(static_cast<bool>(ctx), "failed to allocate trainable MUL_MAT_ID gradient test context");
    ggml_context_ptr weights_ctx(ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true }));
    require(static_cast<bool>(weights_ctx), "failed to allocate trainable MUL_MAT_ID weight context");

    ggml_tensor * weights = ggml_new_tensor_3d(weights_ctx.get(), GGML_TYPE_F32, input_dim, output_dim, n_experts);
    ggml_backend_buffer_ptr weights_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(), ggml_backend_cpu_buffer_type()));
    require(weights && weights_buffer, "failed to allocate trainable MUL_MAT_ID weights");

    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_dim, n_slots, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * upstream = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, output_dim, n_used, n_tokens);
    require(input && ids && upstream, "failed to allocate trainable MUL_MAT_ID test tensors");

    float * weight_data = static_cast<float *>(weights->data);
    float * input_data = static_cast<float *>(input->data);
    float * upstream_data = static_cast<float *>(upstream->data);
    int32_t * ids_data = static_cast<int32_t *>(ids->data);
    for (int64_t expert = 0; expert < n_experts; ++expert) {
        for (int64_t row = 0; row < output_dim; ++row) {
            for (int64_t col = 0; col < input_dim; ++col) {
                weight_data[expert * input_dim * output_dim + row * input_dim + col] =
                    std::sin(static_cast<float>((expert + 1) * (row + 2) * (col + 1)) * 0.09f);
            }
        }
    }
    for (int64_t i = 0; i < ggml_nelements(input); ++i) {
        input_data[i] = std::cos(static_cast<float>(i + 1) * 0.21f);
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            ids_data[token * n_used + used] = static_cast<int32_t>((used + token) % n_experts);
            for (int64_t row = 0; row < output_dim; ++row) {
                upstream_data[token * n_used * output_dim + used * output_dim + row] =
                    std::sin(static_cast<float>(1 + row + 2 * used + 4 * token) * 0.16f);
            }
        }
    }

    ggml_set_param(weights);
    ggml_set_param(input);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_tensor * loss = ggml_sum(ctx.get(), ggml_mul(ctx.get(), output, upstream));
    ggml_set_loss(loss);

    ggml_cgraph * forward = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(forward, loss);
    ggml_cgraph * backward = ggml_graph_dup(ctx.get(), forward, true);
    ggml_build_backward_expand(ctx.get(), backward, nullptr);
    (void) evaluate_loss(ctx.get(), forward, loss);
    ggml_graph_reset(backward);
    compute_graph(ctx.get(), backward);

    const ggml_tensor * gradient = ggml_graph_get_grad(backward, weights);
    require(gradient != nullptr, "trainable MUL_MAT_ID did not produce expert weight gradients");
    require(ggml_are_same_shape(gradient, weights), "broadcast MUL_MAT_ID weight gradient has the wrong shape");
    const std::vector<float> actual = copy_tensor(gradient);

    std::vector<float> expected(static_cast<size_t>(input_dim * output_dim * n_experts), 0.0f);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            const int64_t expert = ids_data[token * n_used + used];
            for (int64_t row = 0; row < output_dim; ++row) {
                const float g = upstream_data[token * n_used * output_dim + used * output_dim + row];
                for (int64_t col = 0; col < input_dim; ++col) {
                    expected[static_cast<size_t>(expert * input_dim * output_dim + row * input_dim + col)] +=
                        input_data[token * input_dim + col] * g;
                }
            }
        }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(actual[i]), "trainable MUL_MAT_ID weight gradient is not finite");
        require(std::fabs(actual[i] - expected[i]) <= 2e-5f,
            "broadcast MUL_MAT_ID weight gradient mismatch at index " + std::to_string(i) +
            ": actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
    }
}


} // namespace

void run_training_math_quantized_test() {
    test_quantized_matmul_activation_gradient();
    test_quantized_mul_mat_id_activation_gradient_with_slots(1);
    test_quantized_mul_mat_id_activation_gradient_with_slots(2);
    test_trainable_mul_mat_id_weight_gradient_with_broadcast();
}

} // namespace qlora_training_math
