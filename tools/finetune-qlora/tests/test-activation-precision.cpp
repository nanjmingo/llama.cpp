#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t input_dim = 64;
constexpr int64_t output_dim = 7;
constexpr int64_t batch_size = 5;

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float round_activation(float value, ggml_prec precision) {
    switch (precision) {
        case GGML_PREC_F32:
            return value;
        case GGML_PREC_F16:
            return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        case GGML_PREC_BF16:
            return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
        default:
            throw std::runtime_error("unsupported reference activation precision");
    }
}

std::vector<float> dequantize_q8_rows(const std::vector<uint8_t> & quantized, int64_t rows) {
    const ggml_type type = GGML_TYPE_Q8_0;
    const size_t row_size = ggml_row_size(type, input_dim);
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    require(traits && traits->to_float, "Q8_0 dequantization is unavailable");

    std::vector<float> result(static_cast<size_t>(rows * input_dim));
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(quantized.data() + static_cast<size_t>(row) * row_size,
                        result.data() + static_cast<size_t>(row * input_dim), input_dim);
    }
    return result;
}

std::vector<float> reference_matmul(const std::vector<float> & weights,
                                    const std::vector<float> & inputs,
                                    ggml_prec precision) {
    std::vector<float> outputs(static_cast<size_t>(output_dim * batch_size), 0.0f);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        for (int64_t row = 0; row < output_dim; ++row) {
            float sum = 0.0f;
            for (int64_t col = 0; col < input_dim; ++col) {
                const float activation = round_activation(
                    inputs[static_cast<size_t>(batch * input_dim + col)], precision);
                sum += weights[static_cast<size_t>(row * input_dim + col)] * activation;
            }
            outputs[static_cast<size_t>(batch * output_dim + row)] = sum;
        }
    }
    return outputs;
}

std::vector<float> reference_q8_matmul(const std::vector<float> & dequantized_weights,
                                       const std::vector<float> & dequantized_inputs) {
    std::vector<float> outputs(static_cast<size_t>(output_dim * batch_size), 0.0f);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        for (int64_t row = 0; row < output_dim; ++row) {
            float sum = 0.0f;
            for (int64_t col = 0; col < input_dim; ++col) {
                sum += dequantized_weights[static_cast<size_t>(row * input_dim + col)] *
                       dequantized_inputs[static_cast<size_t>(batch * input_dim + col)];
            }
            outputs[static_cast<size_t>(batch * output_dim + row)] = sum;
        }
    }
    return outputs;
}

std::vector<float> copy_tensor(const ggml_tensor * tensor) {
    const float * data = static_cast<const float *>(tensor->data);
    return std::vector<float>(data, data + ggml_nelements(tensor));
}

float max_abs_difference(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    require(lhs.size() == rhs.size(), "cannot compare vectors with different sizes");
    float maximum = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        maximum = std::max(maximum, std::fabs(lhs[i] - rhs[i]));
    }
    return maximum;
}

void check_matches_reference(const char * name,
                             const std::vector<float> & actual,
                             const std::vector<float> & expected) {
    require(actual.size() == expected.size(), std::string(name) + " output has an invalid size");
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 2e-4f + 2e-5f * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            std::ostringstream message;
            message << name << " output differs from the explicit reference at index " << i
                    << ": actual=" << actual[i] << " expected=" << expected[i]
                    << " tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
    }
}

void test_quantized_matmul_activation_precision() {
    context_ptr ctx(ggml_init({ 16 * 1024 * 1024, nullptr, false }), &ggml_free);
    require(static_cast<bool>(ctx), "failed to allocate activation precision test context");

    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, input_dim, output_dim);
    ggml_tensor * inputs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, input_dim, batch_size);
    require(weights && inputs, "failed to allocate activation precision test tensors");

    std::vector<float> source_weights(static_cast<size_t>(input_dim * output_dim));
    for (int64_t row = 0; row < output_dim; ++row) {
        for (int64_t col = 0; col < input_dim; ++col) {
            source_weights[static_cast<size_t>(row * input_dim + col)] =
                0.7f * std::sin(static_cast<float>((row + 1) * (col + 3)) * 0.17f) +
                0.4f * std::cos(static_cast<float>((row + 2) * (col + 1)) * 0.11f) +
                0.0017f * static_cast<float>((col % 13) - 6);
        }
    }

    const size_t weight_row_size = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    std::vector<uint8_t> quantized_weights(weight_row_size * static_cast<size_t>(output_dim));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, source_weights.data(), quantized_weights.data(),
                        0, output_dim, input_dim, nullptr);
    std::memcpy(weights->data, quantized_weights.data(), quantized_weights.size());

    std::vector<float> source_inputs(static_cast<size_t>(input_dim * batch_size));
    for (int64_t i = 0; i < static_cast<int64_t>(source_inputs.size()); ++i) {
        source_inputs[static_cast<size_t>(i)] =
            1.0001f * std::sin(static_cast<float>(i + 1) * 0.173f) +
            0.21314159f * std::cos(static_cast<float>(i + 3) * 0.071f) +
            0.00013f * static_cast<float>((i % 9) - 4);
    }
    std::memcpy(inputs->data, source_inputs.data(), source_inputs.size() * sizeof(float));

    ggml_tensor * output_default = ggml_mul_mat(ctx.get(), weights, inputs);
    ggml_tensor * output_q8 = ggml_mul_mat(ctx.get(), weights, inputs);
    ggml_tensor * output_f32 = ggml_mul_mat(ctx.get(), weights, inputs);
    ggml_tensor * output_f16 = ggml_mul_mat(ctx.get(), weights, inputs);
    ggml_tensor * output_bf16 = ggml_mul_mat(ctx.get(), weights, inputs);
    require(output_default && output_q8 && output_f32 && output_f16 && output_bf16,
            "failed to create activation precision matmul operations");
    require(ggml_prec_set_src(output_q8, GGML_PREC_Q8, 1), "failed to set Q8 activation precision");
    require(ggml_prec_set_src(output_f32, GGML_PREC_F32, 1), "failed to set F32 activation precision");
    require(ggml_prec_set_src(output_f16, GGML_PREC_F16, 1), "failed to set F16 activation precision");
    require(ggml_prec_set_src(output_bf16, GGML_PREC_BF16, 1), "failed to set BF16 activation precision");

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    require(graph != nullptr, "failed to allocate activation precision test graph");
    ggml_build_forward_expand(graph, output_default);
    ggml_build_forward_expand(graph, output_q8);
    ggml_build_forward_expand(graph, output_f32);
    ggml_build_forward_expand(graph, output_f16);
    ggml_build_forward_expand(graph, output_bf16);

    const ggml_status status = ggml_graph_compute_with_ctx(ctx.get(), graph, 3);
    require(status == GGML_STATUS_SUCCESS,
            std::string("activation precision graph execution failed: ") + ggml_status_to_string(status));

    const ggml_type_traits * weight_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    require(weight_traits && weight_traits->to_float, "Q8_0 weight dequantization is unavailable");
    std::vector<float> dequantized_weights(static_cast<size_t>(input_dim * output_dim));
    for (int64_t row = 0; row < output_dim; ++row) {
        weight_traits->to_float(quantized_weights.data() + static_cast<size_t>(row) * weight_row_size,
                                dequantized_weights.data() + static_cast<size_t>(row * input_dim), input_dim);
    }

    const std::vector<float> expected_f32 = reference_matmul(dequantized_weights, source_inputs, GGML_PREC_F32);
    const std::vector<float> expected_f16 = reference_matmul(dequantized_weights, source_inputs, GGML_PREC_F16);
    const std::vector<float> expected_bf16 = reference_matmul(dequantized_weights, source_inputs, GGML_PREC_BF16);

    const size_t input_row_size = ggml_row_size(GGML_TYPE_Q8_0, input_dim);
    std::vector<uint8_t> quantized_inputs(input_row_size * static_cast<size_t>(batch_size));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, source_inputs.data(), quantized_inputs.data(),
                        0, batch_size, input_dim, nullptr);
    const std::vector<float> dequantized_inputs = dequantize_q8_rows(quantized_inputs, batch_size);
    const std::vector<float> expected_q8 = reference_q8_matmul(dequantized_weights, dequantized_inputs);

    require(max_abs_difference(expected_f32, expected_f16) > 1e-4f,
            "test inputs do not distinguish F16 activation rounding from F32");
    require(max_abs_difference(expected_f32, expected_bf16) > 1e-3f,
            "test inputs do not distinguish BF16 activation rounding from F32");
    require(max_abs_difference(expected_f32, expected_q8) > 1e-3f,
            "test inputs do not distinguish Q8 activation quantization from F32");

    const std::vector<float> actual_default = copy_tensor(output_default);
    const std::vector<float> actual_q8 = copy_tensor(output_q8);
    const std::vector<float> actual_f32 = copy_tensor(output_f32);
    const std::vector<float> actual_f16 = copy_tensor(output_f16);
    const std::vector<float> actual_bf16 = copy_tensor(output_bf16);

    check_matches_reference("default Q8 activation", actual_default, expected_q8);
    check_matches_reference("explicit Q8 activation", actual_q8, expected_q8);
    check_matches_reference("F32 activation", actual_f32, expected_f32);
    check_matches_reference("F16 activation", actual_f16, expected_f16);
    check_matches_reference("BF16 activation", actual_bf16, expected_bf16);

    require(max_abs_difference(actual_default, actual_q8) <= 1e-6f,
            "default activation precision differs from explicit Q8 activation precision");
}

} // namespace

int main() {
    try {
        test_quantized_matmul_activation_precision();
        std::cout << "activation precision tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
