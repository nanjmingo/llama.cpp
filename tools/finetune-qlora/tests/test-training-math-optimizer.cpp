#include "test-training-math-common.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace qlora_training_math {
namespace {

ggml_opt_optimizer_params test_sgd_optimizer_params(void *) {
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.sgd.alpha = 1e-2f;
    params.sgd.wd = 0.0f;
    return params;
}

ggml_opt_optimizer_params test_adamw_optimizer_params(void *) {
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.adamw.alpha = 1e-2f;
    params.adamw.beta1 = 0.9f;
    params.adamw.beta2 = 0.99f;
    params.adamw.eps = 1e-8f;
    params.adamw.wd = 0.01f;
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
                                                                  bool dynamic_graphs = false,
                                                                  ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_SGD,
                                                                  ggml_type optimizer_state_type = GGML_TYPE_F32) {
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
    ggml_set_name(fixture->a, "test.lora_a");
    ggml_set_name(fixture->b, "test.lora_b");
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
    opt_params.optimizer = optimizer;
    opt_params.optimizer_state_type = optimizer_state_type;
    opt_params.get_opt_pars = optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW
        ? test_adamw_optimizer_params
        : test_sgd_optimizer_params;
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
    require(ggml_opt_has_grad(fixture.opt, fixture.a), "ggml-opt did not detect the LoRA A gradient path");
    require(ggml_opt_has_grad(fixture.opt, fixture.b), "ggml-opt did not detect the LoRA B gradient path");
    ggml_backend_tensor_set(fixture.inputs, input_values.data(), 0, input_values.size() * sizeof(float));
    ggml_tensor * labels = ggml_opt_labels(fixture.opt);
    require(labels && labels_values.size() == static_cast<size_t>(ggml_nelements(labels)),
            "ggml-opt microbatch label size mismatch");
    ggml_backend_tensor_set(labels, labels_values.data(), 0, labels_values.size() * sizeof(float));
    ggml_opt_eval(fixture.opt, fixture.result);
}

void test_ggml_opt_detects_missing_gradient_path() {
    constexpr int64_t physical_positions = 2;
    const std::vector<float> a_values = { 0.2f, -0.1f, 0.3f, 0.4f };
    const std::vector<float> b_values = { -0.2f, 0.1f, 0.3f, 0.4f, -0.1f, 0.2f };
    auto fixture = make_ggml_opt_lora_fixture(
        physical_positions, 1, a_values, b_values, 0.75f, true);

    ggml_reset(fixture->compute_ctx.get());
    ggml_tensor * hidden = ggml_step(fixture->compute_ctx.get(), fixture->a);
    fixture->logits = ggml_mul_mat(fixture->compute_ctx.get(), fixture->b, hidden);
    ggml_cgraph * forward = ggml_new_graph_custom(fixture->compute_ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(forward, fixture->logits);
    ggml_opt_prepare_alloc(
        fixture->opt, fixture->compute_ctx.get(), forward, fixture->inputs, fixture->logits);
    ggml_opt_alloc(fixture->opt, true);

    require(ggml_opt_grad_acc(fixture->opt, fixture->a) != nullptr,
            "ggml-opt did not allocate the disconnected parameter gradient accumulator");
    require(!ggml_opt_has_grad(fixture->opt, fixture->a),
            "ggml-opt reported a gradient path through the non-differentiable STEP op");
    require(ggml_opt_has_grad(fixture->opt, fixture->b),
            "ggml-opt did not detect the connected LoRA B gradient path");
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

void test_dynamic_ggml_opt_supports_partial_accumulation_group() {
    constexpr int64_t physical_positions = 2;
    constexpr int64_t input_dim = 2;
    constexpr int64_t classes = 3;
    constexpr float alpha_over_rank = 0.75f;
    constexpr float learning_rate = 1e-2f;
    constexpr int32_t full_period = 3;
    constexpr int32_t partial_period = 2;
    const std::vector<float> a_values = { -0.3f, 0.1f, 0.4f, -0.2f };
    const std::vector<float> b_values = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f };

    std::vector<float> all_inputs(input_dim * physical_positions * (full_period + partial_period));
    std::vector<float> all_labels(classes * physical_positions * (full_period + partial_period), 0.0f);
    for (size_t i = 0; i < all_inputs.size(); ++i) {
        all_inputs[i] = 0.1f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    const size_t position_count = all_labels.size() / classes;
    for (size_t position = 0; position < position_count; ++position) {
        all_labels[position * classes + position % classes] = 1.0f;
    }

    std::unique_ptr<ggml_opt_lora_fixture> fixture = make_ggml_opt_lora_fixture(
        physical_positions, full_period, a_values, b_values, alpha_over_rank, true);
    require(!ggml_opt_static_graphs(fixture->opt), "partial accumulation fixture unexpectedly uses static graphs");

    const auto slice = [](const std::vector<float> & values, size_t offset, size_t count) {
        return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                                  values.begin() + static_cast<std::ptrdiff_t>(offset + count));
    };
    std::vector<float> expected_window_losses;

    auto run_group = [&](int32_t period, size_t first_window, int32_t window_count) {
        const std::vector<float> a_before = get_backend_tensor_values(fixture->a);
        const std::vector<float> b_before = get_backend_tensor_values(fixture->b);
        const size_t input_stride = static_cast<size_t>(input_dim * physical_positions);
        const size_t label_stride = static_cast<size_t>(classes * physical_positions);
        const size_t first_input = first_window * input_stride;
        const size_t first_label = first_window * label_stride;
        const size_t group_inputs_count = static_cast<size_t>(window_count) * input_stride;
        const size_t group_labels_count = static_cast<size_t>(window_count) * label_stride;
        const std::vector<float> group_inputs = slice(all_inputs, first_input, group_inputs_count);
        const std::vector<float> group_labels = slice(all_labels, first_label, group_labels_count);
        const std::vector<float> base_logits(group_labels.size(), 0.0f);

        lora_batch_graph reference = make_lora_batch_graph(
            group_inputs, group_labels, base_logits, physical_positions * window_count,
            a_before, b_before, alpha_over_rank);
        const float expected_group_loss = evaluate_loss(reference.ctx.get(), reference.forward, reference.loss);
        ggml_graph_reset(reference.backward);
        compute_graph(reference.ctx.get(), reference.backward);
        const std::vector<float> gradient_a = copy_tensor(ggml_graph_get_grad(reference.backward, reference.a));
        const std::vector<float> gradient_b = copy_tensor(ggml_graph_get_grad(reference.backward, reference.b));

        for (int32_t i = 0; i < window_count; ++i) {
            const size_t window_offset = (first_window + static_cast<size_t>(i)) * input_stride;
            const size_t label_offset = (first_window + static_cast<size_t>(i)) * label_stride;
            const std::vector<float> window_inputs = slice(all_inputs, window_offset, input_stride);
            const std::vector<float> window_labels = slice(all_labels, label_offset, label_stride);
            lora_batch_graph window = make_lora_batch_graph(
                window_inputs, window_labels, std::vector<float>(label_stride, 0.0f), physical_positions,
                a_before, b_before, alpha_over_rank);
            expected_window_losses.push_back(evaluate_loss(window.ctx.get(), window.forward, window.loss));
        }

        ggml_opt_set_opt_period(fixture->opt, period);
        for (int32_t i = 0; i < window_count; ++i) {
            const size_t window_offset = (first_window + static_cast<size_t>(i)) * input_stride;
            const size_t label_offset = (first_window + static_cast<size_t>(i)) * label_stride;
            run_ggml_opt_microbatch(*fixture,
                slice(all_inputs, window_offset, input_stride),
                slice(all_labels, label_offset, label_stride), alpha_over_rank);
        }

        const std::vector<float> a_after = get_backend_tensor_values(fixture->a);
        const std::vector<float> b_after = get_backend_tensor_values(fixture->b);
        for (size_t i = 0; i < a_before.size(); ++i) {
            require(std::fabs(a_after[i] - (a_before[i] - learning_rate * gradient_a[i])) < 3e-5f,
                    "partial accumulation A update differs from the full-group reference gradient");
        }
        for (size_t i = 0; i < b_before.size(); ++i) {
            require(std::fabs(b_after[i] - (b_before[i] - learning_rate * gradient_b[i])) < 3e-5f,
                    "partial accumulation B update differs from the full-group reference gradient");
        }
        require(std::isfinite(expected_group_loss), "partial accumulation reference loss is not finite");
    };

    run_group(full_period, 0, full_period);
    run_group(partial_period, full_period, partial_period);

    double actual_loss = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(fixture->result, &actual_loss, &uncertainty);
    double expected_loss = 0.0;
    for (float loss : expected_window_losses) {
        expected_loss += loss;
    }
    expected_loss /= expected_window_losses.size();
    require(std::isfinite(actual_loss) && std::fabs(actual_loss - expected_loss) < 3e-5,
            "loss reporting must support full and partial accumulation periods in one epoch");

    double last_loss = 0.0;
    ggml_opt_result_last_loss(fixture->result, &last_loss);
    require(std::isfinite(last_loss) && std::fabs(last_loss - expected_window_losses.back()) < 3e-5,
            "last loss reporting must restore the unscaled loss for the most recent partial accumulation window");
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


void test_adamw_optimizer_state_roundtrip_matches_continuous_training() {
    constexpr int64_t physical_positions = 2;
    constexpr float alpha_over_rank = 0.75f;
    const std::vector<float> a_initial = { -0.3f, 0.1f, 0.4f, -0.2f };
    const std::vector<float> b_initial = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.2f };
    const std::vector<float> inputs_first = { 0.5f, -1.0f, 1.5f, 0.25f };
    const std::vector<float> labels_first = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    const std::vector<float> inputs_second = { -0.2f, 0.7f, 0.8f, -0.4f };
    const std::vector<float> labels_second = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f };
    const std::vector<ggml_type> state_types = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    };

    for (const ggml_type state_type : state_types) {
        std::unique_ptr<ggml_opt_lora_fixture> continuous = make_ggml_opt_lora_fixture(
            physical_positions, 1, a_initial, b_initial, alpha_over_rank, true,
            GGML_OPT_OPTIMIZER_TYPE_ADAMW, state_type);
        run_ggml_opt_microbatch(*continuous, inputs_first, labels_first, alpha_over_rank);

        const size_t state_size = ggml_opt_state_size(continuous->opt);
        require(state_size > 0, "AdamW optimizer state export reported an empty state");
        std::vector<uint8_t> optimizer_state(state_size);
        require(ggml_opt_state_export(continuous->opt, optimizer_state.data(), optimizer_state.size()),
                "failed to export AdamW optimizer state");
        std::vector<uint8_t> truncated_state(optimizer_state.begin(), optimizer_state.end() - 1);
        require(!ggml_opt_state_import(continuous->opt, truncated_state.data(), truncated_state.size()),
                "truncated AdamW optimizer state was accepted");
        const std::vector<float> a_after_first = get_backend_tensor_values(continuous->a);
        const std::vector<float> b_after_first = get_backend_tensor_values(continuous->b);

        run_ggml_opt_microbatch(*continuous, inputs_second, labels_second, alpha_over_rank);
        const std::vector<float> a_continuous = get_backend_tensor_values(continuous->a);
        const std::vector<float> b_continuous = get_backend_tensor_values(continuous->b);

        std::unique_ptr<ggml_opt_lora_fixture> resumed = make_ggml_opt_lora_fixture(
            physical_positions, 1, a_after_first, b_after_first, alpha_over_rank, true,
            GGML_OPT_OPTIMIZER_TYPE_ADAMW, state_type);
        require(ggml_opt_state_import(resumed->opt, optimizer_state.data(), optimizer_state.size()),
                "failed to import AdamW optimizer state with type " + std::string(ggml_type_name(state_type)));
        run_ggml_opt_microbatch(*resumed, inputs_second, labels_second, alpha_over_rank);
        const std::vector<float> a_resumed = get_backend_tensor_values(resumed->a);
        const std::vector<float> b_resumed = get_backend_tensor_values(resumed->b);

        require(a_continuous.size() == a_resumed.size() && b_continuous.size() == b_resumed.size(),
                "optimizer-state roundtrip changed LoRA tensor dimensions");
        for (size_t i = 0; i < a_continuous.size(); ++i) {
            require(std::fabs(a_continuous[i] - a_resumed[i]) < 1e-6f,
                    "resumed AdamW A update differs after state type " + std::string(ggml_type_name(state_type)));
        }
        for (size_t i = 0; i < b_continuous.size(); ++i) {
            require(std::fabs(b_continuous[i] - b_resumed[i]) < 1e-6f,
                    "resumed AdamW B update differs after state type " + std::string(ggml_type_name(state_type)));
        }
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

std::vector<float> roundtrip_type(ggml_type type, const std::vector<float> & values) {
    if (type == GGML_TYPE_F32) {
        return values;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    require(traits && traits->to_float && traits->from_float_ref, "optimizer test type lacks float conversions");
    const size_t bytes = ggml_row_size(type, static_cast<int64_t>(values.size()));
    std::vector<uint8_t> raw(bytes);
    traits->from_float_ref(values.data(), raw.data(), static_cast<int64_t>(values.size()));
    std::vector<float> result(values.size());
    traits->to_float(raw.data(), result.data(), static_cast<int64_t>(values.size()));
    return result;
}

std::vector<float> read_tensor_values(const ggml_tensor * tensor) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(result.data(), tensor->data, result.size() * sizeof(float));
        return result;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    require(traits && traits->to_float, "optimizer tensor type lacks float conversion");
    traits->to_float(tensor->data, result.data(), ggml_nelements(tensor));
    return result;
}

void write_tensor_values(ggml_tensor * tensor, const std::vector<float> & values) {
    require(values.size() == static_cast<size_t>(ggml_nelements(tensor)), "optimizer test tensor size mismatch");
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    require(traits && traits->from_float_ref, "optimizer tensor type lacks float encoding");
    traits->from_float_ref(values.data(), tensor->data, ggml_nelements(tensor));
}

void test_adamw_supports_parameter_and_state_types() {
    constexpr int64_t n = 32;
    constexpr float lr = 1e-3f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.99f;
    constexpr float eps = 1e-6f;
    constexpr float wd = 0.01f;
    const std::vector<ggml_type> supported_types = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    };

    for (ggml_type weight_type : supported_types) {
        for (ggml_type state_type : supported_types) {
            ggml_context_ptr ctx = make_context();
            require(static_cast<bool>(ctx), "failed to allocate typed AdamW test context");
            ggml_tensor * weights = ggml_new_tensor_1d(ctx.get(), weight_type, n);
            ggml_tensor * gradients = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
            ggml_tensor * m = ggml_new_tensor_1d(ctx.get(), state_type, n);
            ggml_tensor * v = ggml_new_tensor_1d(ctx.get(), state_type, n);
            ggml_tensor * params = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 7);
            require(weights && gradients && m && v && params, "failed to allocate typed AdamW test tensors");

            std::vector<float> expected_w(static_cast<size_t>(n));
            std::vector<float> grad(static_cast<size_t>(n));
            std::vector<float> zero(static_cast<size_t>(n), 0.0f);
            for (int64_t i = 0; i < n; ++i) {
                expected_w[static_cast<size_t>(i)] = 0.4f * std::sin(static_cast<float>(i + 1) * 0.17f);
                grad[static_cast<size_t>(i)] = 0.3f + 0.02f * static_cast<float>(i % 13);
            }
            expected_w = roundtrip_type(weight_type, expected_w);
            write_tensor_values(weights, expected_w);
            write_tensor_values(m, zero);
            write_tensor_values(v, zero);
            std::copy(grad.begin(), grad.end(), static_cast<float *>(gradients->data));
            ggml_set_param(weights);
            ggml_tensor * updated = ggml_opt_step_adamw(ctx.get(), weights, gradients, m, v, params);
            ggml_cgraph * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, updated);

            std::vector<float> expected_m(static_cast<size_t>(n), 0.0f);
            std::vector<float> expected_v(static_cast<size_t>(n), 0.0f);
            for (int step = 1; step <= 2; ++step) {
                const float params_values[] = {
                    lr, beta1, beta2, eps, wd,
                    1.0f / (1.0f - std::pow(beta1, static_cast<float>(step))),
                    1.0f / (1.0f - std::pow(beta2, static_cast<float>(step))),
                };
                std::copy(std::begin(params_values), std::end(params_values), static_cast<float *>(params->data));

                const std::vector<float> old_w = read_tensor_values(weights);
                const std::vector<float> old_m = read_tensor_values(m);
                const std::vector<float> old_v = read_tensor_values(v);
                for (int64_t i = 0; i < n; ++i) {
                    const size_t index = static_cast<size_t>(i);
                    expected_m[index] = old_m[index] * beta1 + grad[index] * (1.0f - beta1);
                    expected_v[index] = old_v[index] * beta2 + grad[index] * grad[index] * (1.0f - beta2);
                    expected_w[index] = old_w[index] * (1.0f - lr * wd) - lr * expected_m[index] * params_values[5] /
                        (std::sqrt(expected_v[index] * params_values[6]) + eps);
                }
                const std::vector<float> encoded_m = roundtrip_type(state_type, expected_m);
                const std::vector<float> encoded_v = roundtrip_type(state_type, expected_v);
                const std::vector<float> encoded_w = roundtrip_type(weight_type, expected_w);
                compute_graph(ctx.get(), graph);

                const std::vector<float> actual_w = read_tensor_values(weights);
                const std::vector<float> actual_m = read_tensor_values(m);
                const std::vector<float> actual_v = read_tensor_values(v);
                const std::string combination = std::string("AdamW failed for weight ") + ggml_type_name(weight_type) +
                    " and state " + ggml_type_name(state_type);
                for (size_t i = 0; i < encoded_w.size(); ++i) {
                    require(std::isfinite(actual_w[i]) && std::fabs(actual_w[i] - encoded_w[i]) < 1e-5f,
                            combination + ": weight update disagrees with the converted reference");
                    require(std::isfinite(actual_m[i]) && std::fabs(actual_m[i] - encoded_m[i]) < 1e-5f,
                            combination + ": first moment disagrees with the converted reference");
                    require(std::isfinite(actual_v[i]) && std::fabs(actual_v[i] - encoded_v[i]) < 1e-5f,
                            combination + ": second moment disagrees with the converted reference");
                }
            }
        }
    }
}


} // namespace

void run_training_math_ggml_opt_tests() {
    test_ggml_opt_cross_entropy_gradient_accumulation();
    test_ggml_opt_detects_missing_gradient_path();
    test_dynamic_ggml_opt_resets_gradients_between_optimizer_steps();
    test_dynamic_ggml_opt_supports_partial_accumulation_group();
}

void run_training_math_adamw_test() {
    test_adamw_first_update_matches_reference();
    test_adamw_optimizer_state_roundtrip_matches_continuous_training();
    test_adamw_supports_parameter_and_state_types();
}

} // namespace qlora_training_math
