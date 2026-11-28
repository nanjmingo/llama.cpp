#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct training_result {
    float parameter = NAN;
    bool has_parameter_gradient = false;
};

static ggml_opt_optimizer_params get_test_optimizer_params(void *) {
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.adamw.alpha = 1e-2f;
    params.adamw.beta1 = 0.9f;
    params.adamw.beta2 = 0.999f;
    params.adamw.eps = 1e-8f;
    params.adamw.wd = 0.0f;
    return params;
}

static training_result run_dynamic_training(
        ggml_backend_sched_t backend_sched,
        ggml_backend_t backend,
        bool activation_checkpointing,
        const std::vector<int32_t> & accumulation_groups) {
    if (accumulation_groups.empty()) {
        return {};
    }

    ggml_init_params params_model = {
        /*.mem_size   =*/ 4 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_model = ggml_init(params_model);
    if (!ctx_model) {
        return {};
    }

    ggml_tensor * parameter = ggml_new_tensor_1d(ctx_model, GGML_TYPE_F32, 1);
    ggml_set_name(parameter, "dynamic_test_parameter");
    ggml_set_param(parameter);

    ggml_tensor * input = ggml_new_tensor_1d(ctx_model, GGML_TYPE_F32, 1);
    ggml_set_name(input, "dynamic_test_input");
    ggml_set_input(input);

    ggml_backend_buffer_t model_buffer = ggml_backend_alloc_ctx_tensors(ctx_model, backend);
    if (!model_buffer) {
        ggml_free(ctx_model);
        return {};
    }

    const float initial_parameter = 1.0f;
    ggml_backend_tensor_set(parameter, &initial_parameter, 0, sizeof(initial_parameter));

    ggml_opt_params opt_params = ggml_opt_default_params(backend_sched, GGML_OPT_LOSS_TYPE_MEAN);
    opt_params.opt_period = accumulation_groups.front();
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    opt_params.get_opt_pars = get_test_optimizer_params;

    ggml_opt_context_t opt_ctx = ggml_opt_init(opt_params);
    if (!opt_ctx) {
        ggml_backend_buffer_free(model_buffer);
        ggml_free(ctx_model);
        return {};
    }
    ggml_opt_set_activation_checkpointing(opt_ctx, activation_checkpointing);

    // Keep the compute contexts alive until the optimizer is freed. Dynamic
    // graph allocation resets the preceding graph at the beginning of a group.
    std::vector<ggml_context *> compute_contexts;
    bool has_parameter_gradient = true;
    for (size_t igroup = 0; igroup < accumulation_groups.size(); ++igroup) {
        const int32_t period = accumulation_groups[igroup];
        if (period < 1) {
            ggml_opt_free(opt_ctx);
            for (ggml_context * ctx : compute_contexts) {
                ggml_free(ctx);
            }
            ggml_backend_buffer_free(model_buffer);
            ggml_free(ctx_model);
            return {};
        }
        if (igroup > 0) {
            ggml_opt_set_opt_period(opt_ctx, period);
        }

        for (int32_t i = 0; i < period; ++i) {
            ggml_init_params params_compute = {
                /*.mem_size   =*/ 4 * 1024 * 1024,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_compute = ggml_init(params_compute);
            if (!ctx_compute) {
                ggml_opt_free(opt_ctx);
                for (ggml_context * ctx : compute_contexts) {
                    ggml_free(ctx);
                }
                ggml_backend_buffer_free(model_buffer);
                ggml_free(ctx_model);
                return {};
            }
            compute_contexts.push_back(ctx_compute);

            ggml_tensor * output = ggml_mul(ctx_compute, parameter, input);
            ggml_set_name(output, "dynamic_test_output");

            ggml_cgraph * graph = ggml_new_graph_custom(ctx_compute, 512, /*grads =*/ true);
            ggml_build_forward_expand(graph, output);

            ggml_opt_prepare_alloc(opt_ctx, ctx_compute, graph, input, output);
            ggml_opt_alloc(opt_ctx, /*backward =*/ true);
            has_parameter_gradient = has_parameter_gradient && ggml_opt_has_grad(opt_ctx, parameter);

            const float input_value = 1.0f;
            ggml_backend_tensor_set(input, &input_value, 0, sizeof(input_value));
            ggml_opt_eval(opt_ctx, nullptr);
        }
    }

    float final_parameter = NAN;
    ggml_backend_tensor_get(parameter, &final_parameter, 0, sizeof(final_parameter));

    ggml_opt_free(opt_ctx);
    for (ggml_context * ctx : compute_contexts) {
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(model_buffer);
    ggml_free(ctx_model);

    return { final_parameter, has_parameter_gradient };
}

static bool test_graph_reset_preserves_adamw_momenta() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * parameter = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_param(parameter);
    ggml_tensor * gradient = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * moment1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * moment2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * optimizer_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);

    ggml_get_data_f32(moment1)[0] = 0.25f;
    ggml_get_data_f32(moment2)[0] = 0.75f;

    ggml_tensor * step = ggml_opt_step_adamw(
        ctx, parameter, gradient, moment1, moment2, optimizer_params);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, /*grads =*/ true);
    ggml_build_forward_expand(graph, step);

    ggml_graph_reset(graph);

    const bool preserved =
        ggml_get_data_f32(moment1)[0] == 0.25f &&
        ggml_get_data_f32(moment2)[0] == 0.75f;
    ggml_free(ctx);
    return preserved;
}

static bool almost_equal(float a, float b, float tolerance = 1e-6f) {
    return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tolerance;
}

static void report(const char * name, bool ok, int & passed, int & total) {
    std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    passed += ok ? 1 : 0;
    ++total;
}

} // namespace

int main() {
    ggml_log_set(nullptr, nullptr);
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize CPU backend\n");
        return 1;
    }

    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, /*parallel =*/ false, /*op_offload =*/ true);
    if (!backend_sched) {
        ggml_backend_free(backend);
        std::fprintf(stderr, "failed to initialize backend scheduler\n");
        return 1;
    }

    int passed = 0;
    int total = 0;

    report("graph_reset_preserves_adamw_momenta",
        test_graph_reset_preserves_adamw_momenta(), passed, total);

    const std::vector<int32_t> period_one = { 1, 1, 1, 1 };
    const std::vector<int32_t> period_two = { 2, 2 };
    const std::vector<int32_t> period_two_with_tail = { 2, 1 };

    const training_result ac_on_period_one = run_dynamic_training(
        backend_sched, backend, true, period_one);
    const training_result ac_off_period_one = run_dynamic_training(
        backend_sched, backend, false, period_one);
    report("dynamic_ac_off_period_1_matches_ac_on_and_updates_parameter",
        std::isfinite(ac_off_period_one.parameter) &&
        ac_off_period_one.parameter < 1.0f &&
        ac_off_period_one.has_parameter_gradient && ac_on_period_one.has_parameter_gradient &&
        almost_equal(ac_off_period_one.parameter, ac_on_period_one.parameter), passed, total);

    const training_result ac_on_period_two = run_dynamic_training(
        backend_sched, backend, true, period_two);
    const training_result ac_off_period_two = run_dynamic_training(
        backend_sched, backend, false, period_two);
    report("dynamic_ac_off_period_2_matches_ac_on_and_updates_parameter",
        std::isfinite(ac_off_period_two.parameter) &&
        ac_off_period_two.parameter < 1.0f &&
        ac_off_period_two.has_parameter_gradient && ac_on_period_two.has_parameter_gradient &&
        almost_equal(ac_off_period_two.parameter, ac_on_period_two.parameter), passed, total);

    const training_result ac_on_tail = run_dynamic_training(
        backend_sched, backend, true, period_two_with_tail);
    const training_result ac_off_tail = run_dynamic_training(
        backend_sched, backend, false, period_two_with_tail);
    report("dynamic_ac_off_period_2_partial_tail_preserves_adamw_state",
        std::isfinite(ac_off_tail.parameter) &&
        ac_off_tail.has_parameter_gradient && ac_on_tail.has_parameter_gradient &&
        almost_equal(ac_off_tail.parameter, ac_on_tail.parameter), passed, total);

    report("period_1_and_period_2_have_different_update_cadence",
        std::isfinite(ac_on_period_one.parameter) && std::isfinite(ac_on_period_two.parameter) &&
        std::fabs(ac_on_period_one.parameter - ac_on_period_two.parameter) > 1e-3f,
        passed, total);

    ggml_backend_sched_free(backend_sched);
    ggml_backend_free(backend);

    std::printf("%d/%d dynamic optimizer tests passed\n", passed, total);
    return passed == total ? 0 : 1;
}
