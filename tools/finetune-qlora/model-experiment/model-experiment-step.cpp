#include "model-experiment-internal.h"

#include <algorithm>
#include <cmath>

namespace qlora_model_experiment {

StepResult Experiment::train_one_step(float learning_rate) {
    detail::require(static_cast<bool>(impl_->train_ctx), "this experiment has no training context");
    detail::require(std::isfinite(learning_rate) && learning_rate > 0.0f, "learning rate must be positive and finite");

    std::vector<std::vector<float>> initial_a;
    std::vector<std::vector<float>> initial_b;
    initial_a.reserve(impl_->pairs.size());
    initial_b.reserve(impl_->pairs.size());
    for (const auto & pair : impl_->pairs) {
        initial_a.push_back(detail::tensor_to_vector(pair.a));
        initial_b.push_back(detail::tensor_to_vector(pair.b));
    }

    impl_->optimizer_params.sgd.alpha = learning_rate;
    std::unique_ptr<ggml_opt_result, decltype(&ggml_opt_result_free)> result(
        ggml_opt_result_init(), ggml_opt_result_free);
    detail::require(static_cast<bool>(result), "failed to allocate optimizer result");
    llama_opt_epoch(impl_->train_ctx.get(), impl_->model->dataset.get(), result.get(), nullptr, 1, nullptr, nullptr);

    StepResult step;
    step.learning_rate = learning_rate;
    double uncertainty = 0.0;
    ggml_opt_result_loss(result.get(), &step.training_loss, &uncertainty);
    detail::require(std::isfinite(step.training_loss), "training graph loss is not finite");

    step.pair_names.reserve(impl_->pairs.size());
    step.initial_a.reserve(impl_->pairs.size());
    step.initial_b.reserve(impl_->pairs.size());
    step.final_a.reserve(impl_->pairs.size());
    step.final_b.reserve(impl_->pairs.size());
    for (size_t pair_index = 0; pair_index < impl_->pairs.size(); ++pair_index) {
        const auto & pair = impl_->pairs[pair_index];
        step.pair_names.push_back(pair.name);
        step.initial_a.push_back(initial_a[pair_index]);
        step.initial_b.push_back(initial_b[pair_index]);
        step.final_a.push_back(detail::tensor_to_vector(pair.a));
        step.final_b.push_back(detail::tensor_to_vector(pair.b));

        for (size_t i = 0; i < step.final_a.back().size(); ++i) {
            step.max_a_update = std::max(step.max_a_update,
                std::fabs(step.final_a.back()[i] - step.initial_a.back()[i]));
        }
        for (size_t i = 0; i < step.final_b.back().size(); ++i) {
            const float update = step.final_b.back()[i] - step.initial_b.back()[i];
            step.squared_parameter_update += static_cast<double>(update) * update;
            const float magnitude = std::fabs(update);
            if (magnitude > step.max_b_update) {
                step.max_b_update = magnitude;
                step.selected_pair = pair_index;
                step.selected_index = i;
            }
        }
    }
    for (size_t pair_index = 0; pair_index < impl_->pairs.size(); ++pair_index) {
        for (size_t i = 0; i < step.final_a[pair_index].size(); ++i) {
            const double update = static_cast<double>(step.final_a[pair_index][i]) - step.initial_a[pair_index][i];
            step.squared_parameter_update += update * update;
        }
    }
    return step;
}

float Experiment::b_parameter(size_t pair_index, size_t tensor_index) {
    detail::require(pair_index < impl_->pairs.size(), "LoRA pair index is out of range");
    const auto values = detail::tensor_to_vector(impl_->pairs[pair_index].b);
    detail::require(tensor_index < values.size(), "LoRA parameter index is out of range");
    return values[tensor_index];
}

void Experiment::set_b_parameter(size_t pair_index, size_t tensor_index, float value) {
    detail::require(pair_index < impl_->pairs.size(), "LoRA pair index is out of range");
    auto values = detail::tensor_to_vector(impl_->pairs[pair_index].b);
    detail::require(tensor_index < values.size(), "LoRA parameter index is out of range");
    values[tensor_index] = value;
    detail::vector_to_tensor(impl_->pairs[pair_index].b, values);
}

void Experiment::set_parameters_at_update_scale(const StepResult & step, float scale) {
    detail::require(step.final_a.size() == impl_->pairs.size() && step.final_b.size() == impl_->pairs.size() &&
                    step.initial_a.size() == impl_->pairs.size() && step.initial_b.size() == impl_->pairs.size(),
                    "step result does not match this model");
    detail::require(std::isfinite(scale), "parameter update scale must be finite");
    for (size_t pair_index = 0; pair_index < impl_->pairs.size(); ++pair_index) {
        const auto & pair = impl_->pairs[pair_index];
        detail::require(step.initial_a[pair_index].size() == pair.initial_a.size() &&
                        step.initial_b[pair_index].size() == pair.initial_b.size(),
                        "step result tensor shape mismatch");
        std::vector<float> scaled_a(pair.initial_a.size());
        std::vector<float> scaled_b(pair.initial_b.size());
        for (size_t i = 0; i < scaled_a.size(); ++i) {
            scaled_a[i] = step.initial_a[pair_index][i] +
                scale * (step.final_a[pair_index][i] - step.initial_a[pair_index][i]);
        }
        for (size_t i = 0; i < scaled_b.size(); ++i) {
            scaled_b[i] = step.initial_b[pair_index][i] +
                scale * (step.final_b[pair_index][i] - step.initial_b[pair_index][i]);
        }
        detail::vector_to_tensor(pair.a, scaled_a);
        detail::vector_to_tensor(pair.b, scaled_b);
    }
}



} // namespace qlora_model_experiment
