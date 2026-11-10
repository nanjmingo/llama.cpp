#pragma once

#include "../model-experiment/model-experiment.h"

namespace qlora_model_test {

struct GradientReference {
    std::string pair_name;
    size_t pair_index = 0;
    size_t tensor_index = 0;
    float center_parameter = 0.0f;
    float epsilon = 0.0f;
    double numeric_gradient = 0.0;
};

bool run_forward_checks(qlora_model_experiment::Model & model);
GradientReference run_gradient_reference(qlora_model_experiment::Model & model, bool & passed);
qlora_model_experiment::StepResult run_training_step_check(qlora_model_experiment::Model & model, bool & passed);
bool run_update_gradient_check(const GradientReference & reference,
                               const qlora_model_experiment::StepResult & step);

} // namespace qlora_model_test
