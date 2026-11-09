#pragma once

#include "../model-experiment/model-experiment.h"

#include <vector>

namespace qlora_model_eval {

struct Options {
    qlora_model_experiment::ModelOptions model;
    qlora_model_experiment::ModelPrecisionConfig precision;
    std::vector<float> learning_rates = { 1.0e-2f, 1.0e-3f, 1.0e-4f, 1.0e-5f };
    int epochs = 10;
};

Options parse_options(int argc, char ** argv);

} // namespace qlora_model_eval
