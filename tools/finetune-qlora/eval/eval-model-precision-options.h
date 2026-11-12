#pragma once

#include "eval-model-precision-model.h"

#include <vector>

namespace qlora_model_eval {

struct Options {
    ModelOptions model;
    ModelPrecisionConfig precision;
    std::vector<float> learning_rates = { 1.0e-2f, 1.0e-3f, 1.0e-4f, 1.0e-5f };
    int epochs = 10;
};

Options parse_options(int argc, char ** argv);

}
