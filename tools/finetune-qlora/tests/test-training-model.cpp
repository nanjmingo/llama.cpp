#include "test-training-model-regression.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace qlora_model_experiment;
using namespace qlora_model_test;

namespace {

ModelOptions parse_regression_options(int argc, char ** argv) {
    ModelOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            options.model_path = argv[++i];
        } else if (arg == "--verbose-logs") {
            options.verbose_library_logs = true;
        } else {
            throw std::runtime_error(std::string("usage: ") + argv[0] +
                " --model MODEL.gguf [--verbose-logs]; regression checks use F32 activations and F32 KV cache");
        }
    }
    if (options.model_path.empty()) {
        throw std::runtime_error(std::string("usage: ") + argv[0] +
            " --model MODEL.gguf [--verbose-logs]; regression checks use F32 activations and F32 KV cache");
    }
    return options;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Model model(parse_regression_options(argc, argv));
        bool passed = run_forward_checks(model);

        bool gradient_reference_passed = false;
        const GradientReference gradient_reference = run_gradient_reference(model, gradient_reference_passed);
        passed = gradient_reference_passed && passed;

        bool step_passed = false;
        const StepResult step = run_training_step_check(model, step_passed);
        passed = step_passed && passed;
        if (gradient_reference_passed && step_passed) {
            passed = run_update_gradient_check(gradient_reference, step) && passed;
        }

        if (!passed) {
            std::cerr << "[FAIL] model-backed QLoRA regression checks failed\n";
            return 1;
        }
        std::cout << "[PASS] model-backed QLoRA regression checks\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
