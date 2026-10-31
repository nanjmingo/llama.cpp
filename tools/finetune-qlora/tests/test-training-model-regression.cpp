#include "test-training-model-regression.h"
#include "../model-experiment/model-experiment-internal.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace qlora_model_test {

using namespace qlora_model_experiment;

bool run_forward_checks(Model & model) {
    Experiment experiment(model, ContextMode::TrainingAndInference);
    const double training_loss = experiment.evaluate_training_graph_loss();
    const double inference_loss = experiment.evaluate_inference_loss();
    const double absolute_delta = std::fabs(training_loss - inference_loss);
    const double tolerance = 2.0e-2 * std::max(1.0, std::fabs(training_loss));
    const bool passed = std::isfinite(training_loss) && std::isfinite(inference_loss) && absolute_delta <= tolerance;

    if (!passed) {
        std::cerr << "[FAIL] training-graph loss differs from independent inference cross-entropy"
                  << " abs_delta=" << absolute_delta << " tolerance=" << tolerance << '\n';
    }
    return passed;
}

GradientReference run_gradient_reference(Model & model, bool & passed) {
    constexpr float epsilon = 1.0e-2f;
    constexpr float reference_learning_rate = 1.0e-2f;
    GradientReference reference;
    passed = false;

    try {
        StepResult probe_step;
        {
            Experiment probe(model, ContextMode::TrainingOnly);
            probe_step = probe.train_one_step(reference_learning_rate);
        }
        detail::require(probe_step.max_b_update > 1.0e-8f, "SGD did not update any LoRA B parameter");
        detail::require(probe_step.max_a_update < 2.0e-7f,
                        "A parameters changed on the first update despite all B tensors being zero-initialized");

        reference.pair_index = probe_step.selected_pair;
        reference.tensor_index = probe_step.selected_index;
        reference.pair_name = probe_step.pair_names.at(reference.pair_index);
        reference.center_parameter = probe_step.initial_b.at(reference.pair_index).at(reference.tensor_index);
        reference.epsilon = epsilon;

        {
            Experiment initial(model, ContextMode::InferenceOnly);
            const auto & pairs = initial.snapshots();
            detail::require(reference.pair_index < pairs.size() &&
                            reference.tensor_index < pairs[reference.pair_index].initial_b.size(),
                            "selected gradient parameter is outside the initial LoRA adapter");
            detail::require(pairs[reference.pair_index].name == reference.pair_name &&
                            pairs[reference.pair_index].initial_b[reference.tensor_index] == reference.center_parameter,
                            "independent LoRA adapters do not have matching initial parameters");

            Experiment independent(model, ContextMode::InferenceOnly);
            const float independent_initial = independent.b_parameter(reference.pair_index, reference.tensor_index);
            detail::require(independent_initial == reference.center_parameter,
                            "independent LoRA adapters do not have matching initial parameters");
            initial.set_b_parameter(reference.pair_index, reference.tensor_index,
                                    reference.center_parameter + epsilon);
            detail::require(independent.b_parameter(reference.pair_index, reference.tensor_index) == independent_initial,
                            "mutating one LoRA adapter changed an independent adapter");
        }

        const FiniteDifferenceResult numeric = evaluate_finite_difference(
            model, reference.pair_index, reference.tensor_index, reference.center_parameter,
            reference.epsilon, ContextMode::TrainingOnly);
        reference.numeric_gradient = numeric.numeric_training_gradient;
        passed = std::isfinite(reference.numeric_gradient) &&
            std::isfinite(numeric.positive_training_loss) && std::isfinite(numeric.negative_training_loss);

        if (!passed) {
            std::cerr << "[FAIL] finite-difference reference contains non-finite measurements"
                      << " epsilon=" << reference.epsilon
                      << " numeric_gradient=" << reference.numeric_gradient
                      << " positive_loss=" << numeric.positive_training_loss
                      << " negative_loss=" << numeric.negative_training_loss << '\n';
        }
    } catch (const std::exception & error) {
        std::cerr << "[FAIL] finite-difference reference: " << error.what() << '\n';
        passed = false;
    }

    return reference;
}

bool run_update_gradient_check(const GradientReference & reference, const StepResult & step) {
    if (reference.pair_index >= step.initial_b.size() ||
            reference.pair_index >= step.final_b.size() ||
            reference.tensor_index >= step.initial_b[reference.pair_index].size() ||
            reference.tensor_index >= step.final_b[reference.pair_index].size()) {
        std::cerr << "[FAIL] gradient probe is outside the adapter tensor\n";
        return false;
    }

    const double estimated_gradient =
        (static_cast<double>(step.initial_b[reference.pair_index][reference.tensor_index]) -
         step.final_b[reference.pair_index][reference.tensor_index]) / step.learning_rate;
    double absolute_error = 0.0;
    double relative_error = 0.0;
    const bool passed = gradient_matches(estimated_gradient, reference.numeric_gradient,
                                         absolute_error, relative_error);
    if (!passed) {
        std::cerr << std::setprecision(6)
                  << "[FAIL] update-derived gradient disagrees with the finite-difference gradient"
                  << " learning_rate=" << step.learning_rate
                  << " update_estimate=" << estimated_gradient
                  << " numeric=" << reference.numeric_gradient
                  << " abs_error=" << absolute_error
                  << " rel_error=" << relative_error << '\n';
    }
    return passed;
}

StepResult run_training_step_check(Model & model, bool & passed) {
    constexpr float learning_rate = 1.0e-2f;
    passed = false;

    Experiment experiment(model, ContextMode::TrainingAndInference);
    const double pre_step_inference_loss = experiment.evaluate_inference_loss();
    const StepResult step = experiment.train_one_step(learning_rate);
    const double post_step_training_loss = experiment.evaluate_training_graph_loss();
    const double post_step_inference_loss = experiment.evaluate_inference_loss();
    const double loss_tolerance = 2.0e-2 * std::max(1.0, std::fabs(step.training_loss));
    const double initial_loss_delta = std::fabs(step.training_loss - pre_step_inference_loss);
    const double post_loss_delta = std::fabs(post_step_training_loss - post_step_inference_loss);
    const bool finite_losses = std::isfinite(pre_step_inference_loss) &&
        std::isfinite(step.training_loss) && std::isfinite(post_step_training_loss) &&
        std::isfinite(post_step_inference_loss);
    const bool update_expected = step.max_b_update > 1.0e-8f && step.max_a_update < 2.0e-7f;
    passed = finite_losses && initial_loss_delta <= loss_tolerance && post_loss_delta <= loss_tolerance &&
        update_expected;

    if (!passed) {
        std::cerr << std::setprecision(6)
                  << "[FAIL] training step loss consistency check"
                  << " pre_loss_delta=" << initial_loss_delta
                  << " post_loss_delta=" << post_loss_delta
                  << " max_A_update=" << step.max_a_update
                  << " max_B_update=" << step.max_b_update
                  << " finite_losses=" << (finite_losses ? "yes" : "no")
                  << " update_expected=" << (update_expected ? "yes" : "no")
                  << " tolerance=" << loss_tolerance << '\n';
    }
    return step;
}

} // namespace qlora_model_test
