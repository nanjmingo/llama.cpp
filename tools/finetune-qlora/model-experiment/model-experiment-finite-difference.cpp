#include "model-experiment-internal.h"

#include <cmath>

namespace qlora_model_experiment {

FiniteDifferenceResult evaluate_finite_difference(Model & model, size_t pair_index,
                                                   size_t tensor_index, float center, float epsilon,
                                                   ContextMode mode) {
    detail::require(std::isfinite(center), "finite-difference center must be finite");
    detail::require(std::isfinite(epsilon) && epsilon > 0.0f,
                    "finite-difference epsilon must be positive and finite");

    FiniteDifferenceResult result;
    const bool has_training = mode != ContextMode::InferenceOnly;
    const bool has_inference = mode != ContextMode::TrainingOnly;
    {
        Experiment positive(model, mode);
        positive.set_b_parameter(pair_index, tensor_index, center + epsilon);
        result.observed_positive_parameter = positive.b_parameter(pair_index, tensor_index);
        if (has_training) {
            result.positive_training_loss = positive.evaluate_training_graph_loss();
        }
        if (has_inference) {
            result.positive_inference_loss = positive.evaluate_inference_loss();
        }
    }
    {
        Experiment negative(model, mode);
        negative.set_b_parameter(pair_index, tensor_index, center - epsilon);
        result.observed_negative_parameter = negative.b_parameter(pair_index, tensor_index);
        if (has_training) {
            result.negative_training_loss = negative.evaluate_training_graph_loss();
            result.numeric_training_gradient =
                (result.positive_training_loss - result.negative_training_loss) / (2.0 * epsilon);
        }
        if (has_inference) {
            result.negative_inference_loss = negative.evaluate_inference_loss();
            result.numeric_inference_gradient =
                (result.positive_inference_loss - result.negative_inference_loss) / (2.0 * epsilon);
        }
    }
    return result;
}



} // namespace qlora_model_experiment
