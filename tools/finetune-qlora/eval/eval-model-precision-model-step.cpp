#include "eval-model-precision-model-internal.h"

#include <cmath>

namespace qlora_model_eval {

void EvaluationRun::train_one_step(float learning_rate) {
    detail::require(static_cast<bool>(impl_->train_ctx), "this evaluation run has no training context");
    detail::require(std::isfinite(learning_rate) && learning_rate > 0.0f, "learning rate must be positive and finite");

    impl_->optimizer_params.sgd.alpha = learning_rate;
    llama_opt_epoch(impl_->train_ctx.get(), impl_->model->dataset.get(), nullptr, nullptr, 1, nullptr, nullptr);
}

}
