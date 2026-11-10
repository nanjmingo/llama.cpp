#include "model-experiment-internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qlora_model_experiment {

double Experiment::evaluate_inference_loss() {
    detail::require(static_cast<bool>(impl_->eval_ctx), "this experiment has no inference context");
    detail::require(impl_->model->tokens.size() == static_cast<size_t>(impl_->model->context_length + 1),
                    "invalid training sample length");

    llama_memory_clear(llama_get_memory(impl_->eval_ctx.get()), true);
    llama_batch batch = llama_batch_init(impl_->model->context_length, 0, 1);
    if (!batch.token) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to allocate inference batch");
    }

    batch.n_tokens = impl_->model->context_length;
    for (int32_t i = 0; i < impl_->model->context_length; ++i) {
        batch.token[i] = impl_->model->tokens[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }

    const int32_t decode_result = llama_decode(impl_->eval_ctx.get(), batch);
    llama_batch_free(batch);
    detail::require(decode_result == 0, "llama_decode failed during independent inference loss evaluation");

    const llama_vocab * vocab = llama_model_get_vocab(impl_->model->model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    double total_loss = 0.0;
    for (int32_t i = 0; i < impl_->model->context_length; ++i) {
        const float * logits = llama_get_logits_ith(impl_->eval_ctx.get(), i);
        detail::require(logits != nullptr, "model did not return logits for every input token");
        const llama_token target = impl_->model->tokens[static_cast<size_t>(i + 1)];
        double max_logit = -INFINITY;
        for (int32_t j = 0; j < n_vocab; ++j) {
            max_logit = std::max(max_logit, static_cast<double>(logits[j]));
        }
        double sum_exp = 0.0;
        for (int32_t j = 0; j < n_vocab; ++j) {
            sum_exp += std::exp(static_cast<double>(logits[j]) - max_logit);
        }
        total_loss += max_logit + std::log(sum_exp) - static_cast<double>(logits[target]);
    }

    const double loss = total_loss / static_cast<double>(impl_->model->context_length);
    detail::require(std::isfinite(loss), "inference loss is not finite");
    return loss;
}

double Experiment::evaluate_training_graph_loss() {
    detail::require(static_cast<bool>(impl_->train_ctx), "this experiment has no training context");
    std::unique_ptr<ggml_opt_result, decltype(&ggml_opt_result_free)> result(
        ggml_opt_result_init(), ggml_opt_result_free);
    detail::require(static_cast<bool>(result), "failed to allocate training-graph evaluation result");
    llama_opt_epoch(impl_->train_ctx.get(), impl_->model->dataset.get(), nullptr, result.get(), 0, nullptr, nullptr);

    double loss = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(result.get(), &loss, &uncertainty);
    detail::require(std::isfinite(loss), "training-graph loss is not finite");
    return loss;
}



} // namespace qlora_model_experiment
