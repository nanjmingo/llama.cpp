#include "model-internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qlora_model_eval {

namespace detail {

AdapterContextDetacher::~AdapterContextDetacher() {
    if (train) {
        llama_set_adapters_lora(train, nullptr, 0, nullptr);
    }
    if (eval) {
        llama_set_adapters_lora(eval, nullptr, 0, nullptr);
    }
}

} // namespace detail

EvaluationRun::Impl::Impl(Model::Impl & shared, EvaluationMode mode) : model(&shared) {
    adapter.reset(llama_adapter_lora_init(model->model.get(), model->temporary_adapter_file.path.string().c_str()));
    detail::require(static_cast<bool>(adapter), "failed to load an independent LoRA adapter instance");

    if (mode != EvaluationMode::InferenceOnly) {
        const llama_context_params train_params = detail::context_params(
            GGML_TYPE_F32, GGML_TYPE_F32, model->options.flash_attention, model->options.n_gpu_layers != 0);
        train_ctx.reset(llama_init_from_model(model->model.get(), train_params));
        detail::require(static_cast<bool>(train_ctx), "failed to create a training context");
    }
    const llama_context_params eval_params = detail::context_params(
        model->precision.inference_cache_type_k, model->precision.inference_cache_type_v,
        model->options.flash_attention, model->options.n_gpu_layers != 0);
    eval_ctx.reset(llama_init_from_model(model->model.get(), eval_params));
    detail::require(static_cast<bool>(eval_ctx), "failed to create an inference context");

    llama_adapter_lora * adapter_ptr = adapter.get();
    float adapter_scale = 1.0f;
    if (train_ctx) {
        detail::require(llama_set_adapters_lora(train_ctx.get(), &adapter_ptr, 1, &adapter_scale) == 0,
                        "failed to attach LoRA to the training context");
        detacher.train = train_ctx.get();
    }
    detail::require(llama_set_adapters_lora(eval_ctx.get(), &adapter_ptr, 1, &adapter_scale) == 0,
                    "failed to attach LoRA to the inference context");
    detacher.eval = eval_ctx.get();

    detail::require(!adapter->ab_map.empty(), "loaded LoRA has no tensor pairs");
    for (const auto & entry : adapter->ab_map) {
        detail::require(entry.second.a && entry.second.b,
                        "loaded LoRA pair is incomplete: " + entry.first);
        if (train_ctx) {
            ggml_set_param(entry.second.a);
            ggml_set_param(entry.second.b);
        }
    }

    if (train_ctx) {
        optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
        optimizer_params.sgd.alpha = 1.0e-2f;
        optimizer_params.sgd.wd = 0.0f;
        llama_opt_params opt_params = {};
        opt_params.n_ctx_train = model->context_length;
        opt_params.param_filter = lora_param_filter;
        opt_params.param_filter_ud = nullptr;
        opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
        opt_params.get_opt_pars_ud = &optimizer_params;
        opt_params.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        llama_opt_init(train_ctx.get(), model->model.get(), opt_params);
    }
}

EvaluationRun::EvaluationRun(Model & model, EvaluationMode mode)
    : impl_(std::make_unique<Impl>(*model.impl_, mode)) {}

EvaluationRun::~EvaluationRun() = default;

EvaluationRun::EvaluationRun(EvaluationRun &&) noexcept = default;

EvaluationRun & EvaluationRun::operator=(EvaluationRun &&) noexcept = default;


double EvaluationRun::evaluate_inference_loss() {
    detail::require(static_cast<bool>(impl_->eval_ctx), "this evaluation run has no inference context");
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

void EvaluationRun::train_one_step(float learning_rate) {
    detail::require(static_cast<bool>(impl_->train_ctx), "this evaluation run has no training context");
    detail::require(std::isfinite(learning_rate) && learning_rate > 0.0f, "learning rate must be positive and finite");

    impl_->optimizer_params.sgd.alpha = learning_rate;
    llama_opt_epoch(impl_->train_ctx.get(), impl_->model->dataset.get(), nullptr, nullptr, 1, nullptr, nullptr);
}

} // namespace qlora_model_eval
