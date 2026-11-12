#include "eval-model-precision-model-internal.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>

namespace qlora_model_eval {
namespace detail {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

BackendScope::BackendScope() { llama_backend_init(); }
BackendScope::~BackendScope() { llama_backend_free(); }

void QuietLibraryLogs::callback(ggml_log_level level, const char * text, void * user_data) {
    auto * self = static_cast<QuietLibraryLogs *>(user_data);
    if (self->previous_callback && (self->verbose || level == GGML_LOG_LEVEL_ERROR)) {
        self->previous_callback(level, text, self->previous_user_data);
    }
}

QuietLibraryLogs::QuietLibraryLogs(bool verbose_logs) : verbose(verbose_logs) {
    llama_log_get(&previous_callback, &previous_user_data);
    llama_log_set(callback, this);
}

QuietLibraryLogs::~QuietLibraryLogs() {
    llama_log_set(previous_callback, previous_user_data);
}

TemporaryAdapterFile::~TemporaryAdapterFile() {
    if (!path.empty()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
}

AdapterContextDetacher::~AdapterContextDetacher() {
    if (train) {
        llama_set_adapters_lora(train, nullptr, 0, nullptr);
    }
    if (eval) {
        llama_set_adapters_lora(eval, nullptr, 0, nullptr);
    }
}

std::vector<llama_token> tokenize_sample(const llama_vocab * vocab, int32_t context_length) {
    const std::string sample_text =
        "Language models learn to predict the next token from preceding context. A precision evaluation measures how "
        "numerical choices during training affect the model's loss. This fixed sample provides a repeatable set "
        "of next-token targets for comparing independent training runs.";
    std::string text;
    std::vector<llama_token> tokens(2048);
    int32_t count = 0;
    do {
        if (!text.empty()) {
            text += '\n';
        }
        text += sample_text;
        count = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
        require(count > 0, "failed to tokenize the model-training fixture");
    } while (count < context_length + 1);
    tokens.resize(static_cast<size_t>(context_length + 1));
    return tokens;
}

llama_context_params context_params(ggml_type type_k, ggml_type type_v, bool flash_attention, bool gpu_offload) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = kRequestedContext;
    params.n_batch = kRequestedContext;
    params.n_ubatch = kRequestedContext;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn_type = flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = type_k;
    params.type_v = type_v;
    params.offload_kqv = gpu_offload;
    params.op_offload = gpu_offload;
    return params;
}

}

namespace {

void validate_model_path(const ModelOptions & options) {
    detail::require(!options.model_path.empty(), "--model is required");
    detail::require(std::filesystem::is_regular_file(options.model_path),
                    "model file does not exist: " + options.model_path);
}

void validate_gpu_backend(const ModelOptions & options) {
    if (options.n_gpu_layers == 0) {
        return;
    }

    bool found_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            found_gpu = true;
            break;
        }
    }
    detail::require(found_gpu, "--n-gpu-layers requests GPU offload, but no GPU backend device is available");
}

std::string format_lora_targets(const std::vector<std::string> & targets) {
    std::ostringstream output;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (i > 0) {
            output << ',';
        }
        output << targets[i];
    }
    return output.str();
}

}

Model::Impl::Impl(ModelOptions opts, ModelPrecisionConfig precision_config)
    : options(std::move(opts)), precision(std::move(precision_config)), quiet_logs(options.verbose_library_logs) {
    validate_model_path(options);
    validate_gpu_backend(options);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    model_params.use_extra_bufts = false;
    model.reset(llama_model_load_from_file(options.model_path.c_str(), model_params));
    detail::require(static_cast<bool>(model), "failed to load the GGUF model");

    active_forward_precision = precision.forward_precision;
    size_t configured_tensors = 0;
    for (const auto & entry : model->tensors_by_name) {
        if (entry.second) {
            ++configured_tensors;
        }
    }
    model->prec_policy.prec_src1.clear();
    model->prec_policy.prec_src1_default = precision.forward_precision;
    model->prec_policy.prec_bwd = precision.backward_precision;
    if (options.report_configuration) {
        std::cout << "[CONFIG] model_path=" << options.model_path
              << " forward_precision=" << precision.forward_precision_name
              << " backward_precision=" << precision.backward_precision_name
              << " inference_cache_type_k=" << ggml_type_name(precision.inference_cache_type_k)
              << " inference_cache_type_v=" << ggml_type_name(precision.inference_cache_type_v)
              << " lora_targets=" << format_lora_targets(options.lora_targets)
              << " flash_attention=" << (options.flash_attention ? "on" : "off")
              << " n_gpu_layers=" << options.n_gpu_layers
              << " gpu_offload=" << (options.n_gpu_layers != 0 ? "on" : "off")
              << " training_cache_type_k=f32 training_cache_type_v=f32"
              << " configured_tensors=" << configured_tensors << '\n';
    }

    lora_tensors prepared;
    {
        std::mt19937 rng(42);
        detail::require(validate_lora_targets(model->tensors_by_name, options.lora_targets,
                                              llama_model_n_layer(model.get()), 0),
                        "invalid --lora-targets");
        prepared = alloc_lora_tensors(model->tensors_by_name, options.lora_targets, detail::kRank, rng,
                                      llama_model_n_layer(model.get()), 0);
        detail::require(!prepared.refs.empty(), "model has no tensors matching --lora-targets");
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        temporary_adapter_file.path = std::filesystem::temp_directory_path() /
            ("qlora-model-precision-" + std::to_string(stamp) + ".gguf");
        detail::require(save_adapter(prepared.refs, temporary_adapter_file.path.string(), model->arch_name(),
                                     detail::kAlpha, options.model_path),
                        "failed to write the temporary LoRA adapter");
        prepared = {};
    }

    context_length = detail::kRequestedContext;

    tokens = detail::tokenize_sample(llama_model_get_vocab(model.get()), context_length);
    training_sample sample;
    sample.tokens = tokens;
    sample.weights.assign(tokens.size(), 1.0f);
    ggml_opt_dataset_t dataset_raw = build_training_dataset(
        { sample }, window_options { context_length, llama_vocab_bos(llama_model_get_vocab(model.get())), 0 });
    detail::require(dataset_raw != nullptr, "failed to build a one-window training dataset");
    dataset.reset(dataset_raw);
    detail::require(ggml_opt_dataset_ndata(dataset.get()) == 1,
                    "model-training fixture must contain one window");
}

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

Model::Model(ModelOptions options, ModelPrecisionConfig precision)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(precision))) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

void Model::Impl::set_forward_precision(ggml_prec forward_precision) {
    if (active_forward_precision == forward_precision) {
        return;
    }
    model->prec_policy.prec_src1_default = forward_precision;
    active_forward_precision = forward_precision;
}

void Model::set_forward_precision(ggml_prec precision) {
    impl_->set_forward_precision(precision);
}

EvaluationRun::EvaluationRun(Model & model, EvaluationMode mode)
    : impl_(std::make_unique<Impl>(*model.impl_, mode)) {}
EvaluationRun::~EvaluationRun() = default;
EvaluationRun::EvaluationRun(EvaluationRun &&) noexcept = default;
EvaluationRun & EvaluationRun::operator=(EvaluationRun &&) noexcept = default;


}
