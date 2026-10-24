#include "../adapter.h"
#include "../dataset-window.h"
#include "../../../src/llama-adapter.h"
#include "../../../src/llama-context.h"
#include "../../../src/llama-model.h"

#include "ggml-backend.h"
#include "ggml-opt.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kRequestedContext = 256;
constexpr int32_t kRank = 2;
constexpr float kAlpha = 2.0f;
constexpr float kLearningRate = 1.0e-2f;
constexpr float kFiniteDifferenceEpsilon = 1.0e-2f;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct backend_scope {
    backend_scope() { llama_backend_init(); }
    ~backend_scope() { llama_backend_free(); }
};

struct scoped_quiet_library_logs {
    ggml_log_callback previous_callback = nullptr;
    void * previous_user_data = nullptr;
    bool verbose = false;

    static void callback(ggml_log_level level, const char * text, void * user_data) {
        auto * self = static_cast<scoped_quiet_library_logs *>(user_data);
        if (self->previous_callback && (self->verbose || level == GGML_LOG_LEVEL_ERROR)) {
            self->previous_callback(level, text, self->previous_user_data);
        }
    }

    explicit scoped_quiet_library_logs(bool verbose_logs) : verbose(verbose_logs) {
        llama_log_get(&previous_callback, &previous_user_data);
        llama_log_set(callback, this);
    }

    ~scoped_quiet_library_logs() {
        llama_log_set(previous_callback, previous_user_data);
    }
};

struct adapter_context_detacher {
    llama_context * train = nullptr;
    llama_context * eval = nullptr;

    ~adapter_context_detacher() {
        if (train) {
            llama_set_adapters_lora(train, nullptr, 0, nullptr);
        }
        if (eval) {
            llama_set_adapters_lora(eval, nullptr, 0, nullptr);
        }
    }
};

struct adapter_pair_snapshot {
    std::string name;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    std::vector<float> initial_a;
    std::vector<float> initial_b;
};

std::vector<float> tensor_to_vector(ggml_tensor * tensor) {
    require(tensor != nullptr, "cannot read a null tensor");
    require(tensor->type == GGML_TYPE_F32, "expected an F32 LoRA tensor");
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    require(ggml_nbytes(tensor) == result.size() * sizeof(float), "LoRA tensor is not contiguous F32 storage");
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

void vector_to_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    require(tensor != nullptr, "cannot write a null tensor");
    require(tensor->type == GGML_TYPE_F32, "expected an F32 LoRA tensor");
    require(values.size() == static_cast<size_t>(ggml_nelements(tensor)), "LoRA tensor value count mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}

std::vector<llama_token> tokenize_sample(const llama_vocab * vocab, int32_t context_length) {
    const std::string sample_text =
        "The small language model learns from examples. A careful test compares its predictions before and after "
        "one training update. The purpose is to verify that gradients from the model's training graph agree with "
        "independent numerical derivatives. The sample includes enough ordinary words to fill a short context "
        "and provides a fixed next-token target at every position.";
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
        require(count > 0, "failed to tokenize the model-gradient fixture");
    } while (count < context_length + 1);
    tokens.resize(static_cast<size_t>(context_length + 1));
    return tokens;
}

llama_context_params cpu_context_params() {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = kRequestedContext;
    params.n_batch = kRequestedContext;
    params.n_ubatch = kRequestedContext;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.offload_kqv = false;
    params.op_offload = false;
    return params;
}

double evaluate_token_loss(llama_context * ctx,
                          const std::vector<llama_token> & sample_tokens,
                          const llama_vocab * vocab,
                          int32_t context_length) {
    require(sample_tokens.size() == static_cast<size_t>(context_length + 1), "invalid gradient-check sample length");
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(context_length, 0, 1);
    require(batch.token != nullptr, "failed to allocate inference batch");
    batch.n_tokens = context_length;
    for (int32_t i = 0; i < context_length; ++i) {
        batch.token[i] = sample_tokens[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }

    const int32_t decode_result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(decode_result == 0, "llama_decode failed during numerical loss evaluation");

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    double total_loss = 0.0;
    for (int32_t i = 0; i < context_length; ++i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        require(logits != nullptr, "model did not return logits for every input token");
        const llama_token target = sample_tokens[static_cast<size_t>(i + 1)];
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
    return total_loss / static_cast<double>(context_length);
}

double evaluate_training_graph_loss(llama_context * ctx, ggml_opt_dataset_t dataset) {
    std::unique_ptr<ggml_opt_result, decltype(&ggml_opt_result_free)> result(
        ggml_opt_result_init(), ggml_opt_result_free);
    require(static_cast<bool>(result), "failed to allocate training-graph evaluation result");

    // Route the dataset through the optimizer's forward-only evaluation path.
    llama_opt_epoch(ctx, dataset, nullptr, result.get(), 0, nullptr, nullptr);

    double loss = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(result.get(), &loss, &uncertainty);
    require(std::isfinite(loss), "training-graph loss is not finite");
    return loss;
}

void set_all_initial_parameters(const std::vector<adapter_pair_snapshot> & pairs) {
    for (const auto & pair : pairs) {
        vector_to_tensor(pair.a, pair.initial_a);
        vector_to_tensor(pair.b, pair.initial_b);
    }
}

double max_parameter_restore_error(const std::vector<adapter_pair_snapshot> & pairs) {
    double max_error = 0.0;
    for (const auto & pair : pairs) {
        const auto current_a = tensor_to_vector(pair.a);
        const auto current_b = tensor_to_vector(pair.b);
        for (size_t i = 0; i < current_a.size(); ++i) {
            max_error = std::max(max_error, std::fabs(
                static_cast<double>(current_a[i]) - pair.initial_a[i]));
        }
        for (size_t i = 0; i < current_b.size(); ++i) {
            max_error = std::max(max_error, std::fabs(
                static_cast<double>(current_b[i]) - pair.initial_b[i]));
        }
    }
    return max_error;
}

void set_parameters_at_update_scale(
        const std::vector<adapter_pair_snapshot> & pairs,
        const std::vector<std::vector<float>> & final_a,
        const std::vector<std::vector<float>> & final_b,
        float scale) {
    for (size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
        const auto & pair = pairs[pair_index];
        std::vector<float> scaled_a(pair.initial_a.size());
        std::vector<float> scaled_b(pair.initial_b.size());
        for (size_t i = 0; i < scaled_a.size(); ++i) {
            scaled_a[i] = pair.initial_a[i] + scale * (final_a[pair_index][i] - pair.initial_a[i]);
        }
        for (size_t i = 0; i < scaled_b.size(); ++i) {
            scaled_b[i] = pair.initial_b[i] + scale * (final_b[pair_index][i] - pair.initial_b[i]);
        }
        vector_to_tensor(pair.a, scaled_a);
        vector_to_tensor(pair.b, scaled_b);
    }
}

struct update_probe {
    size_t pair_index;
    size_t tensor_index;
    float update;
};

std::vector<update_probe> select_update_probes(
        const std::vector<adapter_pair_snapshot> & pairs,
        const std::vector<std::vector<float>> & final_b,
        size_t count) {
    std::vector<update_probe> candidates;
    candidates.reserve(pairs.size());
    for (size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
        const auto & pair = pairs[pair_index];
        size_t selected_index = 0;
        float selected_update = 0.0f;
        for (size_t i = 0; i < pair.initial_b.size(); ++i) {
            const float update = final_b[pair_index][i] - pair.initial_b[i];
            if (std::fabs(update) > std::fabs(selected_update)) {
                selected_index = i;
                selected_update = update;
            }
        }
        if (selected_update != 0.0f) {
            candidates.push_back({ pair_index, selected_index, selected_update });
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const update_probe & lhs, const update_probe & rhs) {
        return std::fabs(lhs.update) > std::fabs(rhs.update);
    });
    if (candidates.size() > count) {
        candidates.resize(count);
    }
    return candidates;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        std::string activation_precision_name = "f32";
        ggml_prec activation_precision = GGML_PREC_F32;
        bool verbose_library_logs = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--activation-precision" && i + 1 < argc) {
                activation_precision_name = argv[++i];
                if (activation_precision_name == "f32") {
                    activation_precision = GGML_PREC_F32;
                } else if (activation_precision_name == "f16") {
                    activation_precision = GGML_PREC_F16;
                } else if (activation_precision_name == "bf16") {
                    activation_precision = GGML_PREC_BF16;
                } else if (activation_precision_name == "q8") {
                    activation_precision = GGML_PREC_Q8;
                } else {
                    std::cerr << "Invalid activation precision: " << activation_precision_name << '\n';
                    return 2;
                }
            } else if (arg == "--verbose-logs") {
                verbose_library_logs = true;
            } else {
                std::cerr << "Usage: " << argv[0]
                          << " --model MODEL.gguf [--activation-precision f32|f16|bf16|q8] [--verbose-logs]\n";
                return 2;
            }
        }
        require(!model_path.empty(), "--model is required");
        require(std::filesystem::is_regular_file(model_path), "model file does not exist: " + model_path);

        scoped_quiet_library_logs quiet_library_logs(verbose_library_logs);
        backend_scope backend;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
        require(static_cast<bool>(model), "failed to load the GGUF model");

        size_t overridden_tensors = 0;
        for (const auto & entry : model->tensors_by_name) {
            ggml_tensor * tensor = entry.second;
            if (tensor && ggml_is_quantized(tensor->type)) {
                model->prec_policy.prec_src1[tensor] = activation_precision;
                ++overridden_tensors;
            }
        }
        std::cout << "[CONFIG] activation_precision=" << activation_precision_name
                  << " quantized_tensors=" << overridden_tensors << '\n';

        llama_context_params context_params = cpu_context_params();
        std::unique_ptr<llama_context, decltype(&llama_free)> train_ctx(
            llama_init_from_model(model.get(), context_params), llama_free);
        std::unique_ptr<llama_context, decltype(&llama_free)> eval_ctx(
            llama_init_from_model(model.get(), context_params), llama_free);
        require(train_ctx && eval_ctx, "failed to create CPU contexts");

        const std::vector<std::string> targets = { "attn_q" };
        lora_tensors prepared;
        {
            std::mt19937 rng(42);
            prepared = alloc_lora_tensors(model->tensors_by_name, targets, kRank, rng,
                                          llama_model_n_layer(model.get()), 0);
            require(!prepared.refs.empty(), "model has no LoRA target matching attn_q");
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto adapter_path = std::filesystem::temp_directory_path() /
                ("qlora-model-gradient-" + std::to_string(stamp) + ".gguf");
            require(save_adapter(prepared.refs, adapter_path.string(), model->arch_name(), kAlpha, model_path),
                    "failed to write the temporary LoRA adapter");
            prepared = {};

            struct adapter_file_scope {
                std::filesystem::path path;
                ~adapter_file_scope() {
                    std::error_code ec;
                    std::filesystem::remove(path, ec);
                }
            } adapter_file{ adapter_path };

            std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)> adapter(
                llama_adapter_lora_init(model.get(), adapter_path.string().c_str()), llama_adapter_lora_free);
            require(static_cast<bool>(adapter), "failed to load the temporary LoRA adapter");

            llama_adapter_lora * adapter_ptr = adapter.get();
            float adapter_scale = 1.0f;
            require(llama_set_adapters_lora(train_ctx.get(), &adapter_ptr, 1, &adapter_scale) == 0,
                    "failed to attach LoRA to the training context");
            require(llama_set_adapters_lora(eval_ctx.get(), &adapter_ptr, 1, &adapter_scale) == 0,
                    "failed to attach LoRA to the evaluation context");
            adapter_context_detacher detach_adapters{ train_ctx.get(), eval_ctx.get() };

            std::vector<std::string> pair_names;
            pair_names.reserve(adapter->ab_map.size());
            for (const auto & entry : adapter->ab_map) {
                pair_names.push_back(entry.first);
            }
            std::sort(pair_names.begin(), pair_names.end());
            require(!pair_names.empty(), "loaded LoRA has no tensor pairs");

            std::vector<adapter_pair_snapshot> snapshots;
            snapshots.reserve(pair_names.size());
            for (const std::string & name : pair_names) {
                auto it = adapter->ab_map.find(name);
                require(it != adapter->ab_map.end() && it->second.a && it->second.b,
                        "loaded LoRA pair is incomplete: " + name);
                ggml_set_param(it->second.a);
                ggml_set_param(it->second.b);
                snapshots.push_back({ name, it->second.a, it->second.b,
                                      tensor_to_vector(it->second.a), tensor_to_vector(it->second.b) });
            }

            const int32_t context_length = static_cast<int32_t>(llama_n_ctx(train_ctx.get()));
            require(context_length == static_cast<int32_t>(llama_n_ctx(eval_ctx.get())),
                    "training and evaluation contexts have different effective context lengths");
            require(context_length <= static_cast<int32_t>(llama_n_batch(train_ctx.get())) &&
                    context_length <= static_cast<int32_t>(llama_n_ubatch(train_ctx.get())),
                    "effective context length exceeds the configured batch capacities");

            const std::vector<llama_token> tokens = tokenize_sample(llama_model_get_vocab(model.get()), context_length);
            training_sample sample;
            sample.tokens = tokens;
            sample.weights.assign(tokens.size(), 1.0f);
            ggml_opt_dataset_t dataset_raw = build_training_dataset(
                { sample }, window_options { context_length, llama_vocab_bos(llama_model_get_vocab(model.get())), 0 });
            require(dataset_raw != nullptr, "failed to build a one-window gradient-check dataset");
            std::unique_ptr<ggml_opt_dataset, decltype(&ggml_opt_dataset_free)> dataset(
                dataset_raw, ggml_opt_dataset_free);
            require(ggml_opt_dataset_ndata(dataset.get()) == 1, "gradient-check fixture must contain one window");

            ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
            optimizer_params.sgd.alpha = kLearningRate;
            optimizer_params.sgd.wd = 0.0f;
            llama_opt_params opt_params = {};
            opt_params.n_ctx_train = context_length;
            opt_params.param_filter = lora_param_filter;
            opt_params.param_filter_ud = nullptr;
            opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
            opt_params.get_opt_pars_ud = &optimizer_params;
            opt_params.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
            llama_opt_init(train_ctx.get(), model.get(), opt_params);

            const double initial_loss = evaluate_token_loss(
                eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
            require(std::isfinite(initial_loss), "initial model loss is not finite");

            std::unique_ptr<ggml_opt_result, decltype(&ggml_opt_result_free)> result(
                ggml_opt_result_init(), ggml_opt_result_free);
            require(static_cast<bool>(result), "failed to allocate optimizer result");
            llama_opt_epoch(train_ctx.get(), dataset.get(), result.get(), nullptr, 1, nullptr, nullptr);

            double train_loss = 0.0;
            double train_uncertainty = 0.0;
            ggml_opt_result_loss(result.get(), &train_loss, &train_uncertainty);
            require(std::isfinite(train_loss), "training graph loss is not finite");
            const double forward_loss_abs_error = std::fabs(train_loss - initial_loss);
            const double forward_loss_tolerance = 2.0e-2 * std::max(1.0, std::fabs(train_loss));
            require(forward_loss_abs_error <= forward_loss_tolerance,
                    "training-graph loss differs from the independent inference cross-entropy: " +
                    std::to_string(train_loss) + " vs " + std::to_string(initial_loss));

            size_t selected_pair = 0;
            size_t selected_index = 0;
            float max_b_update = 0.0f;
            float max_a_update = 0.0f;
            std::vector<std::vector<float>> final_a;
            std::vector<std::vector<float>> final_b;
            final_a.reserve(snapshots.size());
            final_b.reserve(snapshots.size());
            for (size_t pair_index = 0; pair_index < snapshots.size(); ++pair_index) {
                const auto & pair = snapshots[pair_index];
                const auto updated_a = tensor_to_vector(pair.a);
                const auto updated_b = tensor_to_vector(pair.b);
                final_a.push_back(updated_a);
                final_b.push_back(updated_b);
                for (size_t i = 0; i < updated_a.size(); ++i) {
                    max_a_update = std::max(max_a_update, std::fabs(updated_a[i] - pair.initial_a[i]));
                }
                for (size_t i = 0; i < updated_b.size(); ++i) {
                    const float magnitude = std::fabs(updated_b[i] - pair.initial_b[i]);
                    if (magnitude > max_b_update) {
                        max_b_update = magnitude;
                        selected_pair = pair_index;
                        selected_index = i;
                    }
                }
            }
            require(max_b_update > 1.0e-8f, "SGD did not update any LoRA B parameter");
            require(max_a_update < 2.0e-7f,
                    "A parameters changed on the first update despite all B tensors being zero-initialized");

            double squared_parameter_update = 0.0;
            for (size_t pair_index = 0; pair_index < snapshots.size(); ++pair_index) {
                const auto & pair = snapshots[pair_index];
                for (size_t i = 0; i < final_a[pair_index].size(); ++i) {
                    const double delta = static_cast<double>(final_a[pair_index][i]) - pair.initial_a[i];
                    squared_parameter_update += delta * delta;
                }
                for (size_t i = 0; i < final_b[pair_index].size(); ++i) {
                    const double delta = static_cast<double>(final_b[pair_index][i]) - pair.initial_b[i];
                    squared_parameter_update += delta * delta;
                }
            }
            const double predicted_first_order_loss_delta = -squared_parameter_update / kLearningRate;
            const double post_step_training_loss = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
            const double post_step_inference_loss = evaluate_token_loss(
                eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
            std::cout << std::setprecision(10)
                      << "[DIAG-STEP] pre_step_training_graph_loss=" << train_loss
                      << " post_step_training_graph_loss=" << post_step_training_loss
                      << " observed_training_loss_delta=" << (post_step_training_loss - train_loss)
                      << " pre_step_inference_loss=" << initial_loss
                      << " post_step_inference_loss=" << post_step_inference_loss
                      << " observed_inference_loss_delta=" << (post_step_inference_loss - initial_loss)
                      << " squared_parameter_update=" << squared_parameter_update
                      << " predicted_first_order_loss_delta=" << predicted_first_order_loss_delta
                      << " max_A_update=" << max_a_update
                      << " max_B_update=" << max_b_update << '\n';

            const auto & chosen = snapshots[selected_pair];
            const float initial_parameter = chosen.initial_b[selected_index];
            const float updated_parameter = final_b[selected_pair][selected_index];
            const float autodiff_gradient = (initial_parameter - updated_parameter) / kLearningRate;

            set_all_initial_parameters(snapshots);
            const double training_graph_loss = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
            const double training_graph_loss_abs_error = std::fabs(train_loss - training_graph_loss);
            require(training_graph_loss_abs_error <= forward_loss_tolerance,
                    "forward-only training-graph loss differs from the training-graph loss: " +
                    std::to_string(train_loss) + " vs " + std::to_string(training_graph_loss));

            const float epsilon = kFiniteDifferenceEpsilon;
            double numeric_gradient_training = 0.0;
            double numeric_gradient_inference = 0.0;
            double loss_positive_training = 0.0;
            double loss_negative_training = 0.0;
            double loss_positive_inference = 0.0;
            double loss_negative_inference = 0.0;

            set_all_initial_parameters(snapshots);
            const float restored_parameter = tensor_to_vector(chosen.b)[selected_index];
            const double baseline_training = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
            const double baseline_inference = evaluate_token_loss(
                eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
            const double repeated_baseline_training = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
            const double repeated_baseline_inference = evaluate_token_loss(
                eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
            std::cout << std::setprecision(10)
                      << "[DIAG-REPEAT] baseline_training_graph=" << baseline_training
                      << " repeated_training_graph=" << repeated_baseline_training
                      << " training_graph_abs_delta=" << std::fabs(repeated_baseline_training - baseline_training)
                      << " baseline_inference=" << baseline_inference
                      << " repeated_inference=" << repeated_baseline_inference
                      << " inference_abs_delta=" << std::fabs(repeated_baseline_inference - baseline_inference)
                      << '\n';

            auto positive = chosen.initial_b;
            positive[selected_index] = initial_parameter + epsilon;
            vector_to_tensor(chosen.b, positive);
            const float observed_positive_parameter = tensor_to_vector(chosen.b)[selected_index];
            loss_positive_inference = evaluate_token_loss(
                eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
            const float parameter_after_positive_inference = tensor_to_vector(chosen.b)[selected_index];
            loss_positive_training = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
            const float parameter_after_positive_training = tensor_to_vector(chosen.b)[selected_index];

            auto negative = chosen.initial_b;
            negative[selected_index] = initial_parameter - epsilon;
            vector_to_tensor(chosen.b, negative);
            const float observed_negative_parameter = tensor_to_vector(chosen.b)[selected_index];
            loss_negative_inference = evaluate_token_loss(
                eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
            const float parameter_after_negative_inference = tensor_to_vector(chosen.b)[selected_index];
            loss_negative_training = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
            const float parameter_after_negative_training = tensor_to_vector(chosen.b)[selected_index];

            numeric_gradient_training =
                (loss_positive_training - loss_negative_training) / (2.0 * epsilon);
            numeric_gradient_inference =
                (loss_positive_inference - loss_negative_inference) / (2.0 * epsilon);
            std::cout << std::setprecision(10)
                      << "[DIAG-FD] epsilon=" << epsilon
                      << " restored_parameter=" << restored_parameter
                      << " baseline_training_graph=" << baseline_training
                      << " baseline_inference=" << baseline_inference
                      << " observed_positive_parameter=" << observed_positive_parameter
                      << " parameter_after_positive_inference=" << parameter_after_positive_inference
                      << " parameter_after_positive_training_graph=" << parameter_after_positive_training
                      << " loss_positive_training_graph=" << loss_positive_training
                      << " loss_positive_inference=" << loss_positive_inference
                      << " observed_negative_parameter=" << observed_negative_parameter
                      << " parameter_after_negative_inference=" << parameter_after_negative_inference
                      << " parameter_after_negative_training_graph=" << parameter_after_negative_training
                      << " loss_negative_training_graph=" << loss_negative_training
                      << " loss_negative_inference=" << loss_negative_inference
                      << " numeric_gradient_training_graph=" << numeric_gradient_training
                      << " numeric_gradient_inference=" << numeric_gradient_inference
                      << '\n';

            const float sweep_epsilons[] = { 3.0e-3f, 1.0e-3f, 3.0e-4f, 1.0e-4f };
            for (const float sweep_epsilon : sweep_epsilons) {
                set_all_initial_parameters(snapshots);
                auto sweep_positive = chosen.initial_b;
                sweep_positive[selected_index] = initial_parameter + sweep_epsilon;
                vector_to_tensor(chosen.b, sweep_positive);
                const float observed_positive = tensor_to_vector(chosen.b)[selected_index];
                const double loss_positive_training = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
                const double loss_positive_inference = evaluate_token_loss(
                    eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);

                set_all_initial_parameters(snapshots);
                auto sweep_negative = chosen.initial_b;
                sweep_negative[selected_index] = initial_parameter - sweep_epsilon;
                vector_to_tensor(chosen.b, sweep_negative);
                const float observed_negative = tensor_to_vector(chosen.b)[selected_index];
                const double loss_negative_training = evaluate_training_graph_loss(train_ctx.get(), dataset.get());
                const double loss_negative_inference = evaluate_token_loss(
                    eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);

                set_all_initial_parameters(snapshots);
                const double restore_parameter_max_abs = max_parameter_restore_error(snapshots);
                const double numeric_gradient_training_sweep =
                    (loss_positive_training - loss_negative_training) / (2.0 * sweep_epsilon);
                const double numeric_gradient_inference_sweep =
                    (loss_positive_inference - loss_negative_inference) / (2.0 * sweep_epsilon);
                std::cout << std::setprecision(10)
                          << "[DIAG-FD-SWEEP] epsilon=" << sweep_epsilon
                          << " observed_positive_parameter=" << observed_positive
                          << " loss_positive_training=" << loss_positive_training
                          << " loss_negative_training=" << loss_negative_training
                          << " numeric_gradient_training=" << numeric_gradient_training_sweep
                          << " loss_positive_inference=" << loss_positive_inference
                          << " loss_negative_inference=" << loss_negative_inference
                          << " numeric_gradient_inference=" << numeric_gradient_inference_sweep
                          << " gradient_objective_abs_delta="
                          << std::fabs(numeric_gradient_training_sweep - numeric_gradient_inference_sweep)
                          << " observed_negative_parameter=" << observed_negative
                          << " restore_parameter_max_abs=" << restore_parameter_max_abs
                          << " autodiff_from_update=" << autodiff_gradient << '\n';
            }

            const float coordinate_epsilon = 1.0e-3f;
            const auto coordinate_probes = select_update_probes(snapshots, final_b, 3);
            for (const auto & probe : coordinate_probes) {
                if (probe.pair_index == selected_pair && probe.tensor_index == selected_index) {
                    continue;
                }
                const auto & probe_pair = snapshots[probe.pair_index];
                const float parameter_zero = probe_pair.initial_b[probe.tensor_index];
                set_all_initial_parameters(snapshots);
                auto probe_positive = probe_pair.initial_b;
                probe_positive[probe.tensor_index] = parameter_zero + coordinate_epsilon;
                vector_to_tensor(probe_pair.b, probe_positive);
                const double probe_loss_positive = evaluate_token_loss(
                    eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);

                set_all_initial_parameters(snapshots);
                auto probe_negative = probe_pair.initial_b;
                probe_negative[probe.tensor_index] = parameter_zero - coordinate_epsilon;
                vector_to_tensor(probe_pair.b, probe_negative);
                const double probe_loss_negative = evaluate_token_loss(
                    eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);

                set_all_initial_parameters(snapshots);
                const double probe_restore_error = max_parameter_restore_error(snapshots);
                const double probe_numeric_gradient =
                    (probe_loss_positive - probe_loss_negative) / (2.0 * coordinate_epsilon);
                std::cout << std::setprecision(10)
                          << "[DIAG-COORDINATE] tensor=" << probe_pair.name
                          << " parameter=" << probe.tensor_index
                          << " parameter_initial=" << parameter_zero
                          << " update=" << probe.update
                          << " autodiff_from_update=" << -probe.update / kLearningRate
                          << " epsilon=" << coordinate_epsilon
                          << " loss_positive=" << probe_loss_positive
                          << " loss_negative=" << probe_loss_negative
                          << " numeric_gradient_inference=" << probe_numeric_gradient
                          << " restore_parameter_max_abs=" << probe_restore_error << '\n';
            }

            std::cout << std::setprecision(10)
                      << "[DIAG-DIRECTION] scale=0 training_loss=" << baseline_training
                      << " inference_loss=" << baseline_inference
                      << " loss_delta_training=0 loss_delta_inference=0\n";
            for (const float scale : { 0.25f, 0.5f, 2.0f }) {
                set_all_initial_parameters(snapshots);
                set_parameters_at_update_scale(snapshots, final_a, final_b, scale);
                const double direction_loss_inference = evaluate_token_loss(
                    eval_ctx.get(), tokens, llama_model_get_vocab(model.get()), context_length);
                std::cout << std::setprecision(10)
                          << "[DIAG-DIRECTION] scale=" << scale
                          << " inference_loss=" << direction_loss_inference
                          << " loss_delta_inference=" << (direction_loss_inference - baseline_inference)
                          << " squared_parameter_update=" << (scale * scale * squared_parameter_update)
                          << '\n';
            }
            std::cout << std::setprecision(10)
                      << "[DIAG-DIRECTION] scale=1 training_loss=" << post_step_training_loss
                      << " inference_loss=" << post_step_inference_loss
                      << " loss_delta_training=" << (post_step_training_loss - train_loss)
                      << " loss_delta_inference=" << (post_step_inference_loss - initial_loss)
                      << " squared_parameter_update=" << squared_parameter_update << '\n';

            set_all_initial_parameters(snapshots);
            const double numeric_gradient = numeric_gradient_training;
            const double abs_error = std::fabs(autodiff_gradient - numeric_gradient);
            const double relative_error = abs_error / std::max(1.0e-4,
                std::max(std::fabs(static_cast<double>(autodiff_gradient)), std::fabs(numeric_gradient)));
            require(std::isfinite(numeric_gradient) && std::isfinite(autodiff_gradient) &&
                    std::isfinite(training_graph_loss),
                    "gradient comparison produced a non-finite value");

            std::cout << std::setprecision(10)
                      << "[DIAG] tensor=" << chosen.name
                      << " parameter=" << selected_index
                      << " initial_parameter=" << initial_parameter
                      << " updated_parameter=" << updated_parameter
                      << " train_loss=" << train_loss
                      << " inference_loss=" << initial_loss
                      << " training_graph_loss=" << training_graph_loss
                      << " autodiff=" << autodiff_gradient
                      << " numeric_gradient_training_graph=" << numeric_gradient_training
                      << " numeric_gradient_inference=" << numeric_gradient_inference
                      << " abs_error=" << abs_error
                      << " relative_error=" << relative_error << '\n';

            require(relative_error <= 0.15f || abs_error <= 2.0e-3f,
                    "model LoRA gradient disagrees with finite differences: tensor=" + chosen.name +
                    " parameter=" + std::to_string(selected_index) +
                    " autodiff=" + std::to_string(autodiff_gradient) +
                    " numeric=" + std::to_string(numeric_gradient) +
                    " relative_error=" + std::to_string(relative_error));

            std::cout << "[PASS] model-backed LoRA gradient finite difference: tensor=" << chosen.name
                      << " parameter=" << selected_index
                      << " training_loss=" << train_loss
                      << " inference_loss=" << initial_loss
                      << " training_graph_loss=" << training_graph_loss
                      << " autodiff=" << autodiff_gradient
                      << " finite_difference=" << numeric_gradient
                      << " relative_error=" << relative_error
                      << " max_A_update=" << max_a_update << '\n';
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
