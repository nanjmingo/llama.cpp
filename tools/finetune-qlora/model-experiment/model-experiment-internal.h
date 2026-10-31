#pragma once

#include "model-experiment.h"

#include "../adapter.h"
#include "../dataset-window.h"
#include "../../src/llama-adapter.h"
#include "../../src/llama-context.h"
#include "../../src/llama-model.h"

#include "ggml-backend.h"
#include "ggml-opt.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace qlora_model_experiment {
namespace detail {

constexpr int32_t kRequestedContext = 256;
constexpr int32_t kRank = 2;
constexpr float kAlpha = 2.0f;

void require(bool condition, const std::string & message);
std::vector<float> tensor_to_vector(ggml_tensor * tensor);
void vector_to_tensor(ggml_tensor * tensor, const std::vector<float> & values);
std::vector<llama_token> tokenize_sample(const llama_vocab * vocab, int32_t context_length);
llama_context_params context_params(ggml_type type_k, ggml_type type_v, bool flash_attention, bool gpu_offload);

template <typename T, void (*FreeFn)(T *)>
using llama_unique_ptr = std::unique_ptr<T, decltype(FreeFn)>;

struct BackendScope {
    BackendScope();
    ~BackendScope();
};

struct QuietLibraryLogs {
    ggml_log_callback previous_callback = nullptr;
    void * previous_user_data = nullptr;
    bool verbose = false;

    static void callback(ggml_log_level level, const char * text, void * user_data);
    explicit QuietLibraryLogs(bool verbose_logs);
    ~QuietLibraryLogs();
};

struct TemporaryAdapterFile {
    std::filesystem::path path;
    ~TemporaryAdapterFile();
};

struct AdapterContextDetacher {
    llama_context * train = nullptr;
    llama_context * eval = nullptr;
    ~AdapterContextDetacher();
};

struct ParameterPairState {
    std::string name;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    std::vector<float> initial_a;
    std::vector<float> initial_b;
};

} // namespace detail

struct Model::Impl {
    ModelOptions options;
    ModelPrecisionConfig precision;
    ggml_prec active_activation_precision = GGML_PREC_F32;
    detail::QuietLibraryLogs quiet_logs;
    detail::BackendScope backend;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{ nullptr, llama_model_free };
    detail::TemporaryAdapterFile temporary_adapter_file;
    std::unique_ptr<ggml_opt_dataset, decltype(&ggml_opt_dataset_free)> dataset{ nullptr, ggml_opt_dataset_free };
    std::vector<llama_token> tokens;
    int32_t context_length = 0;

    explicit Impl(ModelOptions opts, ModelPrecisionConfig precision_config);
    void set_activation_precision(ggml_prec precision);
};

struct Experiment::Impl {
    Model::Impl * model = nullptr;
    std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)> adapter{ nullptr, llama_adapter_lora_free };
    std::unique_ptr<llama_context, decltype(&llama_free)> train_ctx{ nullptr, llama_free };
    std::unique_ptr<llama_context, decltype(&llama_free)> eval_ctx{ nullptr, llama_free };
    detail::AdapterContextDetacher detacher;
    std::vector<detail::ParameterPairState> pairs;
    std::vector<AdapterPairSnapshot> snapshots;
    ggml_opt_optimizer_params optimizer_params{};
    Impl(Model::Impl & shared, ContextMode mode);
};

} // namespace qlora_model_experiment
