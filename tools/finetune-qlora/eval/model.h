#pragma once

#include "ggml.h"

#include <memory>
#include <string>
#include <vector>

namespace qlora_model_eval {

struct ModelOptions {
    std::string model_path;
    std::vector<std::string> lora_targets = {
        "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"
    };
    bool verbose_library_logs = false;
    bool report_configuration = false;
    bool flash_attention = false;
    int32_t n_gpu_layers = 0;
};

struct ModelPrecisionConfig {
    std::string forward_precision_name = "f32";
    ggml_prec forward_precision = GGML_PREC_F32;
    std::string backward_precision_name = "f32";
    ggml_prec backward_precision = GGML_PREC_F32;
    ggml_type inference_cache_type_k = GGML_TYPE_F32;
    ggml_type inference_cache_type_v = GGML_TYPE_F32;
};

enum class EvaluationMode {
    InferenceOnly,
    TrainingAndInference,
};

class Model {
public:
    struct Impl;

    explicit Model(ModelOptions options, ModelPrecisionConfig precision = {});
    ~Model();

    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    void set_forward_precision(ggml_prec precision);

private:
    std::unique_ptr<Impl> impl_;

    friend class EvaluationRun;
};

class EvaluationRun {
public:
    struct Impl;

    EvaluationRun(Model & model, EvaluationMode mode);
    ~EvaluationRun();

    EvaluationRun(EvaluationRun &&) noexcept;
    EvaluationRun & operator=(EvaluationRun &&) noexcept;
    EvaluationRun(const EvaluationRun &) = delete;
    EvaluationRun & operator=(const EvaluationRun &) = delete;

    double evaluate_inference_loss();
    void train_one_step(float learning_rate);

private:
    std::unique_ptr<Impl> impl_;
};

}
