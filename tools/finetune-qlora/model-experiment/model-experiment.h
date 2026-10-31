#pragma once

#include "ggml.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace qlora_model_experiment {

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
    std::string activation_precision_name = "f32";
    ggml_prec activation_precision = GGML_PREC_F32;
    ggml_type inference_cache_type_k = GGML_TYPE_F32;
    ggml_type inference_cache_type_v = GGML_TYPE_F32;
};

struct AdapterPairSnapshot {
    std::string name;
    std::vector<float> initial_a;
    std::vector<float> initial_b;
};

struct StepResult {
    float learning_rate = 0.0f;
    double training_loss = 0.0;
    float max_a_update = 0.0f;
    float max_b_update = 0.0f;
    double squared_parameter_update = 0.0;
    size_t selected_pair = 0;
    size_t selected_index = 0;
    std::vector<std::string> pair_names;
    std::vector<std::vector<float>> initial_a;
    std::vector<std::vector<float>> initial_b;
    std::vector<std::vector<float>> final_a;
    std::vector<std::vector<float>> final_b;
};

struct UpdateProbe {
    size_t pair_index = 0;
    size_t tensor_index = 0;
    float update = 0.0f;
};

enum class ContextMode {
    TrainingOnly,
    InferenceOnly,
    TrainingAndInference,
};

struct FiniteDifferenceResult {
    float observed_positive_parameter = 0.0f;
    float observed_negative_parameter = 0.0f;
    double positive_training_loss = 0.0;
    double negative_training_loss = 0.0;
    double positive_inference_loss = 0.0;
    double negative_inference_loss = 0.0;
    double numeric_training_gradient = 0.0;
    double numeric_inference_gradient = 0.0;
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

    int32_t context_length() const;
    void set_activation_precision(ggml_prec precision);

private:
    std::unique_ptr<Impl> impl_;

    friend class Experiment;
};

class Experiment {
public:
    struct Impl;

    Experiment(Model & model, ContextMode mode);
    ~Experiment();

    Experiment(Experiment &&) noexcept;
    Experiment & operator=(Experiment &&) noexcept;
    Experiment(const Experiment &) = delete;
    Experiment & operator=(const Experiment &) = delete;

    const std::vector<AdapterPairSnapshot> & snapshots() const;
    double evaluate_inference_loss();
    double evaluate_training_graph_loss();
    StepResult train_one_step(float learning_rate);
    float b_parameter(size_t pair_index, size_t tensor_index);
    void set_b_parameter(size_t pair_index, size_t tensor_index, float value);
    void set_parameters_at_update_scale(const StepResult & step, float scale);

private:
    std::unique_ptr<Impl> impl_;
};


std::vector<UpdateProbe> select_update_probes(const StepResult & step, size_t count);
FiniteDifferenceResult evaluate_finite_difference(Model & model, size_t pair_index,
                                                   size_t tensor_index, float center, float epsilon,
                                                   ContextMode mode);
void gradient_error(double estimated_gradient, double numeric_gradient,
                    double & absolute_error, double & relative_error);
bool gradient_matches(double estimated_gradient, double numeric_gradient,
                      double & absolute_error, double & relative_error);

} // namespace qlora_model_experiment
