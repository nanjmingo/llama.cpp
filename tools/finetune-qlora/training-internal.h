#pragma once

#include "adapter.h"
#include "args.h"
#include "common.h"
#include "ggml-opt.h"
#include "llama.h"
#include "save-worker.h"
#include "schedule.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct qlora_loss_log_entry {
    bool training = true;
    int64_t epoch = 0;
    int64_t window = 0;
    double loss = 0.0;
    double average_loss = 0.0;
};

struct qlora_grad_stats_log_entry {
    int64_t step = 0;
    ggml_opt_grad_stats stats = {};
};

struct qlora_save_job {
    std::filesystem::path output_root;
    std::filesystem::path snapshot_path;
    std::filesystem::path final_adapter_path;
    int32_t snapshot_limit = 0;
    int32_t checkpoint_limit = 0;
    int64_t step = 0;
    double progress_loss = 0.0;
    double progress_average_loss = 0.0;
    int64_t progress_elapsed_us = 0;
    bool has_training_progress = false;
    int64_t next_epoch = 0;
    int64_t next_window = 0;
    bool write_snapshot = false;
    bool write_checkpoint = false;
    bool write_final_adapter = false;
    bool write_grad_stats_log = false;
    bool snapshot_saved = false;
    bool checkpoint_saved = false;
    bool final_adapter_saved = false;
    lora_adapter_snapshot adapter;
    std::vector<uint8_t> optimizer_state;
    std::string optimizer_name;
    std::string optimizer_state_type;
    std::string loss_log;
    std::string grad_stats_log;
    std::string training_state_json;
    std::string manifest_json;
};

struct qlora_training_state {
    const lora_tensor_refs * output_tensors = nullptr;
    std::filesystem::path output_root;
    const std::string * arch = nullptr;
    const std::string * base_model_path = nullptr;
    const qlora_params * qparams = nullptr;
    const common_params * params = nullptr;
    llama_context * ctx = nullptr;
    ggml_opt_dataset_t dataset = nullptr;
    float lora_alpha = 0.0f;
    int32_t save_checkpoint_steps = 0;
    int32_t save_checkpoint_limit = 0;
    int32_t save_snapshot_steps = 0;
    int32_t save_snapshot_limit = 0;
    int32_t gradient_accumulation_steps = 1;
    int64_t dataset_windows = 0;
    std::vector<int64_t> dataset_permutation;
    int64_t epoch = 0;
    int64_t window_start = 0;
    int64_t validation_window_start = 0;
    bool validation_progress_started = false;
    double training_loss_sum = 0.0;
    int64_t training_data_count = 0;
    double last_training_loss = 0.0;
    double last_training_average_loss = 0.0;
    int64_t last_training_elapsed_us = 0;
    bool has_training_progress = false;
    double training_weight_sum = 0.0;
    bool training_statistics_complete = true;
    double validation_weight_sum = 0.0;
    std::vector<qlora_loss_log_entry> loss_history;
    std::vector<qlora_grad_stats_log_entry> grad_stats_history;
    bool abort_requested = false;
    int64_t last_checkpoint_step = -1;
    int64_t last_checkpoint_next_epoch = -1;
    int64_t last_checkpoint_next_window = -1;
    int64_t cached_adapter_step = -1;
    std::filesystem::path cached_adapter_path;
    qlora_lr_schedule * schedule = nullptr;
    qlora_save_worker * worker = nullptr;
    std::shared_ptr<qlora_save_job> active_job;
};

std::string qlora_serialize_loss_log(const std::vector<qlora_loss_log_entry> & history);
std::string qlora_serialize_grad_stats_log(const std::vector<qlora_grad_stats_log_entry> & history);

bool write_qlora_save_job(const std::shared_ptr<qlora_save_job> & job, std::string & error);
