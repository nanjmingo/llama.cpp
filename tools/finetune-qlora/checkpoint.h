#pragma once

#include "args.h"
#include "common.h"
#include "json.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct qlora_checkpoint {
    std::filesystem::path directory;
    common_json metadata = common_json::object();
    std::vector<uint8_t> optimizer_state;
    std::vector<int64_t> dataset_permutation;
    int64_t next_epoch = 0;
    int64_t next_window = 0;
    int64_t scheduler_step = 0;
    int32_t format_version = 0;
    bool training_statistics_complete = true;
    double training_loss_sum = 0.0;
    int64_t training_data_count = 0;
    double training_weight_sum = 0.0;
};

struct qlora_file_identity {
    std::string path;
    int64_t size = -1;
};

std::string qlora_canonical_path_string(const std::filesystem::path & path);
bool qlora_get_file_identity(const std::string & path, qlora_file_identity & result);
common_json qlora_make_dataset_file_metadata(const qlora_params & qparams);

bool is_checkpoint_directory(const std::filesystem::path & directory, int64_t * optimizer_step = nullptr);
bool recover_checkpoint_directories(const std::filesystem::path & root);
bool load_qlora_checkpoint(const std::string & requested_path, qlora_checkpoint & result);
bool validate_resume_config(const qlora_checkpoint & checkpoint,
                            const qlora_params & qparams,
                            const common_params & params,
                            const std::string & arch,
                            float lora_alpha,
                            int32_t n_ctx,
                            int64_t dataset_windows);

bool initialize_training_logs(const std::filesystem::path & output_root, bool write_grad_stats_log);
bool install_file_alias(const std::filesystem::path & source, const std::filesystem::path & destination);
std::filesystem::path qlora_checkpoint_path(const std::filesystem::path & output_root, int64_t step);
std::filesystem::path qlora_snapshot_path(const std::filesystem::path & output_root, int64_t step);
