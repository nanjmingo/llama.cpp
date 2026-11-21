#pragma once

#include "training-internal.h"

class qlora_training_callback_scope {
public:
    explicit qlora_training_callback_scope(qlora_training_state & state);
    ~qlora_training_callback_scope();

    qlora_training_callback_scope(const qlora_training_callback_scope &) = delete;
    qlora_training_callback_scope & operator=(const qlora_training_callback_scope &) = delete;

private:
    qlora_training_state * previous_ = nullptr;
};

bool qlora_should_abort(void * user_data);
void qlora_collect_save_result(qlora_training_state & state, bool wait);
bool qlora_prepare_save_job(qlora_training_state & state,
                            bool write_snapshot,
                            bool write_checkpoint,
                            bool write_final_adapter,
                            int64_t next_epoch,
                            int64_t next_window,
                            std::shared_ptr<qlora_save_job> & job,
                            bool * validation_failed = nullptr);
bool qlora_submit_save_job(qlora_training_state & state, std::shared_ptr<qlora_save_job> job);

void qlora_epoch_callback_progress_line(bool train,
                                        ggml_opt_context_t opt_ctx,
                                        ggml_opt_dataset_t dataset,
                                        ggml_opt_result_t result,
                                        int64_t ibatch,
                                        int64_t ibatch_max,
                                        int64_t t_start_us);
void qlora_save_step_callback(bool train,
                              ggml_opt_context_t opt_ctx,
                              ggml_opt_dataset_t dataset,
                              ggml_opt_result_t result,
                              int64_t ibatch,
                              int64_t ibatch_max,
                              int64_t t_start_us);
void qlora_log_epoch_loss(int epoch,
                          int total_epochs,
                          double training_loss_sum,
                          double context_size,
                          double supervised_weight_sum);
