#include "training-callback.h"

#include "checkpoint.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local qlora_training_state * g_qlora_training_state = nullptr;

}

static std::string format_four_significant_digits(double value);
static void split_progress_time(int64_t elapsed_us, int64_t & hours, int64_t & minutes, int64_t & seconds);

qlora_training_callback_scope::qlora_training_callback_scope(qlora_training_state & state)
    : previous_(g_qlora_training_state) {
    g_qlora_training_state = &state;
}

qlora_training_callback_scope::~qlora_training_callback_scope() {
    g_qlora_training_state = previous_;
}

bool qlora_should_abort(void * user_data) {
    const qlora_training_state * state = static_cast<const qlora_training_state *>(user_data);
    return !state || state->abort_requested;
}

static bool is_optimizer_update(const qlora_training_state & state, int64_t ibatch) {
    const int64_t absolute_window = state.window_start + ibatch;
    return absolute_window % state.gradient_accumulation_steps == 0 || absolute_window == state.dataset_windows;
}

void qlora_collect_save_result(qlora_training_state & state, bool wait) {
    if (!state.worker || !state.active_job) return;
    qlora_save_result result;
    const bool ready = wait ? state.worker->wait(result) : state.worker->poll(result);
    if (!ready) return;

    const std::shared_ptr<qlora_save_job> job = std::move(state.active_job);
    if (job->snapshot_saved) {
        state.cached_adapter_step = job->step;
        state.cached_adapter_path = job->snapshot_path;
        LOG_INF("saved snapshot (optimizer step %ld)\n", static_cast<long>(job->step));
    }
    if (job->checkpoint_saved) {
        state.last_checkpoint_step = job->step;
        state.last_checkpoint_next_epoch = job->next_epoch;
        state.last_checkpoint_next_window = job->next_window;
        state.cached_adapter_step = job->step;
        state.cached_adapter_path = qlora_checkpoint_path(state.output_root, job->step) / "adapter.gguf";
        if (job->has_training_progress) {
            const std::string loss_text = format_four_significant_digits(job->progress_loss);
            const std::string average_loss_text = format_four_significant_digits(job->progress_average_loss);
            int64_t elapsed_h = 0;
            int64_t elapsed_m = 0;
            int64_t elapsed_s = 0;
            split_progress_time(job->progress_elapsed_us, elapsed_h, elapsed_m, elapsed_s);
            LOG_INF("saved resumable checkpoint (optimizer step %ld, loss=%s avg=%s "
                    "elapsed=%02" PRId64 ":%02" PRId64 ":%02" PRId64 ")\n",
                    static_cast<long>(job->step), loss_text.c_str(), average_loss_text.c_str(),
                    elapsed_h, elapsed_m, elapsed_s);
        } else {
            LOG_INF("saved resumable checkpoint (optimizer step %ld)\n", static_cast<long>(job->step));
        }
    }
    if (job->final_adapter_saved) {
        state.cached_adapter_step = job->step;
        state.cached_adapter_path = job->final_adapter_path;
    }
    if (!result.success) {
        LOG_ERR("asynchronous save failed: %s; training will continue\n", result.error.c_str());
    }
}

bool qlora_prepare_save_job(qlora_training_state & state,
                            bool write_snapshot,
                            bool write_checkpoint,
                            bool write_final_adapter,
                            int64_t next_epoch,
                            int64_t next_window,
                            std::shared_ptr<qlora_save_job> & job,
                            bool * validation_failed) {
    try {
        if (validation_failed) *validation_failed = false;
        auto prepared = std::make_shared<qlora_save_job>();
        prepared->output_root = state.output_root;
        prepared->snapshot_limit = state.save_snapshot_limit;
        prepared->checkpoint_limit = state.save_checkpoint_limit;
        prepared->step = state.schedule->step;
        prepared->progress_loss = state.last_training_loss;
        prepared->progress_average_loss = state.last_training_average_loss;
        prepared->progress_elapsed_us = state.last_training_elapsed_us;
        prepared->has_training_progress = state.has_training_progress;
        prepared->next_epoch = next_epoch;
        prepared->next_window = next_window;
        prepared->write_snapshot = write_snapshot;
        prepared->write_checkpoint = write_checkpoint;
        prepared->write_final_adapter = write_final_adapter;
        prepared->write_grad_stats_log = state.qparams->grad_stats_interval > 0;
        if (write_snapshot || write_checkpoint) {
            prepared->loss_log = qlora_serialize_loss_log(state.loss_history);
            if (prepared->write_grad_stats_log) {
                prepared->grad_stats_log = qlora_serialize_grad_stats_log(state.grad_stats_history);
            }
        }
        if (write_snapshot) {
            prepared->snapshot_path = qlora_snapshot_path(state.output_root, prepared->step);
        }
        if (write_final_adapter) {
            prepared->final_adapter_path = state.output_root / "adapter.gguf";
        }

        if (write_snapshot || write_checkpoint) {
            std::vector<uint8_t> snapshot_validation_state;
            std::vector<uint8_t> & optimizer_state = write_checkpoint
                ? prepared->optimizer_state
                : snapshot_validation_state;
            const size_t state_size = llama_opt_state_size(state.ctx);
            if (state_size == 0) {
                if (validation_failed) *validation_failed = true;
                LOG_ERR("%s: optimizer state is not available at a valid optimizer-step boundary\n", __func__);
                return false;
            }
            optimizer_state.resize(state_size);
            if (!llama_opt_state_export(state.ctx, optimizer_state.data(), optimizer_state.size())) {
                LOG_ERR("%s: failed to export optimizer state\n", __func__);
                return false;
            }
            if (!ggml_opt_state_data_is_finite(optimizer_state.data(), optimizer_state.size())) {
                if (validation_failed) *validation_failed = true;
                LOG_ERR("%s: serialized optimizer state contains non-finite values or is invalid\n", __func__);
                return false;
            }
        }
        std::string validation_error;
        if (!capture_lora_adapter_snapshot(*state.output_tensors, *state.arch, state.lora_alpha,
                                           *state.base_model_path, prepared->adapter, &validation_error)) {
            if (validation_failed) *validation_failed = true;
            LOG_ERR("%s: refusing to save invalid adapter tensors: %s\n", __func__, validation_error.c_str());
            return false;
        }

        if (write_checkpoint) {
            const int64_t permutation_size = ggml_opt_dataset_permutation_size(state.dataset);
            if (permutation_size != state.dataset_windows) return false;
            std::vector<int64_t> permutation(static_cast<size_t>(permutation_size));
            if (!ggml_opt_dataset_get_permutation(state.dataset, permutation.data(), permutation_size)) return false;

            common_json metadata = common_json::object();
            try {
                metadata["format_version"] = 4;
                qlora_file_identity base_model_identity;
                if (!qlora_get_file_identity(*state.base_model_path, base_model_identity)) {
                    throw std::runtime_error("cannot inspect base model file: " + *state.base_model_path);
                }
                common_json base_model_file = common_json::object();
                base_model_file["path"] = base_model_identity.path;
                base_model_file["size"] = base_model_identity.size;
                metadata["base_model_path"] = qlora_canonical_path_string(*state.base_model_path);
                metadata["base_model_file"] = base_model_file;
                metadata["architecture"] = *state.arch;
                metadata["lora_rank"] = state.qparams->lora_rank;
                metadata["lora_alpha"] = state.lora_alpha;
                metadata["lora_targets"] = state.qparams->lora_targets;
                metadata["freeze_layers"] = state.qparams->freeze_layers;
                metadata["gradient_accumulation_steps"] = state.qparams->gradient_accumulation_steps;
                metadata["grad_checkpoint_interval"] = state.qparams->grad_checkpoint_interval;
                metadata["max_steps"] = state.qparams->max_steps;
                metadata["val_windows"] = state.qparams->val_windows;
                metadata["context_length"] = llama_n_ctx(state.ctx);
                metadata["dataset_windows"] = state.dataset_windows;
                metadata["dataset_files"] = qlora_make_dataset_file_metadata(*state.qparams);
                metadata["shuffle_dataset"] = state.qparams->shuffle_dataset;
                metadata["skip_invalid_records"] = state.qparams->dataset.skip_invalid;
                metadata["lr_scheduler"] = state.qparams->lr_scheduler;
                metadata["warmup_steps"] = state.qparams->warmup_steps;
                metadata["warmup_init_ratio"] = state.qparams->warmup_init_ratio;
                metadata["learning_rate"] = state.params->lr.lr0;
                metadata["learning_rate_min"] = state.params->lr.lr_min;
                metadata["weight_decay"] = state.params->lr.wd;
                metadata["optimizer"] = ggml_opt_optimizer_name(state.params->optimizer);
                prepared->optimizer_name = ggml_opt_optimizer_name(state.params->optimizer);
                prepared->optimizer_state_type = state.qparams->optimizer_state_type;
                metadata["lora_type"] = state.qparams->lora_type;
                metadata["optimizer_state_type"] = state.qparams->optimizer_state_type;
                metadata["forward_precision"] = state.qparams->forward_precision_name;
                metadata["backward_precision"] = state.qparams->backward_precision_name;
                const ggml_opt_optimizer_params optimizer_defaults = ggml_opt_get_default_optimizer_params(nullptr);
                metadata["adamw_beta1"] = optimizer_defaults.adamw.beta1;
                metadata["adamw_beta2"] = optimizer_defaults.adamw.beta2;
                metadata["adamw_eps"] = optimizer_defaults.adamw.eps;
                metadata["total_epochs"] = state.params->lr.epochs;
                metadata["next_epoch"] = next_epoch;
                metadata["next_window"] = next_window;
                metadata["scheduler_step"] = state.schedule->step;
                metadata["training_statistics_complete"] = state.training_statistics_complete;
                metadata["training_loss_sum"] = state.training_loss_sum;
                metadata["training_data_count"] = state.training_data_count;
                metadata["training_weight_sum"] = state.training_weight_sum;
                common_json order = common_json::array();
                for (const int64_t id : permutation) order.push_back(id);
                metadata["dataset_permutation"] = order;
            } catch (const std::exception & ex) {
                LOG_ERR("%s: failed to build checkpoint metadata: %s\n", __func__, ex.what());
                return false;
            }
            prepared->training_state_json = metadata.dump(2) + "\n";
            common_json manifest = common_json::object();
            manifest["format_version"] = 2;
            manifest["optimizer_file"] = "optimizer.gguf";
            manifest["adapter_file"] = "adapter.gguf";
            manifest["training_state_file"] = "training-state.json";
            manifest["optimizer_step"] = prepared->step;
            prepared->manifest_json = manifest.dump(2) + "\n";
        }

        job = std::move(prepared);
        return true;
    } catch (const std::exception & ex) {
        LOG_ERR("%s: failed to stage asynchronous save data: %s\n", __func__, ex.what());
        return false;
    } catch (...) {
        LOG_ERR("%s: failed to stage asynchronous save data\n", __func__);
        return false;
    }
}

bool qlora_submit_save_job(qlora_training_state & state, std::shared_ptr<qlora_save_job> job) {
    try {
        qlora_collect_save_result(state, true);
        if (state.abort_requested) return false;
        if (!state.worker || !state.worker->submit([job](std::string & error) {
                return write_qlora_save_job(job, error);
            })) {
            LOG_ERR("%s: could not submit asynchronous save job\n", __func__);
            return false;
        }
        state.active_job = std::move(job);
        return true;
    } catch (const std::exception & ex) {
        LOG_ERR("%s: failed to submit asynchronous save job: %s\n", __func__, ex.what());
        return false;
    } catch (...) {
        LOG_ERR("%s: failed to submit asynchronous save job\n", __func__);
        return false;
    }
}

static void update_training_step(qlora_training_state & state, int64_t ibatch) {
    if (is_optimizer_update(state, ibatch)) {
        state.schedule->advance_step();
    }
}

static void save_due_state(qlora_training_state & state, int64_t ibatch) {
    const bool optimizer_update = is_optimizer_update(state, ibatch);
    if (!optimizer_update || state.schedule->step <= 0) return;

    const bool snapshot_due = state.save_snapshot_limit > 0 && state.save_snapshot_steps > 0 &&
        state.schedule->step % state.save_snapshot_steps == 0 && state.cached_adapter_step != state.schedule->step;
    const int64_t consumed = state.window_start + ibatch;
    const int64_t next_epoch = consumed == state.dataset_windows ? state.epoch : state.epoch - 1;
    const int64_t next_window = consumed == state.dataset_windows ? 0 : consumed;
    const bool checkpoint_due = state.save_checkpoint_limit > 0 && state.save_checkpoint_steps > 0 &&
        state.schedule->step % state.save_checkpoint_steps == 0 &&
        !(state.last_checkpoint_step == state.schedule->step &&
          state.last_checkpoint_next_epoch == next_epoch && state.last_checkpoint_next_window == next_window);
    if (!snapshot_due && !checkpoint_due) return;

    qlora_collect_save_result(state, true);
    if (state.abort_requested) return;

    std::shared_ptr<qlora_save_job> job;
    bool validation_failed = false;
    if (!qlora_prepare_save_job(state, snapshot_due, checkpoint_due, false,
                                next_epoch, next_window, job, &validation_failed)) {
        if (validation_failed) {
            state.abort_requested = true;
            LOG_ERR("periodic save validation failed; stopping training without writing a final adapter or checkpoint\n");
        } else {
            LOG_ERR("periodic save snapshot preparation failed; training will continue\n");
        }
        return;
    }
    if (!qlora_submit_save_job(state, std::move(job))) {
        LOG_ERR("periodic save job could not be queued\n");
    }
}

static int64_t progress_number_width(int64_t value) {
    return static_cast<int64_t>(std::to_string(std::max<int64_t>(value, 1)).size());
}

static double calculate_weighted_training_loss(double loss_sum, double context_size, double supervised_weight_sum) {
    if (context_size <= 0.0 || supervised_weight_sum <= 0.0) {
        return 0.0;
    }
    return loss_sum * context_size / supervised_weight_sum;
}

static std::string format_four_significant_digits(double value) {
    if (!std::isfinite(value)) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    }
    if (value == 0.0) {
        return "0.000";
    }

    int exponent = static_cast<int>(std::floor(std::log10(std::fabs(value))));
    if (exponent >= -4 && exponent <= 3) {
        const double scale = std::pow(10.0, 3 - exponent);
        const double rounded = std::round(std::fabs(value) * scale) / scale;
        if (rounded >= std::pow(10.0, exponent + 1)) {
            ++exponent;
        }
    }

    std::ostringstream stream;
    if (exponent >= 4 || exponent < -4) {
        stream << std::scientific << std::setprecision(3) << value;
    } else {
        stream << std::fixed << std::setprecision(3 - exponent) << value;
    }
    return stream.str();
}

static void split_progress_time(int64_t elapsed_us, int64_t & hours, int64_t & minutes, int64_t & seconds) {
    int64_t remaining_seconds = std::max<int64_t>(elapsed_us, 0) / 1000000;
    hours = remaining_seconds / 3600;
    remaining_seconds -= hours * 3600;
    minutes = remaining_seconds / 60;
    seconds = remaining_seconds - minutes * 60;
}

static bool get_window_training_statistics(const qlora_training_state & state,
                                    int64_t window_index,
                                    int64_t & supervised,
                                    double & weight_sum) {
    const ggml_tensor * labels = ggml_opt_dataset_labels(state.dataset);
    const ggml_tensor * weights = ggml_opt_dataset_weights(state.dataset);
    if (!labels || labels->type != GGML_TYPE_I32 || labels->ne[1] != state.dataset_windows ||
        window_index < 0 || window_index >= state.dataset_windows ||
        state.dataset_permutation.size() != static_cast<size_t>(state.dataset_windows)) {
        return false;
    }
    if (weights && (weights->type != GGML_TYPE_F32 || weights->ne[0] != labels->ne[0] ||
                    weights->ne[1] != labels->ne[1])) {
        return false;
    }

    const int64_t data_index = state.dataset_permutation[static_cast<size_t>(window_index)];
    if (data_index < 0 || data_index >= labels->ne[1]) {
        return false;
    }

    const int32_t * label_data = static_cast<const int32_t *>(labels->data) + data_index * labels->ne[0];
    const float * weight_data = weights ?
        static_cast<const float *>(weights->data) + data_index * weights->ne[0] : nullptr;
    supervised = 0;
    weight_sum = 0.0;
    for (int64_t i = 0; i < labels->ne[0]; ++i) {
        if (label_data[i] < 0) {
            continue;
        }
        const float weight = weight_data ? weight_data[i] : 1.0f;
        if (!std::isfinite(weight) || weight < 0.0f) {
            return false;
        }
        if (weight > 0.0f) {
            ++supervised;
            weight_sum += weight;
        }
    }
    return true;
}

void qlora_epoch_callback_progress_line(bool               train,
                                       ggml_opt_context_t opt_ctx,
                                       ggml_opt_dataset_t dataset,
                                       ggml_opt_result_t  result,
                                       int64_t            ibatch,
                                       int64_t            ibatch_max,
                                       int64_t            t_start_us) {
    if (!g_qlora_training_state) {
        ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
        return;
    }

    qlora_training_state & state = *g_qlora_training_state;
    if (!train && !state.validation_progress_started) {
        fprintf(stderr, "\n");
        state.validation_progress_started = true;
    }

    const int64_t window_start = train ? state.window_start : state.validation_window_start;
    const int64_t window_index = window_start + ibatch - 1;
    const int64_t current_window = window_index + 1;
    double loss = 0.0;
    ggml_opt_result_last_loss(result, &loss);
    if (!std::isfinite(loss)) {
        state.abort_requested = true;
        LOG_ERR("%s: non-finite %s loss at epoch %ld window %ld; stopping training without final saves\n",
                __func__, train ? "training" : "validation", static_cast<long>(state.epoch),
                static_cast<long>(current_window));
        return;
    }

    int64_t supervised = 0;
    double weight_sum = 0.0;
    if (!get_window_training_statistics(state, window_index, supervised, weight_sum)) {
        LOG_ERR("%s: failed to read statistics for window %ld\n", __func__, static_cast<long>(current_window));
        return;
    }

    double average_loss = 0.0;
    const double context_size = static_cast<double>(llama_n_ctx(state.ctx));
    if (train) {
        state.training_loss_sum += loss;
        state.training_data_count += llama_n_ctx(state.ctx);
        state.training_weight_sum += weight_sum;
        average_loss = calculate_weighted_training_loss(
            state.training_loss_sum, context_size, state.training_weight_sum);
    } else {
        ggml_opt_result_loss(result, &average_loss, nullptr);
        state.validation_weight_sum += weight_sum;
        if (weight_sum > 0.0 && state.validation_weight_sum > 0.0 && ibatch > 0) {
            average_loss *= context_size * static_cast<double>(ibatch) / state.validation_weight_sum;
        }
    }
    if (!std::isfinite(average_loss) || !std::isfinite(state.training_loss_sum) ||
        !std::isfinite(state.training_weight_sum) || !std::isfinite(state.validation_weight_sum) ||
        !std::isfinite(weight_sum)) {
        state.abort_requested = true;
        LOG_ERR("%s: non-finite accumulated %s loss at epoch %ld window %ld; stopping training without final saves\n",
                __func__, train ? "training" : "validation", static_cast<long>(state.epoch),
                static_cast<long>(current_window));
        return;
    }
    if (weight_sum > 0.0) {
        loss *= context_size / weight_sum;
    }
    if (!std::isfinite(loss)) {
        state.abort_requested = true;
        LOG_ERR("%s: non-finite weighted %s loss at epoch %ld window %ld; stopping training without final saves\n",
                __func__, train ? "training" : "validation", static_cast<long>(state.epoch),
                static_cast<long>(current_window));
        return;
    }
    state.loss_history.push_back({ train, state.epoch, current_window, loss, average_loss });
    const std::string loss_text = format_four_significant_digits(loss);
    const std::string average_loss_text = format_four_significant_digits(average_loss);
    const std::string weight_sum_text = format_four_significant_digits(weight_sum);

    const int64_t elapsed_us = std::max<int64_t>(0, ggml_time_us() - t_start_us);
    if (train) {
        state.last_training_loss = loss;
        state.last_training_average_loss = average_loss;
        state.last_training_elapsed_us = elapsed_us;
        state.has_training_progress = true;
    }
    const int64_t eta_us = ibatch > 0 ? static_cast<int64_t>(
        static_cast<double>(elapsed_us) * static_cast<double>(ibatch_max - ibatch) / static_cast<double>(ibatch)) : 0;
    int64_t elapsed_h = 0;
    int64_t elapsed_m = 0;
    int64_t elapsed_s = 0;
    int64_t eta_h = 0;
    int64_t eta_m = 0;
    int64_t eta_s = 0;
    split_progress_time(elapsed_us, elapsed_h, elapsed_m, elapsed_s);
    split_progress_time(eta_us, eta_h, eta_m, eta_s);

    const int64_t window_width = progress_number_width(state.dataset_windows);
    if (train) {
        const int64_t configured_period = state.gradient_accumulation_steps;
        const int64_t accumulation_group_offset = ((ibatch - 1) / configured_period) * configured_period;
        const int64_t accumulation_period = std::min<int64_t>(
            configured_period, ibatch_max - accumulation_group_offset);
        const int64_t accumulation_step = ibatch - accumulation_group_offset;
        const int64_t accumulation_width = progress_number_width(accumulation_period);

        fprintf(stderr, "[%0*" PRId64 "/%0*" PRId64 " (%0*" PRId64 "/%0*" PRId64 ")] "
                "loss=%s avg=%s supervised=%" PRId64 " weight_sum=%s "
                "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
                static_cast<int>(window_width), current_window,
                static_cast<int>(window_width), state.dataset_windows,
                static_cast<int>(accumulation_width), accumulation_step,
                static_cast<int>(accumulation_width), accumulation_period,
                loss_text.c_str(), average_loss_text.c_str(), supervised, weight_sum_text.c_str(),
                elapsed_h, elapsed_m, elapsed_s, eta_h, eta_m, eta_s);
    } else {
        fprintf(stderr, "[val %0*" PRId64 "/%0*" PRId64 "] "
                "loss=%s avg=%s supervised=%" PRId64 " weight_sum=%s "
                "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
                static_cast<int>(window_width), current_window,
                static_cast<int>(window_width), state.dataset_windows,
                loss_text.c_str(), average_loss_text.c_str(), supervised, weight_sum_text.c_str(),
                elapsed_h, elapsed_m, elapsed_s, eta_h, eta_m, eta_s);
    }
    fflush(stderr);
}

static void log_gradient_stats_if_due(qlora_training_state & state, ggml_opt_result_t result) {
    const int32_t interval = state.qparams->grad_stats_interval;
    const int64_t step = state.schedule->step;
    if (interval <= 0 || step <= 0 || step % interval != 0) {
        return;
    }

    ggml_opt_grad_stats stats = {};
    if (!ggml_opt_result_grad_stats(result, &stats)) {
        LOG_WRN("%s: gradient statistics were not available for optimizer step %ld\n",
                __func__, static_cast<long>(step));
        return;
    }
    state.grad_stats_history.push_back({ step, stats });

    const double zero_percent = stats.n_elements > 0
        ? 100.0 * static_cast<double>(stats.n_zeros) / static_cast<double>(stats.n_elements) : 0.0;
    LOG_INF("gradient_stats step=%ld tensors=%ld elements=%ld scale=%.6g norm=%.6g rms=%.6g max_abs=%.6g "
            "zero=%.4f%% nonfinite=%ld\n",
            static_cast<long>(step), static_cast<long>(stats.n_tensors), static_cast<long>(stats.n_elements),
            stats.scale, stats.norm, stats.rms, stats.max_abs, zero_percent, static_cast<long>(stats.n_nonfinite));
}

void qlora_save_step_callback(bool               train,
                        ggml_opt_context_t opt_ctx,
                        ggml_opt_dataset_t dataset,
                        ggml_opt_result_t  result,
                        int64_t            ibatch,
                        int64_t            ibatch_max,
                        int64_t            t_start_us) {
    if (!g_qlora_training_state) {
        qlora_epoch_callback_progress_line(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
        return;
    }
    qlora_collect_save_result(*g_qlora_training_state, false);
    qlora_epoch_callback_progress_line(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
    if (!train || g_qlora_training_state->abort_requested) {
        return;
    }
    update_training_step(*g_qlora_training_state, ibatch);
    if (is_optimizer_update(*g_qlora_training_state, ibatch)) {
        log_gradient_stats_if_due(*g_qlora_training_state, result);
    }
    save_due_state(*g_qlora_training_state, ibatch);
}

void qlora_log_epoch_loss(int epoch, int total_epochs, double training_loss_sum,
                    double context_size, double supervised_weight_sum) {
    const double train_loss = calculate_weighted_training_loss(
        training_loss_sum, context_size, supervised_weight_sum);
    const std::string train_loss_text = format_four_significant_digits(train_loss);
    LOG_INF("epoch %d/%d: train_loss=%s\n", epoch, total_epochs, train_loss_text.c_str());
}
