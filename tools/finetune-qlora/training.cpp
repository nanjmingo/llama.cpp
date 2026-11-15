#include "training.h"

#include "checkpoint.h"
#include "dataset-pipeline.h"
#include "dataset-window.h"
#include "log.h"
#include "schedule.h"
#include "training-callback.h"
#include "training-internal.h"

#include "common.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

enum ggml_type qlora_type_from_name(const std::string & name) {
    if (name == "f32") return GGML_TYPE_F32;
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "bf16") return GGML_TYPE_BF16;
    throw std::invalid_argument("unsupported tensor type: " + name);
}

namespace {

struct ggml_opt_dataset_deleter {
    void operator()(ggml_opt_dataset * dataset) const {
        ggml_opt_dataset_free(dataset);
    }
};

struct ggml_opt_result_deleter {
    void operator()(ggml_opt_result * result) const {
        ggml_opt_result_free(result);
    }
};

using ggml_opt_dataset_ptr = std::unique_ptr<ggml_opt_dataset, ggml_opt_dataset_deleter>;
using ggml_opt_result_ptr = std::unique_ptr<ggml_opt_result, ggml_opt_result_deleter>;

ggml_opt_dataset_ptr create_training_dataset(const qlora_params & qparams,
                                             const llama_model * model,
                                             const llama_vocab * vocab,
                                             int32_t context_length) {
    common_chat_templates_ptr templates;
    const char * model_template = llama_model_chat_template(model, nullptr);
    if (model_template && model_template[0] != '\0') {
        templates = common_chat_templates_init(model, "");
    } else {
        LOG_WRN("%s: model has no chat template; chat records cannot be rendered\n", __func__);
    }

    dataset_load_result loaded = load_training_samples(qparams.dataset, vocab, templates.get());
    if (!loaded.success || loaded.samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return {};
    }

    const llama_token bos = llama_vocab_bos(vocab);
    ggml_opt_dataset_ptr dataset(build_training_dataset(
        loaded.samples, window_options{context_length, bos, -1}));
    if (!dataset) {
        return {};
    }
    return dataset;
}


} // namespace

bool train_adapter(const qlora_params & qparams,
                   common_params & params,
                   llama_model * model,
                   llama_context * ctx,
                   const lora_tensor_refs & output_refs,
                   const std::string & arch,
                   float lora_alpha,
                   qlora_checkpoint * resume) {
    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_batch = llama_n_batch(ctx);
    const int32_t n_ubatch = llama_n_ubatch(ctx);
    if (n_ubatch != n_ctx || n_batch < n_ctx) {
        LOG_ERR("%s: QLoRA requires one full-context microbatch per training window: "
                "set -ub equal to -c and -b to at least -c (n_batch=%d, n_ubatch=%d, n_ctx=%d)\n",
                __func__, n_batch, n_ubatch, n_ctx);
        return false;
    }

    if (params.lr.lr0 <= 0.0f || (qparams.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n", __func__);
        return false;
    }

    ggml_opt_dataset_ptr dataset = create_training_dataset(qparams, model, llama_model_get_vocab(model), n_ctx);
    if (!dataset) {
        return false;
    }

    qlora_lr_schedule schedule{ &params.lr, qparams.lr_scheduler, qparams.warmup_steps,
                                qparams.warmup_init_ratio, 0, 0 };
    llama_opt_params opt_params{
        0,
        lora_param_filter,
        nullptr,
        qlora_opt_lr_pars,
        &schedule,
        params.optimizer,
        qlora_type_from_name(qparams.optimizer_state_type),
    };
    const int64_t dataset_windows = ggml_opt_dataset_ndata(dataset.get());
    if (dataset_windows <= 0) {
        LOG_ERR("%s: dataset has no training windows\n", __func__);
        return false;
    }

    const int64_t accumulation_steps = qparams.gradient_accumulation_steps;
    const int64_t optimizer_steps_per_epoch = dataset_windows / accumulation_steps +
        (dataset_windows % accumulation_steps != 0);
    if (qparams.max_steps > 0) {
        params.lr.epochs = 1 + (qparams.max_steps - 1) / optimizer_steps_per_epoch;
    }
    if (qparams.max_steps == 0 && params.lr.epochs > 0 && optimizer_steps_per_epoch > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total optimizer step count overflows int64\n", __func__);
        return false;
    }
    schedule.total_steps = qparams.max_steps > 0 ? qparams.max_steps : optimizer_steps_per_epoch * params.lr.epochs;
    if (schedule.total_steps <= 0) {
        LOG_ERR("%s: total training step count must be positive\n", __func__);
        return false;
    }

    if (resume) {
        if (!validate_resume_config(*resume, qparams, params, arch, lora_alpha, n_ctx, dataset_windows)) {
            return false;
        }
        if (!ggml_opt_dataset_set_permutation(dataset.get(), resume->dataset_permutation.data(),
                                               static_cast<int64_t>(resume->dataset_permutation.size()))) {
            LOG_ERR("%s: invalid dataset permutation in checkpoint\n", __func__);
            return false;
        }
    }
    if (qparams.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total training steps %ld\n", __func__, qparams.warmup_steps,
                static_cast<long>(schedule.total_steps));
        return false;
    }
    if (resume) {
        if (resume->next_epoch >= params.lr.epochs) {
            LOG_ERR("%s: checkpoint already completed the requested %d epochs\n", __func__, params.lr.epochs);
            return false;
        }
        if (resume->scheduler_step > schedule.total_steps) {
            LOG_ERR("%s: checkpoint optimizer step exceeds the configured schedule\n", __func__);
            return false;
        }
        if (qparams.max_steps > 0 && resume->scheduler_step >= schedule.total_steps) {
            LOG_ERR("%s: checkpoint has already reached --max-steps=%d\n", __func__, qparams.max_steps);
            return false;
        }
        schedule.step = resume->scheduler_step;
    }
    llama_opt_init(ctx, model, opt_params);
    llama_opt_set_gradient_checkpointing(ctx, qparams.gradient_checkpointing);
    llama_opt_set_opt_period(ctx, qparams.gradient_accumulation_steps);
    llama_opt_set_grad_stats_interval(ctx, qparams.grad_stats_interval);
    if (qparams.gradient_checkpointing) {
        LOG_INF("%s: gradient checkpointing enabled\n", __func__);
    }
    if (qparams.shuffle_dataset && !resume) {
        uint32_t seed = params.sampling.seed;
        if (seed == LLAMA_DEFAULT_SEED) {
            try {
                seed = std::random_device{}();
            } catch (const std::exception & ex) {
                LOG_ERR("%s: failed to generate a dataset shuffle seed: %s\n", __func__, ex.what());
                return false;
            }
        }
        llama_opt_set_seed(ctx, seed);
        LOG_INF("%s: dataset shuffle RNG seed=%u\n", __func__, seed);
    }
    if (resume) {
        if (!llama_opt_state_import(ctx, resume->optimizer_state.data(), resume->optimizer_state.size())) {
            LOG_ERR("%s: failed to restore optimizer state from checkpoint\n", __func__);
            return false;
        }
        LOG_INF("%s: restored optimizer state from checkpoint '%s' (optimizer step %ld)\n",
                __func__, resume->directory.string().c_str(), static_cast<long>(resume->scheduler_step));
    }

    ggml_opt_result_ptr result_train(ggml_opt_result_init());
    ggml_opt_result_ptr result_eval(ggml_opt_result_init());
    if (!result_train || !result_eval) {
        LOG_ERR("%s: failed to allocate optimizer results\n", __func__);
        return false;
    }
    qlora_save_worker save_worker;
    qlora_training_state save_state;
    save_state.worker = &save_worker;
    save_state.output_tensors = &output_refs;
    save_state.output_root = qparams.output_dir;
    save_state.arch = &arch;
    save_state.base_model_path = &params.model.path;
    save_state.qparams = &qparams;
    save_state.params = &params;
    save_state.ctx = ctx;
    save_state.dataset = dataset.get();
    save_state.lora_alpha = lora_alpha;
    save_state.save_checkpoint_steps = qparams.save_checkpoint_steps;
    save_state.save_checkpoint_limit = qparams.save_checkpoint_limit;
    save_state.save_snapshot_steps = qparams.save_snapshot_steps;
    save_state.save_snapshot_limit = qparams.save_snapshot_limit;
    save_state.gradient_accumulation_steps = qparams.gradient_accumulation_steps;
    save_state.dataset_windows = dataset_windows;
    save_state.schedule = &schedule;
    if (!initialize_training_logs(save_state.output_root, qparams.grad_stats_interval > 0)) {
        LOG_ERR("%s: failed to initialize training logs in '%s'\n", __func__,
                save_state.output_root.string().c_str());
        return false;
    }
    qlora_training_callback_scope callback_scope(save_state);

    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=assistant-response-only\n", __func__,
            qparams.lora_rank, lora_alpha, params.lr.epochs);
    LOG_INF("%s: dataset: %ld full-context windows; accumulate %d windows/update = %ld optimizer steps/epoch "
            "(n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, static_cast<long>(dataset_windows), qparams.gradient_accumulation_steps,
            static_cast<long>(optimizer_steps_per_epoch), n_ctx, n_ubatch, n_ctx / 2);
    if (qparams.max_steps > 0) {
        LOG_INF("%s: --max-steps=%d overrides --epochs; training will stop after %d optimizer updates\n",
                __func__, qparams.max_steps, qparams.max_steps);
    }
    if (qparams.val_windows >= 0) {
        LOG_INF("%s: validation window limit=%d (0 disables validation)\n", __func__, qparams.val_windows);
    }
    if (qparams.grad_stats_interval > 0) {
        LOG_INF("%s: logging global LoRA gradient statistics every %d optimizer updates\n",
                __func__, qparams.grad_stats_interval);
    }
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%ld lr=%.3g lr_min=%.3g\n",
            __func__, qparams.lr_scheduler.c_str(), qparams.warmup_steps, static_cast<double>(qparams.warmup_init_ratio),
            static_cast<long>(schedule.total_steps), static_cast<double>(params.lr.lr0),
            static_cast<double>(std::max(0.0f, params.lr.lr_min)));
    if (qparams.save_snapshot_limit > 0 && qparams.save_snapshot_steps > 0) {
        LOG_INF("%s: saving adapter snapshots every %d optimizer steps (limit %d)\n",
                __func__, qparams.save_snapshot_steps, qparams.save_snapshot_limit);
    }
    if (qparams.save_checkpoint_limit > 0) {
        if (qparams.save_checkpoint_steps > 0) {
            LOG_INF("%s: saving resumable checkpoints every %d optimizer steps (limit %d)\n",
                    __func__, qparams.save_checkpoint_steps, qparams.save_checkpoint_limit);
        } else {
            LOG_INF("%s: saving only the final resumable checkpoint\n", __func__);
        }
    }

    const ggml_opt_epoch_callback train_callback = qlora_save_step_callback;
    const int64_t first_epoch = resume ? resume->next_epoch : 0;
    for (params.lr.epoch = first_epoch; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        const int64_t window_start = resume && params.lr.epoch == resume->next_epoch ? resume->next_window : 0;
        save_state.epoch = params.lr.epoch + 1;
        save_state.window_start = window_start;
        const bool resume_epoch_statistics = resume && params.lr.epoch == resume->next_epoch && window_start > 0;
        save_state.training_statistics_complete = resume_epoch_statistics
            ? resume->training_statistics_complete : true;
        if (resume_epoch_statistics && !save_state.training_statistics_complete) {
            LOG_WRN("%s: checkpoint does not contain epoch loss statistics; resumed averages start at the checkpoint\n",
                    __func__);
        }
        save_state.training_loss_sum = resume_epoch_statistics ? resume->training_loss_sum : 0.0;
        save_state.training_data_count = resume_epoch_statistics ? resume->training_data_count : 0;
        save_state.training_weight_sum = resume_epoch_statistics ? resume->training_weight_sum : 0.0;
        save_state.validation_weight_sum = 0.0;
        if (qparams.shuffle_dataset && dataset_windows > 1 && window_start == 0) {
            llama_opt_dataset_shuffle(ctx, dataset.get(), dataset_windows);
        }
        save_state.dataset_permutation.resize(static_cast<size_t>(dataset_windows));
        if (!ggml_opt_dataset_get_permutation(dataset.get(), save_state.dataset_permutation.data(), dataset_windows)) {
            LOG_ERR("%s: failed to read the dataset permutation for progress reporting\n", __func__);
            return false;
        }
        int64_t window_end = dataset_windows;
        if (qparams.max_steps > 0) {
            const int64_t remaining_steps = qparams.max_steps - schedule.step;
            const int64_t remaining_windows = dataset_windows - window_start;
            const int64_t remaining_epoch_steps = remaining_windows / accumulation_steps +
                (remaining_windows % accumulation_steps != 0);
            if (remaining_steps < remaining_epoch_steps) {
                window_end = window_start + remaining_steps * accumulation_steps;
            }
        }
        const int64_t eval_end = qparams.val_windows < 0 ? dataset_windows :
            window_end + std::min<int64_t>(qparams.val_windows, dataset_windows - window_end);
        save_state.validation_window_start = window_end;
        save_state.validation_progress_started = false;
        const bool epoch_completed = llama_opt_epoch_range_ex(
            ctx, dataset.get(), result_train.get(), result_eval.get(),
            window_start, window_end, eval_end,
            train_callback, qlora_epoch_callback_progress_line,
            qlora_should_abort, &save_state);
        std::fprintf(stderr, "\n");
        if (!epoch_completed) {
            if (save_state.abort_requested) {
                LOG_ERR("%s: training aborted after numerical validation failure; final adapter and checkpoint saves were skipped\n",
                        __func__);
            } else {
                LOG_ERR("%s: optimizer epoch execution failed; final adapter and checkpoint saves were skipped\n", __func__);
            }
            return false;
        }
        qlora_log_epoch_loss(params.lr.epoch + 1, params.lr.epochs, save_state.training_loss_sum,
                       static_cast<double>(llama_n_ctx(ctx)), save_state.training_weight_sum);
        ggml_opt_result_reset(result_train.get());
        ggml_opt_result_reset(result_eval.get());
        resume = nullptr;
        if (qparams.max_steps > 0 && schedule.step >= qparams.max_steps) {
            break;
        }
    }

    qlora_collect_save_result(save_state, true);
    if (save_state.abort_requested) {
        LOG_ERR("%s: training was aborted; final adapter and checkpoint saves were skipped\n", __func__);
        return false;
    }

    const int64_t final_next_epoch = params.lr.epochs;
    const int64_t final_next_window = 0;
    if (save_state.save_checkpoint_limit > 0) {
        if (save_state.last_checkpoint_step != schedule.step ||
            save_state.last_checkpoint_next_epoch != final_next_epoch ||
            save_state.last_checkpoint_next_window != final_next_window) {
            std::shared_ptr<qlora_save_job> job;
            if (!qlora_prepare_save_job(save_state, false, true, false,
                                        final_next_epoch, final_next_window, job)) {
                LOG_ERR("%s: final resumable checkpoint data could not be staged\n", __func__);
                return false;
            }
            if (!qlora_submit_save_job(save_state, job)) {
                LOG_ERR("%s: final resumable checkpoint could not be queued\n", __func__);
                return false;
            }
            qlora_collect_save_result(save_state, true);
            if (!job->checkpoint_saved) {
                LOG_ERR("%s: final resumable checkpoint could not be saved\n", __func__);
                return false;
            }
        }

        const std::filesystem::path final_checkpoint = qlora_checkpoint_path(save_state.output_root, schedule.step);
        if (!is_checkpoint_directory(final_checkpoint) ||
            !install_file_alias(final_checkpoint / "adapter.gguf", save_state.output_root / "adapter.gguf")) {
            LOG_ERR("%s: final adapter could not be installed in '%s'\n", __func__,
                    (save_state.output_root / "adapter.gguf").string().c_str());
            return false;
        }
    } else {
        const std::filesystem::path adapter_source = save_state.cached_adapter_step == schedule.step
            ? save_state.cached_adapter_path : std::filesystem::path();
        const std::filesystem::path final_adapter = save_state.output_root / "adapter.gguf";
        if (!adapter_source.empty()) {
            if (!install_file_alias(adapter_source, final_adapter)) {
                LOG_ERR("%s: final adapter could not be installed in '%s'\n", __func__, final_adapter.string().c_str());
                return false;
            }
        } else {
            std::shared_ptr<qlora_save_job> job;
            if (!qlora_prepare_save_job(save_state, false, false, true,
                                        final_next_epoch, final_next_window, job)) {
                LOG_ERR("%s: final adapter data could not be staged\n", __func__);
                return false;
            }
            if (!qlora_submit_save_job(save_state, job)) {
                LOG_ERR("%s: final adapter could not be queued\n", __func__);
                return false;
            }
            qlora_collect_save_result(save_state, true);
            if (!job->final_adapter_saved) {
                LOG_ERR("%s: final adapter could not be saved to '%s'\n", __func__, final_adapter.string().c_str());
                return false;
            }
        }
    }
    LOG_INF("final adapter saved (optimizer step %ld)\n", static_cast<long>(schedule.step));
    return true;
}
