// QLoRA fine-tuning for quantized GGUF models.
//
// Only the F32 LoRA A/B tensors are trained. QLoRA-only arguments are parsed
// locally so this example does not need to modify common/arg.h or common/arg.cpp.
// Periodic saves are numbered adapter snapshots; optimizer-state resume is not
// implemented yet.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "adapter.h"
#include "args.h"
#include "dataset.h"
#include "schedule.h"

// Internal adapter struct is needed to mark the loaded LoRA tensors trainable.
#include "../../src/llama-adapter.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string snapshot_path(const std::string & output_path, int64_t global_window) {
    const size_t separator = output_path.find_last_of("/\\");
    const size_t basename_start = separator == std::string::npos ? 0 : separator + 1;
    const size_t dot = output_path.find_last_of('.');
    const bool has_extension = dot != std::string::npos && dot > basename_start;
    const size_t stem_end = has_extension ? dot : output_path.size();

    std::ostringstream suffix;
    suffix << ".step" << std::setw(8) << std::setfill('0') << global_window;
    return output_path.substr(0, stem_end) + suffix.str() +
           (has_extension ? output_path.substr(dot) : std::string());
}

// Never overwrite an existing numbered snapshot, including snapshots from an
// earlier run that used the same --lora-out path.
std::string unique_snapshot_path(const std::string & output_path, int64_t global_window) {
    const std::string base = snapshot_path(output_path, global_window);
    std::string candidate = base;
    const size_t separator = base.find_last_of("/\\");
    const size_t basename_start = separator == std::string::npos ? 0 : separator + 1;
    const size_t dot = base.find_last_of('.');
    const bool has_extension = dot != std::string::npos && dot > basename_start;
    const size_t stem_end = has_extension ? dot : base.size();
    const std::string stem = base.substr(0, stem_end);
    const std::string extension = has_extension ? base.substr(dot) : std::string();

    for (uint64_t suffix = 1; ; ++suffix) {
        std::error_code ec;
        const bool exists = std::filesystem::exists(candidate, ec);
        if (ec) {
            LOG_ERR("%s: cannot check whether snapshot exists: %s\n", __func__, candidate.c_str());
            return {};
        }
        if (!exists) {
            return candidate;
        }
        candidate = stem + "." + std::to_string(suffix) + extension;
    }
}

std::string unique_init_adapter_path(const std::string & output_path) {
    const std::string base = output_path + ".init";
    std::string candidate = base + ".gguf";
    for (uint64_t suffix = 1; ; ++suffix) {
        std::error_code ec;
        const bool exists = std::filesystem::exists(candidate, ec);
        if (ec) {
            LOG_ERR("%s: cannot check whether temporary adapter exists: %s\n", __func__, candidate.c_str());
            return {};
        }
        const bool temp_exists = std::filesystem::exists(candidate + ".tmp", ec);
        if (ec) {
            LOG_ERR("%s: cannot check whether temporary adapter exists: %s\n", __func__, (candidate + ".tmp").c_str());
            return {};
        }
        if (!exists && !temp_exists) {
            return candidate;
        }
        candidate = base + "." + std::to_string(suffix) + ".gguf";
    }
}

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

struct llama_backend_scope {
    ~llama_backend_scope() {
        llama_backend_free();
    }
};

struct remove_file_scope {
    std::string path;

    void remove_now() {
        if (!path.empty()) {
            std::remove(path.c_str());
            path.clear();
        }
    }

    ~remove_file_scope() {
        remove_now();
    }
};

struct save_context {
    const lora_tensor_refs * output_tensors = nullptr;
    const std::string * output_path = nullptr;
    const std::string * arch = nullptr;
    const std::string * base_model_path = nullptr;
    float lora_alpha = 0.0f;
    int32_t save_every = 0;
    int32_t ubatches_per_window = 1;
    int64_t dataset_windows = 0;
    int64_t epoch = 0; // one-based
    int64_t last_saved_global_window = 0;
    qlora_lr_schedule * schedule = nullptr;
};

// The callback is intentionally an orchestrator only: progress, logical-step
// updates, and snapshot writes stay separate so each can be extended in place.
static thread_local save_context * g_save_context = nullptr;

struct save_context_scope {
    explicit save_context_scope(save_context & state) {
        g_save_context = &state;
    }

    ~save_context_scope() {
        g_save_context = nullptr;
    }
};

void update_training_step(save_context & state, int64_t ibatch) {
    const int64_t ubatch_in_window = ibatch % state.ubatches_per_window;
    if (ibatch > 0 && ubatch_in_window == 0) {
        state.schedule->advance_step();
    }
}

void save_snapshot_if_due(save_context & state, int64_t ibatch) {
    if (state.save_every <= 0 || state.dataset_windows <= 0) {
        return;
    }

    const int64_t window_in_epoch = ibatch / state.ubatches_per_window;
    const int64_t global_window = (state.epoch - 1) * state.dataset_windows + window_in_epoch;
    if (global_window <= 0 || global_window == state.last_saved_global_window ||
        global_window % state.save_every != 0) {
        return;
    }

    state.last_saved_global_window = global_window;
    const std::string path = unique_snapshot_path(*state.output_path, global_window);
    if (path.empty()) {
        LOG_ERR("periodic adapter snapshot failed: could not select a unique path; training will continue\n");
        return;
    }
    if (save_adapter(*state.output_tensors, path, *state.arch, state.lora_alpha, *state.base_model_path)) {
        LOG_INF("periodic adapter snapshot saved: %s (window %ld)\n", path.c_str(), static_cast<long>(global_window));
    } else {
        // Snapshot failures are deliberately non-fatal: keep the training loop running.
        LOG_ERR("periodic adapter snapshot failed: %s; training will continue\n", path.c_str());
    }
}

void save_every_callback(bool               train,
                         ggml_opt_context_t opt_ctx,
                         ggml_opt_dataset_t dataset,
                         ggml_opt_result_t  result,
                         int64_t            ibatch,
                         int64_t            ibatch_max,
                         int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    if (!train || !g_save_context) {
        return;
    }
    update_training_step(*g_save_context, ibatch);
    save_snapshot_if_due(*g_save_context, ibatch);
}

void log_epoch_loss(int epoch, int total_epochs, ggml_opt_result_t train_result) {
    double train_loss = 0.0;
    double uncertainty = 0.0;
    ggml_opt_result_loss(train_result, &train_loss, &uncertainty);
    LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f\n", epoch, total_epochs, train_loss, uncertainty);
}

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

bool train_adapter(const qlora_params & qparams,
                   common_params & params,
                   llama_model * model,
                   llama_context * ctx,
                   const lora_tensor_refs & output_refs,
                   const lora_tensor_refs & training_refs,
                   const std::string & arch,
                   float lora_alpha) {
    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_ubatch = llama_n_ubatch(ctx);
    const bool has_kv_lora = std::any_of(training_refs.begin(), training_refs.end(), [](const auto & kv) {
        return kv.first.find("attn_k") != std::string::npos || kv.first.find("attn_v") != std::string::npos;
    });
    if (has_kv_lora && n_ubatch != n_ctx) {
        LOG_ERR("%s: attn_k/attn_v LoRA requires a full-context microbatch: "
                "set -ub equal to -c (n_ubatch=%d, n_ctx=%d); otherwise K/V gradients are unavailable\n",
                __func__, n_ubatch, n_ctx);
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
    };
    const int64_t dataset_windows = ggml_opt_dataset_ndata(dataset.get());
    if (dataset_windows <= 0) {
        LOG_ERR("%s: dataset has no training windows\n", __func__);
        return false;
    }
    if (params.lr.epochs > 0 && dataset_windows > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total training step count overflows int64\n", __func__);
        return false;
    }
    schedule.total_steps = dataset_windows * params.lr.epochs;
    if (schedule.total_steps <= 0) {
        LOG_ERR("%s: total training step count must be positive\n", __func__);
        return false;
    }
    if (qparams.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total training steps %ld\n", __func__, qparams.warmup_steps,
                static_cast<long>(schedule.total_steps));
        return false;
    }
    llama_opt_init(ctx, model, opt_params);

    ggml_opt_result_ptr result_train(ggml_opt_result_init());
    ggml_opt_result_ptr result_eval(ggml_opt_result_init());
    if (!result_train || !result_eval) {
        LOG_ERR("%s: failed to allocate optimizer results\n", __func__);
        return false;
    }
    const int32_t ubatches_per_window = n_ubatch > 0 ? std::max(1, n_ctx / n_ubatch) : 1;

    save_context save_state;
    save_state.output_tensors = &output_refs;
    save_state.output_path = &qparams.lora_out;
    save_state.arch = &arch;
    save_state.base_model_path = &params.model.path;
    save_state.lora_alpha = lora_alpha;
    save_state.save_every = qparams.save_every;
    save_state.ubatches_per_window = ubatches_per_window;
    save_state.dataset_windows = dataset_windows;
    save_state.schedule = &schedule;
    save_context_scope callback_scope(save_state);

    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=assistant-response-only\n", __func__,
            qparams.lora_rank, lora_alpha, params.lr.epochs);
    LOG_INF("%s: dataset: %ld windows × %d ubatches = %ld optimizer steps/epoch (n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, static_cast<long>(dataset_windows), ubatches_per_window,
            static_cast<long>(dataset_windows * ubatches_per_window), n_ctx, n_ubatch, n_ctx / 2);
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%ld lr=%.3g lr_min=%.3g\n",
            __func__, qparams.lr_scheduler.c_str(), qparams.warmup_steps, static_cast<double>(qparams.warmup_init_ratio),
            static_cast<long>(schedule.total_steps), static_cast<double>(params.lr.lr0),
            static_cast<double>(std::max(0.0f, params.lr.lr_min)));
    if (qparams.save_every > 0) {
        LOG_INF("%s: saving numbered adapter snapshots every %d windows based on %s\n", __func__, qparams.save_every,
                qparams.lora_out.c_str());
    }

    const ggml_opt_epoch_callback train_callback = save_every_callback;
    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        save_state.epoch = params.lr.epoch + 1;
        if (qparams.shuffle_dataset && dataset_windows > 1) {
            llama_opt_dataset_shuffle(ctx, dataset.get(), dataset_windows);
        }
        llama_opt_epoch(ctx, dataset.get(), result_train.get(), result_eval.get(), dataset_windows, train_callback,
                        ggml_opt_epoch_callback_progress_bar);
        std::fprintf(stderr, "\n");
        log_epoch_loss(params.lr.epoch + 1, params.lr.epochs, result_train.get());
        ggml_opt_result_reset(result_train.get());
        ggml_opt_result_reset(result_eval.get());
    }
    return true;
}

}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    qlora_params qparams;
    common_params params;
    {
        std::vector<std::string> common_args_storage;
        std::string parse_error;
        if (!qlora_parse_args(argc, argv, qparams, common_args_storage, parse_error)) {
            LOG_ERR("%s: %s\n", __func__, parse_error.c_str());
            qlora_print_usage(argc, argv);
            return 1;
        }

        std::vector<char *> common_argv;
        common_argv.reserve(common_args_storage.size());
        for (std::string & arg : common_args_storage) {
            common_argv.push_back(arg.data());
        }

        params.escape = false;
        if (!common_params_parse(static_cast<int>(common_argv.size()), common_argv.data(), params,
                                 LLAMA_EXAMPLE_FINETUNE, qlora_print_usage)) {
            return 1;
        }
    }

    qparams.dataset.threads = params.cpuparams.n_threads > 0
        ? static_cast<size_t>(params.cpuparams.n_threads)
        : 0;

    if (params.lora_adapters.size() > 1) {
        LOG_ERR("%s: QLoRA accepts only one input adapter; comma-separated and repeated --lora inputs are unsupported\n", __func__);
        return 1;
    }

    const bool adapter_supplied = !params.lora_adapters.empty();
    const std::string input_adapter_path = adapter_supplied ? params.lora_adapters.front().path : std::string();
    const float input_adapter_scale = adapter_supplied ? params.lora_adapters.front().scale : 1.0f;
    params.lora_adapters.clear();

    if (qparams.dataset.input_paths.empty()) {
        LOG_ERR("%s: --train-file is required\n", __func__);
        return 1;
    }

    // Settings required for training.
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;
    params.warmup = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    const float lora_alpha = qparams.lora_alpha > 0.0f ? qparams.lora_alpha : static_cast<float>(qparams.lora_rank);

    // Step 1: read architecture metadata before loading the model.
    std::string arch;
    {
        ggml_context * meta_ctx_raw = nullptr;
        ggml_context_ptr meta_ctx;
        gguf_init_params gguf_params = { true, &meta_ctx_raw };
        gguf_context_ptr gguf(gguf_init_from_file(params.model.path.c_str(), gguf_params));
        meta_ctx.reset(meta_ctx_raw);
        if (!gguf) {
            LOG_ERR("%s: failed to open model GGUF: %s\n", __func__, params.model.path.c_str());
            return 1;
        }
        const int key_id = gguf_find_key(gguf.get(), "general.architecture");
        if (key_id >= 0) {
            arch = gguf_get_val_str(gguf.get(), key_id);
        }
    }

    if (adapter_supplied) {
        if (!std::isfinite(input_adapter_scale)) {
            LOG_ERR("%s: input adapter scale must be finite\n", __func__);
            return 1;
        }
        if (!arch.empty() && !validate_lora_adapter_rank(input_adapter_path, arch, qparams.lora_rank)) {
            return 1;
        }
    }

    // Load the base model without an adapter. Both new and existing adapters are
    // prepared as a temporary GGUF and then loaded through the upstream loader.
    lora_tensor_refs output_refs;
    lora_tensor_refs training_refs;

    if (adapter_supplied) {
        LOG_INF("%s: preparing existing LoRA adapter: %s (optimizer state starts fresh)\n",
                __func__, input_adapter_path.c_str());
    }

    common_init();
    llama_backend_init();
    llama_backend_scope backend_scope;
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model/context\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    if (arch.empty()) {
        char description[256] = {};
        llama_model_desc(model, description, sizeof(description));
        arch = std::string(description);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    {
        remove_file_scope temporary_adapter_file;
        temporary_adapter_file.path = unique_init_adapter_path(qparams.lora_out);
        if (temporary_adapter_file.path.empty()) {
            return 1;
        }
        const std::vector<std::string> targets = string_split(qparams.lora_targets, ",");
        std::mt19937 rng(42);
        lora_tensors prepared;
        if (adapter_supplied) {
            prepared = transform_lora_tensors(
                input_adapter_path, arch, model->tensors_by_name, targets, qparams.lora_rank,
                lora_alpha, input_adapter_scale, rng, llama_model_n_layer(model), qparams.freeze_layers);
        } else {
            prepared = alloc_lora_tensors(
                model->tensors_by_name, targets, qparams.lora_rank, rng,
                llama_model_n_layer(model), qparams.freeze_layers);
        }
        if (prepared.refs.empty()) {
            return 1;
        }
        if (!save_adapter(prepared.refs, temporary_adapter_file.path, arch, lora_alpha, params.model.path)) {
            LOG_ERR("%s: cannot write temporary adapter; aborting before training\n", __func__);
            return 1;
        }

        llama_adapter_lora_ptr initial_lora(
            llama_adapter_lora_init(model, temporary_adapter_file.path.c_str()));
        if (!initial_lora) {
            LOG_ERR("%s: failed to load prepared adapter '%s' into the loaded model\n",
                    __func__, temporary_adapter_file.path.c_str());
            return 1;
        }

        llama_init->lora().emplace_back(std::move(initial_lora));
        common_adapter_lora_info adapter_info = {};
        adapter_info.path = temporary_adapter_file.path;
        adapter_info.scale = 1.0f;
        adapter_info.ptr = llama_init->lora().back().get();
        params.lora_adapters.push_back(adapter_info);
        common_set_adapter_lora(ctx, params.lora_adapters);

        llama_adapter_lora * loaded = adapter_info.ptr;
        if (!loaded) {
            LOG_ERR("%s: prepared adapter was not loaded into the model\n", __func__);
            return 1;
        }
        for (auto & kv : loaded->ab_map) {
            output_refs[kv.first] = { kv.second.a, kv.second.b };
            if (lora_tensor_matches_targets(
                    kv.first, targets, qparams.freeze_layers, llama_model_n_layer(model))) {
                ggml_set_param(kv.second.a);
                ggml_set_param(kv.second.b);
                training_refs[kv.first] = { kv.second.a, kv.second.b };
            }
        }
        if (training_refs.empty()) {
            LOG_ERR("%s: no adapter tensors matched --lora-targets; nothing can be trained\n", __func__);
            return 1;
        }
        LOG_INF("%s: adapter has %zu tensor pairs; training %zu and keeping %zu frozen\n",
                __func__, output_refs.size(), training_refs.size(), output_refs.size() - training_refs.size());
        temporary_adapter_file.remove_now();
    }

    if (!train_adapter(qparams, params, model, ctx, output_refs, training_refs, arch, lora_alpha)) {
        return 1;
    }

    // A final-save error is reported after training has completed. Periodic save
    // failures above are non-fatal and do not interrupt training.
    const bool final_save_ok = save_adapter(output_refs, qparams.lora_out, arch, lora_alpha, params.model.path);
    if (!final_save_ok) {
        LOG_ERR("%s: training finished, but the final adapter could not be saved to %s\n", __func__,
                qparams.lora_out.c_str());
    }

    return final_save_ok ? 0 : 1;
}
