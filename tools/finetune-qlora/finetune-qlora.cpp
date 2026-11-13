
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
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
#include "dataset-pipeline.h"
#include "schedule.h"
#include "save-worker.h"
#include "checkpoint.h"
#include "training.h"

#include "../../src/llama-adapter.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <limits>
#include <system_error>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <cctype>
#include <vector>

namespace {

std::string unique_init_adapter_path(const std::string & output_path) {
    const std::string base = (std::filesystem::path(output_path) / ".adapter-init").string();
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

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    qlora_params qparams;
    common_params params;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;
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

    std::error_code output_ec;
    std::filesystem::create_directories(qparams.output_dir, output_ec);
    if (output_ec || !std::filesystem::is_directory(qparams.output_dir, output_ec) || output_ec) {
        LOG_ERR("%s: cannot create output directory '%s'\n", __func__, qparams.output_dir.c_str());
        return 1;
    }
    if (!recover_checkpoint_directories(qparams.output_dir)) {
        LOG_ERR("%s: failed to recover checkpoint output directory '%s'\n", __func__, qparams.output_dir.c_str());
        return 1;
    }

    const bool input_adapter_cli = !params.lora_adapters.empty();
    if (!qparams.resume_from.empty() && input_adapter_cli) {
        LOG_ERR("%s: --resume-from cannot be combined with --lora; the checkpoint supplies its adapter\n", __func__);
        return 1;
    }
    qlora_checkpoint resume_checkpoint;
    qlora_checkpoint * resume = nullptr;
    if (!qparams.resume_from.empty()) {
        if (!load_qlora_checkpoint(qparams.resume_from, resume_checkpoint)) {
            return 1;
        }
        resume = &resume_checkpoint;
    }
    const bool adapter_supplied = input_adapter_cli || resume != nullptr;
    const std::string input_adapter_path = resume
        ? (resume_checkpoint.directory / "adapter.gguf").string()
        : (input_adapter_cli ? params.lora_adapters.front().path : std::string());
    const float input_adapter_scale = input_adapter_cli ? params.lora_adapters.front().scale : 1.0f;
    params.lora_adapters.clear();

    if (qparams.dataset.input_paths.empty()) {
        LOG_ERR("%s: --train-file is required\n", __func__);
        return 1;
    }
    if (params.optimizer != GGML_OPT_OPTIMIZER_TYPE_ADAMW && qparams.optimizer_state_type != "f32") {
        LOG_ERR("%s: --optimizer-state-type applies only to AdamW; use f32 with SGD\n", __func__);
        return 1;
    }
    if (params.optimizer != GGML_OPT_OPTIMIZER_TYPE_ADAMW && qparams.lora_type != "f32") {
        LOG_ERR("%s: non-F32 LoRA storage currently requires AdamW; use --optimizer adamw\n", __func__);
        return 1;
    }

    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.warmup = false;
    params.no_alloc_kv_cache = true;

    const float lora_alpha = qparams.lora_alpha > 0.0f ? qparams.lora_alpha : static_cast<float>(qparams.lora_rank);

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

    lora_tensor_refs output_refs;
    lora_tensor_refs training_refs;

    if (resume) {
        LOG_INF("%s: preparing LoRA adapter from checkpoint: %s\n",
                __func__, input_adapter_path.c_str());
    } else if (input_adapter_cli) {
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

    size_t forward_precision_tensors = 0;
    size_t cuda_model_tensors = 0;
    size_t cuda_model_tensor_bytes = 0;
    for (const auto & entry : model->tensors_by_name) {
        const ggml_tensor * tensor = entry.second;
        if (!tensor) {
            continue;
        }
        ++forward_precision_tensors;
        if (tensor->buffer) {
            const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
            if (buffer_name && std::strncmp(buffer_name, "CUDA", 4) == 0) {
                ++cuda_model_tensors;
                cuda_model_tensor_bytes += ggml_nbytes(tensor);
            }
        }
    }
    LOG_INF("%s: model weight placement: requested_gpu_layers=%d gpu_offload_supported=%d total_tensors=%zu cuda_tensors=%zu cuda_tensor_bytes=%zu\n",
            __func__, params.n_gpu_layers, llama_supports_gpu_offload() ? 1 : 0,
            forward_precision_tensors, cuda_model_tensors, cuda_model_tensor_bytes);
    model->prec_policy.prec_src1.clear();
    model->prec_policy.prec_src1_default = qparams.forward_precision;
    model->prec_policy.prec_bwd = qparams.backward_precision;
    LOG_INF("%s: requested forward precision=%s; configured model tensors=%zu; requested backward precision=%s\n",
            __func__, qparams.forward_precision_name.c_str(), forward_precision_tensors,
            qparams.backward_precision_name.c_str());

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    if (arch.empty()) {
        char description[256] = {};
        llama_model_desc(model, description, sizeof(description));
        arch = std::string(description);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    {
        remove_file_scope temporary_adapter_file;
        temporary_adapter_file.path = unique_init_adapter_path(qparams.output_dir);
        if (temporary_adapter_file.path.empty()) {
            return 1;
        }
        const std::vector<std::string> targets = string_split(qparams.lora_targets, ",");
        const enum ggml_type lora_type = qlora_type_from_name(qparams.lora_type);
        std::mt19937 rng(42);
        lora_tensors prepared;
        if (adapter_supplied) {
            prepared = transform_lora_tensors(
                input_adapter_path, arch, model->tensors_by_name, targets, qparams.lora_rank,
                lora_alpha, input_adapter_scale, rng, llama_model_n_layer(model), qparams.freeze_layers, lora_type);
        } else {
            prepared = alloc_lora_tensors(
                model->tensors_by_name, targets, qparams.lora_rank, rng,
                llama_model_n_layer(model), qparams.freeze_layers, lora_type);
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

    if (!train_adapter(qparams, params, model, ctx, output_refs, arch, lora_alpha, resume)) {
        return 1;
    }
    return 0;
}
