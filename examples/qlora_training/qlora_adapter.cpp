#include "qlora_adapter.h"

#include "common.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Tensors whose names contain these substrings use MUL_MAT_ID (sparse MoE expert dispatch)
// which has no backward implementation — exclude them from LoRA targets unconditionally.
static const std::vector<std::string> EXCLUDED_SUBSTRINGS = {
    "_exps",  // MoE expert weight stacks (ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_up_exps)
};

static bool tensor_is_excluded(const char * name) {
    const std::string tensor_name(name);
    return std::any_of(
        EXCLUDED_SUBSTRINGS.begin(), EXCLUDED_SUBSTRINGS.end(),
        [&tensor_name](const std::string & substring) { return tensor_name.find(substring) != std::string::npos; });
}

// Extract the transformer block index from "blk.NN.<rest>"; return -1 on mismatch.
static int tensor_layer_index(const char * name) {
    const char * block = strstr(name, "blk.");
    if (!block) {
        return -1;
    }

    block += 4;  // Skip "blk.".
    char * end         = nullptr;
    long   layer_index = strtol(block, &end, 10);
    if (end == block || (*end != '.' && *end != '\0')) {
        return -1;
    }
    return (int) layer_index;
}

static bool tensor_matches_targets(const char * name, const std::vector<std::string> & targets, int freeze_layers = 0) {
    if (tensor_is_excluded(name)) {
        return false;
    }

    if (freeze_layers > 0) {
        const int layer_index = tensor_layer_index(name);
        if (layer_index >= 0 && layer_index < freeze_layers) {
            return false;
        }
    }

    const std::string tensor_name(name);
    return std::any_of(targets.begin(), targets.end(), [&tensor_name](const std::string & target) {
        return tensor_name.find(target) != std::string::npos;
    });
}

// ---------------------------------------------------------------------------
// LoRA tensor allocation
// ---------------------------------------------------------------------------

lora_tensors alloc_lora_tensors(const std::string &              model_path,
                                const std::vector<std::string> & targets,
                                int32_t                          rank,
                                std::mt19937 &                   rng,
                                int32_t                          freeze_layers) {
    lora_tensors lt;

    // Open the model GGUF to discover tensor names and shapes
    // without needing access to private llama_model internals.
    struct ggml_context *   ctx_meta    = nullptr;
    struct gguf_init_params gguf_params = { /*.no_alloc=*/true, /*.ctx=*/&ctx_meta };
    struct gguf_context *   ctx_gguf    = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open model GGUF for tensor discovery: %s\n", __func__, model_path.c_str());
        return lt;
    }

    // Collect matching 2-D tensors
    struct tensor_info {
        std::string name;
        int64_t     ne0, ne1;
    };

    std::vector<tensor_info> matched;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta); t; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (ggml_n_dims(t) < 2) {
            continue;
        }
        if (!tensor_matches_targets(t->name, targets, freeze_layers)) {
            continue;
        }
        matched.push_back({ t->name, t->ne[0], t->ne[1] });
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    if (matched.empty()) {
        LOG_ERR("%s: no model tensors matched --lora-targets; check spelling\n", __func__);
        return lt;
    }

    if (freeze_layers > 0) {
        LOG_INF("%s: freezing layers blk.0 .. blk.%d (no LoRA allocated; backward already pruned by grads_needed)\n",
                __func__, freeze_layers - 1);
    }
    LOG_INF("%s: allocating LoRA A/B tensors for %zu weight matrices, rank=%d\n", __func__, matched.size(), rank);

    // Allocate ggml context for A+B tensors (2 tensors per matched weight)
    const size_t            mem = (2 * matched.size() + 16) * ggml_tensor_overhead();
    struct ggml_init_params ip  = { mem, nullptr, /*no_alloc=*/true };
    lt.ctx                      = ggml_init(ip);

    for (const auto & tensor : matched) {
        const int64_t input_dim  = tensor.ne0;  // Input features (matrix columns).
        const int64_t output_dim = tensor.ne1;  // Output features (matrix rows).

        // LoRA applies A first, then B: B @ (A @ x).
        // Keep the tensor dimensions consistent with llama-adapter.cpp:48-60.
        ggml_tensor * lora_a = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, input_dim, rank);
        ggml_tensor * lora_b = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, rank, output_dim);

        ggml_set_name(lora_a, (tensor.name + ".lora_a").c_str());
        ggml_set_name(lora_b, (tensor.name + ".lora_b").c_str());

        lt.ab[tensor.name] = { lora_a, lora_b };
    }

    // Allocate backend buffer for all LoRA tensors at once
    lt.buf = ggml_backend_alloc_ctx_tensors_from_buft(lt.ctx, ggml_backend_cpu_buffer_type());

    // Initialize: A ~ N(0, 1/sqrt(rank)), B = 0
    const float                     std_a = 1.0f / std::sqrt((float) rank);
    std::normal_distribution<float> dist(0.0f, std_a);

    for (auto & kv : lt.ab) {
        ggml_tensor * la = kv.second.first;
        ggml_tensor * lb = kv.second.second;

        // Fill A
        float * data_a = (float *) la->data;
        for (int64_t i = 0; i < ggml_nelements(la); ++i) {
            data_a[i] = dist(rng);
        }
        // Zero B
        memset(lb->data, 0, ggml_nbytes(lb));
    }

    return lt;
}

// ---------------------------------------------------------------------------
// Param filter: only train lora_a / lora_b tensors
// ---------------------------------------------------------------------------

bool lora_param_filter(const struct ggml_tensor * t, void * /*ud*/) {
    const char * n   = t->name;
    const size_t len = strlen(n);
    if (len > 7 && strcmp(n + len - 7, ".lora_a") == 0) {
        return true;
    }
    if (len > 7 && strcmp(n + len - 7, ".lora_b") == 0) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Save adapter GGUF
// ---------------------------------------------------------------------------

static std::string basename_from_path(const std::string & p) {
    const size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) {
        return p;
    }
    return p.substr(pos + 1);
}

static bool parse_checkpoint_number(const std::string & path, size_t begin, size_t end, int64_t & value) {
    if (begin >= end) {
        return false;
    }

    const std::string number     = path.substr(begin, end - begin);
    char *            number_end = nullptr;
    errno                        = 0;
    const long long parsed       = strtoll(number.c_str(), &number_end, 10);
    if (errno != 0 || number_end != number.c_str() + number.size() || parsed < 0) {
        return false;
    }

    value = (int64_t) parsed;
    return true;
}

static bool checkpoint_state_from_filename(const std::string & path,
                                           const std::string & mode,
                                           checkpoint_state &  state) {
    const size_t suffix = path.rfind(".gguf");
    if (suffix == std::string::npos || suffix + 5 != path.size()) {
        return false;
    }

    if (mode == "sft") {
        const size_t epoch_tag = path.rfind(".epoch", suffix);
        const size_t ckpt_tag  = path.rfind(".ckpt", suffix);
        if (epoch_tag == std::string::npos || ckpt_tag == std::string::npos || epoch_tag >= ckpt_tag) {
            return false;
        }
        if (!parse_checkpoint_number(path, epoch_tag + 6, ckpt_tag, state.epoch)) {
            return false;
        }
        if (!parse_checkpoint_number(path, ckpt_tag + 5, suffix, state.window)) {
            return false;
        }
    }

    state.mode = mode;
    return true;
}

static bool gguf_get_i64(const gguf_context * gctx, const char * key, int64_t & value) {
    const int64_t key_id = gguf_find_key(gctx, key);
    if (key_id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(gctx, key_id)) {
        case GGUF_TYPE_UINT32:
            value = gguf_get_val_u32(gctx, key_id);
            return true;
        case GGUF_TYPE_INT32:
            value = gguf_get_val_i32(gctx, key_id);
            return true;
        case GGUF_TYPE_UINT64:
            {
                const uint64_t val = gguf_get_val_u64(gctx, key_id);
                if (val > INT64_MAX) {
                    return false;
                }
                value = (int64_t) val;
                return true;
            }
        case GGUF_TYPE_INT64:
            value = gguf_get_val_i64(gctx, key_id);
            return true;
        default:
            return false;
    }
}

bool load_checkpoint_state(const std::string & path, const std::string & expected_mode, checkpoint_state & state) {
    ggml_context *         ctx_meta = nullptr;
    const gguf_init_params params   = { true, &ctx_meta };
    gguf_context *         gctx     = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        LOG_ERR("%s: cannot open checkpoint %s\n", __func__, path.c_str());
        return false;
    }

    const int64_t mode_id = gguf_find_key(gctx, "training.checkpoint.mode");
    if (mode_id >= 0 && gguf_get_kv_type(gctx, mode_id) == GGUF_TYPE_STRING) {
        state.mode         = gguf_get_val_str(gctx, mode_id);
        state.has_metadata = true;
        gguf_get_i64(gctx, "training.checkpoint.epoch", state.epoch);
        gguf_get_i64(gctx, "training.checkpoint.window", state.window);
        gguf_get_i64(gctx, "training.checkpoint.step", state.step);
        gguf_get_i64(gctx, "training.dataset.windows", state.dataset_windows);
        gguf_get_i64(gctx, "training.context_length", state.context_length);
        gguf_get_i64(gctx, "training.scheduler.step", state.schedule_step);

        const int64_t shuffle_id = gguf_find_key(gctx, "training.dataset.shuffle");
        if (shuffle_id >= 0 && gguf_get_kv_type(gctx, shuffle_id) == GGUF_TYPE_BOOL) {
            state.shuffle = gguf_get_val_bool(gctx, shuffle_id);
        }
    }

    const int64_t alpha_id = gguf_find_key(gctx, "adapter.lora.alpha");
    if (alpha_id >= 0 && gguf_get_kv_type(gctx, alpha_id) == GGUF_TYPE_FLOAT32) {
        state.alpha     = gguf_get_val_f32(gctx, alpha_id);
        state.has_alpha = true;
    }

    gguf_free(gctx);
    ggml_free(ctx_meta);

    if (!state.has_metadata && !checkpoint_state_from_filename(path, expected_mode, state)) {
        LOG_ERR("%s: %s has no checkpoint metadata and its filename does not contain a checkpoint position\n", __func__,
                path.c_str());
        return false;
    }
    if (state.mode != expected_mode) {
        LOG_ERR("%s: checkpoint mode is %s, expected %s\n", __func__, state.mode.c_str(), expected_mode.c_str());
        return false;
    }
    if (state.mode == "sft" && state.epoch <= 0) {
        LOG_ERR("%s: invalid checkpoint epoch %ld\n", __func__, (long) state.epoch);
        return false;
    }
    if (state.window < 0 || state.step < 0 || state.schedule_step < -1) {
        LOG_ERR("%s: checkpoint contains a negative training position\n", __func__);
        return false;
    }
    return true;
}

void save_adapter(const lora_tensors &     tensors,
                  const std::string &      out_path,
                  const std::string &      arch,
                  float                    alpha,
                  const std::string &      base_model_path,
                  const checkpoint_state * checkpoint) {
    // Build output GGUF context
    struct gguf_context * gctx = gguf_init_empty();

    // Metadata required by llama_adapter_lora_init
    gguf_set_val_str(gctx, "general.type", "adapter");
    gguf_set_val_str(gctx, "general.architecture", arch.c_str());
    gguf_set_val_str(gctx, "adapter.type", "lora");
    gguf_set_val_f32(gctx, "adapter.lora.alpha", alpha);
    gguf_set_val_str(gctx, "adapter.base_model", basename_from_path(base_model_path).c_str());

    if (checkpoint) {
        gguf_set_val_u32(gctx, "training.checkpoint.version", 1);
        gguf_set_val_str(gctx, "training.checkpoint.mode", checkpoint->mode.c_str());
        gguf_set_val_i64(gctx, "training.checkpoint.epoch", checkpoint->epoch);
        gguf_set_val_i64(gctx, "training.checkpoint.window", checkpoint->window);
        gguf_set_val_i64(gctx, "training.checkpoint.step", checkpoint->step);
        gguf_set_val_i64(gctx, "training.dataset.windows", checkpoint->dataset_windows);
        gguf_set_val_i64(gctx, "training.context_length", checkpoint->context_length);
        gguf_set_val_bool(gctx, "training.dataset.shuffle", checkpoint->shuffle);
        gguf_set_val_i64(gctx, "training.scheduler.step", checkpoint->schedule_step);
    }

    // Register tensors
    for (const auto & kv : tensors.ab) {
        gguf_add_tensor(gctx, kv.second.first);   // lora_a
        gguf_add_tensor(gctx, kv.second.second);  // lora_b
    }

    // Write: meta placeholder → tensor data → rewrite meta
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, out_path.c_str());
        gguf_free(gctx);
        return;
    }

    // Write meta placeholder
    const size_t      meta_size = gguf_get_meta_size(gctx);
    std::vector<char> zeros_buf(meta_size, 0);
    fout.write(zeros_buf.data(), meta_size);

    // Write tensor data — copy to CPU first in case tensors live on GPU
    for (const auto & kv : tensors.ab) {
        for (ggml_tensor * t : { kv.second.first, kv.second.second }) {
            const size_t      nb = ggml_nbytes(t);
            std::vector<char> cpu_buf(nb);
            ggml_backend_tensor_get(t, cpu_buf.data(), 0, nb);
            fout.write(cpu_buf.data(), nb);
            // GGUF tensors are 32-byte aligned
            const size_t pad = GGML_PAD(nb, 32) - nb;
            if (pad > 0) {
                std::vector<char> pad_buf(pad, 0);
                fout.write(pad_buf.data(), pad);
            }
        }
    }

    // Re-write metadata at offset 0
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(gctx, meta.data());
    fout.seekp(0);
    fout.write((const char *) meta.data(), meta_size);

    fout.close();
    gguf_free(gctx);

    LOG_INF("%s: adapter saved to %s\n", __func__, out_path.c_str());
}
