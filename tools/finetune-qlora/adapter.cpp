#include "adapter.h"

#include "common.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
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

// Return true for the special embedding tensor layout used by llama.cpp LoRA.
static bool tensor_is_token_embedding(const std::string & name) {
    static const std::string suffix = "token_embd.weight";
    return name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void initialize_lora_weights(ggml_tensor * lora_a, ggml_tensor * lora_b, bool embedding, std::mt19937 & rng) {
    float * data_a = static_cast<float *>(lora_a->data);
    float * data_b = static_cast<float *>(lora_b->data);

    if (embedding) {
        std::fill(data_a, data_a + ggml_nelements(lora_a), 0.0f);
        std::normal_distribution<float> dist_b(0.0f, 1.0f);
        for (int64_t i = 0; i < ggml_nelements(lora_b); ++i) {
            data_b[i] = dist_b(rng);
        }
    } else {
        const float bound = 1.0f / std::sqrt(static_cast<float>(lora_a->ne[0]));
        std::uniform_real_distribution<float> dist_a(-bound, bound);
        for (int64_t i = 0; i < ggml_nelements(lora_a); ++i) {
            data_a[i] = dist_a(rng);
        }
        std::fill(data_b, data_b + ggml_nelements(lora_b), 0.0f);
    }
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

bool lora_tensor_matches_targets(const std::string & name,
                                 const std::vector<std::string> & targets,
                                 int32_t freeze_layers,
                                 int32_t model_n_layer) {
    const int layer_index = tensor_layer_index(name.c_str());
    if (model_n_layer > 0 && layer_index >= model_n_layer) {
        return false;
    }
    return tensor_matches_targets(name.c_str(), targets, freeze_layers);
}

bool validate_lora_targets(const std::vector<std::pair<std::string, ggml_tensor *>> & model_tensors,
                           const std::vector<std::string> & targets,
                           int32_t model_n_layer,
                           int32_t freeze_layers) {
    if (targets.empty()) {
        LOG_ERR("%s: --lora-targets must contain at least one target\n", __func__);
        return false;
    }

    bool any_trainable_match = false;
    for (const std::string & target : targets) {
        if (target.empty()) {
            LOG_ERR("%s: --lora-targets contains an empty target\n", __func__);
            return false;
        }

        bool found = false;
        for (const auto & entry : model_tensors) {
            const std::string & name = entry.first;
            const ggml_tensor * tensor = entry.second;
            if (!tensor || ggml_n_dims(tensor) < 2 || tensor->ne[3] != 1) {
                continue;
            }
            const int layer_index = tensor_layer_index(name.c_str());
            if (model_n_layer > 0 && layer_index >= model_n_layer) {
                continue;
            }
            if (name.find(target) != std::string::npos) {
                found = true;
                if (tensor_matches_targets(name.c_str(), {target}, freeze_layers)) {
                    any_trainable_match = true;
                }
            }
        }

        if (!found) {
            LOG_ERR("%s: --lora-targets entry '%s' matches no supported model tensor\n",
                    __func__, target.c_str());
            return false;
        }
    }

    if (!any_trainable_match) {
        LOG_ERR("%s: --lora-targets and --freeze-layers leave no trainable tensors\n", __func__);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// LoRA tensor allocation
// ---------------------------------------------------------------------------

lora_tensors alloc_lora_tensors(const std::vector<std::pair<std::string, ggml_tensor *>> & model_tensors,
                                const std::vector<std::string> & targets,
                                int32_t                          rank,
                                std::mt19937 &                   rng,
                                int32_t                          model_n_layer,
                                int32_t                          freeze_layers) {
    lora_tensors lt;

    struct tensor_info {
        std::string name;
        int64_t     ne0, ne1;
        bool        embedding;
    };

    {
        std::vector<tensor_info> matched;
        matched.reserve(model_tensors.size());

        for (const auto & entry : model_tensors) {
            const std::string & name = entry.first;
            ggml_tensor * t = entry.second;
            if (t == nullptr || ggml_n_dims(t) < 2) {
                continue;
            }
            const int layer_index = tensor_layer_index(name.c_str());
            if (model_n_layer > 0 && layer_index >= model_n_layer) {
                continue;
            }
            if (!tensor_matches_targets(name.c_str(), targets, freeze_layers)) {
                continue;
            }
            matched.push_back({ name, t->ne[0], t->ne[1], tensor_is_token_embedding(name) });
        }

        if (matched.empty()) {
            LOG_ERR("%s: no model tensors matched --lora-targets; check spelling\n", __func__);
            return lt;
        }

        if (freeze_layers > 0) {
            LOG_INF("%s: freezing layers blk.0 .. blk.%d (no LoRA allocated; backward already pruned by grads_needed)\n",
                    __func__, freeze_layers - 1);
        }
        LOG_INF("%s: allocating LoRA A/B tensors for %zu weight matrices, rank=%d\n", __func__, matched.size(), rank);

        const size_t mem = (2 * matched.size() + 16) * ggml_tensor_overhead();
        struct ggml_init_params ip = { mem, nullptr, true };
        lt.ctx.reset(ggml_init(ip));
        if (!lt.ctx) {
            LOG_ERR("%s: failed to allocate LoRA tensor context\n", __func__);
            return {};
        }

        for (const auto & tensor : matched) {
            const int64_t input_dim = tensor.ne0;
            const int64_t output_dim = tensor.ne1;
            ggml_tensor * lora_a = tensor.embedding
                ? ggml_new_tensor_2d(lt.ctx.get(), GGML_TYPE_F32, rank, output_dim)
                : ggml_new_tensor_2d(lt.ctx.get(), GGML_TYPE_F32, input_dim, rank);
            ggml_tensor * lora_b = tensor.embedding
                ? ggml_new_tensor_2d(lt.ctx.get(), GGML_TYPE_F32, rank, input_dim)
                : ggml_new_tensor_2d(lt.ctx.get(), GGML_TYPE_F32, rank, output_dim);
            ggml_set_name(lora_a, (tensor.name + ".lora_a").c_str());
            ggml_set_name(lora_b, (tensor.name + ".lora_b").c_str());
            lt.refs[tensor.name] = { lora_a, lora_b };
        }
    }

    // Allocate backend buffer for all LoRA tensors at once
    lt.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(lt.ctx.get(), ggml_backend_cpu_buffer_type()));
    if (!lt.buf) {
        LOG_ERR("%s: failed to allocate LoRA tensor buffer\n", __func__);
        return {};
    }

    for (auto & kv : lt.refs) {
        ggml_tensor * la = kv.second.first;
        ggml_tensor * lb = kv.second.second;
        initialize_lora_weights(la, lb, tensor_is_token_embedding(kv.first), rng);
    }

    return lt;
}

// ---------------------------------------------------------------------------
// Existing adapter conversion
// ---------------------------------------------------------------------------

namespace {

struct source_lora_pair {
    bool has_a = false;
    bool has_b = false;
    int64_t a_ne0 = 0;
    int64_t a_ne1 = 0;
    int64_t b_ne0 = 0;
    int64_t b_ne1 = 0;
    std::vector<float> a;
    std::vector<float> b;
};

struct output_lora_pair {
    std::string name;
    bool embedding = false;
    int64_t a_ne0 = 0;
    int64_t a_ne1 = 0;
    int64_t b_ne0 = 0;
    int64_t b_ne1 = 0;
    int64_t source_rank = 0;
    float source_scale = 1.0f;
    bool has_source = false;
    std::vector<float> source_a;
    std::vector<float> source_b;
};

static bool name_ends_with(const std::string & name, const char * suffix) {
    const size_t n = strlen(suffix);
    return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}

static bool get_string_metadata(const gguf_context * gguf, const char * key, std::string & value) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        return false;
    }
    value = gguf_get_val_str(gguf, id);
    return true;
}

static bool load_source_lora(const std::string & path,
                             const std::string & expected_arch,
                             std::map<std::string, source_lora_pair> & pairs,
                             float & source_alpha,
                             bool load_data = true) {
    ggml_context * tensor_ctx_raw = nullptr;
    const gguf_init_params init_params = { true, &tensor_ctx_raw };
    gguf_context_ptr gguf(gguf_init_from_file(path.c_str(), init_params));
    ggml_context_ptr tensor_ctx(tensor_ctx_raw);
    if (!gguf || !tensor_ctx) {
        LOG_ERR("%s: failed to read input LoRA GGUF: %s\n", __func__, path.c_str());
        return false;
    }

    std::string type;
    std::string arch;
    std::string adapter_type;
    if (!get_string_metadata(gguf.get(), "general.type", type) || type != "adapter" ||
        !get_string_metadata(gguf.get(), "general.architecture", arch) || arch != expected_arch ||
        !get_string_metadata(gguf.get(), "adapter.type", adapter_type) || adapter_type != "lora") {
        LOG_ERR("%s: input file is not a compatible LoRA adapter for architecture '%s'\n",
                __func__, expected_arch.c_str());
        return false;
    }

    const int64_t alpha_id = gguf_find_key(gguf.get(), "adapter.lora.alpha");
    if (alpha_id >= 0 && gguf_get_kv_type(gguf.get(), alpha_id) != GGUF_TYPE_FLOAT32) {
        LOG_ERR("%s: adapter.lora.alpha must be a float32 value\n", __func__);
        return false;
    }
    source_alpha = alpha_id >= 0 ? gguf_get_val_f32(gguf.get(), alpha_id) : 0.0f;
    if (!std::isfinite(source_alpha)) {
        LOG_ERR("%s: adapter.lora.alpha is not finite\n", __func__);
        return false;
    }

    std::ifstream file;
    if (load_data) {
        file.open(path, std::ios::binary);
        if (!file) {
            LOG_ERR("%s: cannot open input LoRA: %s\n", __func__, path.c_str());
            return false;
        }
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); ++i) {
        const std::string tensor_name = gguf_get_tensor_name(gguf.get(), i);
        if (name_ends_with(tensor_name, "_norm.weight")) {
            // Match the upstream LoRA loader, which currently ignores norm-vector tensors.
            continue;
        }
        const bool is_a = name_ends_with(tensor_name, ".lora_a");
        const bool is_b = name_ends_with(tensor_name, ".lora_b");
        if (!is_a && !is_b) {
            LOG_ERR("%s: unsupported tensor '%s' in input adapter; only LoRA A/B tensors are supported\n",
                    __func__, tensor_name.c_str());
            return false;
        }
        if (tensor_is_excluded(tensor_name.c_str())) {
            LOG_ERR("%s: input adapter contains unsupported MoE expert LoRA tensor '%s'\n",
                    __func__, tensor_name.c_str());
            return false;
        }

        ggml_tensor * tensor = ggml_get_tensor(tensor_ctx.get(), tensor_name.c_str());
        if (!tensor || gguf_get_tensor_type(gguf.get(), i) != GGML_TYPE_F32 ||
            tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            LOG_ERR("%s: tensor '%s' must be a two-dimensional F32 LoRA tensor\n",
                    __func__, tensor_name.c_str());
            return false;
        }

        std::vector<float> data;
        if (load_data) {
            const size_t tensor_size = ggml_nbytes(tensor);
            const size_t offset = gguf_get_data_offset(gguf.get()) + gguf_get_tensor_offset(gguf.get(), i);
            data.resize(ggml_nelements(tensor));
            file.seekg(static_cast<std::streamoff>(offset));
            file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(tensor_size));
            if (!file) {
                LOG_ERR("%s: failed to read tensor '%s' from input adapter\n", __func__, tensor_name.c_str());
                return false;
            }
        }

        const size_t suffix_size = is_a ? strlen(".lora_a") : strlen(".lora_b");
        const std::string weight_name = tensor_name.substr(0, tensor_name.size() - suffix_size);
        source_lora_pair & pair = pairs[weight_name];
        if (is_a) {
            if (pair.has_a) {
                LOG_ERR("%s: duplicate LoRA A tensor for '%s'\n", __func__, weight_name.c_str());
                return false;
            }
            pair.has_a = true;
            pair.a_ne0 = tensor->ne[0];
            pair.a_ne1 = tensor->ne[1];
            pair.a = std::move(data);
        } else {
            if (pair.has_b) {
                LOG_ERR("%s: duplicate LoRA B tensor for '%s'\n", __func__, weight_name.c_str());
                return false;
            }
            pair.has_b = true;
            pair.b_ne0 = tensor->ne[0];
            pair.b_ne1 = tensor->ne[1];
            pair.b = std::move(data);
        }
    }

    if (pairs.empty()) {
        LOG_ERR("%s: input adapter contains no LoRA A/B tensors\n", __func__);
        return false;
    }
    for (const auto & kv : pairs) {
        const source_lora_pair & pair = kv.second;
        if (!pair.has_a || !pair.has_b) {
            LOG_ERR("%s: LoRA tensor pair for '%s' is incomplete\n", __func__, kv.first.c_str());
            return false;
        }
        const bool embedding = name_ends_with(kv.first, "token_embd.weight");
        const bool rank_shapes_match = embedding
            ? pair.a_ne0 == pair.b_ne0
            : pair.a_ne1 == pair.b_ne0;
        if (!rank_shapes_match) {
            LOG_ERR("%s: LoRA A/B ranks do not match for '%s'\n", __func__, kv.first.c_str());
            return false;
        }
    }
    return true;
}

} // namespace

bool validate_lora_adapter_rank(const std::string & input_path,
                                const std::string & expected_arch,
                                int32_t requested_rank) {
    if (requested_rank <= 0) {
        LOG_ERR("%s: requested LoRA rank must be positive\n", __func__);
        return false;
    }
    std::map<std::string, source_lora_pair> pairs;
    float source_alpha = 0.0f;
    if (!load_source_lora(input_path, expected_arch, pairs, source_alpha, false)) {
        return false;
    }
    for (const auto & kv : pairs) {
        const source_lora_pair & pair = kv.second;
        const int64_t source_rank = name_ends_with(kv.first, "token_embd.weight") ? pair.a_ne0 : pair.a_ne1;
        if (source_rank > requested_rank) {
            LOG_ERR("%s: cannot reduce rank of '%s' from %ld to %d\n",
                    __func__, kv.first.c_str(), static_cast<long>(source_rank), requested_rank);
            return false;
        }
    }
    return true;
}

lora_tensors transform_lora_tensors(const std::string & input_path,
                                    const std::string & expected_arch,
                                    const std::vector<std::pair<std::string, ggml_tensor *>> & model_tensors,
                                    const std::vector<std::string> & targets,
                                    int32_t rank,
                                    float alpha,
                                    float input_scale,
                                    std::mt19937 & rng,
                                    int32_t model_n_layer,
                                    int32_t freeze_layers) {
    lora_tensors result;
    if (rank <= 0 || alpha <= 0.0f || !std::isfinite(alpha) || !std::isfinite(input_scale)) {
        LOG_ERR("%s: rank and alpha must be positive and adapter scale must be finite\n", __func__);
        return result;
    }
    if (!validate_lora_targets(model_tensors, targets, model_n_layer, freeze_layers)) {
        return result;
    }

    std::map<std::string, source_lora_pair> source_pairs;
    float source_alpha = 0.0f;
    if (!load_source_lora(input_path, expected_arch, source_pairs, source_alpha)) {
        return result;
    }

    std::unordered_map<std::string, ggml_tensor *> model_by_name;
    model_by_name.reserve(model_tensors.size());
    for (const auto & entry : model_tensors) {
        model_by_name.emplace(entry.first, entry.second);
    }

    std::vector<output_lora_pair> output_pairs;
    output_pairs.reserve(source_pairs.size() + model_tensors.size());
    std::unordered_map<std::string, bool> found_targets;
    for (auto & kv : source_pairs) {
        const std::string & name = kv.first;
        source_lora_pair & source = kv.second;
        const auto model_it = model_by_name.find(name);
        if (model_it == model_by_name.end() || !model_it->second) {
            LOG_ERR("%s: input adapter tensor '%s' does not exist in the base model\n",
                    __func__, name.c_str());
            return {};
        }
        ggml_tensor * model_tensor = model_it->second;
        const bool embedding = name_ends_with(name, "token_embd.weight");
        const int64_t old_rank = embedding ? source.a_ne0 : source.a_ne1;
        const bool rank_shapes_match = embedding
            ? source.a_ne0 == source.b_ne0
            : source.a_ne1 == source.b_ne0;
        const bool model_shapes_match = embedding
            ? model_tensor->ne[0] == source.b_ne1 && model_tensor->ne[1] == source.a_ne1
            : model_tensor->ne[0] == source.a_ne0 && model_tensor->ne[1] == source.b_ne1;
        if (!rank_shapes_match || !model_shapes_match || old_rank <= 0) {
            LOG_ERR("%s: LoRA tensor shape mismatch for '%s'\n", __func__, name.c_str());
            return {};
        }
        if (old_rank > rank) {
            LOG_ERR("%s: cannot reduce rank of '%s' from %ld to %d\n",
                    __func__, name.c_str(), static_cast<long>(old_rank), rank);
            return {};
        }

        const float old_scale = source_alpha != 0.0f
            ? input_scale * source_alpha / static_cast<float>(old_rank)
            : input_scale;
        const float new_scale = alpha / static_cast<float>(rank);
        const float scale_ratio = old_scale / new_scale;
        if (!std::isfinite(scale_ratio)) {
            LOG_ERR("%s: scaling conversion for '%s' is not finite\n", __func__, name.c_str());
            return {};
        }
        output_lora_pair output;
        output.name = name;
        output.embedding = embedding;
        output.a_ne0 = embedding ? rank : model_tensor->ne[0];
        output.a_ne1 = embedding ? model_tensor->ne[1] : rank;
        output.b_ne0 = rank;
        output.b_ne1 = embedding ? model_tensor->ne[0] : model_tensor->ne[1];
        output.source_rank = old_rank;
        output.source_scale = scale_ratio;
        output.has_source = true;
        output.source_a = std::move(source.a);
        output.source_b = std::move(source.b);
        output_pairs.push_back(std::move(output));
        found_targets.emplace(name, true);
    }

    for (const auto & entry : model_tensors) {
        const std::string & name = entry.first;
        ggml_tensor * tensor = entry.second;
        if (!tensor || ggml_n_dims(tensor) < 2 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            continue;
        }
        const int layer_index = tensor_layer_index(name.c_str());
        if (model_n_layer > 0 && layer_index >= model_n_layer) {
            continue;
        }
        if (!tensor_matches_targets(name.c_str(), targets, freeze_layers) || found_targets.count(name)) {
            continue;
        }

        output_lora_pair output;
        output.name = name;
        output.embedding = name_ends_with(name, "token_embd.weight");
        output.a_ne0 = output.embedding ? rank : tensor->ne[0];
        output.a_ne1 = output.embedding ? tensor->ne[1] : rank;
        output.b_ne0 = rank;
        output.b_ne1 = output.embedding ? tensor->ne[0] : tensor->ne[1];
        output.source_rank = 0;
        output.has_source = false;
        output_pairs.push_back(std::move(output));
        found_targets.emplace(name, true);
    }

    if (output_pairs.empty()) {
        LOG_ERR("%s: no LoRA tensors to prepare\n", __func__);
        return result;
    }

    const size_t mem = (2 * output_pairs.size() + 16) * ggml_tensor_overhead();
    const ggml_init_params init_params = { mem, nullptr, true };
    result.ctx.reset(ggml_init(init_params));
    if (!result.ctx) {
        LOG_ERR("%s: failed to allocate LoRA conversion context\n", __func__);
        return {};
    }

    for (const output_lora_pair & pair : output_pairs) {
        ggml_tensor * a = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_F32, pair.a_ne0, pair.a_ne1);
        ggml_tensor * b = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_F32, pair.b_ne0, pair.b_ne1);
        ggml_set_name(a, (pair.name + ".lora_a").c_str());
        ggml_set_name(b, (pair.name + ".lora_b").c_str());
        result.refs.emplace(pair.name, std::make_pair(a, b));
    }

    result.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.ctx.get(), ggml_backend_cpu_buffer_type()));
    if (!result.buf) {
        LOG_ERR("%s: failed to allocate converted LoRA tensor buffer\n", __func__);
        return {};
    }

    for (const output_lora_pair & pair : output_pairs) {
        auto it = result.refs.find(pair.name);
        ggml_tensor * a = it->second.first;
        ggml_tensor * b = it->second.second;
        float * dst_a = static_cast<float *>(a->data);
        float * dst_b = static_cast<float *>(b->data);
        initialize_lora_weights(a, b, pair.embedding, rng);

        if (!pair.has_source) {
            continue;
        }

        if (!pair.embedding) {
            const int64_t input_dim = a->ne[0];
            for (int64_t r = 0; r < pair.source_rank; ++r) {
                std::memcpy(dst_a + input_dim * r,
                            pair.source_a.data() + input_dim * r,
                            static_cast<size_t>(input_dim) * sizeof(float));
            }
            const int64_t output_dim = b->ne[1];
            for (int64_t j = 0; j < output_dim; ++j) {
                for (int64_t r = 0; r < pair.source_rank; ++r) {
                    dst_b[r + rank * j] = pair.source_b[r + pair.source_rank * j] * pair.source_scale;
                }
            }
        } else {
            const int64_t embedding_dim = a->ne[1];
            for (int64_t j = 0; j < embedding_dim; ++j) {
                for (int64_t r = 0; r < pair.source_rank; ++r) {
                    dst_a[r + rank * j] = pair.source_a[r + pair.source_rank * j];
                }
            }
            const int64_t vocab_size = b->ne[1];
            for (int64_t j = 0; j < vocab_size; ++j) {
                for (int64_t r = 0; r < pair.source_rank; ++r) {
                    dst_b[r + rank * j] = pair.source_b[r + pair.source_rank * j] * pair.source_scale;
                }
            }
        }
    }

    LOG_INF("%s: prepared %zu LoRA tensor pairs at rank %d and alpha %.4g\n",
            __func__, result.refs.size(), rank, alpha);
    return result;
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

static std::string basename_from_path(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool save_adapter(const lora_tensor_refs & tensors,
                  const std::string & out_path,
                  const std::string & arch,
                  float alpha,
                  const std::string & base_model_path) {
    if (tensors.empty()) {
        LOG_ERR("%s: refusing to save an adapter with no LoRA tensors\n", __func__);
        return false;
    }

    gguf_context_ptr gctx(gguf_init_empty());
    if (!gctx) {
        LOG_ERR("%s: failed to allocate GGUF context\n", __func__);
        return false;
    }

    gguf_set_val_str(gctx.get(), "general.type", "adapter");
    gguf_set_val_str(gctx.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(gctx.get(), "adapter.type", "lora");
    gguf_set_val_f32(gctx.get(), "adapter.lora.alpha", alpha);
    gguf_set_val_str(gctx.get(), "adapter.base_model", basename_from_path(base_model_path).c_str());

    for (const auto & kv : tensors) {
        gguf_add_tensor(gctx.get(), kv.second.first);
        gguf_add_tensor(gctx.get(), kv.second.second);
    }

    // Write to a sibling temporary file first so a failed periodic snapshot
    // does not leave a partly-written GGUF at the requested destination.
    const std::string temp_path = out_path + ".tmp";
    std::remove(temp_path.c_str());
    std::ofstream fout(temp_path, std::ios::binary | std::ios::trunc);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, temp_path.c_str());
        return false;
    }

    const size_t meta_size = gguf_get_meta_size(gctx.get());
    {
        std::vector<char> zeros_buf(meta_size, 0);
        fout.write(zeros_buf.data(), static_cast<std::streamsize>(zeros_buf.size()));
    }

    if (fout.good()) {
        for (const auto & kv : tensors) {
            for (ggml_tensor * tensor : { kv.second.first, kv.second.second }) {
                const size_t nbytes = ggml_nbytes(tensor);
                std::vector<char> cpu_buf(nbytes);
                ggml_backend_tensor_get(tensor, cpu_buf.data(), 0, nbytes);
                fout.write(cpu_buf.data(), static_cast<std::streamsize>(nbytes));
                const size_t padding = GGML_PAD(nbytes, 32) - nbytes;
                if (padding > 0) {
                    std::vector<char> pad_buf(padding, 0);
                    fout.write(pad_buf.data(), static_cast<std::streamsize>(padding));
                }
                if (!fout.good()) {
                    break;
                }
            }
            if (!fout.good()) {
                break;
            }
        }
    }

    if (fout.good()) {
        std::vector<uint8_t> meta(meta_size);
        gguf_get_meta_data(gctx.get(), meta.data());
        fout.seekp(0);
        fout.write(reinterpret_cast<const char *>(meta.data()), static_cast<std::streamsize>(meta.size()));
    }

    fout.close();
    const bool write_ok = fout.good();

    if (!write_ok) {
        LOG_ERR("%s: failed while writing adapter %s\n", __func__, out_path.c_str());
        std::remove(temp_path.c_str());
        return false;
    }

    // POSIX rename atomically replaces an existing destination. On platforms
    // where it does not, temporarily move the old destination aside and restore
    // it if installing the completed file fails; never delete the old adapter
    // before the replacement has succeeded.
    if (std::rename(temp_path.c_str(), out_path.c_str()) != 0) {
        const std::string backup_path = out_path + ".previous";
        if (std::rename(out_path.c_str(), backup_path.c_str()) != 0) {
            LOG_ERR("%s: cannot replace %s with completed adapter file\n", __func__, out_path.c_str());
            std::remove(temp_path.c_str());
            return false;
        }

        if (std::rename(temp_path.c_str(), out_path.c_str()) != 0) {
            const int restore_result = std::rename(backup_path.c_str(), out_path.c_str());
            LOG_ERR("%s: cannot install completed adapter at %s\n", __func__, out_path.c_str());
            if (restore_result != 0) {
                LOG_ERR("%s: previous adapter remains at %s and could not be restored automatically\n",
                        __func__, backup_path.c_str());
            }
            std::remove(temp_path.c_str());
            return false;
        }
        std::remove(backup_path.c_str());
    }

    LOG_INF("%s: adapter saved to %s\n", __func__, out_path.c_str());
    return true;
}
