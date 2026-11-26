#include "llamango-graph-training.h"

#include "llama-model.h"
#include "llama-impl.h"
#include "llama-cparams.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T>
static void fill_attn_mask(
        T * data,
        int64_t n_elements,
        int64_t n_kv,
        const llama_ubatch * ubatch,
        const llama_hparams & hparams,
        int n_swa,
        llama_swa_type swa_type,
        bool causal) {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_kv >= n_tokens);
    std::fill(data, data + n_elements, llama_cast<T>(-INFINITY));

    for (int i1 = 0; i1 < n_tokens; ++i1) {
        const llama_seq_id s1 = ubatch->seq_id[i1][0];
        const llama_pos p1 = ubatch->pos[i1];
        const uint64_t idst = i1*n_kv;

        for (int i0 = 0; i0 < n_tokens; ++i0) {
            const llama_seq_id s0 = ubatch->seq_id[i0][0];
            const llama_pos p0 = ubatch->pos[i0];

            if (s0 != s1 || (causal && p0 > p1) || llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                continue;
            }

            data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
        }
    }
}

static void llm_graph_fill_training_attn_mask(
        ggml_tensor * mask,
        const llama_ubatch * ubatch,
        const llama_hparams & hparams,
        int n_swa,
        llama_swa_type swa_type,
        bool causal) {
    if (!mask || !mask->buffer) {
        return;
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));

    if (mask->type == GGML_TYPE_F16) {
        fill_attn_mask((ggml_fp16_t *) mask->data, ggml_nelements(mask), mask->ne[0], ubatch, hparams, n_swa, swa_type, causal);
    } else {
        GGML_ASSERT(mask->type == GGML_TYPE_F32);
        fill_attn_mask((float *) mask->data, ggml_nelements(mask), mask->ne[0], ubatch, hparams, n_swa, swa_type, causal);
    }
}

static void llm_graph_fill_training_relative_pos(
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        uint32_t extent) {
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_kv = dst->ne[0];

    GGML_ASSERT(n_kv == n_tokens);
    GGML_ASSERT(dst->ne[1] == n_tokens);

    std::vector<int32_t> data(n_kv * n_tokens, (int32_t) extent);
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < n_kv; ++j) {
            if (ubatch->seq_id[i][0] != ubatch->seq_id[j][0]) {
                continue;
            }

            const int64_t distance = int64_t(ubatch->pos[i]) - ubatch->pos[j];
            data[i*n_kv + j] = distance >= 0 && distance < extent ? (int32_t) distance : (int32_t) extent;
        }
    }

    ggml_backend_tensor_set(dst, data.data(), 0, ggml_nbytes(dst));
}

void llm_graph_set_training_attn_mask(
        const llama_kv_cache_context * cache,
        const llm_attn_mask & mask,
        const llama_ubatch * ubatch,
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        int n_swa,
        llama_swa_type swa_type,
        bool causal_training) {
    if (cparams.training) {
        llm_graph_fill_training_attn_mask(mask.raw, ubatch, hparams, n_swa, swa_type, causal_training);
    } else {
        cache->set_input_attn_mask(mask, ubatch, cparams.causal_attn);
    }
}

void llm_graph_set_training_relative_pos(
        const llama_kv_cache_context * cache,
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        uint32_t extent,
        bool training) {
    if (training) {
        llm_graph_fill_training_relative_pos(dst, ubatch, extent);
    } else {
        cache->set_input_relative_pos(dst, ubatch, extent);
    }
}

bool llm_graph_context::use_shared_kv_direct() const {
    return cparams.training && hparams.has_shared_kv();
}

void llm_graph_context::resolve_shared_kv(uint32_t il, ggml_tensor * & k, ggml_tensor * & v) const {
    if (!use_shared_kv_direct()) {
        return;
    }

    GGML_ASSERT(il < shared_kv_k.size());

    if (hparams.has_kv(il)) {
        GGML_ASSERT(k != nullptr);
        shared_kv_k[il] = k;
        shared_kv_v[il] = v;
        return;
    }

    const uint32_t source = hparams.kv_source(il);
    GGML_ASSERT(source < il);
    GGML_ASSERT(source < shared_kv_k.size());
    GGML_ASSERT(shared_kv_k[source] != nullptr);

    k = shared_kv_k[source];
    if (shared_kv_v[source]) {
        v = shared_kv_v[source];
    }
}
