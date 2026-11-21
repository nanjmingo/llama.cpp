#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <locale>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_opt_dataset {
    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    struct ggml_tensor    * data   = nullptr;
    struct ggml_tensor    * labels = nullptr;
    struct ggml_tensor    * weights = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;
    size_t  nbs_weights = -1;

    std::vector<int64_t> permutation;
};

struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    enum ggml_opt_loss_type    loss_type;
    enum ggml_opt_build_type   build_type;
    enum ggml_opt_build_type   build_type_alloc;

    struct ggml_tensor * inputs  = nullptr;
    struct ggml_tensor * outputs = nullptr;
    struct ggml_tensor * labels  = nullptr;

    struct ggml_tensor * loss     = nullptr;
    struct ggml_tensor * pred     = nullptr;
    struct ggml_tensor * ncorrect = nullptr;
    struct ggml_tensor * nvalid   = nullptr;

    struct ggml_cgraph * gf      = nullptr;
    struct ggml_cgraph * gb_grad = nullptr;
    struct ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<struct ggml_tensor *> grad_accs;
    std::vector<std::pair<int32_t, int32_t>> recompute_backend_affinities;
    std::vector<struct ggml_tensor *> grad_m;
    std::vector<struct ggml_tensor *> grad_v;
    std::vector<ggml_backend_buffer_t> bufs_grad_accs;
    std::vector<struct ggml_context *> ctxs_grad_accs;
    std::vector<ggml_backend_buffer_t> bufs_momenta;
    std::vector<struct ggml_context *> ctxs_momenta;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    float   loss_normalizer    = 0.0f;
    bool    loss_per_datapoint = false;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    void *                        get_opt_pars_ud = nullptr;
    struct ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    enum ggml_type optimizer_state_type = GGML_TYPE_F32;
    bool activation_checkpointing = false;
    bool optimizer_state_ready = false;
    std::vector<uint8_t> pending_optimizer_state;
};

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;
    int64_t              naccuracy_data = 0;

    std::vector<int32_t> opt_periods;
    bool                  loss_per_datapoint = false;
    double                loss_sum = 0.0;
    double                loss_squared_sum = 0.0;
};

namespace {
    bool ggml_opt_apply_pending_state(ggml_opt_context_t opt_ctx);

    struct ggml_opt_recompute_state {
        std::unordered_set<ggml_tensor *> forward;
        std::unordered_set<ggml_tensor *> keep;
        std::unordered_map<ggml_tensor *, ggml_tensor *> clones;
    };

    static ggml_tensor * ggml_opt_recompute_clone(
            ggml_context * ctx,
            ggml_tensor * tensor,
            ggml_opt_recompute_state & state) {
        if (!tensor || state.forward.find(tensor) == state.forward.end() ||
                state.keep.find(tensor) != state.keep.end() || tensor->type == GGML_TYPE_I32) {
            return tensor;
        }

        const auto found = state.clones.find(tensor);
        if (found != state.clones.end()) {
            return found->second;
        }

        ggml_tensor * clone = ggml_dup_tensor(ctx, tensor);
        state.clones.emplace(tensor, clone);
        clone->op = tensor->op;
        memcpy(clone->nb, tensor->nb, sizeof(tensor->nb));
        memcpy(clone->op_params, tensor->op_params, sizeof(tensor->op_params));
        clone->flags = tensor->flags & ~(GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_INPUT |
                                         GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_LOSS);
        ggml_format_name(clone, "%.48s (recompute)", tensor->name);
        clone->view_offs = tensor->view_offs;
        clone->view_src = ggml_opt_recompute_clone(ctx, tensor->view_src, state);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            clone->src[i] = ggml_opt_recompute_clone(ctx, tensor->src[i], state);
        }

        return clone;
    }

    static void ggml_opt_recompute_checkpoints(ggml_opt_context_t opt_ctx) {
        opt_ctx->recompute_backend_affinities.clear();
        if (!opt_ctx->activation_checkpointing || opt_ctx->static_graphs ||
                opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD || opt_ctx->gf->n_nodes == 0) {
            return;
        }

        ggml_cgraph * graph = opt_ctx->gb_grad;
        const int n_forward = opt_ctx->gf->n_nodes;
        GGML_ASSERT(graph && n_forward <= graph->n_nodes);

        ggml_opt_recompute_state state;
        state.forward.reserve(static_cast<size_t>(n_forward));
        state.keep.reserve(static_cast<size_t>(n_forward));

        for (int i = 0; i < n_forward; ++i) {
            state.forward.insert(graph->nodes[i]);
        }

        for (int i = 0; i < n_forward; ++i) {
            ggml_tensor * node = graph->nodes[i];
            bool keep = node->op == GGML_OP_GET_ROWS ||
                (node->flags & (GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_INPUT |
                                GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_LOSS)) != 0;

            for (int j = 0; j < GGML_MAX_SRC && !keep && node->view_src == nullptr; ++j) {
                ggml_tensor * src = node->src[j];
                if (!src) {
                    continue;
                }
                ggml_tensor * base = src;
                while (base->view_src) {
                    base = base->view_src;
                }
                keep = state.forward.find(base) == state.forward.end() && base->buffer &&
                    ggml_backend_buffer_get_usage(base->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            }

            if (keep) {
                state.keep.insert(node);
            }
        }

        for (int i = 0; i < n_forward; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if (std::strncmp(node->name, "l_out-", 6) == 0) {
                state.keep.insert(graph->nodes[i]);
            }
        }

        for (int i = n_forward; i < graph->n_nodes; ++i) {
            ggml_tensor * node = graph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                node->src[j] = ggml_opt_recompute_clone(opt_ctx->ctx_compute, node->src[j], state);
            }
            if (node->view_src) {
                node->view_src = ggml_opt_recompute_clone(opt_ctx->ctx_compute, node->view_src, state);
            }
        }

        if (state.clones.empty()) {
            return;
        }

        ggml_cgraph * reordered = ggml_new_graph_custom(opt_ctx->ctx_compute, graph->size, graph->grads != nullptr);
        GGML_ASSERT(reordered && "failed to allocate activation checkpoint graph");
        for (int i = 0; i < n_forward; ++i) {
            ggml_build_forward_expand(reordered, graph->nodes[i]);
        }
        for (int i = n_forward; i < graph->n_nodes; ++i) {
            ggml_build_forward_expand(reordered, graph->nodes[i]);
        }

        if (graph->grads) {
            for (size_t i = 0; i < graph->visited_hash_set.size; ++i) {
                if (!ggml_bitset_get(graph->visited_hash_set.used, i)) {
                    continue;
                }
                ggml_tensor * key = graph->visited_hash_set.keys[i];
                const size_t new_index = ggml_hash_find(&reordered->visited_hash_set, key);
                if (new_index == GGML_HASHSET_FULL ||
                        !ggml_bitset_get(reordered->visited_hash_set.used, new_index)) {
                    continue;
                }
                reordered->grads[new_index] = graph->grads[i];
                reordered->grad_accs[new_index] = graph->grad_accs[i];
            }
        }

        // Remember where each recomputed activation came from. The scheduler
        // otherwise sees all recompute nodes after the forward graph and may
        // extend the final forward backend across the entire recompute section.
        std::unordered_map<ggml_tensor *, int32_t> forward_indices;
        std::unordered_map<ggml_tensor *, int32_t> reordered_indices;
        forward_indices.reserve(static_cast<size_t>(n_forward));
        reordered_indices.reserve(static_cast<size_t>(reordered->n_nodes));
        for (int i = 0; i < n_forward; ++i) {
            forward_indices.emplace(graph->nodes[i], i);
        }
        for (int i = 0; i < reordered->n_nodes; ++i) {
            reordered_indices.emplace(reordered->nodes[i], i);
        }
        for (const auto & entry : state.clones) {
            const auto source = forward_indices.find(entry.first);
            const auto clone = reordered_indices.find(entry.second);
            GGML_ASSERT(source != forward_indices.end());
            GGML_ASSERT(clone != reordered_indices.end());
            opt_ctx->recompute_backend_affinities.emplace_back(source->second, clone->second);
        }

        ggml_graph_clear(graph);
        ggml_graph_cpy(reordered, graph);
    }

    static ggml_tensor * ggml_opt_clone_tensor_for_sched(
            ggml_context * ctx,
            ggml_tensor * tensor,
            std::unordered_map<ggml_tensor *, ggml_tensor *> & clones) {
        if (!tensor) {
            return nullptr;
        }

        const auto found = clones.find(tensor);
        if (found != clones.end()) {
            return found->second;
        }

        ggml_tensor * clone = ggml_dup_tensor(ctx, tensor);
        clones.emplace(tensor, clone);

        // Keep the original buffer metadata for every tensor that already has
        // storage. The scheduler uses this metadata to recover the backend of
        // preallocated graph outputs as well as weights and inputs. In
        // particular, dropping buffers from computed forward nodes makes the
        // placement probe differ from the graph that will actually run; with a
        // separate optimizer graph (gradient accumulation > 1), the recompute
        // section can then inherit one neighbouring GPU and skew VRAM usage.
        // The probe only performs graph splitting, not computation, so it does
        // not need to share the source tensor's data pointer or backend extra.
        clone->buffer = tensor->buffer;
        clone->op     = tensor->op;
        clone->flags  = tensor->flags;
        clone->view_offs = tensor->view_offs;
        memcpy(clone->nb, tensor->nb, sizeof(tensor->nb));
        memcpy(clone->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(clone, tensor->name);

        clone->view_src = ggml_opt_clone_tensor_for_sched(ctx, tensor->view_src, clones);
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            clone->src[i] = ggml_opt_clone_tensor_for_sched(ctx, tensor->src[i], clones);
        }

        return clone;
    }

    static ggml_cgraph * ggml_opt_clone_graph_for_sched(
            ggml_context * ctx,
            ggml_cgraph * graph,
            std::unordered_map<ggml_tensor *, ggml_tensor *> & clones) {
        ggml_cgraph * clone = ggml_new_graph_custom(ctx, graph->size, /* grads = */ false);
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * node = ggml_opt_clone_tensor_for_sched(ctx, graph->nodes[i], clones);
            ggml_build_forward_expand(clone, node);
        }

        GGML_ASSERT(clone->n_nodes == graph->n_nodes);
        for (int i = 0; i < graph->n_nodes; ++i) {
            GGML_ASSERT(clone->nodes[i] == clones.at(graph->nodes[i]));
        }
        return clone;
    }

    static void ggml_opt_set_recompute_backend_affinities(
            ggml_opt_context_t opt_ctx,
            ggml_cgraph * graph) {
        if (opt_ctx->static_graphs || graph != opt_ctx->gb_grad ||
                opt_ctx->recompute_backend_affinities.empty()) {
            return;
        }

        // Probe backend placement on a deep metadata clone of the graph that
        // will actually be scheduled. The scheduler's split pass mutates
        // node->src[] to insert cross-backend copies; the probe must therefore
        // be independent from the graph that will subsequently be evaluated.
        //
        // Do not probe gf alone: with gradient accumulation, gb_grad also
        // contains persistent gradient-accumulator leaves and the backward
        // graph. Those nodes participate in scheduler's graph-wide backend
        // expansion and can change the placement of forward nodes. Probing only
        // gf produces affinities that do not match the real graph, which leaves
        // the appended recompute section concentrated on one GPU.
        const size_t n_tensors = graph->visited_hash_set.size + 8;
        const size_t mem_size = n_tensors * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph->size, /* grads = */ false) + 1024;
        ggml_init_params params = { mem_size, nullptr, true };
        ggml_context * ctx_probe = ggml_init(params);
        GGML_ASSERT(ctx_probe != nullptr);

        std::unordered_map<ggml_tensor *, ggml_tensor *> probe_tensors;
        probe_tensors.reserve(graph->visited_hash_set.size);
        ggml_cgraph * graph_probe = ggml_opt_clone_graph_for_sched(ctx_probe, graph, probe_tensors);

        ggml_backend_sched_split_graph(opt_ctx->backend_sched, graph_probe);

        std::vector<std::pair<int32_t, ggml_backend_t>> backends;
        backends.reserve(opt_ctx->recompute_backend_affinities.size());
        for (const auto & affinity : opt_ctx->recompute_backend_affinities) {
            GGML_ASSERT(affinity.first >= 0 && affinity.first < opt_ctx->gf->n_nodes);
            ggml_tensor * source_node = opt_ctx->gf->nodes[affinity.first];
            // Read the backend chosen for the original forward node in the
            // context of the complete forward + backward + accumulator graph.
            // Pin every recomputed node, including weight-backed operations, to
            // that backend so the appended recompute section mirrors the actual
            // forward placement instead of inheriting one neighbouring GPU.
            ggml_tensor * probe_node = probe_tensors.at(source_node);
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(opt_ctx->backend_sched, probe_node);
            if (backend) {
                backends.emplace_back(affinity.second, backend);
            }
        }

        // Discard the probe assignments before pinning recompute nodes in the
        // real graph. The probe graph is independent, so freeing it is safe.
        ggml_backend_sched_reset(opt_ctx->backend_sched);
        for (const auto & affinity : backends) {
            GGML_ASSERT(affinity.first >= 0 && affinity.first < graph->n_nodes);
            ggml_backend_sched_set_tensor_backend(
                opt_ctx->backend_sched, graph->nodes[affinity.first], affinity.second);
        }
        ggml_free(ctx_probe);
    }

}

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->weights = nullptr;
    result->nbs_weights = 0;

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

ggml_opt_dataset_t ggml_opt_dataset_init_with_weights(
        enum ggml_type type_data,
        enum ggml_type type_label,
        enum ggml_type type_weights,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ne_weights,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ne_weights   >  0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
             3*ggml_tensor_overhead(),
             nullptr,
             true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->weights = ggml_new_tensor_2d(result->ctx, type_weights, ne_weights, ndata);
    result->nbs_weights = ggml_nbytes(result->weights) * ndata_shard/ndata;

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

int64_t ggml_opt_dataset_permutation_size(ggml_opt_dataset_t dataset) {
    return static_cast<int64_t>(dataset->permutation.size());
}

bool ggml_opt_dataset_get_permutation(ggml_opt_dataset_t dataset, int64_t * permutation, int64_t size) {
    if (!dataset || !permutation || size != static_cast<int64_t>(dataset->permutation.size())) {
        return false;
    }
    std::copy(dataset->permutation.begin(), dataset->permutation.end(), permutation);
    return true;
}

bool ggml_opt_dataset_set_permutation(ggml_opt_dataset_t dataset, const int64_t * permutation, int64_t size) {
    if (!dataset || !permutation || size != static_cast<int64_t>(dataset->permutation.size())) {
        return false;
    }
    std::vector<bool> seen(static_cast<size_t>(size), false);
    for (int64_t i = 0; i < size; ++i) {
        const int64_t value = permutation[i];
        if (value < 0 || value >= size || seen[static_cast<size_t>(value)]) {
            return false;
        }
        seen[static_cast<size_t>(value)] = true;
    }
    std::copy(permutation, permutation + size, dataset->permutation.begin());
    return true;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

struct ggml_tensor * ggml_opt_dataset_weights(ggml_opt_dataset_t dataset) {
    return dataset->weights;
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data       = (const char *) dataset->data->data + ishard      *dataset->nbs_data;
        char       * ptr_data_batch = (char       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels       = (const char *) dataset->labels->data + ishard      *dataset->nbs_labels;
        char       * ptr_labels_batch = (char       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

void ggml_opt_dataset_get_batch_host_with_weights(
        ggml_opt_dataset_t dataset,
        void * data_batch,
        size_t nb_data_batch,
        void * labels_batch,
        void * weights_batch,
        size_t nb_weights_batch,
        int64_t ibatch) {
    GGML_ASSERT(dataset->weights != nullptr);
    GGML_ASSERT(weights_batch != nullptr);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    ggml_opt_dataset_get_batch_host(dataset, data_batch, nb_data_batch, labels_batch, ibatch);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;
    GGML_ASSERT(nb_weights_batch == shards_per_batch*dataset->nbs_weights);
    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
        const char * ptr_weights = (const char *) dataset->weights->data + ishard*dataset->nbs_weights;
        char * ptr_weights_batch = (char *) weights_batch + ishard_batch*dataset->nbs_weights;
        memcpy(ptr_weights_batch, ptr_weights, dataset->nbs_weights);
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 5e-5f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    return result;
}


struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct ggml_opt_optimizer_params *) userdata);
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        enum ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
         GGML_TYPE_F32,
    };
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const enum ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    // Parameter gradients only need a persistent accumulator when more than
    // one microbatch contributes to an optimizer step. In dynamic mode the
    // old condition kept a separate F32 GPU buffer even for opt_period == 1,
    // although the optimizer consumes that gradient only once.
    const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        opt_ctx->opt_period > 1;

    const bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_input(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor * node = opt_ctx->gf->nodes[i];
        GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - the loss gradient accumulator
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_const = opt_ctx->static_graphs ?
            9 + (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY ? 11 : 0) : 0;
        const size_t size_meta = (n_loss + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    switch (opt_ctx->loss_type) {
        case GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = false;
            break;
        }
        case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            if (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        ggml_tensor * labels_sum = ggml_sum_rows(ctx_results, opt_ctx->labels);
        ggml_tensor * labels_valid = ggml_step(ctx_results, ggml_abs(ctx_results, labels_sum));
        labels_valid = ggml_reshape_1d(ctx_results, labels_valid, ggml_nelements(opt_ctx->pred));
        ggml_tensor * labels_invalid = ggml_scale_bias(ctx_results, labels_valid, -1.0f, 1.0f);

        ggml_tensor * pred_masked = ggml_mul(ctx_results,
            ggml_cast(ctx_results, opt_ctx->pred, GGML_TYPE_F32), labels_valid);
        pred_masked = ggml_add(ctx_results, pred_masked,
            ggml_scale(ctx_results, labels_invalid, (float) opt_ctx->outputs->ne[0]));
        pred_masked = ggml_cast(ctx_results, pred_masked, GGML_TYPE_I32);

        opt_ctx->ncorrect = ggml_count_equal(ctx_results, pred_masked, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);

        opt_ctx->nvalid = ggml_sum(ctx_results, labels_valid);
        ggml_set_name(opt_ctx->nvalid, "nvalid");
        ggml_set_output(opt_ctx->nvalid);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->nvalid);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);

        // Keep the loss accumulator in the static buffer, but allocate parameter
        // accumulators on the same buffer type as their corresponding parameters.
        // Group tensors by buffer type to avoid creating one backend allocation per
        // parameter while preserving the parameter-to-device placement.
        std::map<ggml_backend_buffer_type_t, std::vector<int>> grad_acc_indices;
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(
                    opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            } else if (accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
                const ggml_backend_buffer_type_t param_buft = node->buffer
                    ? ggml_backend_buffer_get_type(node->buffer)
                    : ggml_backend_cpu_buffer_type();
                grad_acc_indices[param_buft].push_back(i);
                opt_ctx->grad_accs[i] = nullptr;
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        for (const auto & entry : grad_acc_indices) {
            const ggml_backend_buffer_type_t param_buft = entry.first;
            const std::vector<int> & indices = entry.second;
            const size_t size_meta = indices.size() * ggml_tensor_overhead();
            struct ggml_init_params gparams = { size_meta, nullptr, true };
            struct ggml_context * gctx = ggml_init(gparams);
            GGML_ASSERT(gctx != nullptr);

            for (int i : indices) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                opt_ctx->grad_accs[i] = ggml_new_tensor(
                    gctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                ggml_format_name(opt_ctx->grad_accs[i], "grad_acc for %s", node->name);
            }

            ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, param_buft);
            GGML_ASSERT(gbuf != nullptr);
            ggml_backend_buffer_clear(gbuf, 0);
            opt_ctx->bufs_grad_accs.push_back(gbuf);
            opt_ctx->ctxs_grad_accs.push_back(gctx);
        }

        if (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    ggml_backend_buffer_type_t param_buft = node->buffer
                        ? ggml_backend_buffer_get_type(node->buffer)
                        : ggml_backend_cpu_buffer_type();

                    const size_t sz = 2 * ggml_tensor_overhead();
                    struct ggml_init_params mip = { sz, nullptr, true };
                    struct ggml_context * mctx = ggml_init(mip);
                    opt_ctx->grad_m[i] = ggml_new_tensor(mctx, opt_ctx->optimizer_state_type, GGML_MAX_DIMS, node->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(mctx, opt_ctx->optimizer_state_type, GGML_MAX_DIMS, node->ne);
                    ggml_backend_buffer_t mbuf = ggml_backend_alloc_ctx_tensors_from_buft(mctx, param_buft);
                    ggml_backend_buffer_clear(mbuf, 0);
                    opt_ctx->bufs_momenta.push_back(mbuf);
                    opt_ctx->ctxs_momenta.push_back(mctx);
                } else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());
    ggml_opt_recompute_checkpoints(opt_ctx);

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // Static graphs keep a duplicated graph with the optimizer update.
    // Dynamic graphs with accumulation run the update in a separate graph that
    // consumes the persistent gradient accumulators. With opt_period == 1, the
    // update is appended directly to gb_grad below, so it can consume the
    // computed gradients without another accumulator or a second backward pass.
    if (opt_ctx->static_graphs) {
        opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);
    } else if (opt_ctx->opt_period == 1) {
        // With one microbatch the optimizer can consume the freshly computed
        // parameter gradients directly. Append the update to the backward graph
        // so it runs in the same scheduler allocation; a separate graph would
        // pull the gradient-producing subgraph in again (or require a duplicate
        // persistent gradient buffer).
        opt_ctx->gb_opt = nullptr;
    } else {
        const int graph_size = std::max(16, 2 * opt_ctx->gf->n_nodes + 16);
        opt_ctx->gb_opt = ggml_new_graph_custom(opt_ctx->ctx_compute, graph_size, /*grads =*/ false);
        GGML_ASSERT(opt_ctx->gb_opt && "failed to allocate optimizer update graph");
    }

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_input(adamw_params);
    const char * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    ggml_cgraph * optimizer_graph = opt_ctx->static_graphs || opt_ctx->opt_period > 1
        ? opt_ctx->gb_opt : opt_ctx->gb_grad;
    for (int i = opt_ctx->gf->n_nodes - 1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->static_graphs
            ? opt_ctx->gb_opt->nodes[i]
            : opt_ctx->gb_grad->nodes[i];
        struct ggml_tensor * grad = opt_ctx->static_graphs
            ? ggml_graph_get_grad(opt_ctx->gb_opt, node)
            : (opt_ctx->opt_period > 1
                ? ggml_graph_get_grad_acc(opt_ctx->gb_grad, node)
                : ggml_graph_get_grad(opt_ctx->gb_grad, node));

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            if (opt_ctx->loss_normalizer > 0.0f) {
                GGML_ASSERT(opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
                const float scale = static_cast<float>(
                    ggml_nrows(opt_ctx->outputs) * opt_ctx->opt_period) / opt_ctx->loss_normalizer;
                grad = ggml_scale(opt_ctx->ctx_compute, grad, scale);
            }
            struct ggml_tensor * m = nullptr;
            struct ggml_tensor * v = nullptr;
            if (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", node->name);
                ggml_format_name(v, "AdamW v for %s", node->name);
            }
            struct ggml_tensor * opt_step;
            switch (optimizer) {
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
                    break;
                case GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, node, grad, adamw_params);
                    break;
                default:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, node->name);
            ggml_build_forward_expand(optimizer_graph, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        // In dynamic mode the persistent parameter gradient accumulators belong
        // to gb_grad; for opt_period == 1, optimizer updates are also part of gb_grad.
        ggml_graph_reset(opt_ctx->static_graphs ? opt_ctx->gb_opt : opt_ctx->gb_grad);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
    opt_ctx->optimizer_state_ready = true;
    GGML_ASSERT(ggml_opt_apply_pending_state(opt_ctx) && "failed to restore optimizer state");
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->loss_type        = params.loss_type;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs          = params.outputs;
    result->opt_period       = params.opt_period;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;
    result->optimizer        = params.optimizer;
    result->optimizer_state_type = params.optimizer_state_type;
    GGML_ASSERT(result->optimizer_state_type == GGML_TYPE_F32 ||
                result->optimizer_state_type == GGML_TYPE_F16 ||
                result->optimizer_state_type == GGML_TYPE_BF16);

    GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    if (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs);
        return result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    ggml_opt_build(result);

    return result;
}

namespace {

struct ggml_opt_state_tensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
    std::vector<uint8_t> data;
};

struct ggml_opt_state_data {
    uint32_t version = 0;
    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    enum ggml_type optimizer_state_type = GGML_TYPE_F32;
    uint64_t iter = 1;
    std::string rng;
    std::vector<ggml_opt_state_tensor> tensors;
};

constexpr char GGML_OPT_STATE_MAGIC[8] = {'G', 'G', 'O', 'P', 'T', 'S', 'T', '1'};
constexpr uint32_t GGML_OPT_STATE_VERSION = 2;

struct opt_state_reader {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    template<typename T>
    bool read_int(T & value) {
        static_assert(std::is_integral<T>::value, "integer type required");
        if (pos > size || size - pos < sizeof(T)) return false;
        uint64_t result = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            result |= static_cast<uint64_t>(data[pos++]) << (i * 8);
        }
        value = static_cast<T>(result);
        return true;
    }

    bool read_string(std::string & value, size_t len) {
        if (pos > size || size - pos < len) return false;
        value.assign(reinterpret_cast<const char *>(data + pos), len);
        pos += len;
        return true;
    }

    bool read_view(const uint8_t *& value, size_t len) {
        if (pos > size || size - pos < len) return false;
        value = data + pos;
        pos += len;
        return true;
    }

    bool skip_bytes(size_t len) {
        if (pos > size || size - pos < len) return false;
        pos += len;
        return true;
    }

    bool read_bytes(std::vector<uint8_t> & value, size_t len) {
        const uint8_t * bytes = nullptr;
        if (!read_view(bytes, len)) return false;
        value.assign(bytes, bytes + len);
        return true;
    }
};

bool ggml_opt_state_parse(const void * data, size_t size, ggml_opt_state_data & state) {
    if (!data || size < sizeof(GGML_OPT_STATE_MAGIC)) return false;
    opt_state_reader reader{static_cast<const uint8_t *>(data), size, 0};
    std::string magic;
    if (!reader.read_string(magic, sizeof(GGML_OPT_STATE_MAGIC)) ||
        magic.compare(0, sizeof(GGML_OPT_STATE_MAGIC), GGML_OPT_STATE_MAGIC, sizeof(GGML_OPT_STATE_MAGIC)) != 0 ||
        !reader.read_int(state.version) || (state.version != 1 && state.version != GGML_OPT_STATE_VERSION)) {
        return false;
    }
    uint32_t optimizer = 0;
    uint32_t optimizer_state_type = GGML_TYPE_F32;
    uint64_t rng_size = 0;
    uint32_t tensor_count = 0;
    if (!reader.read_int(optimizer) || optimizer >= GGML_OPT_OPTIMIZER_TYPE_COUNT ||
        (state.version >= 2 && (!reader.read_int(optimizer_state_type) || optimizer_state_type >= GGML_TYPE_COUNT)) ||
        !reader.read_int(state.iter) || state.iter < 1 ||
        !reader.read_int(rng_size) || rng_size > size ||
        !reader.read_string(state.rng, static_cast<size_t>(rng_size)) ||
        !reader.read_int(tensor_count) || tensor_count > size / 16) {
        return false;
    }
    state.optimizer = static_cast<ggml_opt_optimizer_type>(optimizer);
    state.optimizer_state_type = static_cast<ggml_type>(optimizer_state_type);
    state.tensors.clear();
    state.tensors.reserve(tensor_count);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        ggml_opt_state_tensor tensor;
        uint32_t name_size = 0;
        uint32_t type = 0;
        uint32_t n_dims = 0;
        uint64_t nbytes = 0;
        if (!reader.read_int(name_size) || name_size == 0 || name_size > GGML_MAX_NAME ||
            !reader.read_string(tensor.name, name_size) ||
            !reader.read_int(type) || type >= GGML_TYPE_COUNT ||
            !reader.read_int(n_dims) || n_dims > GGML_MAX_DIMS) {
            return false;
        }
        tensor.type = static_cast<ggml_type>(type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            uint64_t ne = 0;
            if (!reader.read_int(ne) || ne == 0 || ne > static_cast<uint64_t>(INT64_MAX)) return false;
            tensor.ne[d] = static_cast<int64_t>(ne);
            if (d >= static_cast<int>(n_dims) && tensor.ne[d] != 1) return false;
        }
        if (!reader.read_int(nbytes) || nbytes > size || nbytes > SIZE_MAX ||
            !reader.read_bytes(tensor.data, static_cast<size_t>(nbytes))) {
            return false;
        }
        state.tensors.push_back(std::move(tensor));
    }
    return reader.pos == reader.size;
}

std::vector<ggml_tensor *> ggml_opt_moment_tensors(ggml_opt_context_t opt_ctx) {
    std::vector<ggml_tensor *> result;
    for (ggml_tensor * t : opt_ctx->grad_m) if (t) result.push_back(t);
    for (ggml_tensor * t : opt_ctx->grad_v) if (t) result.push_back(t);
    return result;
}

bool ggml_opt_state_check_tensor_data(const uint8_t * data, size_t size, ggml_type type,
                                      const int64_t ne[GGML_MAX_DIMS], float * values,
                                      size_t value_capacity, void * aligned_data) {
    if (!data || (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)) return false;

    size_t element_count = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (ne[d] <= 0 || static_cast<uint64_t>(ne[d]) > SIZE_MAX / element_count) return false;
        element_count *= static_cast<size_t>(ne[d]);
    }

    const size_t element_size = ggml_type_size(type);
    if (element_size == 0 || element_count > SIZE_MAX / element_size || element_count * element_size != size) {
        return false;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && (!traits || !traits->to_float)) return false;

    if (!values || value_capacity == 0 || (type != GGML_TYPE_F32 && !aligned_data)) return false;
    for (size_t offset = 0; offset < element_count;) {
        const size_t count = std::min(element_count - offset, value_capacity);
        const uint8_t * chunk = data + offset * element_size;
        if (type == GGML_TYPE_F32) {
            std::memcpy(values, chunk, count * sizeof(float));
        } else {
            std::memcpy(aligned_data, chunk, count * element_size);
            traits->to_float(aligned_data, values, static_cast<int64_t>(count));
        }
        if (!std::all_of(values, values + count,
                         [](float value) { return std::isfinite(value); })) {
            return false;
        }
        offset += count;
    }
    return true;
}

bool ggml_opt_apply_pending_state(ggml_opt_context_t opt_ctx) {
    if (opt_ctx->pending_optimizer_state.empty()) return true;
    if (!opt_ctx->optimizer_state_ready) return true;

    ggml_opt_state_data state;
    if (!ggml_opt_state_parse(opt_ctx->pending_optimizer_state.data(), opt_ctx->pending_optimizer_state.size(), state) ||
        state.optimizer != opt_ctx->optimizer || state.optimizer_state_type != opt_ctx->optimizer_state_type ||
        state.tensors.size() != ggml_opt_moment_tensors(opt_ctx).size()) {
        return false;
    }
    std::unordered_map<std::string, ggml_tensor *> current;
    for (ggml_tensor * t : ggml_opt_moment_tensors(opt_ctx)) {
        if (!t || t->name[0] == '\0' || !current.emplace(t->name, t).second) return false;
    }
    for (const ggml_opt_state_tensor & entry : state.tensors) {
        const auto it = current.find(entry.name);
        if (it == current.end()) return false;
        ggml_tensor * tensor = it->second;
        if (tensor->type != entry.type || ggml_nbytes(tensor) != entry.data.size()) return false;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] != entry.ne[d]) return false;
        }
        ggml_backend_tensor_set(tensor, entry.data.data(), 0, entry.data.size());
        current.erase(it);
    }
    if (!current.empty()) return false;
    opt_ctx->pending_optimizer_state.clear();
    return true;
}

struct ggml_opt_state_layout {
    std::string rng;
    std::vector<ggml_tensor *> tensors;
    size_t size = 0;
};

bool opt_state_size_add(size_t & size, size_t value) {
    if (value > SIZE_MAX - size) return false;
    size += value;
    return true;
}

bool ggml_opt_state_make_layout(ggml_opt_context_t opt_ctx, ggml_opt_state_layout & layout) {
    if (!opt_ctx || !opt_ctx->optimizer_state_ready || opt_ctx->eval_ready || opt_ctx->opt_i != 0 ||
        !opt_ctx->pending_optimizer_state.empty()) {
        return false;
    }

    std::ostringstream rng_stream;
    rng_stream.imbue(std::locale::classic());
    rng_stream << opt_ctx->rng;
    if (!rng_stream) return false;
    layout.rng = rng_stream.str();
    layout.tensors = ggml_opt_moment_tensors(opt_ctx);
    if (layout.tensors.size() > UINT32_MAX || layout.rng.size() > UINT64_MAX) return false;

    layout.size = 0;
    if (!opt_state_size_add(layout.size, sizeof(GGML_OPT_STATE_MAGIC)) ||
        !opt_state_size_add(layout.size, sizeof(uint32_t) * 3) ||
        !opt_state_size_add(layout.size, sizeof(uint64_t) * 2) ||
        !opt_state_size_add(layout.size, layout.rng.size()) ||
        !opt_state_size_add(layout.size, sizeof(uint32_t))) {
        return false;
    }

    for (const ggml_tensor * tensor : layout.tensors) {
        if (!tensor || tensor->name[0] == '\0') return false;
        const size_t name_size = std::strlen(tensor->name);
        const int n_dims = ggml_n_dims(tensor);
        if (name_size == 0 || name_size > GGML_MAX_NAME || n_dims < 1 || n_dims > GGML_MAX_DIMS ||
            tensor->type != opt_ctx->optimizer_state_type) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor->ne[d] <= 0) return false;
        }
        if (!opt_state_size_add(layout.size, sizeof(uint32_t)) ||
            !opt_state_size_add(layout.size, name_size) ||
            !opt_state_size_add(layout.size, sizeof(uint32_t) * 2) ||
            !opt_state_size_add(layout.size, sizeof(uint64_t) * GGML_MAX_DIMS) ||
            !opt_state_size_add(layout.size, sizeof(uint64_t)) ||
            !opt_state_size_add(layout.size, ggml_nbytes(tensor))) {
            return false;
        }
    }
    return true;
}

struct opt_state_writer {
    uint8_t * data;
    size_t size;
    size_t pos = 0;

    template<typename T>
    bool write_int(T value) {
        static_assert(std::is_integral<T>::value, "integer type required");
        if (pos > size || size - pos < sizeof(T)) return false;
        for (size_t i = 0; i < sizeof(T); ++i) {
            data[pos++] = static_cast<uint8_t>((static_cast<uint64_t>(value) >> (i * 8)) & 0xff);
        }
        return true;
    }

    bool write_bytes(const void * source, size_t count) {
        if (pos > size || count > size - pos) return false;
        if (count != 0) std::memcpy(data + pos, source, count);
        pos += count;
        return true;
    }
};

}

size_t ggml_opt_state_size(ggml_opt_context_t opt_ctx) {
    ggml_opt_state_layout layout;
    return ggml_opt_state_make_layout(opt_ctx, layout) ? layout.size : 0;
}

bool ggml_opt_state_export(ggml_opt_context_t opt_ctx, void * data, size_t size) {
    if (!data) return false;
    ggml_opt_state_layout layout;
    if (!ggml_opt_state_make_layout(opt_ctx, layout) || layout.size != size) return false;

    opt_state_writer writer{static_cast<uint8_t *>(data), size, 0};
    if (!writer.write_bytes(GGML_OPT_STATE_MAGIC, sizeof(GGML_OPT_STATE_MAGIC)) ||
        !writer.write_int<uint32_t>(GGML_OPT_STATE_VERSION) ||
        !writer.write_int<uint32_t>(static_cast<uint32_t>(opt_ctx->optimizer)) ||
        !writer.write_int<uint32_t>(static_cast<uint32_t>(opt_ctx->optimizer_state_type)) ||
        !writer.write_int<uint64_t>(opt_ctx->iter) ||
        !writer.write_int<uint64_t>(static_cast<uint64_t>(layout.rng.size())) ||
        !writer.write_bytes(layout.rng.data(), layout.rng.size()) ||
        !writer.write_int<uint32_t>(static_cast<uint32_t>(layout.tensors.size()))) {
        return false;
    }

    for (const ggml_tensor * tensor : layout.tensors) {
        const size_t name_size = std::strlen(tensor->name);
        const size_t nbytes = ggml_nbytes(tensor);
        if (!writer.write_int<uint32_t>(static_cast<uint32_t>(name_size)) ||
            !writer.write_bytes(tensor->name, name_size) ||
            !writer.write_int<uint32_t>(static_cast<uint32_t>(tensor->type)) ||
            !writer.write_int<uint32_t>(static_cast<uint32_t>(ggml_n_dims(tensor)))) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (!writer.write_int<uint64_t>(static_cast<uint64_t>(tensor->ne[d]))) return false;
        }
        if (!writer.write_int<uint64_t>(static_cast<uint64_t>(nbytes)) ||
            writer.pos > writer.size || writer.size - writer.pos < nbytes) {
            return false;
        }
        ggml_backend_tensor_get(tensor, writer.data + writer.pos, 0, nbytes);
        writer.pos += nbytes;
    }
    return writer.pos == writer.size;
}

bool ggml_opt_state_data_is_finite(const void * data, size_t size) {
    if (!data || size < sizeof(GGML_OPT_STATE_MAGIC)) return false;

    opt_state_reader reader{static_cast<const uint8_t *>(data), size, 0};
    const uint8_t * magic = nullptr;
    uint32_t version = 0;
    uint32_t optimizer = 0;
    uint32_t optimizer_state_type = GGML_TYPE_F32;
    uint64_t iter = 0;
    uint64_t rng_size = 0;
    uint32_t tensor_count = 0;
    if (!reader.read_view(magic, sizeof(GGML_OPT_STATE_MAGIC)) ||
        std::memcmp(magic, GGML_OPT_STATE_MAGIC, sizeof(GGML_OPT_STATE_MAGIC)) != 0 ||
        !reader.read_int(version) || (version != 1 && version != GGML_OPT_STATE_VERSION) ||
        !reader.read_int(optimizer) || optimizer >= GGML_OPT_OPTIMIZER_TYPE_COUNT ||
        (version >= 2 && (!reader.read_int(optimizer_state_type) || optimizer_state_type >= GGML_TYPE_COUNT)) ||
        !reader.read_int(iter) || iter < 1 ||
        !reader.read_int(rng_size) || rng_size > SIZE_MAX ||
        !reader.skip_bytes(static_cast<size_t>(rng_size)) ||
        !reader.read_int(tensor_count) || tensor_count > size / 16 ||
        (optimizer_state_type != GGML_TYPE_F32 && optimizer_state_type != GGML_TYPE_F16 &&
         optimizer_state_type != GGML_TYPE_BF16)) {
        return false;
    }

    std::vector<float> values(tensor_count == 0 ? 0 : 4096);
    std::vector<uint64_t> aligned_data(
        tensor_count == 0 || optimizer_state_type == GGML_TYPE_F32 ? 0 : 1024);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint32_t name_size = 0;
        uint32_t type_value = 0;
        uint32_t n_dims = 0;
        uint64_t nbytes = 0;
        const uint8_t * unused = nullptr;
        int64_t ne[GGML_MAX_DIMS] = {};
        if (!reader.read_int(name_size) || name_size == 0 || name_size > GGML_MAX_NAME ||
            !reader.read_view(unused, name_size) ||
            !reader.read_int(type_value) || type_value >= GGML_TYPE_COUNT ||
            !reader.read_int(n_dims) || n_dims < 1 || n_dims > GGML_MAX_DIMS) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            uint64_t dimension = 0;
            if (!reader.read_int(dimension) || dimension == 0 || dimension > static_cast<uint64_t>(INT64_MAX) ||
                (d >= static_cast<int>(n_dims) && dimension != 1)) {
                return false;
            }
            ne[d] = static_cast<int64_t>(dimension);
        }
        if (!reader.read_int(nbytes) || nbytes > SIZE_MAX) return false;
        const uint8_t * tensor_data = nullptr;
        if (!reader.read_view(tensor_data, static_cast<size_t>(nbytes)) ||
            type_value != optimizer_state_type ||
            !ggml_opt_state_check_tensor_data(tensor_data, static_cast<size_t>(nbytes),
                                              static_cast<ggml_type>(type_value), ne,
                                              values.data(), values.size(), aligned_data.data())) {
            return false;
        }
    }
    return reader.pos == reader.size;
}

bool ggml_opt_state_import(ggml_opt_context_t opt_ctx, const void * data, size_t size) {
    if (!opt_ctx || !data || size == 0 || opt_ctx->eval_ready || opt_ctx->opt_i != 0) return false;
    ggml_opt_state_data state;
    if (!ggml_opt_state_parse(data, size, state) || state.optimizer != opt_ctx->optimizer ||
        state.optimizer_state_type != opt_ctx->optimizer_state_type) return false;
    std::istringstream rng_stream(state.rng);
    rng_stream.imbue(std::locale::classic());
    rng_stream >> opt_ctx->rng;
    if (!rng_stream) return false;
    opt_ctx->iter = state.iter;
    opt_ctx->pending_optimizer_state.assign(static_cast<const uint8_t *>(data),
                                            static_cast<const uint8_t *>(data) + size);
    return ggml_opt_apply_pending_state(opt_ctx);
}

void ggml_opt_set_opt_period(ggml_opt_context_t opt_ctx, int32_t opt_period) {
    GGML_ASSERT(opt_ctx != nullptr);
    GGML_ASSERT(opt_period >= 1);
    GGML_ASSERT(!opt_ctx->static_graphs && "accumulation period can only change for dynamic graphs");
    GGML_ASSERT(!opt_ctx->eval_ready);
    GGML_ASSERT(opt_ctx->opt_i == 0 && "accumulation period can only change between optimizer steps");
    GGML_ASSERT(opt_ctx->loss_normalizer == 0.0f && "accumulation period cannot change after setting the loss normalizer");
    opt_ctx->opt_period = opt_period;
}

void ggml_opt_set_activation_checkpointing(ggml_opt_context_t opt_ctx, bool enabled) {
    GGML_ASSERT(opt_ctx != nullptr);
    GGML_ASSERT(!opt_ctx->static_graphs && "activation checkpointing requires dynamic graphs");
    GGML_ASSERT(!opt_ctx->eval_ready);
    GGML_ASSERT(opt_ctx->opt_i == 0);
    opt_ctx->activation_checkpointing = enabled;
}

void ggml_opt_set_seed(ggml_opt_context_t opt_ctx, uint32_t seed) {
    GGML_ASSERT(opt_ctx != nullptr);
    GGML_ASSERT(!opt_ctx->eval_ready);
    GGML_ASSERT(opt_ctx->opt_i == 0);
    GGML_ASSERT(opt_ctx->iter == 1);
    opt_ctx->rng.seed(seed);
}

void ggml_opt_set_loss_normalizer(ggml_opt_context_t opt_ctx, float loss_normalizer) {
    GGML_ASSERT(opt_ctx != nullptr);
    GGML_ASSERT(std::isfinite(loss_normalizer) && loss_normalizer > 0.0f);
    GGML_ASSERT(opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    GGML_ASSERT(!opt_ctx->static_graphs && "loss normalization is only supported for dynamic graphs");
    GGML_ASSERT(!opt_ctx->eval_ready);
    GGML_ASSERT(opt_ctx->opt_i == 0 && "loss normalizer can only change between optimizer steps");
    GGML_ASSERT(opt_ctx->loss_normalizer == 0.0f && "loss normalizer is already set for this accumulation group");
    opt_ctx->loss_normalizer = loss_normalizer;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    for (ggml_backend_buffer_t buf : opt_ctx->bufs_grad_accs) {
        ggml_backend_buffer_free(buf);
    }
    for (struct ggml_context * ctx : opt_ctx->ctxs_grad_accs) {
        ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buf : opt_ctx->bufs_momenta) {
        ggml_backend_buffer_free(buf);
    }
    for (struct ggml_context * ctx : opt_ctx->ctxs_momenta) {
        ggml_free(ctx);
    }
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    opt_ctx->loss_normalizer = 0.0f;
    if (optimizer) {
        ggml_graph_reset(opt_ctx->static_graphs ? opt_ctx->gb_opt : opt_ctx->gb_grad);
        for (ggml_backend_buffer_t buf : opt_ctx->bufs_momenta) {
            ggml_backend_buffer_clear(buf, 0);
        }
        opt_ctx->iter = 1;
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    const ggml_cgraph * graph = opt_ctx->static_graphs ? opt_ctx->gb_opt : opt_ctx->gb_grad;
    return ggml_graph_get_grad_acc(graph, node);
}

bool ggml_opt_has_grad(ggml_opt_context_t opt_ctx, const struct ggml_tensor * node) {
    if (!opt_ctx || !opt_ctx->allocated_graph || !node || !opt_ctx->allocated_graph->grads) {
        return false;
    }

    const struct ggml_cgraph * graph = opt_ctx->allocated_graph;
    const struct ggml_tensor * grad = ggml_graph_get_grad(graph, node);
    return grad && grad != ggml_graph_get_grad_acc(graph, node);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->opt_periods.clear();
    result->pred.clear();
    result->ncorrect = 0;
    result->naccuracy_data = 0;
    result->loss_sum = 0.0;
    result->loss_squared_sum = 0.0;
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    GGML_ASSERT(result->opt_periods.size() == result->loss.size());
    const double sum = result->loss_sum;
    const double sum_squared = result->loss_squared_sum;
    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_last_loss(ggml_opt_result_t result, double * loss) {
    if (result->loss.empty()) {
        *loss = 0.0;
        return;
    }

    GGML_ASSERT(result->opt_periods.size() == result->loss.size());
    const size_t last = result->loss.size() - 1;
    *loss = result->loss_per_datapoint ?
        double(result->loss[last]) * result->opt_periods[last] : result->loss[last];
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 && result->naccuracy_data > 0 ?
        double(result->ncorrect) / double(result->naccuracy_data) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->naccuracy_data >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->naccuracy_data - 1)) : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);

        if (opt_ctx->gb_grad && opt_ctx->opt_i == 0) {
            ggml_graph_reset(opt_ctx->gb_grad);
        }
    }

    struct ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->static_graphs ? opt_ctx->gb_opt : opt_ctx->gb_grad;
        } break;
    }
    GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph
    ggml_opt_set_recompute_backend_affinities(opt_ctx, graph);

    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

static void ggml_opt_result_add(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);
    result->opt_periods.push_back(opt_ctx->opt_period);
    const float loss_scaled = result->loss_per_datapoint ? loss * opt_ctx->opt_period : loss;
    result->loss_sum += loss_scaled;
    result->loss_squared_sum += loss_scaled * loss_scaled;

    if (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    GGML_ASSERT(ggml_is_scalar(opt_ctx->nvalid));
    GGML_ASSERT(opt_ctx->nvalid->type == GGML_TYPE_F32);
    int64_t ncorrect;
    float nvalid;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    ggml_backend_tensor_get(opt_ctx->nvalid, &nvalid, 0, ggml_nbytes(opt_ctx->nvalid));
    result->ncorrect += ncorrect;
    result->naccuracy_data += (int64_t) std::llround(nvalid);
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);

    const bool optimizer_step = opt_ctx->static_graphs
        ? opt_ctx->allocated_graph == opt_ctx->gb_opt
        : opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT;
    const bool separate_optimizer_graph = optimizer_step && !opt_ctx->static_graphs && opt_ctx->opt_period > 1;

    if (optimizer_step) {
        const ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        switch (opt_ctx->optimizer) {
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                float * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
            } break;
            case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                float * sgd = ggml_get_data_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } break;
            default:
                GGML_ABORT("fatal error");
        }
    }

    // The selected graph may include the optimizer update (opt_period == 1),
    // or only forward/backward work when using a separate accumulated update.
    // Copy metrics before the scheduler allocation is reused for that update.
    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    ggml_opt_result_add(opt_ctx, result);

    if (separate_optimizer_graph) {
        ggml_backend_sched_reset(opt_ctx->backend_sched);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->gb_opt) &&
            "failed to allocate optimizer update graph");
        ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->gb_opt);
    }

    opt_ctx->iter += optimizer_step;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    if (optimizer_step) {
        opt_ctx->loss_normalizer = 0.0f;
    }

    if (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
        opt_ctx->recompute_backend_affinities.clear();
    }

    opt_ctx->eval_ready = false;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    struct ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_loss_type         loss_type,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    return c->optimizer;
}

GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o) {
    switch (o) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            return "adamw";
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            return "sgd";
        default:
            return "undefined";
    };
}
