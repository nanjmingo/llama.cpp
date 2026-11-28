#include "llamango-ggml.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

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
};

bool opt_state_tensor_data_is_finite(
        const uint8_t * data,
        size_t size,
        ggml_type type,
        const int64_t ne[GGML_MAX_DIMS],
        float * values,
        size_t value_capacity,
        void * aligned_data) {
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
        if (!std::all_of(values, values + count, [](float value) { return std::isfinite(value); })) {
            return false;
        }
        offset += count;
    }
    return true;
}

}

extern "C" {

enum ggml_prec ggml_prec_compute_type_for_grad(const struct ggml_tensor * tensor) {
    const enum ggml_prec compute_type = ggml_prec_get_src(tensor, 1);
    return compute_type == GGML_PREC_Q8 || compute_type == GGML_PREC_Q4
        ? GGML_PREC_F16
        : compute_type;
}

bool ggml_cpu_mul_mat_use_fp32_dot_src1(const struct ggml_tensor * tensor) {
    if (!tensor->src[0] || !tensor->src[1] ||
        !ggml_is_quantized(tensor->src[0]->type) ||
        tensor->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    const enum ggml_prec prec = ggml_prec_get_src(tensor, 1);
    return prec == GGML_PREC_F32 || prec == GGML_PREC_F16 || prec == GGML_PREC_BF16;
}

bool ggml_cpu_use_fp32_dot_src1_for_quantized_matmul(const struct ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_MUL_MAT || !tensor->src[0] || !tensor->src[1] ||
        !ggml_is_quantized(tensor->src[0]->type) || tensor->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    const enum ggml_prec prec = ggml_prec_get_src(tensor, 1);
    return prec == GGML_PREC_F32 || prec == GGML_PREC_F16 || prec == GGML_PREC_BF16;
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
            !opt_state_tensor_data_is_finite(tensor_data, static_cast<size_t>(nbytes),
                                             static_cast<ggml_type>(type_value), ne,
                                             values.data(), values.size(), aligned_data.data())) {
            return false;
        }
    }
    return reader.pos == reader.size;
}

struct ggml_tensor * ggml_mul_mat_id_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * as,
        struct ggml_tensor  * grad,
        struct ggml_tensor  * ids,
        struct ggml_tensor  * b_like) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(as->ne[3] == 1 && b_like->ne[3] == 1);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(as->ne[1] == grad->ne[0]);
    GGML_ASSERT(grad->ne[1] == ids->ne[0]);
    GGML_ASSERT(grad->ne[2] == ids->ne[1]);
    GGML_ASSERT(as->ne[0] == b_like->ne[0]);
    GGML_ASSERT(b_like->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->ne[0] % b_like->ne[1] == 0);

    const int64_t ne[4] = { b_like->ne[0], b_like->ne[1], b_like->ne[2], 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    result->op = GGML_OP_EXT_MUL_MAT_ID_BACK;
    result->src[0] = as;
    result->src[1] = grad;
    result->src[2] = ids;
    result->src[3] = b_like;
    return result;
}

struct ggml_tensor * ggml_out_prod_id(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * ids,
        int64_t               n_expert) {
    GGML_ASSERT(a->type   == GGML_TYPE_F32);
    GGML_ASSERT(b->type   == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(a->ne[1]  == b->ne[1]);
    GGML_ASSERT(a->ne[2]  == b->ne[2]);
    GGML_ASSERT(ids->ne[0] == a->ne[1]);
    GGML_ASSERT(ids->ne[1] == a->ne[2]);
    GGML_ASSERT(n_expert > 0);

    const int64_t ne[4] = { a->ne[0], b->ne[0], n_expert, 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_EXT_OUT_PROD_ID;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}



struct ggml_tensor * ggml_cross_entropy_loss_sparse(
        struct ggml_context * ctx,
        struct ggml_tensor  * logits,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_nelements(labels) == ggml_nrows(logits));
    GGML_ASSERT(ggml_nelements(weights) == ggml_nrows(logits));
    GGML_ASSERT(ggml_nrows(logits) > 0);

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    result->op     = GGML_OP_EXT_CROSS_ENTROPY_LOSS_SPARSE;
    result->src[0] = logits;
    result->src[1] = labels;
    result->src[2] = weights;
    return result;
}

struct ggml_tensor * ggml_cross_entropy_loss_sparse_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * grad,
        struct ggml_tensor  * logits,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights) {
    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_nelements(labels) == ggml_nrows(logits));
    GGML_ASSERT(ggml_nelements(weights) == ggml_nrows(logits));
    GGML_ASSERT(ggml_nrows(logits) > 0);

    struct ggml_tensor * result = ggml_dup_tensor(ctx, logits);
    result->op     = GGML_OP_EXT_CROSS_ENTROPY_LOSS_SPARSE_BACK;
    result->src[0] = grad;
    result->src[1] = logits;
    result->src[2] = labels;
    result->src[3] = weights;
    return result;
}


struct ggml_tensor * ggml_fused_linear_cross_entropy_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * grad,
        struct ggml_tensor  * weight,
        struct ggml_tensor  * input,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * forward_stats) {
    GGML_ASSERT(grad->type == GGML_TYPE_F32 && ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_nelements(grad) == 2 * input->ne[1]);
    GGML_ASSERT(weight->ne[0] == input->ne[0]);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(labels->type == GGML_TYPE_I32 && ggml_is_contiguous(labels));
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_nelements(labels) == input->ne[1]);
    GGML_ASSERT(ggml_nelements(weights) == input->ne[1]);
    GGML_ASSERT(forward_stats->type == GGML_TYPE_F32 && ggml_is_contiguous(forward_stats));
    GGML_ASSERT(ggml_nelements(forward_stats) == 2 * input->ne[1]);

    struct ggml_tensor * result = ggml_dup_tensor(ctx, input);
    result->op     = GGML_OP_EXT_FUSED_LINEAR_CROSS_ENTROPY_BACK;
    result->src[0] = grad;
    result->src[1] = weight;
    result->src[2] = input;
    result->src[3] = labels;
    result->src[4] = weights;
    result->src[5] = forward_stats;
    std::memcpy(result->op_params, forward_stats->op_params, sizeof(result->op_params));
    return result;
}


} // extern "C"
