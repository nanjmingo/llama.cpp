#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif


GGML_API enum ggml_prec ggml_prec_compute_type_for_grad(const struct ggml_tensor * tensor);
GGML_API bool ggml_cpu_mul_mat_use_fp32_dot_src1(const struct ggml_tensor * tensor);
GGML_API bool ggml_cpu_use_fp32_dot_src1_for_quantized_matmul(const struct ggml_tensor * tensor);
GGML_API bool ggml_opt_state_data_is_finite(const void * data, size_t size);

GGML_API struct ggml_tensor * ggml_mul_mat_id_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * as,
        struct ggml_tensor  * grad,
        struct ggml_tensor  * ids,
        struct ggml_tensor  * b_like);

GGML_API struct ggml_tensor * ggml_out_prod_id(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * ids,
        int64_t               n_expert);

GGML_API struct ggml_tensor * ggml_gated_delta_net_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        struct ggml_tensor  * d);

GGML_API struct ggml_tensor * ggml_cross_entropy_loss_sparse(
        struct ggml_context * ctx,
        struct ggml_tensor  * logits,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights);
GGML_API struct ggml_tensor * ggml_cross_entropy_loss_sparse_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * grad,
        struct ggml_tensor  * logits,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights);
GGML_API struct ggml_tensor * ggml_fused_linear_cross_entropy(
        struct ggml_context * ctx,
        struct ggml_tensor  * weight,
        struct ggml_tensor  * input,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights,
        float                 logit_softcap,
        int32_t               vocab_chunk_size);
GGML_API struct ggml_tensor * ggml_fused_linear_cross_entropy_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * grad,
        struct ggml_tensor  * weight,
        struct ggml_tensor  * input,
        struct ggml_tensor  * labels,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * forward_stats);

GGML_API void ggml_flash_attn_ext_set_back_prec(
        struct ggml_tensor * a,
        enum ggml_prec       prec);

GGML_API enum ggml_prec ggml_flash_attn_ext_get_back_prec(
        const struct ggml_tensor * a);

#ifdef __cplusplus
}
#endif
