#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

static bool ggml_cpu_use_fp32_dot_src1_for_quantized_matmul(const struct ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1] ||
        !ggml_is_quantized(op->src[0]->type) || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }

    const enum ggml_prec prec = ggml_prec_get_src(op, 1);
    return prec == GGML_PREC_F32 || prec == GGML_PREC_F16 || prec == GGML_PREC_BF16;
}

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    if (ggml_cpu_use_fp32_dot_src1_for_quantized_matmul(op)) {
        return false;
    }

    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    if (ggml_cpu_use_fp32_dot_src1_for_quantized_matmul(op)) {
        return false;
    }

    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
