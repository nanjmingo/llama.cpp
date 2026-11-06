#include "ggml-impl.h"
#include "opt-step-adamw.cuh"

#include <cstdint>

static __global__ void opt_step_adamw_f32(
        float * __restrict__ x, const float * __restrict__ g, float * __restrict__ g_m, float * __restrict__ g_v,
        const float * __restrict__ pars, const int64_t k) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const float alpha = pars[0];
    const float beta1 = pars[1];
    const float beta2 = pars[2];
    const float eps = pars[3];
    const float wd = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const float gi = g[i];
    const float mi = g_m[i] * beta1 + gi * (1.0f - beta1);
    const float vi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);
    g_m[i] = mi;
    g_v[i] = vi;
    x[i] = x[i] * (1.0f - alpha * wd) - alpha * (mi * beta1h) / (sqrtf(vi * beta2h) + eps);
}

static void opt_step_adamw_f32_cuda(
        float * x, const float * g, float * g_m, float * g_v, const float * pars, const int64_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 block_nums((n + block_size - 1) / block_size, 1, 1);
    opt_step_adamw_f32<<<block_nums, block_dims, 0, stream>>>(x, g, g_m, g_v, pars, n);
}

template<ggml_type type>
static __device__ __forceinline__ float opt_adamw_load(const void * data, const int64_t i) {
    if constexpr (type == GGML_TYPE_F32) {
        return static_cast<const float *>(data)[i];
    } else if constexpr (type == GGML_TYPE_F16) {
        return __half2float(static_cast<const half *>(data)[i]);
    } else if constexpr (type == GGML_TYPE_BF16) {
        const uint32_t bits = static_cast<const ggml_bf16_t *>(data)[i].bits;
        return __uint_as_float(bits << 16);
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "unsupported AdamW tensor type");
        const block_q8_0 * blocks = static_cast<const block_q8_0 *>(data);
        const block_q8_0 & block = blocks[i / QK8_0];
        return __half2float(block.d) * block.qs[i % QK8_0];
    }
}

template<ggml_type type>
static __device__ __forceinline__ void opt_adamw_store(void * data, const int64_t i, const float value) {
    if constexpr (type == GGML_TYPE_F32) {
        static_cast<float *>(data)[i] = value;
    } else if constexpr (type == GGML_TYPE_F16) {
        static_cast<half *>(data)[i] = __float2half_rn(value);
    } else if constexpr (type == GGML_TYPE_BF16) {
        uint32_t bits = __float_as_uint(value);
        bits += 0x7fff + ((bits >> 16) & 1);
        static_cast<ggml_bf16_t *>(data)[i].bits = static_cast<uint16_t>(bits >> 16);
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "unsupported AdamW tensor type");
        block_q8_0 * blocks = static_cast<block_q8_0 *>(data);
        const float amax = warp_reduce_max(fabsf(value));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int q = static_cast<int>(roundf(value * id));
        q = q < -128 ? -128 : (q > 127 ? 127 : q);
        block_q8_0 & block = blocks[i / QK8_0];
        block.qs[i % QK8_0] = static_cast<int8_t>(q);
        if ((threadIdx.x % QK8_0) == 0) {
            block.d = __float2half_rn(d);
        }
    }
}

template<ggml_type weight_type, ggml_type state_type>
static __global__ void opt_step_adamw_mixed(
        void * __restrict__ weights,
        const float * __restrict__ grad,
        void * __restrict__ moment_m,
        void * __restrict__ moment_v,
        const float * __restrict__ pars,
        const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = i < n;

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    float w = 0.0f;
    float g = 0.0f;
    float m = 0.0f;
    float v = 0.0f;
    if (valid) {
        w = opt_adamw_load<weight_type>(weights, i);
        g = grad[i];
        m = opt_adamw_load<state_type>(moment_m, i);
        v = opt_adamw_load<state_type>(moment_v, i);
        m = m * beta1 + g * (1.0f - beta1);
        v = v * beta2 + g * g * (1.0f - beta2);
        const float mh = m * beta1h;
        const float vh = sqrtf(v * beta2h) + eps;
        w = w * (1.0f - alpha * wd) - alpha * mh / vh;
    }

    if constexpr (weight_type == GGML_TYPE_Q8_0) {
        opt_adamw_store<weight_type>(weights, i, w);
    } else if (valid) {
        opt_adamw_store<weight_type>(weights, i, w);
    }
    if constexpr (state_type == GGML_TYPE_Q8_0) {
        opt_adamw_store<state_type>(moment_m, i, m);
        opt_adamw_store<state_type>(moment_v, i, v);
    } else if (valid) {
        opt_adamw_store<state_type>(moment_m, i, m);
        opt_adamw_store<state_type>(moment_v, i, v);
    }
}

template<ggml_type weight_type, ggml_type state_type>
static void opt_step_adamw_mixed_cuda(
        void * weights, const float * grad, void * moment_m, void * moment_v,
        const float * pars, const int64_t n, cudaStream_t stream) {
    const dim3 block_dims(CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((n + CUDA_OPT_STEP_ADAMW_BLOCK_SIZE - 1) / CUDA_OPT_STEP_ADAMW_BLOCK_SIZE, 1, 1);
    opt_step_adamw_mixed<weight_type, state_type><<<block_nums, block_dims, 0, stream>>>(
        weights, grad, moment_m, moment_v, pars, n);
}

template<ggml_type weight_type>
static void opt_step_adamw_dispatch_state(
        void * weights, const float * grad, void * moment_m, void * moment_v,
        const float * pars, const int64_t n, const ggml_type state_type, cudaStream_t stream) {
    switch (state_type) {
        case GGML_TYPE_F32: opt_step_adamw_mixed_cuda<weight_type, GGML_TYPE_F32>(weights, grad, moment_m, moment_v, pars, n, stream); break;
        case GGML_TYPE_F16: opt_step_adamw_mixed_cuda<weight_type, GGML_TYPE_F16>(weights, grad, moment_m, moment_v, pars, n, stream); break;
        case GGML_TYPE_BF16: opt_step_adamw_mixed_cuda<weight_type, GGML_TYPE_BF16>(weights, grad, moment_m, moment_v, pars, n, stream); break;
        case GGML_TYPE_Q8_0: opt_step_adamw_mixed_cuda<weight_type, GGML_TYPE_Q8_0>(weights, grad, moment_m, moment_v, pars, n, stream); break;
        default: GGML_ABORT("unsupported AdamW state type");
    }
}

void ggml_cuda_opt_step_adamw(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_m->type == src0_grad_v->type);
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
    GGML_ASSERT(ggml_is_contiguous(adamw_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    const int64_t ne = ggml_nelements(src0);
    const ggml_type state_type = src0_grad_m->type;
    if (state_type == GGML_TYPE_Q8_0) {
        GGML_ASSERT(ggml_nelements(src0_grad_m) >= ne && ggml_nelements(src0_grad_v) >= ne);
        GGML_ASSERT(ggml_nelements(src0_grad_m) % QK8_0 == 0 && ggml_nelements(src0_grad_v) % QK8_0 == 0);
    } else {
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
        GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    }

    void * weights = src0->data;
    const float * grad = static_cast<const float *>(src0_grad->data);
    void * moment_m = src0_grad_m->data;
    void * moment_v = src0_grad_v->data;
    const float * pars = static_cast<const float *>(adamw_params->data);
    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_F32 && state_type == GGML_TYPE_F32) {
        opt_step_adamw_f32_cuda(static_cast<float *>(weights), grad, static_cast<float *>(moment_m),
                                static_cast<float *>(moment_v), pars, ne, stream);
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_F32:  opt_step_adamw_dispatch_state<GGML_TYPE_F32>(weights, grad, moment_m, moment_v, pars, ne, state_type, stream); break;
        case GGML_TYPE_F16:  opt_step_adamw_dispatch_state<GGML_TYPE_F16>(weights, grad, moment_m, moment_v, pars, ne, state_type, stream); break;
        case GGML_TYPE_BF16: opt_step_adamw_dispatch_state<GGML_TYPE_BF16>(weights, grad, moment_m, moment_v, pars, ne, state_type, stream); break;
        case GGML_TYPE_Q8_0: opt_step_adamw_dispatch_state<GGML_TYPE_Q8_0>(weights, grad, moment_m, moment_v, pars, ne, state_type, stream); break;
        default: GGML_ABORT("unsupported AdamW parameter type");
    }
}
