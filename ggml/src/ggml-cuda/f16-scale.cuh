#pragma once

#include "common.cuh"

static __global__ void ggml_cuda_f16_scale_absmax_kernel(
        const char * src,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const size_t nb0, const size_t nb1, const size_t nb2, const size_t nb3,
        const int64_t n_elements, unsigned int * absmax_bits) {
    unsigned int value_max = 0;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        int64_t i3 = i / (ne0 * ne1 * ne2);
        int64_t i2 = (i / (ne0 * ne1)) % ne2;
        int64_t i1 = (i / ne0) % ne1;
        int64_t i0 = i % ne0;
        const float value = *(const float *) (src + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
        const float abs_value = fabsf(value);
        if (isfinite(abs_value)) {
            const unsigned int abs_value_bits = __float_as_uint(abs_value);
            value_max = value_max > abs_value_bits ? value_max : abs_value_bits;
        }
    }
    if (value_max != 0) {
        atomicMax(absmax_bits, value_max);
    }
}

static __global__ void ggml_cuda_f16_scale_compute_kernel(
        const unsigned int * absmax_bits, float * scales) {
    const float max_abs = __uint_as_float(*absmax_bits);
    int exponent = 0;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        exponent = 14 - ilogbf(max_abs);
        exponent = exponent < -126 ? -126 : exponent;
        exponent = exponent > 126 ? 126 : exponent;
    }
    scales[0] = ldexpf(1.0f, exponent);
    scales[1] = ldexpf(1.0f, -exponent);
}

static __global__ void ggml_cuda_f16_scale_tensor_kernel(
        char * data,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const size_t nb0, const size_t nb1, const size_t nb2, const size_t nb3,
        const int64_t n_elements, const float * scale) {
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements; i += stride) {
        const int64_t i3 = i / (ne0 * ne1 * ne2);
        const int64_t i2 = (i / (ne0 * ne1)) % ne2;
        const int64_t i1 = (i / ne0) % ne1;
        const int64_t i0 = i % ne0;
        float * value = (float *) (data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
        *value *= *scale;
    }
}

static void ggml_cuda_f16_scale_init(
        const ggml_tensor * src, unsigned int * absmax_bits, float * scales, cudaStream_t stream) {
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    const int64_t n_elements = ggml_nelements(src);
    GGML_ASSERT(n_elements > 0);
    constexpr int threads = 256;
    const int blocks = (int) std::min<int64_t>((n_elements + threads - 1) / threads, 4096);
    CUDA_CHECK(cudaMemsetAsync(absmax_bits, 0, sizeof(unsigned int), stream));
    ggml_cuda_f16_scale_absmax_kernel<<<blocks, threads, 0, stream>>>(
        (const char *) src->data,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
        n_elements, absmax_bits);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_f16_scale_compute_kernel<<<1, 1, 0, stream>>>(absmax_bits, scales);
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_f16_scale_tensor(
        ggml_tensor * tensor, const float * scale, cudaStream_t stream) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    const int64_t n_elements = ggml_nelements(tensor);
    if (n_elements == 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = (int) std::min<int64_t>((n_elements + threads - 1) / threads, 4096);
    ggml_cuda_f16_scale_tensor_kernel<<<blocks, threads, 0, stream>>>(
        (char *) tensor->data,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
        n_elements, scale);
    CUDA_CHECK(cudaGetLastError());
}
