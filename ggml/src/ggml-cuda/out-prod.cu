#include "out-prod.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

static __global__ void out_prod_absmax_src1(
        const char * src1, const int64_t n_rows, const int64_t n_cols,
        const size_t nb10, const size_t nb11, unsigned int * absmax_bits) {
    float value_max = 0.0f;
    for (int64_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const float * row_data = (const float *) (src1 + row * nb11);
        for (int64_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
            value_max = fmaxf(value_max, fabsf(*(const float *) ((const char *) row_data + col * nb10)));
        }
    }

    value_max = warp_reduce_max(value_max);
    __shared__ float warp_max[WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) {
        warp_max[threadIdx.x / WARP_SIZE] = value_max;
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        value_max = threadIdx.x < blockDim.x / WARP_SIZE ? warp_max[threadIdx.x] : 0.0f;
        value_max = warp_reduce_max(value_max);
        if (threadIdx.x == 0) {
            atomicMax(absmax_bits, __float_as_uint(value_max));
        }
    }
}

static __global__ void out_prod_compute_scales(
        const unsigned int * absmax_bits, float * scales) {
    const float max_abs = __uint_as_float(*absmax_bits);
    int exponent = 0;
    if (max_abs > 0.0f && isfinite(max_abs)) {
        exponent = max(-126, min(126, 14 - ilogbf(max_abs)));
    }

    scales[0] = ldexpf(1.0f, exponent);
    scales[1] = ldexpf(1.0f, -exponent);
    scales[2] = 0.0f;
    scales[3] = 1.0f;
}

template<typename compute_t>
static __global__ void out_prod_convert_src1_tile(
        const char * src1, compute_t * dst1, const int64_t ne10,
        const int64_t n_rows, const size_t nb10, const size_t nb11,
        const float * scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = ne10 * n_rows;
    if (i >= n) {
        return;
    }

    const int64_t row = i / ne10;
    const int64_t col = i % ne10;
    const float value = *(const float *) (src1 + col * nb10 + row * nb11);
    if constexpr (std::is_same_v<compute_t, half>) {
        dst1[i] = __float2half_rn(value * *scale);
    } else if constexpr (std::is_same_v<compute_t, nv_bfloat16>) {
        dst1[i] = __float2bfloat16(value);
    } else {
        dst1[i] = value;
    }
}

template<typename compute_t>
static void ggml_cuda_out_prod_lowp(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const cudaDataType_t data_type) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    constexpr ggml_type target_type = std::is_same_v<compute_t, half> ? GGML_TYPE_F16 :
                                     std::is_same_v<compute_t, nv_bfloat16> ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    const bool convert_src0 = src0->type != target_type;
    to_t_cuda_t<compute_t> to_compute = nullptr;
    to_t_nc_cuda_t<compute_t> to_compute_nc = nullptr;
    const bool contiguous_quantized_src0 = convert_src0 && ggml_is_quantized(src0->type) && ggml_is_contiguous(src0);
    if (convert_src0 && contiguous_quantized_src0) {
        if constexpr (std::is_same_v<compute_t, half>) {
            to_compute = ggml_get_to_fp16_cuda(src0->type);
        } else if constexpr (std::is_same_v<compute_t, nv_bfloat16>) {
            to_compute = ggml_get_to_bf16_cuda(src0->type);
        } else {
            to_compute = ggml_get_to_fp32_cuda(src0->type);
        }
    } else if (convert_src0) {
        if constexpr (std::is_same_v<compute_t, half>) {
            to_compute_nc = ggml_get_to_fp16_nc_cuda(src0->type);
        } else if constexpr (std::is_same_v<compute_t, nv_bfloat16>) {
            to_compute_nc = ggml_get_to_bf16_nc_cuda(src0->type);
        } else {
            to_compute_nc = ggml_get_to_fp32_nc_cuda(src0->type);
        }
    }

    GGML_ASSERT(!convert_src0 || (contiguous_quantized_src0 ? to_compute != nullptr : to_compute_nc != nullptr));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00 && ne1 == ne10);
    GGML_ASSERT(ne2 % src0->ne[2] == 0 && ne3 % src0->ne[3] == 0);
    GGML_ASSERT(ne2 == src1->ne[2] && ne3 == src1->ne[3]);

    constexpr size_t workspace_max = 64ull * 1024 * 1024;
    const int64_t elements_per_row = (convert_src0 ? ne00 : 0) + ne10;
    const int64_t rows_per_tile = std::max<int64_t>(1,
        std::min<int64_t>(ne01, workspace_max / (sizeof(compute_t) * elements_per_row)));
    const int64_t src0_tile_elements = convert_src0 ? ne00 * rows_per_tile : 0;
    ggml_cuda_pool_alloc<compute_t> workspace(ctx.pool(), src0_tile_elements + ne10 * rows_per_tile);
    compute_t * src0_temp = convert_src0 ? workspace.ptr : nullptr;
    compute_t * src1_temp = workspace.ptr + src0_tile_elements;
    ggml_cuda_pool_alloc<unsigned int> absmax(ctx.pool());
    ggml_cuda_pool_alloc<float> scales(ctx.pool());
    if constexpr (std::is_same_v<compute_t, half>) {
        absmax.alloc(1);
        scales.alloc(4);
    }

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;
    const int threads = 256;
    const float alpha = 1.0f;
    const size_t compute_size = sizeof(compute_t);
    const int64_t ldc = nb1 / sizeof(float);
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    if constexpr (std::is_same_v<compute_t, half>) {
        CUBLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    }

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            const char * src0_batch = (const char *) src0->data +
                (i2 / dps2) * nb02 + (i3 / dps3) * nb03;
            const char * src1_batch = (const char *) src1->data + i2 * nb12 + i3 * nb13;
            float * dst_batch = (float *) ((char *) dst->data + i2 * nb2 + i3 * nb3);
            const int64_t lda = convert_src0 ? ne00 : nb01 / compute_size;

            if constexpr (std::is_same_v<compute_t, half>) {
                CUDA_CHECK(cudaMemsetAsync(absmax.ptr, 0, sizeof(unsigned int), stream));
                const int nblocks = (int) std::min<int64_t>(ne11, 4096);
                out_prod_absmax_src1<<<nblocks, threads, 0, stream>>>(
                    src1_batch, ne11, ne10, nb10, nb11, absmax.ptr);
                CUDA_CHECK(cudaGetLastError());
                out_prod_compute_scales<<<1, 1, 0, stream>>>(absmax.ptr, scales.ptr);
                CUDA_CHECK(cudaGetLastError());
            }

            for (int64_t row_begin = 0; row_begin < ne01; row_begin += rows_per_tile) {
                const int64_t n_rows = std::min<int64_t>(rows_per_tile, ne01 - row_begin);
                const char * src0_tile = src0_batch + row_begin * nb01;
                const char * src1_tile = src1_batch + row_begin * nb11;
                const compute_t * src0_data;
                if (convert_src0) {
                    if (contiguous_quantized_src0) {
                        to_compute(src0_tile, src0_temp, ne00 * n_rows, stream);
                    } else {
                        to_compute_nc(src0_tile, src0_temp, ne00, n_rows, 1, 1, nb01 / ggml_type_size(src0->type), 0, 0, stream);
                    }
                    src0_data = src0_temp;
                } else {
                    src0_data = (const compute_t *) src0_tile;
                }

                const int64_t src1_elements = ne10 * n_rows;
                out_prod_convert_src1_tile<<<
                    (src1_elements + threads - 1) / threads, threads, 0, stream>>>(
                        src1_tile, src1_temp, ne10, n_rows, nb10, nb11,
                        std::is_same_v<compute_t, half> ? scales.ptr : nullptr);
                CUDA_CHECK(cudaGetLastError());

                const float beta = row_begin == 0 ? 0.0f : 1.0f;
                if constexpr (std::is_same_v<compute_t, float>) {
                    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        static_cast<int>(ne00), static_cast<int>(ne10), static_cast<int>(n_rows),
                        &alpha, src0_data, static_cast<int>(lda),
                                src1_temp, static_cast<int>(ne10),
                        &beta, dst_batch, static_cast<int>(ldc)));
                } else if constexpr (std::is_same_v<compute_t, half>) {
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        static_cast<int>(ne00), static_cast<int>(ne10), static_cast<int>(n_rows),
                        scales.ptr + 1, src0_data, data_type, static_cast<int>(lda),
                                        src1_temp, data_type, static_cast<int>(ne10),
                        scales.ptr + (row_begin == 0 ? 2 : 3), dst_batch, CUDA_R_32F, static_cast<int>(ldc),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                } else {
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        static_cast<int>(ne00), static_cast<int>(ne10), static_cast<int>(n_rows),
                        &alpha, src0_data, data_type, static_cast<int>(lda),
                                src1_temp, data_type, static_cast<int>(ne10),
                        &beta, dst_batch, CUDA_R_32F, static_cast<int>(ldc),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }
        }
    }

    if constexpr (std::is_same_v<compute_t, half>) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, pointer_mode));
    }
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_prec precision = ggml_prec_get_bwd(dst);

    if (precision == GGML_PREC_F32) {
        ggml_cuda_out_prod_lowp<float>(ctx, dst, CUDA_R_32F);
        return;
    }
    if (precision == GGML_PREC_F16) {
        if (fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
            ggml_cuda_out_prod_lowp<half>(ctx, dst, CUDA_R_16F);
        } else {
            ggml_cuda_out_prod_lowp<float>(ctx, dst, CUDA_R_32F);
        }
        return;
    }
    if (precision == GGML_PREC_BF16) {
        if (fast_bf16_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
            ggml_cuda_out_prod_lowp<nv_bfloat16>(ctx, dst, CUDA_R_16BF);
        } else {
            ggml_cuda_out_prod_lowp<float>(ctx, dst, CUDA_R_32F);
        }
        return;
    }

    if (src0->type == GGML_TYPE_BF16) {
        ggml_cuda_out_prod_lowp<nv_bfloat16>(ctx, dst, CUDA_R_16BF);
        return;
    }
    if (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) {
        ggml_cuda_out_prod_lowp<half>(ctx, dst, CUDA_R_16F);
        return;
    }

    ggml_cuda_out_prod_lowp<float>(ctx, dst, CUDA_R_32F);
}

template<typename lowp_t>
static __global__ void out_prod_id_convert_f32_kernel(
        const float * src, lowp_t * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if constexpr (std::is_same_v<lowp_t, half>) {
        dst[i] = __float2half_rn(src[i]);
    } else {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void ggml_cuda_out_prod_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t cols       = src0->ne[0];
    const int64_t n_exp_used = src0->ne[1];
    const int64_t n_tokens   = src0->ne[2];
    const int64_t rows       = src1->ne[0];
    const int64_t n_expert   = dst->ne[2];

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    const size_t ids_nbytes = ggml_nbytes(ids);
    std::vector<char> ids_host(ids_nbytes);
    if (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer)) {
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids_nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        memcpy(ids_host.data(), ids->data, ids_nbytes);
    }

    std::vector<std::vector<int64_t>> expert_tokens(n_expert);
    for (int64_t itok = 0; itok < n_tokens; ++itok) {
        for (int64_t iexp = 0; iexp < n_exp_used; ++iexp) {
            const int32_t eid = *(const int32_t *)(ids_host.data()
                + itok * ids->nb[1] + iexp * ids->nb[0]);
            if (eid == -1) {
                continue;
            }
            GGML_ASSERT(eid >= 0 && eid < (int32_t)n_expert);
            expert_tokens[eid].push_back(iexp * n_tokens + itok);
        }
    }

    const int64_t a_stride_exp = src0->nb[1] / sizeof(float);
    const int64_t a_stride_tok = src0->nb[2] / sizeof(float);
    const int64_t b_stride_exp = src1->nb[1] / sizeof(float);
    const int64_t b_stride_tok = src1->nb[2] / sizeof(float);
    const int64_t dst_stride_e = dst->nb[2]  / sizeof(float);

    const float alpha_one = 1.0f;
    const float beta_acc  = 1.0f;
    const ggml_prec precision = ggml_prec_get_bwd(dst);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_fp16 = precision == GGML_PREC_F16 && fast_fp16_hardware_available(cc);
    const bool use_bf16 = precision == GGML_PREC_BF16 && fast_bf16_hardware_available(cc);

    const float * a_base = (const float *) src0->data;
    const float * b_base = (const float *) src1->data;
    float       * d_base = (float       *)  dst->data;

    for (int64_t e = 0; e < n_expert; ++e) {
        const auto & toks = expert_tokens[e];
        if (toks.empty()) {
            continue;
        }

        const int64_t ntoks_e = (int64_t) toks.size();

        ggml_cuda_pool_alloc<float> a_gathered(ctx.pool(), cols * ntoks_e);
        ggml_cuda_pool_alloc<float> b_gathered(ctx.pool(), rows * ntoks_e);

        for (int64_t ti = 0; ti < ntoks_e; ++ti) {
            const int64_t flat = toks[ti];
            const int64_t iexp = flat / n_tokens;
            const int64_t itok = flat % n_tokens;
            CUDA_CHECK(cudaMemcpyAsync(
                a_gathered.ptr + ti * cols,
                a_base + iexp * a_stride_exp + itok * a_stride_tok,
                cols * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                b_gathered.ptr + ti * rows,
                b_base + iexp * b_stride_exp + itok * b_stride_tok,
                rows * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        if (use_fp16) {
            ggml_cuda_pool_alloc<half> a_gathered_f16(ctx.pool(), static_cast<size_t>(cols * ntoks_e));
            ggml_cuda_pool_alloc<half> b_gathered_f16(ctx.pool(), static_cast<size_t>(rows * ntoks_e));
            const int64_t a_count = cols * ntoks_e;
            const int64_t b_count = rows * ntoks_e;
            out_prod_id_convert_f32_kernel<<<(a_count + 255) / 256, 256, 0, stream>>>(
                a_gathered.ptr, a_gathered_f16.ptr, a_count);
            out_prod_id_convert_f32_kernel<<<(b_count + 255) / 256, 256, 0, stream>>>(
                b_gathered.ptr, b_gathered_f16.ptr, b_count);
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)cols, (int)rows, (int)ntoks_e,
                &alpha_one, a_gathered_f16.ptr, CUDA_R_16F, (int)cols,
                            b_gathered_f16.ptr, CUDA_R_16F, (int)rows,
                &beta_acc,  d_base + e*dst_stride_e, CUDA_R_32F, (int)cols,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if (use_bf16) {
            ggml_cuda_pool_alloc<nv_bfloat16> a_gathered_bf16(ctx.pool(), static_cast<size_t>(cols * ntoks_e));
            ggml_cuda_pool_alloc<nv_bfloat16> b_gathered_bf16(ctx.pool(), static_cast<size_t>(rows * ntoks_e));
            const int64_t a_count = cols * ntoks_e;
            const int64_t b_count = rows * ntoks_e;
            out_prod_id_convert_f32_kernel<<<(a_count + 255) / 256, 256, 0, stream>>>(
                a_gathered.ptr, a_gathered_bf16.ptr, a_count);
            out_prod_id_convert_f32_kernel<<<(b_count + 255) / 256, 256, 0, stream>>>(
                b_gathered.ptr, b_gathered_bf16.ptr, b_count);
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)cols, (int)rows, (int)ntoks_e,
                &alpha_one, a_gathered_bf16.ptr, CUDA_R_16BF, (int)cols,
                            b_gathered_bf16.ptr, CUDA_R_16BF, (int)rows,
                &beta_acc,  d_base + e*dst_stride_e, CUDA_R_32F, (int)cols,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)cols, (int)rows, (int)ntoks_e,
                &alpha_one, a_gathered.ptr,         (int)cols,
                            b_gathered.ptr,         (int)rows,
                &beta_acc,  d_base + e*dst_stride_e, (int)cols));
        }
    }
}
