#include "out-prod.cuh"
#include "convert.cuh"
#include "f16-scale.cuh"
#include "mmid.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
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
    const ggml_prec precision = ggml_prec_get_src(dst, 1);

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

// The helper is called with tile_width == n_routes. Its tile_spec layout is
// therefore one optional seam tile at index 0 plus one tail tile per expert
// at index 1 + expert. The host combines these small metadata records once;
// route indices themselves remain on the GPU.
template<typename src_t, typename lowp_t>
static __global__ void out_prod_id_gather_kernel(
        const src_t * src, lowp_t * dst, const int32_t * route_indices,
        const int64_t n_cols, const int64_t n_expert_used,
        const int64_t stride_exp, const int64_t stride_tok,
        const int64_t n_elements, const float * scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) {
        return;
    }

    const int64_t token_i = i / n_cols;
    const int64_t col = i - token_i * n_cols;
    const int64_t flat = route_indices[token_i];
    const int64_t expert_slot = flat % n_expert_used;
    const int64_t token = flat / n_expert_used;
    const src_t value = src[expert_slot * stride_exp + token * stride_tok + col];

    if constexpr (std::is_same_v<src_t, float>) {
        if constexpr (std::is_same_v<lowp_t, half>) {
            dst[i] = __float2half_rn(value * (scale == nullptr ? 1.0f : *scale));
        } else if constexpr (std::is_same_v<lowp_t, nv_bfloat16>) {
            dst[i] = __float2bfloat16(value);
        } else {
            dst[i] = value;
        }
    } else {
        dst[i] = (lowp_t) value;
    }
}

template<typename src_t, typename lowp_t>
static __global__ void out_prod_id_gather_batched_kernel(
        const src_t * src, lowp_t * dst,
        const int32_t * route_indices,
        const ggml_cuda_mm_id_tile_spec * expert_ranges,
        const int64_t n_cols, const int64_t n_expert_used,
        const int64_t stride_exp, const int64_t stride_tok,
        const int64_t tile_capacity, const int64_t token_begin,
        const int expert_begin, const int64_t n_elements, const float * scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) {
        return;
    }

    const int64_t expert_stride = n_cols * tile_capacity;
    const int64_t expert_in_group = i / expert_stride;
    const int64_t expert = expert_begin + expert_in_group;
    const int64_t local_i = i - expert_in_group * expert_stride;
    const int64_t token_i = local_i / n_cols;
    const int64_t col = local_i - token_i * n_cols;

    const ggml_cuda_mm_id_tile_spec spec = expert_ranges[expert];
    const int64_t expert_tokens = spec.col_high - spec.col_low;
    if (token_begin + token_i >= expert_tokens) {
        if constexpr (std::is_same_v<lowp_t, half>) {
            dst[i] = __float2half_rn(0.0f);
        } else if constexpr (std::is_same_v<lowp_t, nv_bfloat16>) {
            dst[i] = __float2bfloat16(0.0f);
        } else {
            dst[i] = 0.0f;
        }
        return;
    }

    const int64_t flat = route_indices[spec.col_low + token_begin + token_i];
    const int64_t expert_slot = flat % n_expert_used;
    const int64_t token = flat / n_expert_used;
    const src_t value = src[expert_slot * stride_exp + token * stride_tok + col];

    if constexpr (std::is_same_v<src_t, float>) {
        if constexpr (std::is_same_v<lowp_t, half>) {
            dst[i] = __float2half_rn(value * (scale == nullptr ? 1.0f : *scale));
        } else if constexpr (std::is_same_v<lowp_t, nv_bfloat16>) {
            dst[i] = __float2bfloat16(value);
        } else {
            dst[i] = value;
        }
    } else {
        dst[i] = (lowp_t) value;
    }
}

// Group experts into contiguous batches. Each batch has its own maximum route
// count, so padding is limited to experts within that batch rather than all
// experts in the model.
template<typename src0_t, typename lowp_t>
static void ggml_cuda_out_prod_id_grouped_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * route_indices,
        const ggml_cuda_mm_id_tile_spec * expert_ranges,
        const std::vector<ggml_cuda_mm_id_tile_spec> & expert_ranges_host,
        const size_t max_expert_tokens, const int group_size,
        const cudaDataType_t data_type, const float * scale) {
    const int64_t cols = src0->ne[0];
    const int64_t n_expert_used = src0->ne[1];
    const int64_t rows = src1->ne[0];

    const int64_t src0_type_size = ggml_type_size(src0->type);
    const int64_t a_stride_exp = src0->nb[1] / src0_type_size;
    const int64_t a_stride_tok = src0->nb[2] / src0_type_size;
    const int64_t b_stride_exp = src1->nb[1] / sizeof(float);
    const int64_t b_stride_tok = src1->nb[2] / sizeof(float);
    const int64_t dst_stride_e = dst->nb[2] / sizeof(float);

    if (max_expert_tokens == 0) {
        return;
    }

    GGML_ASSERT(cols > 0 && rows > 0 && group_size > 1);
    GGML_ASSERT(cols <= std::numeric_limits<int>::max());
    GGML_ASSERT(rows <= std::numeric_limits<int>::max());

    constexpr size_t workspace_max = 64ull * 1024 * 1024;
    const size_t bytes_per_expert_token = sizeof(lowp_t) *
        (static_cast<size_t>(cols) + static_cast<size_t>(rows));
    const size_t group_bytes_per_token = bytes_per_expert_token * static_cast<size_t>(group_size);
    GGML_ASSERT(group_bytes_per_token <= workspace_max);
    const size_t max_tile_capacity = std::max<size_t>(1,
        std::min(max_expert_tokens, workspace_max / group_bytes_per_token));
    GGML_ASSERT(max_tile_capacity <= static_cast<size_t>(std::numeric_limits<int>::max()));

    const size_t a_max_elements = static_cast<size_t>(cols) * max_tile_capacity * group_size;
    const size_t b_max_elements = static_cast<size_t>(rows) * max_tile_capacity * group_size;
    ggml_cuda_pool_alloc<lowp_t> a_gathered(ctx.pool(), a_max_elements);
    ggml_cuda_pool_alloc<lowp_t> b_gathered(ctx.pool(), b_max_elements);

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const int threads = 256;
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }

    const src0_t * a_base = (const src0_t *) src0->data;
    const float * b_base = (const float *) src1->data;
    float * d_base = (float *) dst->data;

    for (int expert_begin = 0; expert_begin < static_cast<int>(expert_ranges_host.size());
         expert_begin += group_size) {
        const int group_count = std::min(group_size,
            static_cast<int>(expert_ranges_host.size()) - expert_begin);
        size_t group_max_routes = 0;
        for (int i = 0; i < group_count; ++i) {
            const ggml_cuda_mm_id_tile_spec & range =
                expert_ranges_host[static_cast<size_t>(expert_begin + i)];
            group_max_routes = std::max(group_max_routes,
                static_cast<size_t>(range.col_high - range.col_low));
        }
        if (group_max_routes == 0) {
            continue;
        }

        const size_t tile_capacity = std::min(group_max_routes, max_tile_capacity);
        const size_t a_batch_stride = static_cast<size_t>(cols) * tile_capacity;
        const size_t b_batch_stride = static_cast<size_t>(rows) * tile_capacity;
        const int64_t a_count = static_cast<int64_t>(a_batch_stride) * group_count;
        const int64_t b_count = static_cast<int64_t>(b_batch_stride) * group_count;
        const int a_blocks = static_cast<int>((a_count + threads - 1) / threads);
        const int b_blocks = static_cast<int>((b_count + threads - 1) / threads);

        for (size_t token_begin = 0; token_begin < group_max_routes; token_begin += tile_capacity) {
            out_prod_id_gather_batched_kernel<src0_t, lowp_t><<<a_blocks, threads, 0, stream>>>(
                a_base, a_gathered.ptr, route_indices, expert_ranges, cols, n_expert_used,
                a_stride_exp, a_stride_tok, static_cast<int64_t>(tile_capacity),
                static_cast<int64_t>(token_begin), expert_begin, a_count, nullptr);
            out_prod_id_gather_batched_kernel<float, lowp_t><<<b_blocks, threads, 0, stream>>>(
                b_base, b_gathered.ptr, route_indices, expert_ranges, rows, n_expert_used,
                b_stride_exp, b_stride_tok, static_cast<int64_t>(tile_capacity),
                static_cast<int64_t>(token_begin), expert_begin, b_count, scale);
            CUDA_CHECK(cudaGetLastError());

            if constexpr (std::is_same_v<lowp_t, float>) {
                CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(tile_capacity),
                    &alpha, (const float *) a_gathered.ptr, static_cast<int>(cols),
                            static_cast<long long>(a_batch_stride),
                            (const float *) b_gathered.ptr, static_cast<int>(rows),
                            static_cast<long long>(b_batch_stride),
                    &beta, d_base + static_cast<int64_t>(expert_begin) * dst_stride_e,
                            static_cast<int>(cols), static_cast<long long>(dst_stride_e),
                    group_count));
            } else {
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(tile_capacity),
                    &alpha, a_gathered.ptr, data_type, static_cast<int>(cols),
                            static_cast<long long>(a_batch_stride),
                            b_gathered.ptr, data_type, static_cast<int>(rows),
                            static_cast<long long>(b_batch_stride),
                    &beta, d_base + static_cast<int64_t>(expert_begin) * dst_stride_e,
                            CUDA_R_32F, static_cast<int>(cols), static_cast<long long>(dst_stride_e),
                    group_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }

    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, pointer_mode));
    }
}

template<typename lowp_t>
static void ggml_cuda_out_prod_id_grouped(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * route_indices,
        const ggml_cuda_mm_id_tile_spec * expert_ranges,
        const std::vector<ggml_cuda_mm_id_tile_spec> & expert_ranges_host,
        const size_t max_expert_tokens, const int group_size,
        const cudaDataType_t data_type, const float * scale) {
    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_out_prod_id_grouped_impl<half, lowp_t>(
            ctx, src0, src1, dst, route_indices, expert_ranges, expert_ranges_host,
            max_expert_tokens, group_size, data_type, scale);
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        ggml_cuda_out_prod_id_grouped_impl<float, lowp_t>(
            ctx, src0, src1, dst, route_indices, expert_ranges, expert_ranges_host,
            max_expert_tokens, group_size, data_type, scale);
    }
}

// Fallback for unbalanced routing and small workloads. Reuse one pair of
// gather buffers and process only experts which actually received routes.
template<typename src0_t, typename lowp_t>
static void ggml_cuda_out_prod_id_per_expert_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * route_indices,
        const std::vector<ggml_cuda_mm_id_tile_spec> & expert_ranges,
        const size_t max_expert_tokens,
        const cudaDataType_t data_type, const float * scale) {
    const int64_t cols = src0->ne[0];
    const int64_t n_expert_used = src0->ne[1];
    const int64_t rows = src1->ne[0];

    const int64_t src0_type_size = ggml_type_size(src0->type);
    const int64_t a_stride_exp = src0->nb[1] / src0_type_size;
    const int64_t a_stride_tok = src0->nb[2] / src0_type_size;
    const int64_t b_stride_exp = src1->nb[1] / sizeof(float);
    const int64_t b_stride_tok = src1->nb[2] / sizeof(float);
    const int64_t dst_stride_e = dst->nb[2] / sizeof(float);

    if (max_expert_tokens == 0) {
        return;
    }

    GGML_ASSERT(cols > 0 && rows > 0);
    GGML_ASSERT(cols <= std::numeric_limits<int>::max());
    GGML_ASSERT(rows <= std::numeric_limits<int>::max());

    constexpr size_t workspace_max = 64ull * 1024 * 1024;
    const size_t bytes_per_token = sizeof(lowp_t) *
        (static_cast<size_t>(cols) + static_cast<size_t>(rows));
    GGML_ASSERT(bytes_per_token > 0);
    const size_t tile_capacity = std::max<size_t>(1,
        std::min(max_expert_tokens, workspace_max / bytes_per_token));
    ggml_cuda_pool_alloc<lowp_t> a_gathered(ctx.pool(), static_cast<size_t>(cols) * tile_capacity);
    ggml_cuda_pool_alloc<lowp_t> b_gathered(ctx.pool(), static_cast<size_t>(rows) * tile_capacity);

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const int threads = 256;
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }

    const src0_t * a_base = (const src0_t *) src0->data;
    const float * b_base = (const float *) src1->data;
    float * d_base = (float *) dst->data;

    for (size_t expert = 0; expert < expert_ranges.size(); ++expert) {
        const ggml_cuda_mm_id_tile_spec & range = expert_ranges[expert];
        const size_t expert_routes = static_cast<size_t>(range.col_high - range.col_low);
        if (expert_routes == 0) {
            continue;
        }
        const int64_t dst_offset = static_cast<int64_t>(expert) * dst_stride_e;

        for (size_t token_begin = 0; token_begin < expert_routes; token_begin += tile_capacity) {
            const int64_t tile_routes = static_cast<int64_t>(
                std::min(tile_capacity, expert_routes - token_begin));
            const int64_t a_count = cols * tile_routes;
            const int64_t b_count = rows * tile_routes;

            out_prod_id_gather_kernel<src0_t, lowp_t><<<
                static_cast<int>((a_count + threads - 1) / threads), threads, 0, stream>>>(
                    a_base, a_gathered.ptr,
                    route_indices + range.col_low + token_begin, cols, n_expert_used,
                    a_stride_exp, a_stride_tok, a_count, nullptr);
            out_prod_id_gather_kernel<float, lowp_t><<<
                static_cast<int>((b_count + threads - 1) / threads), threads, 0, stream>>>(
                    b_base, b_gathered.ptr,
                    route_indices + range.col_low + token_begin, rows, n_expert_used,
                    b_stride_exp, b_stride_tok, b_count, scale);
            CUDA_CHECK(cudaGetLastError());

            if constexpr (std::is_same_v<lowp_t, float>) {
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(tile_routes),
                    &alpha, (const float *) a_gathered.ptr, static_cast<int>(cols),
                            (const float *) b_gathered.ptr, static_cast<int>(rows),
                    &beta, d_base + dst_offset, static_cast<int>(cols)));
            } else {
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(tile_routes),
                    &alpha, a_gathered.ptr, data_type, static_cast<int>(cols),
                            b_gathered.ptr, data_type, static_cast<int>(rows),
                    &beta, d_base + dst_offset, CUDA_R_32F, static_cast<int>(cols),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }

    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, pointer_mode));
    }
}

template<typename lowp_t>
static void ggml_cuda_out_prod_id_per_expert(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * route_indices,
        const std::vector<ggml_cuda_mm_id_tile_spec> & expert_ranges,
        const size_t max_expert_tokens,
        const cudaDataType_t data_type, const float * scale) {
    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_out_prod_id_per_expert_impl<half, lowp_t>(
            ctx, src0, src1, dst, route_indices, expert_ranges, max_expert_tokens, data_type, scale);
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        ggml_cuda_out_prod_id_per_expert_impl<float, lowp_t>(
            ctx, src0, src1, dst, route_indices, expert_ranges, max_expert_tokens, data_type, scale);
    }
}

void ggml_cuda_out_prod_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t n_exp_used = src0->ne[1];
    const int64_t n_tokens   = src0->ne[2];
    const int64_t n_expert   = dst->ne[2];

    GGML_ASSERT(ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ids->nb[1] % sizeof(int32_t) == 0);
    GGML_ASSERT(ids->ne[0] == n_exp_used && ids->ne[1] == n_tokens);
    GGML_ASSERT(src1->ne[1] == n_exp_used && src1->ne[2] == n_tokens);
    GGML_ASSERT(n_exp_used >= 0 && n_tokens >= 0);
    GGML_ASSERT(n_exp_used == 0 || n_tokens <= std::numeric_limits<int64_t>::max() / n_exp_used);
    const int64_t n_routes = n_exp_used * n_tokens;
    GGML_ASSERT(n_routes <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(n_expert > 0 && n_expert <= std::numeric_limits<int>::max());
    GGML_ASSERT(n_tokens < (1 << 22) && n_exp_used < (1 << 10));
    GGML_ASSERT(ids->nb[1] / sizeof(int32_t) <= std::numeric_limits<int>::max());

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    if (n_routes == 0) {
        return;
    }

    // Reuse the same GPU-side route grouping used by MUL_MAT_ID_BACK.
    // route_indices contains flattened (expert-slot, token) pairs sorted by
    // expert, while tile_spec describes each expert's contiguous range.
    ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool());
    const bool ids_on_device = ids->buffer && !ggml_backend_buffer_is_host(ids->buffer);
    const int32_t * ids_ptr = (const int32_t *) ids->data;
    if (!ids_on_device) {
        ids_device.alloc(ggml_nbytes(ids) / sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(ids_device.ptr, ids->data, ggml_nbytes(ids),
            cudaMemcpyHostToDevice, stream));
        ids_ptr = ids_device.ptr;
    }

    ggml_cuda_pool_alloc<int32_t> route_indices(ctx.pool(), static_cast<size_t>(n_routes));
    ggml_cuda_pool_alloc<int32_t> ids_src(ctx.pool(), static_cast<size_t>(n_routes));
    const size_t n_tile_specs = static_cast<size_t>(n_expert + 1);
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), n_tile_specs);
    CUDA_CHECK(cudaMemsetAsync(tile_spec.ptr, 0xFF,
        n_tile_specs * sizeof(ggml_cuda_mm_id_tile_spec), stream));

    ggml_cuda_launch_mm_ids_helper(
        ids_ptr,
        ids_src.ptr,
        route_indices.ptr,
        tile_spec.ptr,
        static_cast<int>(n_expert),
        static_cast<int>(n_tokens),
        static_cast<int>(n_exp_used),
        static_cast<int>(n_exp_used),
        static_cast<int>(n_routes),
        static_cast<int>(ids->nb[1] / sizeof(int32_t)),
        static_cast<int>(n_exp_used),
        false,
        stream);
    CUDA_CHECK(cudaGetLastError());

    // A single D2H transfer of the small tile-spec array lets the host build
    // exact expert ranges and estimate the padding cost before choosing a GEMM path.
    std::vector<ggml_cuda_mm_id_tile_spec> tile_spec_host(n_tile_specs);
    CUDA_CHECK(cudaMemcpyAsync(tile_spec_host.data(), tile_spec.ptr,
        n_tile_specs * sizeof(ggml_cuda_mm_id_tile_spec), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<ggml_cuda_mm_id_tile_spec> expert_ranges_host(static_cast<size_t>(n_expert));
    size_t max_expert_tokens = 0;
    size_t total_active_routes = 0;
    size_t active_experts = 0;
    for (int64_t expert = 0; expert < n_expert; ++expert) {
        int32_t route_begin = std::numeric_limits<int32_t>::max();
        int32_t route_end = 0;
        bool found = false;

        const ggml_cuda_mm_id_tile_spec tail = tile_spec_host[static_cast<size_t>(1 + expert)];
        const ggml_cuda_mm_id_tile_spec seam = tile_spec_host[0];
        if (tail.expert == expert) {
            route_begin = tail.col_low;
            route_end = tail.col_high;
            found = true;
        }
        if (seam.expert == expert) {
            route_begin = found ? std::min(route_begin, seam.col_low) : seam.col_low;
            route_end = found ? std::max(route_end, seam.col_high) : seam.col_high;
            found = true;
        }

        GGML_ASSERT(found);
        GGML_ASSERT(route_begin >= 0 && route_end >= route_begin && route_end <= n_routes);
        if (expert > 0) {
            GGML_ASSERT(expert_ranges_host[static_cast<size_t>(expert - 1)].col_high == route_begin);
        }
        expert_ranges_host[static_cast<size_t>(expert)] = {
            static_cast<int32_t>(expert), route_begin, route_end, 0
        };

        const size_t expert_routes = static_cast<size_t>(route_end - route_begin);
        max_expert_tokens = std::max(max_expert_tokens, expert_routes);
        total_active_routes += expert_routes;
        active_experts += expert_routes != 0;
    }

    GGML_ASSERT(total_active_routes <= static_cast<size_t>(n_routes));
    if (total_active_routes == 0) {
        return;
    }

    const ggml_prec precision = ggml_prec_get_src(dst, 1);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_fp16 = (src0->type == GGML_TYPE_F16 || precision == GGML_PREC_F16) &&
        fast_fp16_hardware_available(cc);
    const bool use_bf16 = precision == GGML_PREC_BF16 && fast_bf16_hardware_available(cc);

    // Estimate grouped GEMM padding for contiguous expert batches. Smaller
    // batches reduce route-count imbalance; larger batches reduce launch count.
    const size_t workspace_max = 64ull * 1024 * 1024;
    const size_t lowp_size = use_fp16 || use_bf16 ? sizeof(uint16_t) : sizeof(float);
    const size_t bytes_per_route = lowp_size *
        (static_cast<size_t>(src0->ne[0]) + static_cast<size_t>(src1->ne[0]));
    GGML_ASSERT(bytes_per_route > 0);

    // Prefer the largest contiguous batch size whose predicted GEMM work is
    // at most 20% above useful work. Cap a batch at 32 experts rather than
    // issuing one padded GEMM for the complete expert set.
    constexpr size_t min_grouped_routes = 4096;
    constexpr size_t min_grouped_experts = 32;
    constexpr long double max_grouped_padding_ratio = 1.20L;
    const int group_size_candidates[] = { 32, 16, 8, 4 };
    int grouped_expert_batch = 0;

    if (active_experts >= min_grouped_experts && total_active_routes >= min_grouped_routes) {
        for (const int candidate : group_size_candidates) {
            const size_t group_bytes_per_route = bytes_per_route * static_cast<size_t>(candidate);
            if (group_bytes_per_route > workspace_max) {
                continue;
            }
            const size_t max_tile_capacity = std::max<size_t>(1,
                std::min(max_expert_tokens, workspace_max / group_bytes_per_route));
            long double padded_route_slots = 0.0L;

            for (int expert_begin = 0; expert_begin < n_expert; expert_begin += candidate) {
                const int group_count = std::min(candidate, static_cast<int>(n_expert) - expert_begin);
                size_t group_max_routes = 0;
                for (int i = 0; i < group_count; ++i) {
                    const ggml_cuda_mm_id_tile_spec & range =
                        expert_ranges_host[static_cast<size_t>(expert_begin + i)];
                    group_max_routes = std::max(group_max_routes,
                        static_cast<size_t>(range.col_high - range.col_low));
                }
                if (group_max_routes == 0) {
                    continue;
                }

                const size_t tile_capacity = std::min(group_max_routes, max_tile_capacity);
                const size_t tile_count = (group_max_routes + tile_capacity - 1) / tile_capacity;
                padded_route_slots += static_cast<long double>(group_count) *
                    static_cast<long double>(tile_count) * static_cast<long double>(tile_capacity);
            }

            const long double padding_ratio = padded_route_slots /
                static_cast<long double>(total_active_routes);
            if (padding_ratio <= max_grouped_padding_ratio) {
                grouped_expert_batch = candidate;
                break;
            }
        }
    }
    const bool use_grouped = grouped_expert_batch > 1;

    if (use_grouped) {
        ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> expert_ranges(ctx.pool(),
            static_cast<size_t>(n_expert));
        CUDA_CHECK(cudaMemcpyAsync(expert_ranges.ptr, expert_ranges_host.data(),
            static_cast<size_t>(n_expert) * sizeof(ggml_cuda_mm_id_tile_spec),
            cudaMemcpyHostToDevice, stream));

        if (use_fp16) {
            ggml_cuda_pool_alloc<unsigned int> absmax(ctx.pool(), 1);
            ggml_cuda_pool_alloc<float> scales(ctx.pool(), 2);
            ggml_cuda_f16_scale_init(src1, absmax.ptr, scales.ptr, stream);
            ggml_cuda_out_prod_id_grouped<half>(ctx, src0, src1, dst,
                route_indices.ptr, expert_ranges.ptr, expert_ranges_host, max_expert_tokens,
                grouped_expert_batch, CUDA_R_16F, scales.ptr);
            ggml_cuda_f16_scale_tensor(dst, scales.ptr + 1, stream);
            return;
        }
        if (use_bf16) {
            ggml_cuda_out_prod_id_grouped<nv_bfloat16>(ctx, src0, src1, dst,
                route_indices.ptr, expert_ranges.ptr, expert_ranges_host, max_expert_tokens,
                grouped_expert_batch, CUDA_R_16BF, nullptr);
            return;
        }

        ggml_cuda_out_prod_id_grouped<float>(ctx, src0, src1, dst,
            route_indices.ptr, expert_ranges.ptr, expert_ranges_host, max_expert_tokens,
            grouped_expert_batch, CUDA_R_32F, nullptr);
        return;
    }

    if (use_fp16) {
        ggml_cuda_pool_alloc<unsigned int> absmax(ctx.pool(), 1);
        ggml_cuda_pool_alloc<float> scales(ctx.pool(), 2);
        ggml_cuda_f16_scale_init(src1, absmax.ptr, scales.ptr, stream);
        ggml_cuda_out_prod_id_per_expert<half>(ctx, src0, src1, dst,
            route_indices.ptr, expert_ranges_host, max_expert_tokens, CUDA_R_16F, scales.ptr);
        ggml_cuda_f16_scale_tensor(dst, scales.ptr + 1, stream);
        return;
    }
    if (use_bf16) {
        ggml_cuda_out_prod_id_per_expert<nv_bfloat16>(ctx, src0, src1, dst,
            route_indices.ptr, expert_ranges_host, max_expert_tokens, CUDA_R_16BF, nullptr);
        return;
    }

    ggml_cuda_out_prod_id_per_expert<float>(ctx, src0, src1, dst,
        route_indices.ptr, expert_ranges_host, max_expert_tokens, CUDA_R_32F, nullptr);
}
