#include "out-prod.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

static __global__ void k_compute_out_prod_ptrs(
        const float * src0_d, const float * src1_d, float * dst_d,
        const float ** ptrs_a, const float ** ptrs_b, float ** ptrs_c,
        const int64_t ne2, const int64_t ne3,
        const int64_t dps2, const int64_t dps3,
        const size_t s02, const size_t s03,
        const size_t s12, const size_t s13,
        const size_t s2,  const size_t s3) {
    const int64_t i2 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i3 = blockIdx.y*blockDim.y + threadIdx.y;

    if (i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t idx = i3*ne2 + i2;

    ptrs_a[idx] = src0_d + (i3/dps3)*s03 + (i2/dps2)*s02;
    ptrs_b[idx] = src1_d +  i3      *s13 +  i2      *s12;
    ptrs_c[idx] = dst_d  +  i3      *s3  +  i2      *s2;
}

template<typename lowp_t>
static __global__ void out_prod_convert_src1_tile(
        const char * src1, lowp_t * dst1, const int64_t ne10,
        const int64_t n_rows, const size_t nb10, const size_t nb11) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = ne10 * n_rows;
    if (i >= n) {
        return;
    }

    const int64_t row = i / ne10;
    const int64_t col = i % ne10;
    const float value = *(const float *) (src1 + col * nb10 + row * nb11);
    if constexpr (std::is_same_v<lowp_t, half>) {
        dst1[i] = __float2half_rn(value);
    } else {
        dst1[i] = __float2bfloat16(value);
    }
}

template<typename lowp_t>
static void ggml_cuda_out_prod_lowp(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const cudaDataType_t data_type) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool dequantize_src0 = ggml_is_quantized(src0->type);
    to_t_cuda_t<lowp_t> to_lowp = nullptr;
    if (dequantize_src0) {
        if constexpr (std::is_same_v<lowp_t, half>) {
            to_lowp = ggml_get_to_fp16_cuda(src0->type);
        } else if constexpr (std::is_same_v<lowp_t, nv_bfloat16>) {
            to_lowp = ggml_get_to_bf16_cuda(src0->type);
        }
    }
    GGML_ASSERT((src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || dequantize_src0));
    GGML_ASSERT(!dequantize_src0 || (to_lowp != nullptr && ggml_is_contiguous(src0)));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT((dequantize_src0 || nb00 == sizeof(lowp_t)) && ne01 == ne11);
    GGML_ASSERT(ne0 == ne00 && ne1 == ne10);
    GGML_ASSERT(ne2 % src0->ne[2] == 0 && ne3 % src0->ne[3] == 0);
    GGML_ASSERT(ne2 == src1->ne[2] && ne3 == src1->ne[3]);

    constexpr size_t workspace_max = 64ull * 1024 * 1024;
    const int64_t elements_per_row = (dequantize_src0 ? ne00 : 0) + ne10;
    const int64_t rows_per_tile = std::max<int64_t>(1,
        std::min<int64_t>(ne01, workspace_max / (sizeof(lowp_t) * elements_per_row)));
    const int64_t src0_tile_elements = dequantize_src0 ? ne00 * rows_per_tile : 0;
    ggml_cuda_pool_alloc<lowp_t> workspace(ctx.pool(), src0_tile_elements + ne10 * rows_per_tile);
    lowp_t * src0_temp = dequantize_src0 ? workspace.ptr : nullptr;
    lowp_t * src1_temp = workspace.ptr + src0_tile_elements;

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;
    const int threads = 256;
    const float alpha = 1.0f;
    const size_t lowp_size = sizeof(lowp_t);
    const int64_t ldc = nb1 / sizeof(float);

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            const char * src0_batch = (const char *) src0->data +
                (i2 / dps2) * nb02 + (i3 / dps3) * nb03;
            const char * src1_batch = (const char *) src1->data + i2 * nb12 + i3 * nb13;
            float * dst_batch = (float *) ((char *) dst->data + i2 * nb2 + i3 * nb3);
            const int64_t lda = dequantize_src0 ? ne00 : nb01 / lowp_size;

            for (int64_t row_begin = 0; row_begin < ne01; row_begin += rows_per_tile) {
                const int64_t n_rows = std::min<int64_t>(rows_per_tile, ne01 - row_begin);
                const char * src0_tile = src0_batch + row_begin * nb01;
                const char * src1_tile = src1_batch + row_begin * nb11;
                const lowp_t * src0_data;
                if (dequantize_src0) {
                    to_lowp(src0_tile, src0_temp, ne00 * n_rows, stream);
                    src0_data = src0_temp;
                } else {
                    src0_data = (const lowp_t *) src0_tile;
                }

                const int64_t src1_elements = ne10 * n_rows;
                out_prod_convert_src1_tile<<<
                    (src1_elements + threads - 1) / threads, threads, 0, stream>>>(
                        src1_tile, src1_temp, ne10, n_rows, nb10, nb11);
                CUDA_CHECK(cudaGetLastError());

                const float beta = row_begin == 0 ? 0.0f : 1.0f;
                CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    ne00, ne10, n_rows,
                    &alpha, src0_data, data_type, static_cast<int>(lda),
                            src1_temp, data_type, static_cast<int>(ne10),
                    &beta, dst_batch, CUDA_R_32F, static_cast<int>(ldc),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type == GGML_TYPE_BF16) {
        ggml_cuda_out_prod_lowp<nv_bfloat16>(ctx, dst, CUDA_R_16BF);
        return;
    }
    if (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) {
        ggml_cuda_out_prod_lowp<half>(ctx, dst, CUDA_R_16F);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float * src0_d = (const float *) src0->data;
    const int64_t lda = nb01 / sizeof(float);

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));

    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (dps2 == 1 && ne2 > 1) {
        // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
        GGML_ASSERT(ne2 <= std::numeric_limits<int>::max());
        const int batch_count = (int) ne2;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            CUBLAS_CHECK(
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03, lda, s02,
                                src1_d +  i3     *s13, ldb, s12,
                        &beta,  dst_d  +  i3     *s3,  ldc, s2,
                        batch_count));
        }
    } else if (ne2 > 1 || ne3 > 1) {
        // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
        // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
        GGML_ASSERT(ne3 > 0);
        GGML_ASSERT(ne2 <= (int64_t) std::numeric_limits<int>::max() / ne3);
        const int batch_count = (int) (ne2 * ne3);

        ggml_cuda_pool_alloc<const float *> ptrs_a(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<const float *> ptrs_b(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<      float *> ptrs_c(ctx.pool(), batch_count);

        const dim3 block_dims(16, 16);
        const dim3 grid_dims((ne2 + block_dims.x - 1)/block_dims.x, (ne3 + block_dims.y - 1)/block_dims.y);
        k_compute_out_prod_ptrs<<<grid_dims, block_dims, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ptrs_a.get(), ptrs_b.get(), ptrs_c.get(),
            ne2, ne3, dps2, dps3, s02, s03, s12, s13, s2, s3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
            cublasSgemmBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                    ne0, ne1, ne01,
                    &alpha, ptrs_a.get(), lda,
                            ptrs_b.get(), ldb,
                    &beta,  ptrs_c.get(), ldc,
                    batch_count));
    } else {
        // ne2 == 1 && ne3 == 1: single GEMM
        CUBLAS_CHECK(
            cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                    ne0, ne1, ne01,
                    &alpha, src0_d, lda,
                            src1_d, ldb,
                    &beta,  dst_d,  ldc));
    }
}

// ggml_cuda_out_prod_id
//
// Scattered outer-product for the MUL_MAT_ID backward pass (gradient w.r.t. expert weights).
//
//   src0 = a   [cols, n_expert_used, n_tokens]  F32  — token activations
//   src1 = b   [rows, n_expert_used, n_tokens]  F32  — upstream gradient
//   src2 = ids [n_expert_used, n_tokens]        I32  — expert dispatch indices
//   dst        [cols, rows, n_expert, 1]         F32  — gradient w.r.t. expert weight matrices
//
//   dst[:, :, e] += sum_{(i,t): ids[i,t]==e} a[:, i, t] ⊗ b[:, i, t]
//
// Algorithm:
//   For each expert e: gather the token columns where ids[i,t]==e into contiguous
//   GPU buffers, then use cublasSgemm (beta=1) to accumulate the outer product.
//   ids may be CPU-resident (common in backward graphs where they are leaf tensors).
void ggml_cuda_out_prod_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // a   [cols, n_exp_used, n_tokens]
    const ggml_tensor * src1 = dst->src[1]; // b   [rows, n_exp_used, n_tokens]
    const ggml_tensor * ids  = dst->src[2]; // ids [n_exp_used, n_tokens]  i32

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

    // Zero destination tensor before accumulating
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    // Read ids to host — ids may be CPU-resident (backward graph leaf) or GPU-resident
    const size_t ids_nbytes = ggml_nbytes(ids);
    std::vector<char> ids_host(ids_nbytes);
    if (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer)) {
        // GPU-resident: copy to host and synchronize so we can inspect the values
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ids_nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        memcpy(ids_host.data(), ids->data, ids_nbytes);
    }

    // Build per-expert token list: expert_tokens[e] = list of flat indices (iexp*n_tokens+itok)
    // whose dispatch id equals e.
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

    // Strides (in elements, not bytes)
    const int64_t a_stride_exp = src0->nb[1] / sizeof(float); // cols
    const int64_t a_stride_tok = src0->nb[2] / sizeof(float); // cols * n_exp_used
    const int64_t b_stride_exp = src1->nb[1] / sizeof(float); // rows
    const int64_t b_stride_tok = src1->nb[2] / sizeof(float); // rows * n_exp_used
    const int64_t dst_stride_e = dst->nb[2]  / sizeof(float); // cols * rows

    const float alpha_one = 1.0f;
    const float beta_acc  = 1.0f; // accumulate — dst is already zeroed above

    const float * a_base = (const float *) src0->data;
    const float * b_base = (const float *) src1->data;
    float       * d_base = (float       *)  dst->data;

    for (int64_t e = 0; e < n_expert; ++e) {
        const auto & toks = expert_tokens[e];
        if (toks.empty()) {
            continue;
        }

        const int64_t ntoks_e = (int64_t) toks.size();

        // Allocate contiguous gather buffers on GPU: a_e [cols, ntoks_e], b_e [rows, ntoks_e]
        ggml_cuda_pool_alloc<float> a_gathered(ctx.pool(), cols * ntoks_e);
        ggml_cuda_pool_alloc<float> b_gathered(ctx.pool(), rows * ntoks_e);

        // Gather token vectors from GPU src0/src1 into contiguous buffers
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

        // dst[:, :, e] += a_gathered @ b_gathered^T
        // cuBLAS column-major: A=[cols, ntoks_e] lda=cols, B=[rows, ntoks_e] ldb=rows
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)cols, (int)rows, (int)ntoks_e,
            &alpha_one, a_gathered.ptr,         (int)cols,
                        b_gathered.ptr,         (int)rows,
            &beta_acc,  d_base + e*dst_stride_e, (int)cols));
    }
}
