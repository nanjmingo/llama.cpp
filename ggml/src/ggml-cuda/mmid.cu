#include "common.cuh"
#include "convert.cuh"
#include "mmid.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used != -1 && expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used != -1 && expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}


static __global__ void mul_mat_id_back_gather_kernel(
        const float * __restrict__ grad,
        const int32_t * __restrict__ pairs,
        float * __restrict__ gathered,
        const int64_t output_dim,
        const int64_t n_used,
        const int64_t grad_nb0,
        const int64_t grad_nb1,
        const int64_t grad_nb2) {
    const int64_t route_idx = blockIdx.y;
    const int64_t route = pairs[route_idx];
    const int64_t used = route % n_used;
    const int64_t token = route / n_used;

    const float * grad_route = grad + used * grad_nb1 + token * grad_nb2;
    float * gathered_route = gathered + route_idx * output_dim;

    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < output_dim; row += gridDim.x * blockDim.x) {
        gathered_route[row] = grad_route[row * grad_nb0];
    }
}

template<typename dst_t>
static __global__ void mul_mat_id_back_convert_f32_kernel(
        const float * __restrict__ src,
        dst_t * __restrict__ dst,
        const int64_t src_stride,
        const int64_t src_offset,
        const int64_t chunk_dim,
        const int64_t route_count) {
    const int64_t n = chunk_dim * route_count;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        const int64_t route = i / chunk_dim;
        const int64_t row = i % chunk_dim;
        const float value = src[route * src_stride + src_offset + row];
        if constexpr (std::is_same_v<dst_t, half>) {
            dst[i] = __float2half_rn(value);
        } else {
            dst[i] = __float2bfloat16(value);
        }
    }
}

static __global__ void mul_mat_id_back_scatter_kernel(
        const float * __restrict__ input_gradient,
        const int32_t * __restrict__ pairs,
        float * __restrict__ dst,
        const int64_t input_dim,
        const int64_t n_used,
        const int64_t n_slots,
        const int64_t dst_nb1,
        const int64_t dst_nb2) {
    const int64_t route_idx = blockIdx.y;
    const int64_t route = pairs[route_idx];
    const int64_t used = route % n_used;
    const int64_t token = route / n_used;
    const int64_t slot = used % n_slots;
    float * dst_route = (float *) ((char *) dst + slot * dst_nb1 + token * dst_nb2);
    const float * src_route = input_gradient + route_idx * input_dim;

    for (int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
         col < input_dim; col += gridDim.x * blockDim.x) {
        // Several routed slots may contribute to one broadcast input vector.
        atomicAdd(dst_route + col, src_route[col]);
    }
}

void ggml_cuda_mul_mat_id_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as = dst->src[0];
    const ggml_tensor * grad = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * b_like = dst->src[3];

    GGML_ASSERT(grad->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->nb[0] == sizeof(int32_t) && ids->nb[1] % sizeof(int32_t) == 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(as->ne[0] == b_like->ne[0] && as->ne[1] == grad->ne[0]);
    GGML_ASSERT(grad->ne[1] == ids->ne[0] && grad->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->ne[0] % b_like->ne[1] == 0);

    const int64_t input_dim = as->ne[0];
    const int64_t output_dim = as->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_slots = b_like->ne[1];
    const int64_t n_experts = as->ne[2];
    const int64_t n_routes = n_used * n_tokens;
    GGML_ASSERT(input_dim > 0 && output_dim > 0 && n_slots > 0);
    GGML_ASSERT(n_routes <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(input_dim <= std::numeric_limits<int>::max());
    GGML_ASSERT(output_dim <= std::numeric_limits<int>::max());

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool());
    const bool ids_on_device = ids->buffer && !ggml_backend_buffer_is_host(ids->buffer);
    const int32_t * ids_ptr = (const int32_t *) ids->data;
    if (!ids_on_device) {
        ids_device.alloc(ggml_nbytes(ids) / sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(ids_device.ptr, ids->data, ggml_nbytes(ids), cudaMemcpyHostToDevice, stream));
        ids_ptr = ids_device.ptr;
    }

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    // Group routing pairs on the GPU. Only the expert boundaries are copied
    // back to the host, avoiding a full routing-ID round trip and CPU sort.
    ggml_cuda_pool_alloc<int32_t> pairs(ctx.pool(), static_cast<size_t>(n_routes));
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), static_cast<size_t>(n_experts + 1));
    ggml_cuda_pool_alloc<int32_t> ids_src(ctx.pool(), static_cast<size_t>(n_routes));

    const int si1 = static_cast<int>(ids->nb[1] / sizeof(int32_t));
    ggml_cuda_launch_mm_ids_helper(
        ids_ptr,
        ids_src.ptr,
        pairs.ptr,
        expert_bounds.ptr,
        static_cast<int>(n_experts),
        static_cast<int>(n_tokens),
        static_cast<int>(n_used),
        static_cast<int>(n_used),
        si1,
        static_cast<int>(n_used),
        /*write_inverse =*/ false,
        stream);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int32_t> offsets_host(static_cast<size_t>(n_experts + 1));
    CUDA_CHECK(cudaMemcpyAsync(offsets_host.data(), expert_bounds.ptr,
        offsets_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int64_t n_active_routes = offsets_host.back();
    if (n_active_routes == 0) {
        return;
    }

    const float * grad_ptr = (const float *) grad->data;
    float * dst_ptr = (float *) dst->data;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_bf16 = as->type == GGML_TYPE_BF16 && fast_bf16_hardware_available(cc);
    const to_fp16_cuda_t to_fp16 = as->type == GGML_TYPE_F32 || as->type == GGML_TYPE_F16 || as->type == GGML_TYPE_BF16
        ? nullptr : ggml_get_to_fp16_cuda(as->type);
    const bool use_fp16 = as->type != GGML_TYPE_F32 && as->type != GGML_TYPE_BF16 &&
        (as->type == GGML_TYPE_F16 || to_fp16 != nullptr) && fast_fp16_hardware_available(cc);
    const to_fp32_cuda_t to_fp32 = as->type == GGML_TYPE_F32 || use_fp16 || use_bf16
        ? nullptr : ggml_get_to_fp32_cuda(as->type);
    if (as->type != GGML_TYPE_F32) {
        GGML_ASSERT(use_fp16 || use_bf16 || to_fp32 != nullptr);
    }
    // Convert only a bounded slice of each expert matrix at a time. This is
    // important for large quantized experts, where a full FP16/FP32 copy can
    // require hundreds of megabytes of additional device memory.
    constexpr int64_t min_weight_chunk_dim = 1024;
    constexpr int64_t max_weight_chunk_dim = 8192;
    constexpr int64_t max_batch_routes = 64;
    constexpr size_t max_weight_temp_bytes = 64u * 1024u * 1024u;

    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t weight_temp_budget = std::min(max_weight_temp_bytes, free_memory / 16);
    const size_t bytes_per_output = static_cast<size_t>(input_dim) * (use_fp16 || use_bf16 ? sizeof(half) : sizeof(float));
    const int64_t memory_limited_chunk = bytes_per_output == 0 ? min_weight_chunk_dim :
        static_cast<int64_t>(weight_temp_budget / bytes_per_output);
    const int64_t max_chunk_dim = std::max(min_weight_chunk_dim,
        std::min(max_weight_chunk_dim, memory_limited_chunk));
    const int64_t weight_chunk_capacity = std::min(output_dim, max_chunk_dim);

    ggml_cuda_pool_alloc<float> weights_f32(ctx.pool());
    ggml_cuda_pool_alloc<half> weights_f16(ctx.pool());
    if (as->type != GGML_TYPE_F32) {
        if (use_fp16 && as->type != GGML_TYPE_F16) {
            weights_f16.alloc(static_cast<size_t>(input_dim * weight_chunk_capacity));
        } else if (!use_fp16 && !use_bf16) {
            weights_f32.alloc(static_cast<size_t>(input_dim * weight_chunk_capacity));
        }
    }

    // Bound temporary memory independently of the number of tokens routed to
    // one expert. Large expert batches are processed as several GEMM chunks.
    ggml_cuda_pool_alloc<float> gathered(ctx.pool());
    ggml_cuda_pool_alloc<half> gathered_f16(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> gathered_bf16(ctx.pool());
    ggml_cuda_pool_alloc<float> input_gradient(ctx.pool());
    gathered.alloc(static_cast<size_t>(max_batch_routes * output_dim));
    if (use_fp16) {
        gathered_f16.alloc(static_cast<size_t>(max_batch_routes * weight_chunk_capacity));
    } else if (use_bf16) {
        gathered_bf16.alloc(static_cast<size_t>(max_batch_routes * weight_chunk_capacity));
    }
    input_gradient.alloc(static_cast<size_t>(max_batch_routes * input_dim));

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;

    const int64_t grad_nb0 = grad->nb[0] / sizeof(float);
    const int64_t grad_nb1 = grad->nb[1] / sizeof(float);
    const int64_t grad_nb2 = grad->nb[2] / sizeof(float);
    const int64_t dst_nb1 = dst->nb[1] / sizeof(float);
    const int64_t dst_nb2 = dst->nb[2] / sizeof(float);

    for (int64_t expert = 0; expert < n_experts; ++expert) {
        const int64_t route_begin = offsets_host[static_cast<size_t>(expert)];
        const int64_t route_end = offsets_host[static_cast<size_t>(expert + 1)];
        const int64_t expert_routes = route_end - route_begin;
        if (expert_routes == 0) {
            continue;
        }

        const char * expert_weights = (const char *) as->data + expert * as->nb[2];
        const float * weights_ptr = as->type == GGML_TYPE_F32 ? (const float *) expert_weights : nullptr;

        const bool cache_full_weights = expert_routes > max_batch_routes &&
            weight_chunk_capacity == output_dim && ggml_is_contiguous(as) &&
            as->type != GGML_TYPE_F32 && as->type != GGML_TYPE_F16 && !use_bf16;
        const half * cached_weights_f16 = nullptr;
        const float * cached_weights_f32 = nullptr;
        if (cache_full_weights) {
            const int64_t n_weights = input_dim * output_dim;
            if (use_fp16) {
                to_fp16(expert_weights, weights_f16.ptr, n_weights, stream);
                cached_weights_f16 = weights_f16.ptr;
            } else {
                to_fp32(expert_weights, weights_f32.ptr, n_weights, stream);
                cached_weights_f32 = weights_f32.ptr;
            }
        }

        const int32_t * expert_pairs = pairs.ptr + route_begin;
        for (int64_t route_offset = 0; route_offset < expert_routes; route_offset += max_batch_routes) {
            const int64_t route_count = std::min(max_batch_routes, expert_routes - route_offset);
            const int32_t * route_pairs = expert_pairs + route_offset;
            const dim3 gather_grid((output_dim + 255) / 256, route_count, 1);
            mul_mat_id_back_gather_kernel<<<gather_grid, 256, 0, stream>>>(
                grad_ptr, route_pairs, gathered.ptr, output_dim, n_used,
                grad_nb0, grad_nb1, grad_nb2);
            CUDA_CHECK(cudaGetLastError());

            if (as->type == GGML_TYPE_F32) {
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(input_dim), static_cast<int>(route_count), static_cast<int>(output_dim),
                    &alpha, weights_ptr, static_cast<int>(input_dim),
                    gathered.ptr, static_cast<int>(output_dim),
                    &beta_zero, input_gradient.ptr, static_cast<int>(input_dim)));
            } else {
                for (int64_t output_offset = 0; output_offset < output_dim; output_offset += weight_chunk_capacity) {
                    const int64_t chunk_dim = std::min(weight_chunk_capacity, output_dim - output_offset);
                    const char * weight_chunk = expert_weights + output_offset * as->nb[1];
                    const float * beta = output_offset == 0 ? &beta_zero : &beta_one;

                    if (use_fp16) {
                        const half * weights_f16_ptr;
                        if (as->type == GGML_TYPE_F16) {
                            weights_f16_ptr = (const half *) weight_chunk;
                        } else if (cached_weights_f16 != nullptr) {
                            weights_f16_ptr = cached_weights_f16 + output_offset * input_dim;
                        } else {
                            to_fp16(weight_chunk, weights_f16.ptr, input_dim * chunk_dim, stream);
                            weights_f16_ptr = weights_f16.ptr;
                        }

                        const int64_t gathered_count = route_count * chunk_dim;
                        const dim3 convert_grid((gathered_count + 255) / 256, 1, 1);
                        mul_mat_id_back_convert_f32_kernel<<<convert_grid, 256, 0, stream>>>(
                            gathered.ptr, gathered_f16.ptr, output_dim, output_offset, chunk_dim, route_count);
                        CUDA_CHECK(cudaGetLastError());

                        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(input_dim), static_cast<int>(route_count), static_cast<int>(chunk_dim),
                            &alpha, weights_f16_ptr, CUDA_R_16F, static_cast<int>(input_dim),
                            gathered_f16.ptr, CUDA_R_16F, static_cast<int>(chunk_dim),
                            beta, input_gradient.ptr, CUDA_R_32F, static_cast<int>(input_dim),
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    } else if (use_bf16) {
                        const int64_t gathered_count = route_count * chunk_dim;
                        const dim3 convert_grid((gathered_count + 255) / 256, 1, 1);
                        mul_mat_id_back_convert_f32_kernel<<<convert_grid, 256, 0, stream>>>(
                            gathered.ptr, gathered_bf16.ptr, output_dim, output_offset, chunk_dim, route_count);
                        CUDA_CHECK(cudaGetLastError());

                        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(input_dim), static_cast<int>(route_count), static_cast<int>(chunk_dim),
                            &alpha, (const nv_bfloat16 *) weight_chunk, CUDA_R_16BF, static_cast<int>(input_dim),
                            gathered_bf16.ptr, CUDA_R_16BF, static_cast<int>(chunk_dim),
                            beta, input_gradient.ptr, CUDA_R_32F, static_cast<int>(input_dim),
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    } else {
                        const float * weights_f32_ptr;
                        if (cached_weights_f32 != nullptr) {
                            weights_f32_ptr = cached_weights_f32 + output_offset * input_dim;
                        } else {
                            to_fp32(weight_chunk, weights_f32.ptr, input_dim * chunk_dim, stream);
                            weights_f32_ptr = weights_f32.ptr;
                        }

                        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(input_dim), static_cast<int>(route_count), static_cast<int>(chunk_dim),
                            &alpha, weights_f32_ptr, static_cast<int>(input_dim),
                            gathered.ptr + output_offset, static_cast<int>(output_dim),
                            beta, input_gradient.ptr, static_cast<int>(input_dim)));
                    }
                }
            }

            const dim3 scatter_grid((input_dim + 255) / 256, route_count, 1);
            mul_mat_id_back_scatter_kernel<<<scatter_grid, 256, 0, stream>>>(
                input_gradient.ptr, route_pairs, dst_ptr, input_dim, n_used, n_slots,
                dst_nb1, dst_nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}
