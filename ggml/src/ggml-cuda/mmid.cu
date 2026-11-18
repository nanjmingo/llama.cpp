#include "common.cuh"
#include "convert.cuh"
#include "f16-scale.cuh"
#include "mmid.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

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
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse) {
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
                // -1 is used for an unused route and must not shift the
                // compact ranges of all following experts.
                nex_prev += expert_used >= 0 && expert_used < expert;
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
            // Match the generic path: ignore unused routes when computing
            // the expert's compact offset.
            nex_prev += expert_used >= 0 && expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = ggml_cuda_shfl_up(it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += ggml_cuda_shfl(it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
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

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), it_compact);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + it_compact);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
    }
}


template<typename compute_t>
static __global__ void mul_mat_id_back_gather_grouped_kernel(
        const float * __restrict__ grad,
        const int32_t * __restrict__ pairs,
        const ggml_cuda_mm_id_tile_spec * __restrict__ expert_ranges,
        compute_t * __restrict__ gathered,
        const int expert_start,
        const int64_t route_offset,
        const int64_t chunk_dim,
        const int64_t batch_routes,
        const int64_t n_used,
        const int64_t output_offset,
        const int64_t grad_nb0,
        const int64_t grad_nb1,
        const int64_t grad_nb2,
        const float * scale,
        const int64_t n_elements) {
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t elements_per_expert = batch_routes * chunk_dim;

    for (int64_t i = i0; i < n_elements; i += stride) {
        const int local_expert = static_cast<int>(i / elements_per_expert);
        const int64_t local = i - (int64_t) local_expert * elements_per_expert;
        const int64_t route_in_batch = local / chunk_dim;
        const int64_t row = local - route_in_batch * chunk_dim;
        const ggml_cuda_mm_id_tile_spec spec = expert_ranges[expert_start + local_expert];
        const int64_t expert_routes = spec.col_high - spec.col_low;

        float value = 0.0f;
        if (route_offset + route_in_batch < expert_routes) {
            const int32_t flat = pairs[spec.col_low + route_offset + route_in_batch];
            const int64_t used = flat % n_used;
            const int64_t token = flat / n_used;
            value = grad[used * grad_nb1 + token * grad_nb2 + (output_offset + row) * grad_nb0];
            if constexpr (std::is_same_v<compute_t, half>) {
                if (scale != nullptr) {
                    value *= *scale;
                }
            }
        }

        if constexpr (std::is_same_v<compute_t, half>) {
            gathered[i] = __float2half_rn(value);
        } else if constexpr (std::is_same_v<compute_t, nv_bfloat16>) {
            gathered[i] = __float2bfloat16(value);
        } else {
            gathered[i] = value;
        }
    }
}

static __global__ void mmid_finalize_ranges_kernel(
        const ggml_cuda_mm_id_tile_spec * tile_spec,
        ggml_cuda_mm_id_tile_spec * expert_ranges,
        int32_t * max_expert_routes,
        const int n_experts) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= n_experts) {
        return;
    }

    const ggml_cuda_mm_id_tile_spec tail = tile_spec[1 + expert];
    const ggml_cuda_mm_id_tile_spec seam = tile_spec[0];
    int32_t route_begin = 0x7fffffff;
    int32_t route_end = 0;
    bool found = false;

    if (tail.expert == expert) {
        route_begin = tail.col_low;
        route_end = tail.col_high;
        found = true;
    }
    if (seam.expert == expert) {
        route_begin = found ? min(route_begin, seam.col_low) : seam.col_low;
        route_end = found ? max(route_end, seam.col_high) : seam.col_high;
        found = true;
    }
    if (!found) {
        route_begin = 0;
        route_end = 0;
    }

    expert_ranges[expert] = { expert, route_begin, route_end, 0 };
    atomicMax(max_expert_routes, route_end - route_begin);
}

void ggml_cuda_finalize_mm_ids_ranges(
        const ggml_cuda_mm_id_tile_spec * tile_spec,
        ggml_cuda_mm_id_tile_spec * expert_ranges,
        int32_t * max_expert_routes,
        const int n_experts,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (n_experts + threads - 1) / threads;
    mmid_finalize_ranges_kernel<<<blocks, threads, 0, stream>>>(
        tile_spec, expert_ranges, max_expert_routes, n_experts);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void mul_mat_id_back_scatter_grouped_kernel(
        const float * __restrict__ input_gradient,
        const int32_t * __restrict__ pairs,
        const ggml_cuda_mm_id_tile_spec * __restrict__ expert_ranges,
        float * __restrict__ dst,
        float * __restrict__ route_gradient,
        const int expert_start,
        const int64_t route_offset,
        const int64_t batch_routes,
        const int64_t input_dim,
        const int64_t n_used,
        const int64_t n_slots,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const float * scale,
        const int64_t n_elements) {
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t elements_per_expert = batch_routes * input_dim;

    for (int64_t i = i0; i < n_elements; i += stride) {
        const int local_expert = static_cast<int>(i / elements_per_expert);
        const int64_t local = i - (int64_t) local_expert * elements_per_expert;
        const int64_t route_in_batch = local / input_dim;
        const int64_t col = local - route_in_batch * input_dim;
        const ggml_cuda_mm_id_tile_spec spec = expert_ranges[expert_start + local_expert];
        const int64_t expert_routes = spec.col_high - spec.col_low;
        if (route_offset + route_in_batch >= expert_routes) {
            continue;
        }

        const int64_t sorted_route = spec.col_low + route_offset + route_in_batch;
        const int64_t flat = pairs[sorted_route];
        const int64_t used = flat % n_used;
        const int64_t token = flat / n_used;
        const int64_t slot = used % n_slots;
        const float value = input_gradient[i] * (scale == nullptr ? 1.0f : *scale);

        if (route_gradient != nullptr) {
            route_gradient[sorted_route * input_dim + col] = value;
        } else {
            float * dst_route = dst + slot * dst_nb1 + token * dst_nb2;
            if (n_slots == n_used) {
                // Each token/slot has at most one route, so no atomic is needed.
                dst_route[col] = value;
            } else {
                atomicAdd(dst_route + col, value);
            }
        }
    }
}

static __global__ void mul_mat_id_back_reduce_routes_kernel(
        const float * __restrict__ route_gradient,
        const int32_t * __restrict__ route_to_sorted,
        float * __restrict__ dst,
        const int64_t input_dim,
        const int64_t n_used,
        const int64_t n_slots,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t total_elements) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements; i += stride) {
        const int64_t token = i / (n_slots * input_dim);
        const int64_t slot = (i / input_dim) % n_slots;
        const int64_t col = i % input_dim;
        float sum = 0.0f;

        for (int64_t used = slot; used < n_used; used += n_slots) {
            const int64_t flat = token * n_used + used;
            const int32_t sorted_route = route_to_sorted[flat];
            if (sorted_route >= 0) {
                sum += route_gradient[(int64_t) sorted_route * input_dim + col];
            }
        }

        float * dst_route = dst + slot * dst_nb1 + token * dst_nb2;
        dst_route[col] = sum;
    }
}

template<typename lowp_t>
static __global__ void mul_mat_id_back_convert_weights_f32_kernel(
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
    GGML_ASSERT(n_used > 0 && n_tokens >= 0 && n_experts > 0);
    GGML_ASSERT(input_dim > 0 && output_dim > 0 && n_slots > 0);
    GGML_ASSERT(n_used <= std::numeric_limits<int>::max());
    GGML_ASSERT(n_tokens <= std::numeric_limits<int>::max());
    GGML_ASSERT(n_experts <= std::numeric_limits<int>::max());
    GGML_ASSERT(n_tokens == 0 || n_used <= std::numeric_limits<int64_t>::max() / n_tokens);
    const int64_t n_routes = n_used * n_tokens;
    GGML_ASSERT(n_routes <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(input_dim <= std::numeric_limits<int>::max());
    GGML_ASSERT(output_dim <= std::numeric_limits<int>::max());
    GGML_ASSERT(ids->nb[1] / sizeof(int32_t) <= std::numeric_limits<int>::max());

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    if (n_routes == 0) {
        return;
    }

    ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool());
    const bool ids_on_device = ids->buffer && !ggml_backend_buffer_is_host(ids->buffer);
    const int32_t * ids_ptr = (const int32_t *) ids->data;
    if (!ids_on_device) {
        ids_device.alloc(ggml_nbytes(ids) / sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(ids_device.ptr, ids->data, ggml_nbytes(ids), cudaMemcpyHostToDevice, stream));
        ids_ptr = ids_device.ptr;
    }

    ggml_cuda_pool_alloc<int32_t> pairs(ctx.pool(), static_cast<size_t>(n_routes));
    ggml_cuda_pool_alloc<int32_t> route_to_sorted(ctx.pool(), static_cast<size_t>(n_routes));
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), static_cast<size_t>(n_experts + 1));
    const size_t n_tile_specs = static_cast<size_t>(n_experts + 1);
    CUDA_CHECK(cudaMemsetAsync(tile_spec.ptr, 0xFF,
        n_tile_specs * sizeof(ggml_cuda_mm_id_tile_spec), stream));
    CUDA_CHECK(cudaMemsetAsync(route_to_sorted.ptr, 0xFF,
        static_cast<size_t>(n_routes) * sizeof(int32_t), stream));

    const int si1 = static_cast<int>(ids->nb[1] / sizeof(int32_t));
    ggml_cuda_launch_mm_ids_helper(
        ids_ptr,
        route_to_sorted.ptr,
        pairs.ptr,
        tile_spec.ptr,
        static_cast<int>(n_experts),
        static_cast<int>(n_tokens),
        static_cast<int>(n_used),
        static_cast<int>(n_used),
        static_cast<int>(n_routes),
        si1,
        static_cast<int>(n_used),
        true,
        stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> expert_ranges(ctx.pool(), static_cast<size_t>(n_experts));
    ggml_cuda_pool_alloc<int32_t> max_expert_routes_device(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(max_expert_routes_device.ptr, 0, sizeof(int32_t), stream));
    ggml_cuda_finalize_mm_ids_ranges(
        tile_spec.ptr, expert_ranges.ptr, max_expert_routes_device.ptr,
        static_cast<int>(n_experts), stream);

    int32_t max_expert_routes = 0;
    CUDA_CHECK(cudaMemcpyAsync(&max_expert_routes, max_expert_routes_device.ptr,
        sizeof(max_expert_routes), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    GGML_ASSERT(max_expert_routes >= 0 && max_expert_routes <= n_routes);
    if (max_expert_routes == 0) {
        return;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec compute_type = ggml_prec_get_src(dst, 1);
    const bool use_fp16 = compute_type == GGML_PREC_F16
        ? fast_fp16_hardware_available(cc)
        : compute_type == GGML_PREC_UNDEFINED && as->type != GGML_TYPE_F32 && as->type != GGML_TYPE_BF16 &&
            fast_fp16_hardware_available(cc);
    const bool use_bf16 = compute_type == GGML_PREC_BF16
        ? fast_bf16_hardware_available(cc)
        : compute_type == GGML_PREC_UNDEFINED && as->type == GGML_TYPE_BF16 &&
            fast_bf16_hardware_available(cc);
    const bool use_fp32 = !use_fp16 && !use_bf16;
    const to_fp16_cuda_t to_fp16 = use_fp16 && as->type != GGML_TYPE_F16
        ? ggml_get_to_fp16_cuda(as->type) : nullptr;
    const to_bf16_cuda_t to_bf16 = use_bf16 && as->type != GGML_TYPE_BF16
        ? ggml_get_to_bf16_cuda(as->type) : nullptr;
    const bool use_fp32_staging =
        (use_fp16 && as->type != GGML_TYPE_F16 && to_fp16 == nullptr) ||
        (use_bf16 && as->type != GGML_TYPE_BF16 && to_bf16 == nullptr);
    const to_fp32_cuda_t to_fp32 = (use_fp32 && as->type != GGML_TYPE_F32) || use_fp32_staging
        ? ggml_get_to_fp32_cuda(as->type) : nullptr;
    GGML_ASSERT(!use_fp16 || as->type == GGML_TYPE_F16 || to_fp16 != nullptr || to_fp32 != nullptr);
    GGML_ASSERT(!use_bf16 || as->type == GGML_TYPE_BF16 || to_bf16 != nullptr || to_fp32 != nullptr);
    GGML_ASSERT(!use_fp32 || as->type == GGML_TYPE_F32 || to_fp32 != nullptr);

    ggml_cuda_pool_alloc<unsigned int> grad_absmax(ctx.pool());
    ggml_cuda_pool_alloc<float> grad_scales(ctx.pool());
    if (use_fp16) {
        grad_absmax.alloc(1);
        grad_scales.alloc(2);
        ggml_cuda_f16_scale_init(grad, grad_absmax.ptr, grad_scales.ptr, stream);
    }

    constexpr int64_t max_batch_routes = 64;
    constexpr int64_t max_weight_chunk_dim = 8192;
    constexpr size_t max_workspace_bytes = 64ull * 1024 * 1024;
    constexpr size_t max_route_gradient_bytes = 256ull * 1024 * 1024;
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    const size_t workspace_budget = std::min(max_workspace_bytes, std::max<size_t>(1, free_memory / 16));
    const size_t compute_type_size = use_fp16 || use_bf16 ? sizeof(half) : sizeof(float);
    GGML_ASSERT(static_cast<size_t>(input_dim) <= std::numeric_limits<size_t>::max() / compute_type_size);
    const size_t bytes_per_output = static_cast<size_t>(input_dim) * compute_type_size;
    const int64_t memory_limited_chunk = static_cast<int64_t>(workspace_budget / bytes_per_output);
    const int64_t max_chunk_dim = std::max<int64_t>(1,
        std::min<int64_t>(max_weight_chunk_dim, memory_limited_chunk));
    const int64_t weight_chunk_capacity = std::min<int64_t>(output_dim, max_chunk_dim);
    GGML_ASSERT(weight_chunk_capacity > 0);

    const bool direct_weights = (use_fp32 && as->type == GGML_TYPE_F32) ||
        (use_fp16 && as->type == GGML_TYPE_F16) || (use_bf16 && as->type == GGML_TYPE_BF16);
    const size_t weight_elements_per_expert = static_cast<size_t>(input_dim) * static_cast<size_t>(weight_chunk_capacity);
    const size_t weight_bytes_per_expert = direct_weights ? 0 : weight_elements_per_expert * compute_type_size;
    const size_t gathered_bytes_per_expert = static_cast<size_t>(max_batch_routes) *
        static_cast<size_t>(weight_chunk_capacity) * compute_type_size;
    const size_t input_gradient_bytes_per_expert = static_cast<size_t>(max_batch_routes) *
        static_cast<size_t>(input_dim) * sizeof(float);
    const size_t workspace_per_expert = weight_bytes_per_expert + gathered_bytes_per_expert + input_gradient_bytes_per_expert;
    const size_t expert_batch_size = std::max<size_t>(1, std::min<size_t>(
        static_cast<size_t>(n_experts), workspace_budget / std::max<size_t>(1, workspace_per_expert)));

    const size_t group_weight_elements = weight_elements_per_expert * expert_batch_size;
    const size_t group_gather_elements = static_cast<size_t>(max_batch_routes) *
        static_cast<size_t>(weight_chunk_capacity) * expert_batch_size;
    const size_t group_input_gradient_elements = static_cast<size_t>(max_batch_routes) *
        static_cast<size_t>(input_dim) * expert_batch_size;
    ggml_cuda_pool_alloc<float> weights_f32(ctx.pool());
    ggml_cuda_pool_alloc<half> weights_f16(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> weights_bf16(ctx.pool());
    if (use_fp32 && as->type != GGML_TYPE_F32) {
        weights_f32.alloc(group_weight_elements);
    } else if (use_fp16 && as->type != GGML_TYPE_F16) {
        weights_f16.alloc(group_weight_elements);
        if (use_fp32_staging) {
            weights_f32.alloc(weight_elements_per_expert);
        }
    } else if (use_bf16 && as->type != GGML_TYPE_BF16) {
        weights_bf16.alloc(group_weight_elements);
        if (use_fp32_staging) {
            weights_f32.alloc(weight_elements_per_expert);
        }
    }

    ggml_cuda_pool_alloc<float> gathered_f32(ctx.pool());
    ggml_cuda_pool_alloc<half> gathered_f16(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> gathered_bf16(ctx.pool());
    if (use_fp32) {
        gathered_f32.alloc(group_gather_elements);
    } else if (use_fp16) {
        gathered_f16.alloc(group_gather_elements);
    } else {
        gathered_bf16.alloc(group_gather_elements);
    }
    ggml_cuda_pool_alloc<float> input_gradient(ctx.pool(), group_input_gradient_elements);

    const bool route_gradient_size_valid = static_cast<size_t>(n_routes) <=
        std::numeric_limits<size_t>::max() / static_cast<size_t>(input_dim) / sizeof(float);
    const size_t route_gradient_elements = route_gradient_size_valid ?
        static_cast<size_t>(n_routes) * static_cast<size_t>(input_dim) : 0;
    const size_t route_gradient_bytes = route_gradient_elements * sizeof(float);
    const bool use_route_gradient = n_slots < n_used && route_gradient_size_valid &&
        route_gradient_bytes <= std::min(max_route_gradient_bytes, free_memory / 8);
    ggml_cuda_pool_alloc<float> route_gradient(ctx.pool());
    if (use_route_gradient) {
        route_gradient.alloc(route_gradient_elements);
    }

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    }
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;
    const int threads = 256;
    const int64_t grad_nb0 = grad->nb[0] / sizeof(float);
    const int64_t grad_nb1 = grad->nb[1] / sizeof(float);
    const int64_t grad_nb2 = grad->nb[2] / sizeof(float);
    const int64_t dst_nb1 = dst->nb[1] / sizeof(float);
    const int64_t dst_nb2 = dst->nb[2] / sizeof(float);
    const int64_t n_route_tiles = (max_expert_routes + max_batch_routes - 1) / max_batch_routes;

    for (int expert_start = 0; expert_start < n_experts; expert_start += static_cast<int>(expert_batch_size)) {
        const int group_count = std::min<int>(static_cast<int>(expert_batch_size), static_cast<int>(n_experts) - expert_start);
        const size_t group_weight_stride = weight_elements_per_expert;

        for (int64_t route_tile = 0; route_tile < n_route_tiles; ++route_tile) {
            const int64_t route_offset = route_tile * max_batch_routes;
            for (int64_t output_offset = 0; output_offset < output_dim; output_offset += weight_chunk_capacity) {
                const int64_t chunk_dim = std::min(weight_chunk_capacity, output_dim - output_offset);
                const size_t weight_elements = static_cast<size_t>(input_dim) * static_cast<size_t>(chunk_dim);
                const void * weights_ptr = nullptr;
                int64_t weight_batch_stride = 0;
                cudaDataType_t data_type = CUDA_R_32F;

                if (use_fp32) {
                    if (as->type == GGML_TYPE_F32) {
                        weights_ptr = (const char *) as->data + expert_start * as->nb[2] + output_offset * as->nb[1];
                        weight_batch_stride = static_cast<int64_t>(as->nb[2] / sizeof(float));
                    } else {
                        for (int local_expert = 0; local_expert < group_count; ++local_expert) {
                            const int expert = expert_start + local_expert;
                            const char * weight_chunk = (const char *) as->data + expert * as->nb[2] + output_offset * as->nb[1];
                            to_fp32(weight_chunk, weights_f32.ptr + local_expert * group_weight_stride, weight_elements, stream);
                        }
                        weights_ptr = weights_f32.ptr;
                        weight_batch_stride = static_cast<int64_t>(group_weight_stride);
                    }
                } else if (use_fp16) {
                    data_type = CUDA_R_16F;
                    if (as->type == GGML_TYPE_F16) {
                        weights_ptr = (const char *) as->data + expert_start * as->nb[2] + output_offset * as->nb[1];
                        weight_batch_stride = static_cast<int64_t>(as->nb[2] / sizeof(half));
                    } else {
                        for (int local_expert = 0; local_expert < group_count; ++local_expert) {
                            const int expert = expert_start + local_expert;
                            const char * weight_chunk = (const char *) as->data + expert * as->nb[2] + output_offset * as->nb[1];
                            half * weight_dst = weights_f16.ptr + local_expert * group_weight_stride;
                            if (to_fp16 != nullptr) {
                                to_fp16(weight_chunk, weight_dst, weight_elements, stream);
                            } else {
                                GGML_ASSERT(to_fp32 != nullptr);
                                to_fp32(weight_chunk, weights_f32.ptr, weight_elements, stream);
                                const int blocks = static_cast<int>((weight_elements + threads - 1) / threads);
                                mul_mat_id_back_convert_weights_f32_kernel<half><<<blocks, threads, 0, stream>>>(
                                    weights_f32.ptr, weight_dst, weight_elements);
                                CUDA_CHECK(cudaGetLastError());
                            }
                        }
                        weights_ptr = weights_f16.ptr;
                        weight_batch_stride = static_cast<int64_t>(group_weight_stride);
                    }
                } else {
                    data_type = CUDA_R_16BF;
                    if (as->type == GGML_TYPE_BF16) {
                        weights_ptr = (const char *) as->data + expert_start * as->nb[2] + output_offset * as->nb[1];
                        weight_batch_stride = static_cast<int64_t>(as->nb[2] / sizeof(nv_bfloat16));
                    } else {
                        for (int local_expert = 0; local_expert < group_count; ++local_expert) {
                            const int expert = expert_start + local_expert;
                            const char * weight_chunk = (const char *) as->data + expert * as->nb[2] + output_offset * as->nb[1];
                            nv_bfloat16 * weight_dst = weights_bf16.ptr + local_expert * group_weight_stride;
                            if (to_bf16 != nullptr) {
                                to_bf16(weight_chunk, weight_dst, weight_elements, stream);
                            } else {
                                GGML_ASSERT(to_fp32 != nullptr);
                                to_fp32(weight_chunk, weights_f32.ptr, weight_elements, stream);
                                const int blocks = static_cast<int>((weight_elements + threads - 1) / threads);
                                mul_mat_id_back_convert_weights_f32_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
                                    weights_f32.ptr, weight_dst, weight_elements);
                                CUDA_CHECK(cudaGetLastError());
                            }
                        }
                        weights_ptr = weights_bf16.ptr;
                        weight_batch_stride = static_cast<int64_t>(group_weight_stride);
                    }
                }

                void * gathered_ptr = use_fp32 ? (void *) gathered_f32.ptr :
                    use_fp16 ? (void *) gathered_f16.ptr : (void *) gathered_bf16.ptr;
                const int64_t gathered_stride = static_cast<int64_t>(chunk_dim * max_batch_routes);
                const int64_t output_stride = static_cast<int64_t>(input_dim * max_batch_routes);
                const int64_t gather_elements = static_cast<int64_t>(group_count) * max_batch_routes * chunk_dim;
                const int gather_blocks = static_cast<int>((gather_elements + threads - 1) / threads);

                if (use_fp32) {
                    mul_mat_id_back_gather_grouped_kernel<float><<<gather_blocks, threads, 0, stream>>>(
                        (const float *) grad->data, pairs.ptr, expert_ranges.ptr, (float *) gathered_ptr,
                        expert_start, route_offset, chunk_dim, max_batch_routes, n_used,
                        output_offset, grad_nb0, grad_nb1, grad_nb2, nullptr, gather_elements);
                } else if (use_fp16) {
                    mul_mat_id_back_gather_grouped_kernel<half><<<gather_blocks, threads, 0, stream>>>(
                        (const float *) grad->data, pairs.ptr, expert_ranges.ptr, (half *) gathered_ptr,
                        expert_start, route_offset, chunk_dim, max_batch_routes, n_used,
                        output_offset, grad_nb0, grad_nb1, grad_nb2, grad_scales.ptr, gather_elements);
                } else {
                    mul_mat_id_back_gather_grouped_kernel<nv_bfloat16><<<gather_blocks, threads, 0, stream>>>(
                        (const float *) grad->data, pairs.ptr, expert_ranges.ptr, (nv_bfloat16 *) gathered_ptr,
                        expert_start, route_offset, chunk_dim, max_batch_routes, n_used,
                        output_offset, grad_nb0, grad_nb1, grad_nb2, nullptr, gather_elements);
                }
                CUDA_CHECK(cudaGetLastError());

                const float * beta = output_offset == 0 ? &beta_zero : &beta_one;
                if (use_fp32) {
                    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        static_cast<int>(input_dim), static_cast<int>(max_batch_routes), static_cast<int>(chunk_dim),
                        &alpha, (const float *) weights_ptr, static_cast<int>(input_dim), weight_batch_stride,
                        (const float *) gathered_ptr, static_cast<int>(chunk_dim), gathered_stride,
                        beta, input_gradient.ptr, static_cast<int>(input_dim), output_stride, group_count));
                } else {
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        static_cast<int>(input_dim), static_cast<int>(max_batch_routes), static_cast<int>(chunk_dim),
                        &alpha, weights_ptr, data_type, static_cast<int>(input_dim), weight_batch_stride,
                        gathered_ptr, data_type, static_cast<int>(chunk_dim), gathered_stride,
                        beta, input_gradient.ptr, CUDA_R_32F, static_cast<int>(input_dim), output_stride,
                        group_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
            }

            const int64_t scatter_elements = static_cast<int64_t>(group_count) * max_batch_routes * input_dim;
            const int scatter_blocks = static_cast<int>((scatter_elements + threads - 1) / threads);
            mul_mat_id_back_scatter_grouped_kernel<<<scatter_blocks, threads, 0, stream>>>(
                input_gradient.ptr, pairs.ptr, expert_ranges.ptr, (float *) dst->data,
                use_route_gradient ? route_gradient.ptr : nullptr,
                expert_start, route_offset, max_batch_routes, input_dim,
                n_used, n_slots, dst_nb1, dst_nb2,
                use_fp16 ? grad_scales.ptr + 1 : nullptr, scatter_elements);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (use_route_gradient) {
        const int64_t total_elements = input_dim * n_slots * n_tokens;
        const int reduce_blocks = static_cast<int>(std::min<int64_t>(65535,
            (total_elements + threads - 1) / threads));
        mul_mat_id_back_reduce_routes_kernel<<<reduce_blocks, threads, 0, stream>>>(
            route_gradient.ptr, route_to_sorted.ptr, (float *) dst->data,
            input_dim, n_used, n_slots, dst_nb1, dst_nb2, total_elements);
        CUDA_CHECK(cudaGetLastError());
    }

    if (pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        CUBLAS_CHECK(cublasSetPointerMode(handle, pointer_mode));
    }
}
