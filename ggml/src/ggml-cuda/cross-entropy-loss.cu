#include "common.cuh"
#include "cross-entropy-loss.cuh"
#include "sum.cuh"

#include <cmath>
#include <cstdint>

template <bool use_shared>
static __global__ void cross_entropy_loss_f32(
        const float * __restrict__ logits, const float * __restrict__ labels, float * __restrict__ dst, const int nclasses, const int k) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x)*nclasses;
    labels += int64_t(blockIdx.x)*nclasses;

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        max_logit = fmaxf(max_logit, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    max_logit = warp_reduce_max(max_logit);

    // Calculate log(softmax(logits)) which is just logits - max:
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        sum += expf(logit_i - max_logit);
    }
    sum = warp_reduce_sum(sum);
    sum = logf(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        loss += (logit_i - max_logit - sum) * labels[i];
    }
    loss = -warp_reduce_sum(loss) / (float)k;

    if (threadIdx.x != 0) {
        return;
    }

    dst[blockIdx.x] = loss;
}

template <bool use_shared>
static __global__ void cross_entropy_loss_back_f32(
        const float * __restrict__ grad, const float * __restrict__ logits, const float * __restrict__ labels,
        float * __restrict__ dst, const int nclasses) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x)*nclasses;
    labels += int64_t(blockIdx.x)*nclasses;
    dst    += int64_t(blockIdx.x)*nclasses;

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        maxval = fmaxf(maxval, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    float label_sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf((use_shared ? tmp[i] : logits[i]) - maxval);
        sum += val;
        label_sum += labels[i];

        if (use_shared) {
            tmp[i] = val;
        } else {
            dst[i] = val;
        }
    }
    sum = warp_reduce_sum(sum);
    label_sum = warp_reduce_sum(label_sum);
    const float sm_scale = label_sum/sum;

    const float d_by_nrows = *grad/gridDim.x;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = use_shared ? tmp[i] : dst[i];
        dst[i] = (val*sm_scale - labels[i])*d_by_nrows;
    }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    if (nbytes_shared <= smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_f32<true>), smpbo);
        cross_entropy_loss_f32<true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);
    } else {
        cross_entropy_loss_f32<false><<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];

    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    const float * src1f_d = (const float *) src1f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_back_f32<true>), smpbo);
        cross_entropy_loss_back_f32<true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(grad_d, src0f_d, src1f_d, dst_d, ne00);
    } else {
        cross_entropy_loss_back_f32<false><<<blocks_num, blocks_dim, 0, stream>>>(grad_d, src0f_d, src1f_d, dst_d, ne00);
    }
}


static __global__ void cross_entropy_loss_sparse_f32(
        const float * __restrict__ logits,
        const int32_t * __restrict__ labels,
        const float * __restrict__ weights,
        float * __restrict__ dst,
        const int nclasses,
        const int nrows) {
    const int row = blockIdx.x;
    const int label = labels[row];
    const float weight = weights[row];

    if (label < -1 || label >= nclasses || !isfinite(weight) || weight < 0.0f) {
        if (threadIdx.x == 0) dst[row] = NAN;
        return;
    }
    if (label == -1 || weight == 0.0f) {
        if (threadIdx.x == 0) dst[row] = 0.0f;
        return;
    }

    logits += int64_t(row)*nclasses;

    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        max_logit = fmaxf(max_logit, logits[i]);
    }
    max_logit = warp_reduce_max(max_logit);

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        sum_exp += expf(logits[i] - max_logit);
    }
    sum_exp = warp_reduce_sum(sum_exp);

    const float loss = -weight * (logits[label] - max_logit - logf(sum_exp)) / (float) nrows;
    if (threadIdx.x == 0) dst[row] = loss;
}

static __global__ void cross_entropy_loss_sparse_back_f32(
        const float * __restrict__ grad,
        const float * __restrict__ logits,
        const int32_t * __restrict__ labels,
        const float * __restrict__ weights,
        float * __restrict__ dst,
        const int nclasses,
        const int nrows) {
    const int row = blockIdx.x;
    const int label = labels[row];
    const float weight = weights[row];

    logits += int64_t(row)*nclasses;
    dst    += int64_t(row)*nclasses;

    if (label < -1 || label >= nclasses || !isfinite(weight) || weight < 0.0f) {
        for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
            dst[i] = NAN;
        }
        return;
    }
    if (label == -1 || weight == 0.0f) {
        for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
            dst[i] = 0.0f;
        }
        return;
    }

    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        max_logit = fmaxf(max_logit, logits[i]);
    }
    max_logit = warp_reduce_max(max_logit);

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        sum_exp += expf(logits[i] - max_logit);
    }
    sum_exp = warp_reduce_sum(sum_exp);

    const float scale = *grad * weight / (float) nrows;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float probability = expf(logits[i] - max_logit) / sum_exp;
        dst[i] = (probability - (i == label ? 1.0f : 0.0f)) * scale;
    }
}

void ggml_cuda_cross_entropy_loss_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits  = dst->src[0];
    const ggml_tensor * labels  = dst->src[1];
    const ggml_tensor * weights = dst->src[2];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_nelements(labels) == ggml_nrows(logits));
    GGML_ASSERT(ggml_nelements(weights) == ggml_nrows(logits));
    GGML_ASSERT(ggml_is_scalar(dst) && dst->type == GGML_TYPE_F32);

    const int nclasses = (int) logits->ne[0];
    const int nrows = (int) ggml_nrows(logits);
    const float * logits_d = (const float *) logits->data;
    const int32_t * labels_d = (const int32_t *) labels->data;
    const float * weights_d = (const float *) weights->data;
    float * dst_d = (float *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    ggml_cuda_pool_alloc<float> row_losses(pool, blocks_num.x);

    cross_entropy_loss_sparse_f32<<<blocks_num, blocks_dim, 0, stream>>>(
        logits_d, labels_d, weights_d, row_losses.ptr, nclasses, nrows);
    CUDA_CHECK(cudaGetLastError());
    sum_f32_cuda(pool, row_losses.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_sparse_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad    = dst->src[0];
    const ggml_tensor * logits  = dst->src[1];
    const ggml_tensor * labels  = dst->src[2];
    const ggml_tensor * weights = dst->src[3];

    GGML_ASSERT(ggml_is_scalar(grad) && grad->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(labels->type == GGML_TYPE_I32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(ggml_is_contiguous(labels));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_are_same_shape(logits, dst));
    GGML_ASSERT(ggml_nelements(labels) == ggml_nrows(logits));
    GGML_ASSERT(ggml_nelements(weights) == ggml_nrows(logits));

    const int nclasses = (int) logits->ne[0];
    const int nrows = (int) ggml_nrows(logits);
    const float * grad_d = (const float *) grad->data;
    const float * logits_d = (const float *) logits->data;
    const int32_t * labels_d = (const int32_t *) labels->data;
    const float * weights_d = (const float *) weights->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();
    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    cross_entropy_loss_sparse_back_f32<<<blocks_num, blocks_dim, 0, stream>>>(
        grad_d, logits_d, labels_d, weights_d, dst_d, nclasses, nrows);
    CUDA_CHECK(cudaGetLastError());
}
