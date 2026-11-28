#include "fattn-back.cuh"
#include "f16-scale.cuh"
#include "llamango-ggml.h"

#define FATTN_BACK_MAX_CHUNKS 16
#define FATTN_BACK_WARPS      4

struct fattn_back_params {
    int64_t DK, DV, N, KV, HQ, HK, SEQ, rk2, rk3;
    int64_t q1, q2, q3, k1, k2, k3, v1, v2, v3, o1, o2, o3, m1, m2, m3;
    int64_t m_ne2, m_ne3;
    float scale_qk, scale, softcap, max_bias, m0, m1f;
    uint32_t n_head_log2;
};

template <typename T> static __device__ __forceinline__ float fb_load(const T * p, int64_t i);
template <> __device__ __forceinline__ float fb_load<float>(const float * p, int64_t i) { return p[i]; }
template <> __device__ __forceinline__ float fb_load<half>(const half * p, int64_t i) { return __half2float(p[i]); }

static __device__ __forceinline__ float fb_slope(const fattn_back_params & P, int64_t h) {
    if (P.max_bias <= 0.0f) {
        return 1.0f;
    }
    return h < P.n_head_log2 ? powf(P.m0, h + 1) : powf(P.m1f, 2*(h - P.n_head_log2) + 1);
}

template <typename TK>
static __device__ __forceinline__ float fb_score(const fattn_back_params & P, const float * qreg, const TK * kr, float mv, float & t) {
    const int lane = threadIdx.x;
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        if (e < P.DK) {
            dot += qreg[c]*fb_load(kr, e);
        }
    }
    dot = warp_reduce_sum(dot)*P.scale_qk;
    if (P.softcap != 0.0f) {
        t = tanhf(dot);
        dot = P.softcap*t;
    }
    return dot + mv;
}

template <typename TK, typename TV>
static __global__ void fattn_back_rows_kernel(const fattn_back_params P,
        const float * q, const TK * k, const TV * v, const half * mask, const float * sinks,
        const float * o, const float * d, float * dq, float * dsinks, float * Lbuf, float * Dbuf) {
    const int64_t r = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= P.N*P.HQ*P.SEQ) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t iq1 = r % P.N;
    const int64_t iq2 = (r / P.N) % P.HQ;
    const int64_t iq3 = r / (P.N*P.HQ);
    const int64_t ik2 = iq2 / P.rk2;
    const int64_t ik3 = iq3 / P.rk3;

    const float * qr = q + iq1*P.q1 + iq2*P.q2 + iq3*P.q3;
    const float * dr = d + iq2*P.o1 + iq1*P.o2 + iq3*P.o3;
    const float * orow = o + iq2*P.o1 + iq1*P.o2 + iq3*P.o3;
    const half  * mr = mask ? mask + iq1*P.m1 + (iq2 % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 : nullptr;
    const float slope = fb_slope(P, iq2);
    const float sink  = sinks ? sinks[iq2] : -INFINITY;

    float qreg[FATTN_BACK_MAX_CHUNKS], dreg[FATTN_BACK_MAX_CHUNKS], dqreg[FATTN_BACK_MAX_CHUNKS];
    float Dp = 0.0f;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        qreg[c]  = e < P.DK ? qr[e] : 0.0f;
        dreg[c]  = e < P.DV ? dr[e] : 0.0f;
        dqreg[c] = 0.0f;
        Dp += e < P.DV ? dreg[c]*orow[e] : 0.0f;
    }
    const float D = warp_reduce_sum(Dp);

    float M = sink, S = sinks ? 1.0f : 0.0f;
    for (int64_t j = 0; j < P.KV; ++j) {
        const float mv = mr ? slope*__half2float(mr[j]) : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }
        float t;
        const float s = fb_score(P, qreg, k + j*P.k1 + ik2*P.k2 + ik3*P.k3, mv, t);
        if (s > M) {
            S = S*expf(M - s) + 1.0f;
            M = s;
        } else {
            S += expf(s - M);
        }
    }
    const float L = M == -INFINITY ? -INFINITY : M + logf(S);

    if (L != -INFINITY) {
        for (int64_t j = 0; j < P.KV; ++j) {
            const float mv = mr ? slope*__half2float(mr[j]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }
            const TK * kr = k + j*P.k1 + ik2*P.k2 + ik3*P.k3;
            const TV * vr = v + j*P.v1 + ik2*P.v2 + ik3*P.v3;
            float t = 0.0f;
            const float p = expf(fb_score(P, qreg, kr, mv, t) - L);
            float dp = 0.0f;
#pragma unroll
            for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                const int64_t e = lane + c*WARP_SIZE;
                if (e < P.DV) {
                    dp += dreg[c]*fb_load(vr, e);
                }
            }
            dp = warp_reduce_sum(dp);
            float g = p*(dp - D)*P.scale;
            if (P.softcap != 0.0f) {
                g *= 1.0f - t*t;
            }
#pragma unroll
            for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                const int64_t e = lane + c*WARP_SIZE;
                if (e < P.DK) {
                    dqreg[c] += g*fb_load(kr, e);
                }
            }
        }
    }

    float * dqr = dq + ((iq3*P.HQ + iq2)*P.N + iq1)*P.DK;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        if (e < P.DK) {
            dqr[e] = dqreg[c];
        }
    }
    if (lane == 0) {
        Lbuf[r] = L;
        Dbuf[r] = D;
        if (sinks && L != -INFINITY) {
            atomicAdd(dsinks + iq2, -expf(sink - L)*D);
        }
    }
}

template <typename TK, typename TV>
static __global__ void fattn_back_keys_kernel(const fattn_back_params P,
        const float * q, const TK * k, const TV * v, const half * mask,
        const float * d, const float * Lbuf, const float * Dbuf, float * dk, float * dv) {
    const int64_t r = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= P.KV*P.HK*P.SEQ / P.rk3) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t j   = r % P.KV;
    const int64_t ik2 = (r / P.KV) % P.HK;
    const int64_t ik3 = r / (P.KV*P.HK);

    const TK * kr = k + j*P.k1 + ik2*P.k2 + ik3*P.k3;
    const TV * vr = v + j*P.v1 + ik2*P.v2 + ik3*P.v3;
    float vreg[FATTN_BACK_MAX_CHUNKS], dkreg[FATTN_BACK_MAX_CHUNKS], dvreg[FATTN_BACK_MAX_CHUNKS];
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        vreg[c]  = e < P.DV ? fb_load(vr, e) : 0.0f;
        dkreg[c] = 0.0f;
        dvreg[c] = 0.0f;
    }

    for (int64_t iq3 = ik3*P.rk3; iq3 < (ik3 + 1)*P.rk3; ++iq3) {
        for (int64_t iq2 = ik2*P.rk2; iq2 < (ik2 + 1)*P.rk2; ++iq2) {
            const float slope = fb_slope(P, iq2);
            for (int64_t iq1 = 0; iq1 < P.N; ++iq1) {
                const int64_t row = (iq3*P.HQ + iq2)*P.N + iq1;
                const float L = Lbuf[row];
                if (L == -INFINITY) {
                    continue;
                }
                const float mv = mask ? slope*__half2float(mask[iq1*P.m1 + (iq2 % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 + j]) : 0.0f;
                if (mv == -INFINITY) {
                    continue;
                }
                const float * qr = q + iq1*P.q1 + iq2*P.q2 + iq3*P.q3;
                const float * dr = d + iq2*P.o1 + iq1*P.o2 + iq3*P.o3;
                float qreg[FATTN_BACK_MAX_CHUNKS];
#pragma unroll
                for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                    const int64_t e = lane + c*WARP_SIZE;
                    qreg[c] = e < P.DK ? qr[e] : 0.0f;
                }
                float t = 0.0f;
                const float p = expf(fb_score(P, qreg, kr, mv, t) - L);
                float dp = 0.0f;
#pragma unroll
                for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                    const int64_t e = lane + c*WARP_SIZE;
                    if (e < P.DV) {
                        dp += dr[e]*vreg[c];
                    }
                }
                dp = warp_reduce_sum(dp);
                float g = p*(dp - Dbuf[row])*P.scale;
                if (P.softcap != 0.0f) {
                    g *= 1.0f - t*t;
                }
#pragma unroll
                for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
                    const int64_t e = lane + c*WARP_SIZE;
                    if (e < P.DK) {
                        dkreg[c] += g*qreg[c];
                    }
                    if (e < P.DV) {
                        dvreg[c] += p*dr[e];
                    }
                }
            }
        }
    }

    float * dkr = dk + ((ik3*P.HK + ik2)*P.KV + j)*P.DK;
    float * dvr = dv + ((ik3*P.HK + ik2)*P.KV + j)*P.DV;
#pragma unroll
    for (int c = 0; c < FATTN_BACK_MAX_CHUNKS; ++c) {
        const int64_t e = lane + c*WARP_SIZE;
        if (e < P.DK) {
            dkr[e] = dkreg[c];
        }
        if (e < P.DV) {
            dvr[e] = dvreg[c];
        }
    }
}

template <typename TK, typename TV>
static void launch_fattn_back(ggml_backend_cuda_context & ctx, const fattn_back_params & P, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * o = dst->src[5];
    const ggml_tensor * d = dst->src[6];

    float * out = (float *) dst->data;
    float * dq = out;
    float * dk = out + ggml_flash_attn_back_offset(dst, 1);
    float * dv = out + ggml_flash_attn_back_offset(dst, 2);
    float * ds = out + ggml_flash_attn_back_offset(dst, 3);

    const int64_t n_rows = P.N*P.HQ*P.SEQ;
    ggml_cuda_pool_alloc<float> Lbuf(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<float> Dbuf(ctx.pool(), n_rows);

    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(ds, 0, P.HQ*sizeof(float), stream));

    const dim3 block(WARP_SIZE, FATTN_BACK_WARPS);
    fattn_back_rows_kernel<TK, TV><<<(n_rows + FATTN_BACK_WARPS - 1)/FATTN_BACK_WARPS, block, 0, stream>>>(P,
        (const float *) q->data, (const TK *) k->data, (const TV *) v->data, mask ? (const half *) mask->data : nullptr,
        sinks ? (const float *) sinks->data : nullptr, (const float *) o->data, (const float *) d->data,
        dq, ds, Lbuf.get(), Dbuf.get());

    const int64_t n_keys = P.KV*P.HK*(P.SEQ/P.rk3);
    fattn_back_keys_kernel<TK, TV><<<(n_keys + FATTN_BACK_WARPS - 1)/FATTN_BACK_WARPS, block, 0, stream>>>(P,
        (const float *) q->data, (const TK *) k->data, (const TV *) v->data, mask ? (const half *) mask->data : nullptr,
        (const float *) d->data, Lbuf.get(), Dbuf.get(), dk, dv);
}

static __device__ __forceinline__ float fb_block_reduce(float v, bool is_max, float * sh) {
    v = is_max ? warp_reduce_max(v) : warp_reduce_sum(v);
    const int w = threadIdx.x / WARP_SIZE, l = threadIdx.x % WARP_SIZE;
    __syncthreads();
    if (l == 0) {
        sh[w] = v;
    }
    __syncthreads();
    const int nw = blockDim.x / WARP_SIZE;
    v = l < nw ? sh[l] : (is_max ? -INFINITY : 0.0f);
    v = is_max ? warp_reduce_max(v) : warp_reduce_sum(v);
    return v;
}

static __global__ void fattn_back_gemm_rows(const fattn_back_params P, float * S, half * S_f16,
        const half * mask, const float * sinks, const float * o, const float * d, const half * d_f16,
        const float * d_inv_scale, float * Lbuf, float * Dbuf, float * dsinks,
        const int64_t q0, const int64_t R, const int64_t iq3) {
    __shared__ float sh[32];
    const int64_t i = blockIdx.x;
    const int64_t h = blockIdx.y;
    float * srow = S + (h*R + i)*P.KV;
    const half * mrow = mask ? mask + (q0 + i)*P.m1 + (h % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 : nullptr;
    const float slope = fb_slope(P, h);
    const float sink  = sinks ? sinks[h] : -INFINITY;

    float M = sink;
    for (int64_t j = threadIdx.x; j < P.KV; j += blockDim.x) {
        const float mv = mrow ? slope*__half2float(mrow[j]) : 0.0f;
        float s = -INFINITY;
        if (mv != -INFINITY) {
            s = srow[j]*P.scale_qk;
            if (P.softcap != 0.0f) {
                s = P.softcap*tanhf(s);
            }
            s += mv;
        }
        srow[j] = s;
        M = fmaxf(M, s);
    }
    M = fb_block_reduce(M, true, sh);
    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j < P.KV; j += blockDim.x) {
        sum += srow[j] == -INFINITY ? 0.0f : expf(srow[j] - M);
    }
    sum = fb_block_reduce(sum, false, sh);
    if (sinks && M != -INFINITY) {
        sum += expf(sink - M);
    }
    const float L = M == -INFINITY ? -INFINITY : M + logf(sum);
    for (int64_t j = threadIdx.x; j < P.KV; j += blockDim.x) {
        const float p = L == -INFINITY || srow[j] == -INFINITY ? 0.0f : expf(srow[j] - L);
        srow[j] = p;
        if (S_f16) {
            S_f16[(h*R + i)*P.KV + j] = __float2half_rn(p);
        }
    }

    const float * orow = o + h*P.o1 + (q0 + i)*P.o2 + iq3*P.o3;
    const float * drow = d + h*P.o1 + (q0 + i)*P.o2 + iq3*P.o3;
    float Dp = 0.0f;
    const half * drow_f16 = d_f16 ? d_f16 + h*P.o1 + (q0 + i)*P.o2 + iq3*P.o3 : nullptr;
    for (int64_t e = threadIdx.x; e < P.DV; e += blockDim.x) {
        const float de = drow_f16 ? __half2float(drow_f16[e]) : drow[e];
        Dp += orow[e]*de;
    }
    const float D_scaled = fb_block_reduce(Dp, false, sh);
    const float D = d_inv_scale ? D_scaled * *d_inv_scale : D_scaled;
    if (threadIdx.x == 0) {
        Lbuf[h*R + i] = L;

        Dbuf[h*R + i] = D_scaled;
        if (sinks && L != -INFINITY) {
            atomicAdd(dsinks + h, -expf(sink - L)*D);
        }
    }
}

static __global__ void fattn_back_gemm_ds(const fattn_back_params P, const float * S, float * dP,
        half * dP_f16, const float * dP_convert_scale,
        const half * mask, const float * Lbuf, const float * Dbuf,
        const int64_t q0, const int64_t R, const int64_t rows, const int64_t iq3) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= P.KV*R*P.HQ) {
        return;
    }
    const int64_t j = idx % P.KV;
    const int64_t i = (idx / P.KV) % R;
    const int64_t h = idx / (P.KV*R);
    if (i >= rows) {
        return;
    }
    const float p = S[idx];
    float g = 0.0f;
    if (p > 0.0f) {
        g = p*(dP[idx] - Dbuf[h*R + i])*P.scale;
        if (P.softcap != 0.0f) {
            const float mv = mask ? fb_slope(P, h)*__half2float(mask[(q0 + i)*P.m1 + (h % P.m_ne2)*P.m2 + (iq3 % P.m_ne3)*P.m3 + j]) : 0.0f;
            const float t = (logf(p) + Lbuf[h*R + i] - mv)/P.softcap;
            g *= 1.0f - t*t;
        }
    }

    if (dP_f16) {

        dP_f16[idx] = __float2half_rn(g * *dP_convert_scale);
    } else {
        dP[idx] = g;
    }
}

static __global__ void fattn_back_max_v_norm2(
        const half * v, const int64_t DV, const int64_t KV, const int64_t HK, const int64_t SEQ,
        const int64_t s1, const int64_t s2, const int64_t s3, unsigned int * max_norm2_bits) {
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float warp_norm2[WARPS_PER_BLOCK];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t n_rows = KV*HK*SEQ;
    const int64_t row = (int64_t) blockIdx.x*WARPS_PER_BLOCK + warp;
    float norm2 = 0.0f;
    if (row < n_rows) {
        const int64_t j = row % KV;
        const int64_t h = (row / KV) % HK;
        const int64_t sq = row / (KV*HK);
        for (int64_t e = lane; e < DV; e += WARP_SIZE) {
            const float value = __half2float(v[e + j*s1 + h*s2 + sq*s3]);
            norm2 = fmaf(value, value, norm2);
        }
    }
    norm2 = warp_reduce_sum(norm2);
    if (!isfinite(norm2)) {
        norm2 = INFINITY;
    }
    if (lane == 0) {
        warp_norm2[warp] = norm2;
    }
    __syncthreads();

    float block_max_norm2 = lane < WARPS_PER_BLOCK ? warp_norm2[lane] : 0.0f;
    block_max_norm2 = warp_reduce_max(block_max_norm2);
    if (threadIdx.x == 0) {

        const float bound = isfinite(block_max_norm2) ? block_max_norm2 : INFINITY;
        atomicMax(max_norm2_bits, __float_as_uint(bound));
    }
}

template <typename T>
static __global__ void fattn_back_to_f32(const T * x, float * y, const int64_t D, const int64_t KV, const int64_t H, const int64_t SEQ,
        const int64_t s1, const int64_t s2, const int64_t s3) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= D*KV*H*SEQ) {
        return;
    }
    const int64_t e = idx % D, j = (idx / D) % KV, h = (idx / (D*KV)) % H, sq = idx / (D*KV*H);
    y[idx] = fb_load(x, e + j*s1 + h*s2 + sq*s3);
}

template <typename T>
static __global__ void fattn_back_to_f16_strided(const T * x, half * y,
        const int64_t D, const int64_t KV, const int64_t H, const int64_t SEQ,
        const int64_t s1, const int64_t s2, const int64_t s3) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= D*KV*H*SEQ) {
        return;
    }
    const int64_t e = idx % D, j = (idx / D) % KV, h = (idx / (D*KV)) % H, sq = idx / (D*KV*H);
    y[idx] = __float2half_rn(fb_load(x, e + j*s1 + h*s2 + sq*s3));
}

template <typename T>
static __global__ void fattn_back_to_f16_linear(const T * x, half * y, const int64_t n, const float * scale) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = __float2half_rn(fb_load(x, idx) * *scale);
    }
}

static __global__ void fattn_back_init_gemm_scalars(float * scalars) {
    scalars[0] = 1.0f;
    scalars[1] = 0.0f;
}

static __global__ void fattn_back_compute_dP_scale(
        const unsigned int * max_v_norm2_bits, const unsigned int * d_absmax_bits,
        const float * d_scales, float * dP_convert_scale, float * alpha_dqdk,
        const int64_t DV, const float attn_scale) {
    const float max_v_norm = sqrtf(__uint_as_float(*max_v_norm2_bits));
    const float max_d_norm_bound = sqrtf((float) DV) * __uint_as_float(*d_absmax_bits);

    const float max_dP = 2.0f * max_v_norm * max_d_norm_bound * fabsf(attn_scale);
    int exponent = 0;
    if (max_dP > 0.0f && isfinite(max_dP)) {

        exponent = 13 - ilogbf(max_dP);
        exponent = exponent < -126 ? -126 : exponent;
        exponent = exponent > 126 ? 126 : exponent;
    } else if (!isfinite(max_dP)) {
        exponent = -126;
    }
    const float dP_scale = ldexpf(1.0f, exponent);
    const float dP_inv_scale = ldexpf(1.0f, -exponent);

    *dP_convert_scale = d_scales[1] * dP_scale;
    *alpha_dqdk = dP_inv_scale;
}

static void fattn_back_gemm_strided_batched(
        cublasHandle_t handle, const bool use_f16,
        const cublasOperation_t transa, const cublasOperation_t transb,
        const int m, const int n, const int k,
        const float * alpha,
        const void * A, const int lda, const long long strideA,
        const void * B, const int ldb, const long long strideB,
        const float * beta,
        float * C, const int ldc, const long long strideC,
        const int batch_count) {
    if (use_f16) {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, transa, transb,
            m, n, k,
            alpha, A, CUDA_R_16F, lda, strideA,
                   B, CUDA_R_16F, ldb, strideB,
            beta,  C, CUDA_R_32F, ldc, strideC,
            batch_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, transa, transb,
            m, n, k,
            alpha, (const float *) A, lda, strideA,
                   (const float *) B, ldb, strideB,
            beta, C, ldc, strideC, batch_count));
    }
}

static void fattn_back_gemm(ggml_backend_cuda_context & ctx, const fattn_back_params & P, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * o = dst->src[5];
    const ggml_tensor * d = dst->src[6];

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const ggml_prec back_prec = ggml_flash_attn_ext_get_back_prec(o);
    const bool wants_f16 = back_prec == GGML_PREC_Q8 || back_prec == GGML_PREC_Q4 ||
                           back_prec == GGML_PREC_F16 || back_prec == GGML_PREC_BF16;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    const bool use_f16 = wants_f16 && fp16_mma_hardware_available(cc);

    ggml_cuda_pool_alloc<unsigned int> d_absmax(ctx.pool());
    ggml_cuda_pool_alloc<float> d_scales(ctx.pool());
    if (use_f16) {
        d_absmax.alloc(1);
        d_scales.alloc(2);

        ggml_cuda_f16_scale_init(d, d_absmax.ptr, d_scales.ptr, stream);
    }

    float * out = (float *) dst->data;
    float * dq = out;
    float * dk = out + ggml_flash_attn_back_offset(dst, 1);
    float * dv = out + ggml_flash_attn_back_offset(dst, 2);
    float * ds = out + ggml_flash_attn_back_offset(dst, 3);
    CUDA_CHECK(cudaMemsetAsync(out, 0, ggml_nbytes(dst), stream));

    const int64_t SEQK = P.SEQ / P.rk3;
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool()), v_f32(ctx.pool());
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool()), v_f16(ctx.pool());
    const void * k_data = k->data;
    const void * v_data = v->data;
    int64_t k1 = P.k1, k2 = P.k2, k3 = P.k3;
    int64_t v1 = P.v1, v2 = P.v2, v3 = P.v3;

    if (use_f16) {
        if (k->type != GGML_TYPE_F16) {
            const int64_t nk = P.DK*P.KV*P.HK*SEQK;
            const int64_t nv = P.DV*P.KV*P.HK*SEQK;
            k_f16.alloc(nk);
            v_f16.alloc(nv);
            fattn_back_to_f16_strided<float><<<(nk + 255)/256, 256, 0, stream>>>(
                (const float *) k->data, k_f16.get(), P.DK, P.KV, P.HK, SEQK, P.k1, P.k2, P.k3);
            fattn_back_to_f16_strided<float><<<(nv + 255)/256, 256, 0, stream>>>(
                (const float *) v->data, v_f16.get(), P.DV, P.KV, P.HK, SEQK, P.v1, P.v2, P.v3);
            k_data = k_f16.get();
            v_data = v_f16.get();
            k1 = P.DK; k2 = P.DK*P.KV; k3 = P.DK*P.KV*P.HK;
            v1 = P.DV; v2 = P.DV*P.KV; v3 = P.DV*P.KV*P.HK;
        }
    } else {
        if (k->type != GGML_TYPE_F32) {
            const int64_t nk = P.DK*P.KV*P.HK*SEQK;
            const int64_t nv = P.DV*P.KV*P.HK*SEQK;
            k_f32.alloc(nk);
            v_f32.alloc(nv);
            fattn_back_to_f32<half><<<(nk + 255)/256, 256, 0, stream>>>(
                (const half *) k->data, k_f32.get(), P.DK, P.KV, P.HK, SEQK, P.k1, P.k2, P.k3);
            fattn_back_to_f32<half><<<(nv + 255)/256, 256, 0, stream>>>(
                (const half *) v->data, v_f32.get(), P.DV, P.KV, P.HK, SEQK, P.v1, P.v2, P.v3);
            k_data = k_f32.get();
            v_data = v_f32.get();
            k1 = P.DK; k2 = P.DK*P.KV; k3 = P.DK*P.KV*P.HK;
            v1 = P.DV; v2 = P.DV*P.KV; v3 = P.DV*P.KV*P.HK;
        }
    }

    ggml_cuda_pool_alloc<half> q_f16(ctx.pool()), d_f16(ctx.pool());
    const void * q_data = q->data;
    const void * d_data = d->data;
    int64_t q1 = P.q1, q2 = P.q2, q3 = P.q3;
    if (use_f16) {
        const int64_t nq = P.DK*P.N*P.HQ*P.SEQ;
        const int64_t nd = P.DV*P.HQ*P.N*P.SEQ;
        q_f16.alloc(nq);
        d_f16.alloc(nd);
        fattn_back_to_f16_strided<float><<<(nq + 255)/256, 256, 0, stream>>>(
            (const float *) q->data, q_f16.get(), P.DK, P.N, P.HQ, P.SEQ, P.q1, P.q2, P.q3);
        fattn_back_to_f16_linear<float><<<(nd + 255)/256, 256, 0, stream>>>(
            (const float *) d->data, d_f16.get(), nd, d_scales.ptr);
        CUDA_CHECK(cudaGetLastError());
        q_data = q_f16.get();
        d_data = d_f16.get();
        q1 = P.DK; q2 = P.DK*P.N; q3 = P.DK*P.N*P.HQ;
    }

    const int64_t R = std::max<int64_t>(1, std::min<int64_t>(P.N, (int64_t(32) << 20) / std::max<int64_t>(1, P.KV*P.HQ)));
    ggml_cuda_pool_alloc<float> S(ctx.pool(), P.KV*R*P.HQ), dP(ctx.pool(), P.KV*R*P.HQ);
    ggml_cuda_pool_alloc<half> S_f16(ctx.pool()), dP_f16(ctx.pool());
    ggml_cuda_pool_alloc<unsigned int> max_v_norm2(ctx.pool());
    ggml_cuda_pool_alloc<float> dP_convert_scale(ctx.pool());
    ggml_cuda_pool_alloc<float> gemm_scalars(ctx.pool());
    if (use_f16) {
        S_f16.alloc(P.KV*R*P.HQ);
        dP_f16.alloc(P.KV*R*P.HQ);
        max_v_norm2.alloc(1);
        dP_convert_scale.alloc(1);

        gemm_scalars.alloc(3);
        fattn_back_init_gemm_scalars<<<1, 1, 0, stream>>>(gemm_scalars.ptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(max_v_norm2.ptr, 0, sizeof(unsigned int), stream));
        const int64_t n_v_rows = P.KV*P.HK*SEQK;
        const int n_v_blocks = (int) ((n_v_rows + 7) / 8);
        fattn_back_max_v_norm2<<<n_v_blocks, 256, 0, stream>>>(
            (const half *) v_data, P.DV, P.KV, P.HK, SEQK, v1, v2, v3, max_v_norm2.ptr);
        CUDA_CHECK(cudaGetLastError());

        fattn_back_compute_dP_scale<<<1, 1, 0, stream>>>(
            max_v_norm2.ptr, d_absmax.ptr, d_scales.ptr, dP_convert_scale.ptr,
            gemm_scalars.ptr + 2, P.DV, P.scale);
        CUDA_CHECK(cudaGetLastError());
    }
    ggml_cuda_pool_alloc<float> Lbuf(ctx.pool(), R*P.HQ), Dbuf(ctx.pool(), R*P.HQ);
    const float host_one = 1.0f, host_zero = 0.0f;
    const float * alpha_one = use_f16 ? gemm_scalars.ptr : &host_one;
    const float * beta_zero = use_f16 ? gemm_scalars.ptr + 1 : &host_zero;
    const float * beta_one = alpha_one;
    const float * alpha_dv = use_f16 ? d_scales.ptr + 1 : &host_one;
    const float * alpha_dqdk = use_f16 ? gemm_scalars.ptr + 2 : &host_one;
    cublasPointerMode_t old_pointer_mode;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &old_pointer_mode));
    CUBLAS_CHECK(cublasSetPointerMode(handle,
        use_f16 ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST));
    const float * od = (const float *) o->data;
    const float * dd = (const float *) d->data;

    auto offset = [use_f16](const void * ptr, int64_t elements) -> const void * {
        return use_f16
            ? (const void *) ((const half *) ptr + elements)
            : (const void *) ((const float *) ptr + elements);
    };

    for (int64_t iq3 = 0; iq3 < P.SEQ; ++iq3) {
        const int64_t ik3 = iq3 / P.rk3;
        for (int64_t q0 = 0; q0 < P.N; q0 += R) {
            const int64_t rows = std::min(R, P.N - q0);
            for (int64_t g = 0; g < P.rk2; ++g) {
                fattn_back_gemm_strided_batched(handle, use_f16, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int) P.KV, (int) rows, (int) P.DK,
                    alpha_one, offset(k_data, ik3*k3), (int) k1, (long long) k2,
                          offset(q_data, iq3*q3 + q0*q1 + g*q2), (int) q1, (long long) (P.rk2*q2),
                    beta_zero, S.get() + g*P.KV*R, (int) P.KV, (long long) (P.rk2*P.KV*R), (int) P.HK);
            }
            fattn_back_gemm_rows<<<dim3(rows, P.HQ), 256, 0, stream>>>(P, S.get(), use_f16 ? S_f16.get() : nullptr,
                mask ? (const half *) mask->data : nullptr, sinks ? (const float *) sinks->data : nullptr,
                od, dd, use_f16 ? d_f16.get() : nullptr,
                use_f16 ? d_scales.ptr + 1 : nullptr,
                Lbuf.get(), Dbuf.get(), ds, q0, R, iq3);
            for (int64_t g = 0; g < P.rk2; ++g) {
                const void * dO = offset(d_data, iq3*P.o3 + q0*P.o2 + g*P.DV);
                fattn_back_gemm_strided_batched(handle, use_f16, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int) P.KV, (int) rows, (int) P.DV,
                    alpha_one, offset(v_data, ik3*v3), (int) v1, (long long) v2,
                          dO, (int) P.o2, (long long) (P.rk2*P.DV),
                    beta_zero, dP.get() + g*P.KV*R, (int) P.KV, (long long) (P.rk2*P.KV*R), (int) P.HK);
                fattn_back_gemm_strided_batched(handle, use_f16, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int) P.DV, (int) P.KV, (int) rows,
                    alpha_dv, dO, (int) P.o2, (long long) (P.rk2*P.DV),
                          use_f16 ? (const void *) (S_f16.get() + g*P.KV*R) : (const void *) (S.get() + g*P.KV*R),
                          (int) P.KV, (long long) (P.rk2*P.KV*R),
                    beta_one, dv + ik3*P.HK*P.KV*P.DV, (int) P.DV, (long long) (P.KV*P.DV), (int) P.HK);
            }
            const int64_t n = P.KV*R*P.HQ;
            fattn_back_gemm_ds<<<(n + 255)/256, 256, 0, stream>>>(P, S.get(), dP.get(),
                use_f16 ? dP_f16.get() : nullptr,
                use_f16 ? dP_convert_scale.ptr : nullptr,
                mask ? (const half *) mask->data : nullptr, Lbuf.get(), Dbuf.get(),
                q0, R, rows, iq3);
            CUDA_CHECK(cudaGetLastError());
            for (int64_t g = 0; g < P.rk2; ++g) {
                const void * dP_input = use_f16
                    ? (const void *) (dP_f16.get() + g*P.KV*R)
                    : (const void *) (dP.get() + g*P.KV*R);
                fattn_back_gemm_strided_batched(handle, use_f16, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int) P.DK, (int) rows, (int) P.KV,
                    alpha_dqdk, offset(k_data, ik3*k3), (int) k1, (long long) k2,
                          dP_input, (int) P.KV, (long long) (P.rk2*P.KV*R),
                    beta_zero, dq + (iq3*P.HQ + g)*P.N*P.DK + q0*P.DK, (int) P.DK,
                    (long long) (P.rk2*P.N*P.DK), (int) P.HK);
                fattn_back_gemm_strided_batched(handle, use_f16, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int) P.DK, (int) P.KV, (int) rows,
                    alpha_dqdk, offset(q_data, iq3*q3 + q0*q1 + g*q2), (int) q1, (long long) (P.rk2*q2),
                          dP_input, (int) P.KV, (long long) (P.rk2*P.KV*R),
                    beta_one, dk + ik3*P.HK*P.KV*P.DK, (int) P.DK, (long long) (P.KV*P.DK), (int) P.HK);
            }
        }
    }
    CUBLAS_CHECK(cublasSetPointerMode(handle, old_pointer_mode));
}

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    return q->type == GGML_TYPE_F32 && k->type == v->type && (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) &&
        k->ne[0] <= FATTN_BACK_MAX_CHUNKS*WARP_SIZE && v->ne[0] <= FATTN_BACK_MAX_CHUNKS*WARP_SIZE &&
        q->nb[0] == sizeof(float) && k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
        (!mask || (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask))) &&
        ggml_is_contiguous(dst->src[6]) && dst->src[5]->nb[0] == sizeof(float);
}

void ggml_cuda_op_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * o = dst->src[5];
    GGML_ASSERT(ggml_cuda_flash_attn_back_supported(dst));
    GGML_ASSERT(ggml_are_same_stride(o, dst->src[6]) || ggml_is_contiguous(o));

    fattn_back_params P;
    P.DK = k->ne[0]; P.DV = v->ne[0]; P.N = q->ne[1]; P.KV = k->ne[1];
    P.HQ = q->ne[2]; P.HK = k->ne[2]; P.SEQ = q->ne[3];
    P.rk2 = q->ne[2]/k->ne[2]; P.rk3 = q->ne[3]/k->ne[3];
    const size_t tk = ggml_type_size(k->type), tv = ggml_type_size(v->type), fs = sizeof(float);
    P.q1 = q->nb[1]/fs; P.q2 = q->nb[2]/fs; P.q3 = q->nb[3]/fs;
    P.k1 = k->nb[1]/tk; P.k2 = k->nb[2]/tk; P.k3 = k->nb[3]/tk;
    P.v1 = v->nb[1]/tv; P.v2 = v->nb[2]/tv; P.v3 = v->nb[3]/tv;
    P.o1 = P.DV; P.o2 = P.DV*P.HQ; P.o3 = P.DV*P.HQ*P.N;
    GGML_ASSERT(o->nb[1] == P.o1*fs && o->nb[2] == P.o2*fs && o->nb[3] == P.o3*fs);
    P.m1 = mask ? mask->nb[1]/sizeof(half) : 0; P.m2 = mask ? mask->nb[2]/sizeof(half) : 0; P.m3 = mask ? mask->nb[3]/sizeof(half) : 0;
    P.m_ne2 = mask ? mask->ne[2] : 1; P.m_ne3 = mask ? mask->ne[3] : 1;
    P.scale    = ggml_get_op_params_f32(dst, 0);
    P.max_bias = ggml_get_op_params_f32(dst, 1);
    P.softcap  = ggml_get_op_params_f32(dst, 2);
    P.scale_qk = P.softcap != 0.0f ? P.scale/P.softcap : P.scale;
    P.n_head_log2 = 1u << (uint32_t) floorf(log2f((float) P.HQ));
    P.m0  = powf(2.0f, -(P.max_bias       )/P.n_head_log2);
    P.m1f = powf(2.0f, -(P.max_bias/2.0f)/P.n_head_log2);

    static const bool naive = getenv("GGML_CUDA_FA_BACK_NAIVE") != nullptr;
    if (!naive) {
        fattn_back_gemm(ctx, P, dst);
    } else if (k->type == GGML_TYPE_F16) {
        launch_fattn_back<half, half>(ctx, P, dst);
    } else {
        launch_fattn_back<float, float>(ctx, P, dst);
    }
}
