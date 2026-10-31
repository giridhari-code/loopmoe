// Sentinel custom CUDA kernels — f32 / f16 / bf16.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- helpers ----
__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + expf(-x));
}

// ===================== silu_mul =====================

extern "C" __global__ void sentinel_silu_mul_f32(
    float *out, const float *gate, const float *up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = gate[i];
    out[i] = silu_f(x) * up[i];
}

extern "C" __global__ void sentinel_silu_mul_f16(
    __half *out, const __half *gate, const __half *up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __half2float(gate[i]);
    float y = __half2float(up[i]);
    out[i] = __float2half(silu_f(x) * y);
}

extern "C" __global__ void sentinel_silu_mul_bf16(
    __nv_bfloat16 *out, const __nv_bfloat16 *gate, const __nv_bfloat16 *up, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __bfloat162float(gate[i]);
    float y = __bfloat162float(up[i]);
    out[i] = __float2bfloat16(silu_f(x) * y);
}

// ===================== scale =====================

extern "C" __global__ void sentinel_scale_f32(
    float *out, const float *x, float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * scale;
}

extern "C" __global__ void sentinel_scale_f16(
    __half *out, const __half *x, float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(x[i]) * scale);
}

extern "C" __global__ void sentinel_scale_bf16(
    __nv_bfloat16 *out, const __nv_bfloat16 *x, float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(__bfloat162float(x[i]) * scale);
}

// ===================== scatter_add_rows =====================
// out[idx[row], col] += values[row, col]
// idx: int32[n], values: [n, h], out: [t, h]

extern "C" __global__ void sentinel_scatter_add_rows_f32(
    float *out, const int *idx, const float *values,
    unsigned int n, unsigned int h
) {
    unsigned int row = blockIdx.x;
    if (row >= n) return;
    int dest = idx[row];
    if (dest < 0) return;
    for (unsigned int c = threadIdx.x; c < h; c += blockDim.x) {
        atomicAdd(&out[(unsigned int)dest * h + c], values[row * h + c]);
    }
}

// f16: atomic via float bit-width CAS is heavy; accumulate with native atomicAdd on sm_70+
// Using float intermediate atomics into out interpreted carefully — prefer convert path:
// load-add-store with atomicCAS on uint32 representation of __half.
__device__ __forceinline__ void atomicAddHalf(__half *addr, __half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, val);
#else
    // fallback: CAS loop
    unsigned short *addr_as_us = (unsigned short *)addr;
    unsigned short old = *addr_as_us, assumed;
    do {
        assumed = old;
        float sum = __half2float(__ushort_as_half(assumed)) + __half2float(val);
        old = atomicCAS(addr_as_us, assumed, __half_as_ushort(__float2half(sum)));
    } while (assumed != old);
#endif
}

extern "C" __global__ void sentinel_scatter_add_rows_f16(
    __half *out, const int *idx, const __half *values,
    unsigned int n, unsigned int h
) {
    unsigned int row = blockIdx.x;
    if (row >= n) return;
    int dest = idx[row];
    if (dest < 0) return;
    for (unsigned int c = threadIdx.x; c < h; c += blockDim.x) {
        atomicAddHalf(&out[(unsigned int)dest * h + c], values[row * h + c]);
    }
}

// bf16 atomicAdd requires sm_80+
__device__ __forceinline__ void atomicAddBf16(__nv_bfloat16 *addr, __nv_bfloat16 val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, val);
#else
    unsigned short *addr_as_us = (unsigned short *)addr;
    unsigned short old = *addr_as_us, assumed;
    do {
        assumed = old;
        float sum = __bfloat162float(*((__nv_bfloat16 *)&assumed)) + __bfloat162float(val);
        __nv_bfloat16 s = __float2bfloat16(sum);
        old = atomicCAS(addr_as_us, assumed, *((unsigned short *)&s));
    } while (assumed != old);
#endif
}

extern "C" __global__ void sentinel_scatter_add_rows_bf16(
    __nv_bfloat16 *out, const int *idx, const __nv_bfloat16 *values,
    unsigned int n, unsigned int h
) {
    unsigned int row = blockIdx.x;
    if (row >= n) return;
    int dest = idx[row];
    if (dest < 0) return;
    for (unsigned int c = threadIdx.x; c < h; c += blockDim.x) {
        atomicAddBf16(&out[(unsigned int)dest * h + c], values[row * h + c]);
    }
}

// ===================== rmsnorm f32 (reference) =====================

extern "C" __global__ void sentinel_rmsnorm_f32(
    float *out, const float *x, const float *weight,
    float eps, unsigned int rows, unsigned int dim
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float local = 0.f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x[row * dim + i];
        local += v * v;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(shared[0] / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float w = weight ? weight[i] : 1.f;
        out[row * dim + i] = x[row * dim + i] * inv_rms * w;
    }
}
