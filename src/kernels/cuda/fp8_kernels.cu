// FP8 E4M3 kernels for Sentinel / DoBu
#include <math.h>
#include "fp8.cuh"

#define TILE 16

// Quantize f32 → fp8 e4m3 bytes (row-major, same shape)
extern "C" __global__ void sentinel_quantize_fp8_e4m3_f32(
    unsigned char *out, const float *in, float inv_scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = float_to_fp8_e4m3(in[i] * inv_scale);
}

// Dequant fp8 → f32
extern "C" __global__ void sentinel_dequant_fp8_e4m3_f32(
    float *out, const unsigned char *in, float scale, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fp8_e4m3_to_float(in[i]) * scale;
}

// Weight-only GEMM: C[M,N] = A[M,K](f32) @ (fp8_to_float(B[K,N]) * scale)
extern "C" __global__ void sentinel_gemm_nn_fp8w_f32(
    float *C, const float *A, const unsigned char *B,
    float scale, int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? fp8_e4m3_to_float(B[b_row * N + col]) * scale
                : 0.f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// A bf16 activations, B fp8 weights, out bf16
extern "C" __global__ void sentinel_gemm_nn_fp8w_bf16(
    __nv_bfloat16 *C, const __nv_bfloat16 *A, const unsigned char *B,
    float scale, int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __bfloat162float(A[row * K + a_col]) : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? fp8_e4m3_to_float(B[b_row * N + col]) * scale
                : 0.f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = __float2bfloat16(acc);
}
