// Sentinel custom CUDA kernels — Fertile / DoBu hot paths (f32).
// Compiled to PTX by build.rs (nvcc -ptx).

extern "C" __global__ void sentinel_silu_mul_f32(
    float *out,
    const float *gate,
    const float *up,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = gate[i];
    // silu(x) = x * sigmoid(x)
    float s = 1.f / (1.f + expf(-x));
    out[i] = (x * s) * up[i];
}

// Row-wise scatter-add: out[idx[i], :] += values[i, :]
// idx: (n,) int32, values: (n, h), out: (t, h)
extern "C" __global__ void sentinel_scatter_add_rows_f32(
    float *out,
    const int *idx,
    const float *values,
    unsigned int n,
    unsigned int h
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    if (row >= n) return;
    int dest = idx[row];
    if (dest < 0) return;
    // grid-stride over hidden dim
    for (unsigned int c = col; c < h; c += blockDim.x) {
        atomicAdd(&out[(unsigned int)dest * h + c], values[row * h + c]);
    }
}

// RMSNorm: out[b, d] = x[b,d] * rsqrt(mean(x^2)+eps) * weight[d]
extern "C" __global__ void sentinel_rmsnorm_f32(
    float *out,
    const float *x,
    const float *weight,
    float eps,
    unsigned int rows,
    unsigned int dim
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    // Shared reduction for sum of squares
    extern __shared__ float shared[];
    float local = 0.f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x[row * dim + i];
        local += v * v;
    }
    shared[threadIdx.x] = local;
    __syncthreads();

    // Tree reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) {
        float w = weight ? weight[i] : 1.f;
        out[row * dim + i] = x[row * dim + i] * inv_rms * w;
    }
}

// Elementwise scale: out = x * scale
extern "C" __global__ void sentinel_scale_f32(
    float *out,
    const float *x,
    float scale,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * scale;
}

// Fused bias-add after GEMM-like buffer: out[i] = x[i] + bias[i % bias_len]
extern "C" __global__ void sentinel_bias_add_f32(
    float *out,
    const float *x,
    const float *bias,
    unsigned int n,
    unsigned int bias_len
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] + bias[i % bias_len];
}
