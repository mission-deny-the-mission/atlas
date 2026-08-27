// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3 Flash Kimi Delta Attention decode kernels.
//
// The decay is intentionally per key dimension.  This is not Qwen GDN's
// scalar-per-head gate: `f_b` and `dt_bias` produce one log-decay for every
// key component.  State is `[head, key_dim, value_dim]` in FP32.

#include <cuda_bf16.h>

namespace {
constexpr unsigned int kBlock = 128;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? shared[lane] : 0.f;
    if (warp == 0) value = warp_sum(value);
    return __shfl_sync(0xffffffff, value, 0);
}
}  // namespace

// One CTA processes one KDA head for one decode row.  `qkv` is the sequential
// `[Q | K | V]` layout after GLM's three causal convolutions.  `forget` is
// `f_b(f_a(x))`; `beta_input` is `b_proj(x)`.
extern "C" __global__ void kda_recurrent_decode(
    float* __restrict__ state,
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ forget,
    const __nv_bfloat16* __restrict__ beta_input,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float lower_bound,
    const unsigned int has_lower_bound
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int v = threadIdx.x;
    if (head >= num_heads || v >= head_dim) return;

    const unsigned int width = num_heads * head_dim;
    const unsigned long long row_base = (unsigned long long)row * width;
    const unsigned int base = head * head_dim;
    const __nv_bfloat16* q = qkv + row_base + base;
    const __nv_bfloat16* k = qkv + row_base + width + base;
    const __nv_bfloat16* value = qkv + row_base + 2ull * width + base;
    const __nv_bfloat16* f = forget + row_base + base;
    const __nv_bfloat16* b = beta_input + (unsigned long long)row * num_heads + head;
    float* h = state + ((unsigned long long)row * num_heads + head) * head_dim * head_dim;

    __shared__ float q_shared[kBlock];
    __shared__ float k_shared[kBlock];
    __shared__ float reduction[4];
    __shared__ float q_inv_norm;
    __shared__ float k_inv_norm;
    __shared__ float beta;

    const float qv = static_cast<float>(q[v]);
    const float kv = static_cast<float>(k[v]);
    const float q_sum = block_sum(qv * qv, reduction);
    if (v == 0) q_inv_norm = rsqrtf(q_sum + 1.e-6f);
    const float k_sum = block_sum(kv * kv, reduction);
    if (v == 0) {
        k_inv_norm = rsqrtf(k_sum + 1.e-6f);
        beta = 1.f / (1.f + expf(-static_cast<float>(*b)));
    }
    __syncthreads();
    q_shared[v] = qv * q_inv_norm;
    k_shared[v] = kv * k_inv_norm;
    __syncthreads();

    // Apply the per-key-component decay before forming the correction, which
    // is the defining Kimi delta rule difference from a scalar GDN decay.
    const float decay_rate = expf(a_log[head]);
    float memory = 0.f;
    for (unsigned int key = 0; key < head_dim; ++key) {
        const float raw = static_cast<float>(f[key]) + dt_bias[base + key];
        const float log_decay = has_lower_bound
            ? lower_bound / (1.f + expf(-decay_rate * raw))
            : -decay_rate * (raw > 20.f ? raw : log1pf(expf(raw)));
        const float decay = expf(log_decay);
        const unsigned long long index = (unsigned long long)key * head_dim + v;
        h[index] *= decay;
        memory += h[index] * k_shared[key];
    }
    const float delta = (static_cast<float>(value[v]) - memory) * beta;
    float attended = 0.f;
    for (unsigned int key = 0; key < head_dim; ++key) {
        const unsigned long long index = (unsigned long long)key * head_dim + v;
        const float updated = h[index] + k_shared[key] * delta;
        h[index] = updated;
        attended += updated * q_shared[key];
    }
    output[row_base + base + v] = __float2bfloat16(attended * rsqrtf((float)head_dim));
}

// GLM applies weighted RMSNorm before sigmoid gating.  One CTA handles one
// output head; `weight` is shared across the heads.
extern "C" __global__ void kda_gated_rms_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float eps
) {
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (head >= num_heads || dim >= head_dim) return;
    const unsigned int width = num_heads * head_dim;
    const unsigned long long base = (unsigned long long)row * width + head * head_dim;
    __shared__ float reduction[4];
    __shared__ float inv_rms;
    const float x = static_cast<float>(input[base + dim]);
    const float sum = block_sum(x * x, reduction);
    if (dim == 0) inv_rms = rsqrtf(sum / (float)head_dim + eps);
    __syncthreads();
    const float sigmoid = 1.f / (1.f + expf(-static_cast<float>(gate[base + dim])));
    output[base + dim] = __float2bfloat16(x * inv_rms * static_cast<float>(weight[dim]) * sigmoid);
}
