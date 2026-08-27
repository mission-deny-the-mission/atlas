// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Write raw GLM index K+gate entries using the same physical slot mapping as
// the paged KV cache. K and gate are separate BF16 vectors; the sidecar packs
// them as [K(128) | gate(128)].
extern "C" __global__ void glm_index_pool_write_raw(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ gate,
    const long long* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ index_pool,
    unsigned tokens,
    unsigned dim,
    unsigned block_size,
    unsigned raw_block_stride_bytes) {
    const unsigned token = blockIdx.y;
    const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens || channel >= 2 * dim) return;
    const long long slot = slot_mapping[token];
    if (slot < 0) return;
    const unsigned block = static_cast<unsigned>(slot / block_size);
    const unsigned offset = static_cast<unsigned>(slot % block_size);
    const size_t byte_offset = static_cast<size_t>(block) * raw_block_stride_bytes
        + static_cast<size_t>(offset) * 2 * dim * sizeof(__nv_bfloat16);
    index_pool[byte_offset / sizeof(__nv_bfloat16) + channel] =
        channel < dim ? key[token * dim + channel] : gate[token * dim + channel - dim];
}

extern "C" __global__ void glm_index_key_layer_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned dim,
    float eps) {
    __shared__ float sum[256];
    __shared__ float sumsq[256];
    const unsigned tid = threadIdx.x;
    float xsum = 0.f, xsumsq = 0.f;
    for (unsigned d = tid; d < dim; d += blockDim.x) {
        const float x = __bfloat162float(input[d]);
        xsum += x;
        xsumsq += x * x;
    }
    sum[tid] = xsum;
    sumsq[tid] = xsumsq;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { sum[tid] += sum[tid + stride]; sumsq[tid] += sumsq[tid + stride]; }
        __syncthreads();
    }
    const float mean = sum[0] / dim;
    const float inv_std = rsqrtf(fmaxf(sumsq[0] / dim - mean * mean, 0.f) + eps);
    for (unsigned d = tid; d < dim; d += blockDim.x)
        output[d] = __float2bfloat16((__bfloat162float(input[d]) - mean) * inv_std *
            __bfloat162float(weight[d]) + __bfloat162float(bias[d]));
}
