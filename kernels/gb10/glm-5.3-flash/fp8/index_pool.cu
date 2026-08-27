// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void glm_index_pool_compress(
    const __nv_bfloat16* __restrict__ raw_k,
    const __nv_bfloat16* __restrict__ raw_gate,
    __nv_bfloat16* __restrict__ pooled_k,
    unsigned tokens,
    unsigned dim,
    unsigned kpool) {
    const unsigned pool = blockIdx.x;
    const unsigned channel = threadIdx.x + blockIdx.y * blockDim.x;
    if (channel >= dim) return;
    const unsigned begin = pool * kpool;
    if (begin >= tokens) return;
    const unsigned end = min(begin + kpool, tokens);
    float max_gate = -INFINITY;
    for (unsigned t = begin; t < end; ++t)
        max_gate = fmaxf(max_gate, __bfloat162float(raw_gate[t * dim + channel]));
    float denom = 0.0f;
    float value = 0.0f;
    for (unsigned t = begin; t < end; ++t) {
        const float w = expf(__bfloat162float(raw_gate[t * dim + channel]) - max_gate);
        denom += w;
        value += w * __bfloat162float(raw_k[t * dim + channel]);
    }
    pooled_k[pool * dim + channel] = __float2bfloat16(value / fmaxf(denom, 1e-12f));
}

extern "C" __global__ void glm_index_pool_compress_block(
    const __nv_bfloat16* __restrict__ raw_pool,
    __nv_bfloat16* __restrict__ pooled_pool,
    __nv_bfloat16* __restrict__ pooled_gate,
    unsigned dim,
    unsigned gate_dim,
    unsigned kpool,
    unsigned block_size) {
    const unsigned pool = blockIdx.x;
    const unsigned channel = threadIdx.x + blockIdx.y * blockDim.x;
    const unsigned slots = block_size / kpool;
    if (pool >= slots || channel >= dim) return;
    const unsigned begin = pool * kpool;
    float max_gate = -INFINITY;
    for (unsigned t = begin; t < begin + kpool; ++t) {
        float gate = 0.0f;
        for (unsigned h = 0; h < gate_dim; ++h)
            gate += __bfloat162float(raw_pool[t * (dim + gate_dim) + dim + h]);
        max_gate = fmaxf(max_gate, gate / gate_dim);
    }
    float denom = 0.0f, value = 0.0f;
    for (unsigned t = begin; t < begin + kpool; ++t) {
        float gate = 0.0f;
        for (unsigned h = 0; h < gate_dim; ++h)
            gate += __bfloat162float(raw_pool[t * (dim + gate_dim) + dim + h]);
        const float w = expf(gate / gate_dim - max_gate);
        denom += w;
        value += w * __bfloat162float(raw_pool[t * (dim + gate_dim) + channel]);
    }
    pooled_pool[pool * dim + channel] = __float2bfloat16(value / fmaxf(denom, 1e-12f));
    if (channel < gate_dim) {
        float gate_sum = 0.0f;
        for (unsigned t = begin; t < begin + kpool; ++t)
            gate_sum += __bfloat162float(raw_pool[t * (dim + gate_dim) + dim + channel]);
        pooled_gate[pool * gate_dim + channel] = __float2bfloat16(gate_sum / kpool);
    }
}
