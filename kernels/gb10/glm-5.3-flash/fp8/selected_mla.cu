// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

// Reference selected-token latent MLA attention. Each CTA handles one query
// head and accumulates over the IndexPool-selected logical token positions.
// The output remains in latent KV space; the existing W_UV/O projection path
// can consume it unchanged.
extern "C" __global__ void glm_selected_mla_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const unsigned* __restrict__ block_table,
    const int* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    unsigned selected_count,
    unsigned block_size,
    unsigned kv_lora,
    unsigned num_heads,
    float inv_sqrt_d) {
    const unsigned head = blockIdx.x;
    if (head >= num_heads || threadIdx.x != 0) return;
    float denom = 0.0f;
    float max_score = -INFINITY;
    float acc[512];
    for (unsigned d = 0; d < kv_lora && d < 512; ++d) acc[d] = 0.0f;
    for (unsigned i = 0; i < selected_count; ++i) {
        const int logical = token_indices[i];
        if (logical < 0) continue;
        const unsigned token = static_cast<unsigned>(logical);
        const unsigned block = block_table[token / block_size];
        const unsigned slot = token % block_size;
        const size_t off = (static_cast<size_t>(block) * block_size + slot) * kv_lora;
        float score = 0.0f;
        for (unsigned d = 0; d < kv_lora; ++d)
            score += __bfloat162float(q[head * kv_lora + d]) *
                     __bfloat162float(k_pool[off + d]);
        score *= inv_sqrt_d;
        if (score > max_score) {
            const float scale = isfinite(max_score) ? expf(max_score - score) : 0.0f;
            denom *= scale;
            for (unsigned d = 0; d < kv_lora && d < 512; ++d) acc[d] *= scale;
            max_score = score;
        }
        const float weight = expf(score - max_score);
        denom += weight;
        for (unsigned d = 0; d < kv_lora && d < 512; ++d)
            acc[d] += weight * __bfloat162float(v_pool[off + d]);
    }
    for (unsigned d = 0; d < kv_lora && d < 512; ++d)
        output[head * kv_lora + d] = __float2bfloat16(acc[d] / fmaxf(denom, 1e-12f));
}
