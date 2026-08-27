// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// GLM IndexPool score. Candidate keys are reconstructed from the block-table
// aligned raw [K|gate] sidecar so selection remains correct across prefix-shared
// physical KV blocks. `weights` is query-local (weights_proj(hidden)).
extern "C" __global__ void glm_index_pool_select_weighted(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ raw_pool,
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ ape,
    const unsigned* __restrict__ block_table,
    int* __restrict__ selected,
    unsigned candidates,
    unsigned block_size,
    unsigned raw_block_stride_bytes,
    unsigned heads,
    unsigned dim,
    unsigned kpool,
    unsigned topk) {
    extern __shared__ unsigned char smem[];
    float* scores = reinterpret_cast<float*>(smem);
    int* ids = reinterpret_cast<int*>(scores + topk);
    if (threadIdx.x != 0) return;
    unsigned kept = 0;
    for (unsigned c = 0; c < candidates; ++c) {
        const unsigned first_token = c * kpool;
        const unsigned block = block_table[first_token / block_size];
        const unsigned first_slot = first_token % block_size;
        const __nv_bfloat16* raw = reinterpret_cast<const __nv_bfloat16*>(
            reinterpret_cast<const char*>(raw_pool) + (size_t)block * raw_block_stride_bytes);
        float score = 0.0f;
        for (unsigned h = 0; h < heads; ++h) {
            float dot = 0.0f;
            for (unsigned d = 0; d < dim; ++d) {
                float max_gate = -INFINITY;
                for (unsigned t = 0; t < kpool; ++t)
                    max_gate = fmaxf(max_gate, __bfloat162float(raw[(first_slot + t) * 2 * dim + dim + d]) +
                        __bfloat162float(ape[t * dim + d]));
                float denom = 0.f, pooled = 0.f;
                for (unsigned t = 0; t < kpool; ++t) {
                    const float gate = __bfloat162float(raw[(first_slot + t) * 2 * dim + dim + d]) +
                        __bfloat162float(ape[t * dim + d]);
                    const float w = expf(gate - max_gate);
                    denom += w;
                    pooled += w * __bfloat162float(raw[(first_slot + t) * 2 * dim + d]);
                }
                dot += __bfloat162float(query[h * dim + d]) * pooled / fmaxf(denom, 1.e-12f);
            }
            score += fmaxf(dot * rsqrtf((float)dim), 0.0f) *
                __bfloat162float(weights[h]) * rsqrtf((float)heads);
        }
        unsigned pos = kept < topk ? kept : topk - 1;
        if (kept < topk) ++kept;
        else if (score <= scores[pos]) continue;
        while (pos > 0 && score > scores[pos - 1]) {
            scores[pos] = scores[pos - 1];
            ids[pos] = ids[pos - 1];
            --pos;
        }
        scores[pos] = score;
        ids[pos] = static_cast<int>(c);
    }
    for (unsigned i = 0; i < topk; ++i) selected[i] = i < kept ? ids[i] : -1;
}
