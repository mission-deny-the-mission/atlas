// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Reference IndexPool selector. One CTA scans the compressed candidates for a
// query and keeps the highest-scoring pools in shared memory. The production
// implementation can replace this with a multi-stage reduction without
// changing the ABI or selected-index layout.
extern "C" __global__ void glm_index_pool_select(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ pooled_k,
    int* __restrict__ selected,
    unsigned candidates,
    unsigned dim,
    unsigned topk) {
    extern __shared__ unsigned char smem[];
    float* scores = reinterpret_cast<float*>(smem);
    int* ids = reinterpret_cast<int*>(scores + topk);
    if (threadIdx.x != 0) return;
    unsigned kept = 0;
    for (unsigned c = 0; c < candidates; ++c) {
        float score = 0.0f;
        for (unsigned d = 0; d < dim; ++d)
            score += __bfloat162float(query[d]) *
                     __bfloat162float(pooled_k[c * dim + d]);
        unsigned pos = kept < topk ? kept : topk - 1;
        if (kept < topk) ++kept;
        else if (score <= scores[pos]) continue;
        while (pos > 0 && score > scores[pos - 1]) {
            scores[pos] = scores[pos - 1]; ids[pos] = ids[pos - 1]; --pos;
        }
        scores[pos] = score; ids[pos] = static_cast<int>(c);
    }
    for (unsigned i = 0; i < topk; ++i) selected[i] = i < kept ? ids[i] : -1;
}
