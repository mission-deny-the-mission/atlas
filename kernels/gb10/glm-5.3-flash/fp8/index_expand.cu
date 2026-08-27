// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_runtime.h>

// Expand selected pool IDs into logical token positions. The final
// `tail_count` positions cover the incomplete causal pool at the end of the
// sequence. Invalid slots are filled with -1 for the selected-attention
// gather kernel to skip.
extern "C" __global__ void glm_index_pool_expand(
    const int* __restrict__ selected_pools,
    int* __restrict__ token_indices,
    unsigned selected_count,
    unsigned kpool,
    unsigned seq_len,
    unsigned tail_count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned width = selected_count * kpool + tail_count;
    if (i >= width) return;
    if (i < selected_count * kpool) {
        const unsigned pool = i / kpool;
        const int pool_id = selected_pools[pool];
        token_indices[i] = pool_id < 0
            ? -1
            : pool_id * static_cast<int>(kpool) + static_cast<int>(i % kpool);
    } else {
        const unsigned tail = i - selected_count * kpool;
        const unsigned start = seq_len >= tail_count ? seq_len - tail_count : 0;
        token_indices[i] = (start + tail < seq_len) ? static_cast<int>(start + tail) : -1;
    }
}
