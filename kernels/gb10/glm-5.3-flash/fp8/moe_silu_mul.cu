// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash uses the checkpoint's SwiGLU limit (10.0) for both dense and
// routed experts. This model-local shadow keeps the shared activation kernel
// from applying an implicit no-limit policy to GLM's gate/up projections.

#include <cuda_bf16.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = __bfloat162float(gate[idx]);
    const float u = fminf(fmaxf(__bfloat162float(up[idx]), -10.0f), 10.0f);
    g = fminf(g, 10.0f);
    const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    output[idx] = __float2bfloat16(g * sigmoid_g * u);
}
