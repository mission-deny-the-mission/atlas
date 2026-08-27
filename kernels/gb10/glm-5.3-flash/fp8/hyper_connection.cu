// SPDX-License-Identifier: AGPL-3.0-only

// GLM uses the shared mHC site operators with the reference Sinkhorn endpoint,
// then a plain mean rather than DeepSeek-V4's learned HC head.
#define HC_REFERENCE_SINKHORN 1
#include "../../deepseek-v4-flash/nvfp4/hyper_connection.cu"

extern "C" __global__ void hc_mean(
    const float* __restrict__ streams,
    __nv_bfloat16* __restrict__ output,
    const unsigned int hidden_size,
    const unsigned int hc_mult
) {
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const float* input = streams + (size_t)token * hc_mult * hidden_size;
    __nv_bfloat16* out = output + (size_t)token * hidden_size;
    for (unsigned int dim = tid; dim < hidden_size; dim += blockDim.x) {
        float sum = 0.f;
        for (unsigned int stream = 0; stream < hc_mult; ++stream) {
            sum += input[(size_t)stream * hidden_size + dim];
        }
        out[dim] = __float2bfloat16(sum / (float)hc_mult);
    }
}
