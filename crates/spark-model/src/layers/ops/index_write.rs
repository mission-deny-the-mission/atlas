// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side GLM IndexPool raw-entry writes.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn index_pool_write_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    key: DevicePtr,
    gate: DevicePtr,
    slot_mapping: DevicePtr,
    index_pool: DevicePtr,
    tokens: u32,
    dim: u32,
    block_size: u32,
    raw_block_stride_bytes: u32,
    stream: u64,
) -> Result<()> {
    ensure!(tokens > 0 && dim > 0 && block_size > 0);
    KernelLaunch::new(gpu, kernel)
        .grid([(2 * dim).div_ceil(256), tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(key)
        .arg_ptr(gate)
        .arg_ptr(slot_mapping)
        .arg_ptr(index_pool)
        .arg_u32(tokens)
        .arg_u32(dim)
        .arg_u32(block_size)
        .arg_u32(raw_block_stride_bytes)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn index_key_layer_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    bias: DevicePtr,
    output: DevicePtr,
    dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        dim > 0 && dim <= 1024,
        "GLM IndexPool LayerNorm dim must be 1..=1024"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(dim)
        .arg_f32(eps)
        .launch(stream)
}
