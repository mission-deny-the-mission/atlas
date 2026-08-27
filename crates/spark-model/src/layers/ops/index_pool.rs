// SPDX-License-Identifier: AGPL-3.0-only

//! GLM IndexPool compression dispatch.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn index_pool_compress(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw_k: DevicePtr,
    raw_gate: DevicePtr,
    pooled_k: DevicePtr,
    tokens: u32,
    dim: u32,
    kpool: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens.div_ceil(kpool), dim.div_ceil(256), 1])
        .block([256, 1, 1])
        .arg_ptr(raw_k)
        .arg_ptr(raw_gate)
        .arg_ptr(pooled_k)
        .arg_u32(tokens)
        .arg_u32(dim)
        .arg_u32(kpool)
        .launch(stream)
}

pub fn index_pool_compress_block(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw_pool: DevicePtr,
    pooled_pool: DevicePtr,
    pooled_gate: DevicePtr,
    dim: u32,
    gate_dim: u32,
    kpool: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        dim > 0 && gate_dim > 0 && kpool > 0 && block_size > 0 && block_size.is_multiple_of(kpool)
    );
    KernelLaunch::new(gpu, kernel)
        .grid([block_size / kpool, dim.div_ceil(256), 1])
        .block([256, 1, 1])
        .arg_ptr(raw_pool)
        .arg_ptr(pooled_pool)
        .arg_ptr(pooled_gate)
        .arg_u32(dim)
        .arg_u32(gate_dim)
        .arg_u32(kpool)
        .arg_u32(block_size)
        .launch(stream)
}
