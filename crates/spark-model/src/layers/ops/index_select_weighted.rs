// SPDX-License-Identifier: AGPL-3.0-only

//! GLM multi-head gated IndexPool selection dispatch.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn index_pool_select_weighted(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    raw_pool: DevicePtr,
    weights: DevicePtr,
    ape: DevicePtr,
    block_table: DevicePtr,
    selected: DevicePtr,
    candidates: u32,
    block_size: u32,
    raw_block_stride_bytes: u32,
    heads: u32,
    dim: u32,
    kpool: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    ensure!(heads > 0 && dim > 0 && kpool > 0 && topk > 0 && topk <= 512 && block_size > 0);
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .shared_mem(topk * 8)
        .arg_ptr(query)
        .arg_ptr(raw_pool)
        .arg_ptr(weights)
        .arg_ptr(ape)
        .arg_ptr(block_table)
        .arg_ptr(selected)
        .arg_u32(candidates)
        .arg_u32(block_size)
        .arg_u32(raw_block_stride_bytes)
        .arg_u32(heads)
        .arg_u32(dim)
        .arg_u32(kpool)
        .arg_u32(topk)
        .launch(stream)
}
