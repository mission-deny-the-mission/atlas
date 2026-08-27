// SPDX-License-Identifier: AGPL-3.0-only

//! GLM IndexPool candidate selection dispatch.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub fn index_pool_select(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    pooled_k: DevicePtr,
    selected: DevicePtr,
    candidates: u32,
    dim: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        topk > 0 && topk <= 512,
        "GLM IndexPool selector top-k must be 1..=512"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .shared_mem(topk * 8)
        .arg_ptr(query)
        .arg_ptr(pooled_k)
        .arg_ptr(selected)
        .arg_u32(candidates)
        .arg_u32(dim)
        .arg_u32(topk)
        .launch(stream)
}
